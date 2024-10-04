#reference: https://github.com/NVlabs/AFNO-transformer

import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.img_utils import PeriodicPad2d

def dht2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        # 1D case (input is a 3D tensor)
        D, M, N = x.size()
        N = M  # For 1D case, M and N should be the same size
        n = torch.arange(N, device=x.device).float()

        # Hartley kernel for 1D
        cas = torch.cos(2 * torch.pi * n.view(-1, 1) * n / N) + torch.sin(2 * torch.pi * n.view(-1, 1) * n / N)

        # Perform the DHT
        X = torch.matmul(cas, x.reshape(N, -1))
        return X.reshape(D, M, N)

    elif x.ndim == 4:
        # 2D case (input is a 4D tensor)
        B, D, M, N = x.size()
        m = torch.arange(M, device=x.device).float()
        n = torch.arange(N, device=x.device).float()

        # Hartley kernels for rows and columns
        cas_row = torch.cos(2 * torch.pi * m.view(-1, 1) * m / M) + torch.sin(2 * torch.pi * m.view(-1, 1) * m / M)
        cas_col = torch.cos(2 * torch.pi * n.view(-1, 1) * n / N) + torch.sin(2 * torch.pi * n.view(-1, 1) * n / N)

        # Perform the DHT
        x_reshaped = x.reshape(B * D, M, N)
        intermediate = torch.matmul(x_reshaped, cas_col)
        X = torch.matmul(cas_row, intermediate)
        return X.reshape(B, D, M, N)

    elif x.ndim == 5:
        # 3D case (input is a 5D tensor)
        B, C, D, M, N = x.size()
        d = torch.arange(D, device=x.device).float()
        m = torch.arange(M, device=x.device).float()
        n = torch.arange(N, device=x.device).float()

        # Hartley kernels for depth, rows, and columns
        cas_depth = torch.cos(2 * torch.pi * d.view(-1, 1, 1) * d / D) + torch.sin(2 * torch.pi * d.view(-1, 1, 1) * d / D)
        cas_row = torch.cos(2 * torch.pi * m.view(1, -1, 1) * m / M) + torch.sin(2 * torch.pi * m.view(1, -1, 1) * m / M)
        cas_col = torch.cos(2 * torch.pi * n.view(1, 1, -1) * n / N) + torch.sin(2 * torch.pi * n.view(1, 1, -1) * n / N)

        # Perform the DHT
        x_reshaped = x.reshape(B * C, D, M, N)
        intermediate = torch.einsum('bcde,cfde->bcfe', x_reshaped, cas_col)
        intermediate = torch.einsum('bcfe,cfm->bcme', intermediate, cas_row)
        X = torch.einsum('bcme,cfm->bcme', intermediate, cas_depth)
        return X.reshape(B, C, D, M, N)

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")


def idht2d(x: torch.Tensor) -> torch.Tensor:
    # Compute the DHT
    transformed = dht2d(x)
    
    # Determine normalization factor
    if x.ndim == 3:
        # 1D case (3D tensor input)
        N = x.size(1)  # N is the size of the last dimension
        normalization_factor = N
    elif x.ndim == 4:
        # 2D case (4D tensor input)
        M, N = x.size(2), x.size(3)
        normalization_factor = M * N
    elif x.ndim == 5:
        # 3D case (5D tensor input)
        D, M, N = x.size(2), x.size(3), x.size(4)
        normalization_factor = D * M * N
    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

    # Normalize the transformed result
    return transformed / normalization_factor

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    
    # Perform the convolution using DHT components
    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) - 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) + 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))
    
    return result

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        bias = x
    
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = dht2d(x)
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
    
        X_H_k = x 
        X_H_neg_k = torch.roll(torch.flip(x, dims=[1, 2]), shifts=(1, 1), dims=[1, 2])
    
        block_size = self.block_size
        hidden_size_factor = self.hidden_size_factor
    
        # Ensure o1 and o2 dimensions match the expected sizes
        o1_H_k = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_H_neg_k = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
    
        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
    
        # Reshape and align the dimensions of X_H_k and X_H_neg_k for broadcasting
        X_H_k = X_H_k.reshape(B, H, W, self.num_blocks, block_size)
        X_H_neg_k = X_H_neg_k.reshape(B, H, W, self.num_blocks, block_size)
    
        o1_H_k[:, :, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0])
            ) + self.b1[0]
        )
    
        o1_H_neg_k[:, :, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0])
            ) + self.b1[1]
        )
    
        # Perform second multiplication similar to the first
        o2_H_k = torch.zeros(X_H_k.shape, device=x.device)
        o2_H_neg_k = torch.zeros(X_H_k.shape, device=x.device)
    
        o2_H_k[:, :, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_k[:, :, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[0])
            ) + self.b2[0]
        )
    
        o2_H_neg_k[:, :, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o2_H_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o2_H_k[:, :, :kept_modes], self.w2[0])
            ) + self.b2[1]
        )
    
        # Combine positive and negative frequency components back
        x = o2_H_k + o2_H_neg_k
        x = F.softshrink(x, lambd=self.sparsity_threshold)
    
        # Transform back to spatial domain (assuming DHT-based iDHT here)
        x = idht2d(x)
        x = x.reshape(B, H, W, C)
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x.real + bias.real
        
class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AFNONet(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    model = AFNONet(img_size=(720, 1440), patch_size=(4,4), in_chans=3, out_chans=10)
    sample = torch.randn(1, 3, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))

