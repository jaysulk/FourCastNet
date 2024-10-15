import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def dht2d(x: torch.Tensor, dim=None) -> torch.Tensor:
    result = torch.fft.fftn(x, dim=dim)
    return result.real + result.imag


def idht2d(x: torch.Tensor, dim=None) -> torch.Tensor:
    transformed = dht2d(x, dim=dim)
    if dim is None:
        normalization_factor = x.numel()
    else:
        if isinstance(dim, int):
            dim = [dim]
        normalization_factor = 1
        for d in dim:
            normalization_factor *= x.size(d)
    return transformed / normalization_factor


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

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        X_H_k = dht2d(x)
        X_H_neg_k = torch.roll(torch.flip(x, dims=[1, 2]), shifts=(1, 1), dims=[1, 2])

        block_size = self.block_size
        hidden_size_factor = self.hidden_size_factor

        o1_H_k = torch.zeros([B, H, W, self.num_blocks, self.block_size * hidden_size_factor], device=x.device)
        o1_H_neg_k = torch.zeros([B, H, W, self.num_blocks, self.block_size * hidden_size_factor], device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        X_H_k = X_H_k.reshape(B, H, W, self.num_blocks, block_size)
        X_H_neg_k = X_H_neg_k.reshape(B, H, W, self.num_blocks, block_size)

        # First multiplication for positive and negative frequency components
        o1_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[0])
            ) + self.b1[0]
        )

        o1_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[0])
            ) + self.b1[1]
        )

        # Second multiplication for both positive and negative frequency components
        o2_H_k = torch.zeros(X_H_k.shape, device=x.device)
        o2_H_neg_k = torch.zeros(X_H_k.shape, device=x.device)

        o2_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0])
            ) + self.b2[0]
        )

        o2_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o2_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o2_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0])
            ) + self.b2[1]
        )

        # Combine positive and negative frequency components back
        x = o2_H_k + o2_H_neg_k

        if self.sparsity_threshold > 0:
            x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = x.reshape(B, H, W, C)
        x = idht2d(x)
        x = x.type(dtype)

        return x + bias


class AFNONet(nn.Module):
    def __init__(self, img_size=(720, 1440), patch_size=(16, 16), in_chans=2, out_chans=2, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.blocks = nn.ModuleList([
            Block(embed_dim, mlp_ratio=4.) for _ in range(12)
        ])
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, 'b p e -> b e p')

        for blk in self.blocks:
            x = blk(x)

        x = self.head(x)
        x = rearrange(x, "b p (h w c) -> b c h w", h=self.img_size[0] // self.patch_size[0], w=self.img_size[1] // self.patch_size[1])
        return x


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
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
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


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(720, 1440), patch_size=(16, 16), in_chans=2, embed_dim=768):
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
    model = AFNONet(img_size=(720, 1440), patch_size=(16, 16), in_chans=2, out_chans=2, embed_dim=768)
    sample = torch.randn(1, 2, 720, 1440)  # Shape: [B, D, H, W]
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
