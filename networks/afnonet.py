import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor):
    # Compute the 2D FFT
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    
    # Calculate the Discrete Hartley Transform (DHT) using the real and imaginary parts of the FFT
    H = fft.real - fft.imag
    return H

def idht2d(x: torch.Tensor, H: int, W: int):
    # Perform the inverse DHT by applying the inverse FFT
    # Combine the real and imaginary parts for reconstruction
    real_part = x
    imag_part = torch.zeros_like(x)  # No imaginary part in DHT, but still required for FFT
    complex_x = torch.complex(real_part, imag_part)

    # Apply inverse FFT to reconstruct the original input
    x_reconstructed = torch.fft.ifft2(complex_x, s=(H, W), norm="ortho")
    return x_reconstructed.real

import torch
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor):
    # Compute the 2D FFT
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    
    # Calculate the Discrete Hartley Transform (DHT) using the real and imaginary parts of the FFT
    H = fft.real - fft.imag
    return H

def idht2d(x: torch.Tensor, H: int, W: int):
    # Perform the inverse DHT by applying the inverse FFT
    # Combine the real and imaginary parts for reconstruction
    real_part = x
    imag_part = torch.zeros_like(x)  # No imaginary part in DHT, but still required for FFT
    complex_x = torch.complex(real_part, imag_part)

    # Apply inverse FFT to reconstruct the original input
    x_reconstructed = torch.fft.ifft2(complex_x, s=(H, W), norm="ortho")
    return x_reconstructed.real

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
        x = x.float()  # Convert to float32 for processing
        B, H, W, C = x.shape

        # Apply the DHT instead of FFT
        X_H_k = dht2d(x)  # DHT of x (real-valued transform)

        block_size = self.block_size
        hidden_size_factor = self.hidden_size_factor

        # Ensure o1 dimensions match the expected sizes
        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * hidden_size_factor], device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Reshape and align the dimensions of X_H_k for broadcasting
        X_H_k = X_H_k.reshape(B, H, W // 2 + 1, self.num_blocks, block_size)

        # First multiplication for the real part
        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', X_H_k[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w1[0]) +
            self.b1[0]
        )

        # Second multiplication for the real part
        o2_real = torch.zeros_like(o1_real)
        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) +
            self.b2[0]
        )

        # Since DHT has only real parts, no imaginary part handling is necessary.
        x = F.softshrink(o2_real, lambd=self.sparsity_threshold)

        # Reshape back to the original shape
        x = x.reshape(B, H, W // 2 + 1, C)

        # Compute the inverse DHT to reconstruct the original input
        x = idht2d(x, H, W)
        x = x.type(dtype)  # Convert back to the original data type

        return x + bias


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

        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

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
    model = AFNONet(img_size=(720, 1440), patch_size=(4, 4), in_chans=20, out_chans=10)
    sample = torch.randn(1, 20, 720, 1440)  # Shape: [B, D, H, W]
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
