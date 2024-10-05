import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def dht2d(x: torch.Tensor) -> torch.Tensor:
    
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

def idht2d(x: torch.Tensor) -> torch.Tensor:
    transformed = dht2d(x)
    
    # Determine normalization factor
    B, D, M, N = x.size()
    normalization_factor = M * N
    
    # Normalize the transformed result
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
        B, D, H, W = x.shape  # Changed to reflect the input shape [B, D, H, W]
        
        x = dht2d(x)
        x = x.reshape(B, D, H, W // 2 + 1, self.num_blocks, self.block_size)

        # Reshape and align the dimensions of X_H_k and X_H_neg_k for broadcasting
        X_H_k = x 
        X_H_neg_k = torch.roll(torch.flip(x, dims=[2]), shifts=(1, 0), dims=[2])

        kept_modes = int(H * self.hard_thresholding_fraction)

        # Ensure o1 and o2 dimensions match the expected sizes
        o1_H_k = torch.zeros([B, D, H, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_H_neg_k = torch.zeros([B, D, H, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)

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
        x = x.reshape(B, D, H, W)
        return x

class AFNONet(nn.Module):
    def __init__(self, params, img_size=(720, 1440), patch_size=(16, 16), in_chans=2, out_chans=2, embed_dim=768, depth=12, mlp_ratio=4., drop_rate=0., drop_path_rate=0., num_blocks=16, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
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
