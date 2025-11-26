import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

class FourierCoordConv(nn.Module):
    """Adds spatial information via Fourier features"""
    def __init__(self, scales=[1.0, 2.0, 4.0]):
        super().__init__()
        self.scales = scales
        
    def forward(self, x):
        b, c, h, w = x.shape
        coord_channels = []
        
        y_range = t.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        x_range = t.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_coords = repeat(y_range, 'h -> h w', w=w)
        x_coords = repeat(x_range, 'w -> h w', h=h)
        
        for scale in self.scales:
            coord_channels.extend([
                t.sin(scale * np.pi * y_coords),
                t.cos(scale * np.pi * y_coords),
                t.sin(scale * np.pi * x_coords),
                t.cos(scale * np.pi * x_coords)
            ])
        
        coords = t.stack(coord_channels, dim=0)
        coords = repeat(coords, 'c h w -> b c h w', b=b)
        return t.cat([x, coords], dim=1)


class StochasticDepth(nn.Module):
    """Stochastic Depth (Huang et al. 2016)"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0:
            return x + residual
        
        keep_prob = 1 - self.drop_prob
        mask = t.bernoulli(t.full((x.shape[0], 1, 1, 1), keep_prob, device=x.device))
        return x + residual * mask / keep_prob


class SpatialAttentionMask(nn.Module):
    """Learn which spatial regions to focus on (Sigmoid Gating)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = t.sigmoid(self.conv(x))
        return x * attention


class SEBlock(nn.Module):
    """Squeeze and Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, drop_prob=0.0, layer_scale_init=1e-4):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(channels)
        
        # Layer Scale (CaiT)
        self.gamma = nn.Parameter(layer_scale_init * t.ones(channels, 1, 1))
        
        # Stochastic Depth
        self.stochastic_depth = StochasticDepth(drop_prob)

    def forward(self, x):
        out = F.silu(self.gn1(x))
        out = self.conv1(out)
        out = F.silu(self.gn2(out))
        out = self.conv2(out)
        out = self.se(out)
        return self.stochastic_depth(x, self.gamma * out)


class AxialAttentionBlock(nn.Module):
    """Axial Attention with sequential row->col processing and residual connection"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.row_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        
        # 1. Row attention
        x_row = x.permute(0, 2, 3, 1).reshape(b * h, w, c) # (B*H, W, C)
        x_row_out, _ = self.row_attn(x_row, x_row, x_row)
        x_row_out = x_row_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 2. Column attention
        # Sequential: Feed output of row attention into column attention
        x_col = x_row_out.permute(0, 3, 2, 1).reshape(b * w, h, c) # (B*W, H, C)
        x_col_out, _ = self.col_attn(x_col, x_col, x_col)
        x_col_out = x_col_out.reshape(b, w, h, c).permute(0, 3, 2, 1)
        
        # 3. Add original residual
        return residual + x_col_out


class RelativePositionBias(nn.Module):
    """Relative positional bias (Decomposed for 2D H+W)"""
    def __init__(self, num_heads, max_distance=32):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        num_buckets = 2 * max_distance + 1
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    def forward(self, h, w):
        coords_h = t.arange(h, device=self.relative_attention_bias.weight.device)
        coords_w = t.arange(w, device=self.relative_attention_bias.weight.device)
        coords = t.stack(t.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.clamp(-self.max_distance, self.max_distance)
        relative_coords = relative_coords + self.max_distance
        
        # Look up biases for H and W separately and sum them
        bias_h = self.relative_attention_bias(relative_coords[0])
        bias_w = self.relative_attention_bias(relative_coords[1])
        
        bias = bias_h + bias_w
        return bias.permute(2, 0, 1)


class SpatialAttentionPool(nn.Module):
    """Attention pooling with content-dependent query"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.query_transform = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.rel_pos_bias = RelativePositionBias(num_heads)
    
    def forward(self, x):
        # x: (B, H*W, C)
        b, hw, c = x.shape
        h = w = int(np.sqrt(hw))
        
        x = self.norm(x)
        
        # Generate dynamic query from Global Average Pool
        global_ctx = x.mean(dim=1, keepdim=True)
        q = self.query_transform(global_ctx)
        
        # Get relative position bias
        bias = self.rel_pos_bias(h, w)
        # Average bias for global query
        bias = bias.mean(dim=1, keepdim=True)
        bias = repeat(bias, 'heads 1 hw -> (b heads) 1 hw', b=b)
        
        x, _ = self.attn(q, x, x, attn_mask=bias)
        return x.squeeze(1)


class PixelControlHead(nn.Module):
    """Predict pixel changes in spatial grid cells (UNREAL paper)"""
    def __init__(self, embed_dim, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        
        self.deconv = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 32 * grid_size * grid_size)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus() # Softplus for positive pixel changes
        )
        
    def forward(self, features):
        b = features.shape[0]
        x = self.deconv(features)
        x = x.view(b, 32, self.grid_size, self.grid_size)
        x = self.spatial(x)
        return x.squeeze(1)


class ModelBlock(nn.Module):
    """Standard ResNet-style block container"""
    def __init__(self, in_channels, out_channels, num_residual=2, 
                 drop_prob_start=0.0, drop_prob_end=0.0,
                 use_spatial_attention=False, use_axial_attention=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        drop_probs = np.linspace(drop_prob_start, drop_prob_end, num_residual)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels, drop_prob=drop_probs[i])
            for i in range(num_residual)
        ])
        
        self.spatial_attention = SpatialAttentionMask(out_channels) if use_spatial_attention else None
        self.axial_attention = AxialAttentionBlock(out_channels) if use_axial_attention else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        
        if self.axial_attention is not None:
            x = self.axial_attention(x)
            
        return x


class RandomShifts(nn.Module):
    """Data augmentation via random shifts"""
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if not self.training:
            return x

        n, c, h, w = x.size()
        padding = ([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        
        v = t.linspace(-1.0 + eps, 1.0-eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        x_chan = repeat(v, 'i -> j i 1', j=h)
        y_chan = repeat(v, 'i -> i j 1', j=h)

        base_grid = t.cat([x_chan, y_chan], dim=2)
        base_grid = repeat(base_grid, 'h w c -> n h w c', n=n)

        shift = t.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False, mode='nearest')
