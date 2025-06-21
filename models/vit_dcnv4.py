# --------------------------------------------------------
# DCNv4 + ViT Hybrid Model for Remote Sensing Image Classification
# --------------------------------------------------------

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum
from .DCNv4.modules.dcnv4 import DCNv4


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """Pre-LayerNorm for Transformer blocks"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed Forward Network for Transformer"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            qkv
        )
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer Encoder"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class DCNv4Block(nn.Module):
    """DCNv4 Residual Block with BatchNorm and Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=4, offset_scale=1.0, dw_kernel_size=None):
        super().__init__()
        # 1×1 projection so that main branch and residual have same channels
        if stride > 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.proj = nn.Identity()
        
        # DCNv4 expects sequence of length H*W with C channels == out_channels
        self.dcnv4 = DCNv4(
            channels=out_channels,
            kernel_size=kernel_size,
            stride=1,             # spatial downsample handled by proj
            pad=padding,
            group=groups,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (N, in_ch, H, W)
        x_proj = self.proj(x)            # (N, out_ch, H_out, W_out)
        N, C, H, W = x_proj.shape

        # prepare for DCNv4: sequence of length H*W with C channels
        seq = x_proj.permute(0, 2, 3, 1).reshape(N, H * W, C)  # (N, L, C)
        # apply DCNv4
        seq = self.dcnv4(seq, shape=(H, W))                  # (N, L, C)
        feat = seq.reshape(N, H, W, C).permute(0, 3, 1, 2)     # (N, C, H, W)

        out = self.bn(feat)
        out = self.act(out)
        return out + x_proj


class DCNv4Backbone(nn.Module):
    """DCNv4 Backbone for Feature Extraction"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        self.layer1 = DCNv4Block(base_channels, base_channels * 2, stride=2, groups=4)
        self.layer2 = DCNv4Block(base_channels * 2, base_channels * 4, stride=2, groups=4)
        self.layer3 = DCNv4Block(base_channels * 4, base_channels * 8, stride=2, groups=4)
        self.layer4 = DCNv4Block(base_channels * 8, base_channels * 8, stride=1, groups=4)
        self.out_channels = base_channels * 8

    def forward(self, x):
        # Input: (B, 3, 512, 512)
        x = self.conv1(x)      # (B,  64, 256, 256)
        x = self.layer1(x)     # (B, 128, 128, 128)
        x = self.layer2(x)     # (B, 256,  64,  64)
        x = self.layer3(x)     # (B, 512,  32,  32)
        x = self.layer4(x)     # (B, 512,  32,  32)
        return x


class ViTDCNv4(nn.Module):
    """DCNv4 + ViT Hybrid Model for Remote Sensing Classification"""
    def __init__(self,
                 image_size=512,
                 num_classes=5,
                 dim=512,
                 depth=6,
                 heads=8,
                 mlp_dim=1024,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1,
                 dcnv4_channels=64):
        super().__init__()

        # 1. DCNv4 Backbone
        self.backbone = DCNv4Backbone(channels, dcnv4_channels)

        # 2. Compute feature map dimensions
        img_h, img_w    = pair(image_size)
        down = 16  # 2*2*2*2 strides
        fh, fw         = img_h // down, img_w // down  # e.g. 32,32
        num_patches    = fh * fw                       # 1024
        patch_dim      = dcnv4_channels * 8           # 512

        assert pool in {'cls', 'mean'}

        # 3. Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(patch_dim, dim)
        )  # (B, 1024, dim)

        # 4. CLS token + positional embedding
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)

        # 5. Transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 6. Pool & Head
        self.pool     = pool
        self.to_latent= nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img: (B, 3, 512, 512)
        feat = self.backbone(img)                    # (B,512,32,32)
        x    = self.to_patch_embedding(feat)         # (B,1024,dim)
        b, n, _ = x.shape

        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x   = torch.cat((cls, x), dim=1)             # (B,1025, dim)
        x   = x + self.pos_embedding[:, :n+1]
        x   = self.dropout(x)

        x   = self.transformer(x)                    # (B,1025, dim)
        x   = x.mean(1) if self.pool == 'mean' else x[:, 0]  # (B, dim)
        x   = self.to_latent(x)
        return self.mlp_head(x)                      # (B, num_classes)


def create_vit_dcnv4_model(num_classes=5, image_size=512):
    return ViTDCNv4(image_size=image_size,
                    num_classes=num_classes,
                    dim=512,
                    depth=6,
                    heads=8,
                    mlp_dim=1024,
                    pool='cls',
                    channels=3,
                    dim_head=64,
                    dropout=0.1,
                    emb_dropout=0.1,
                    dcnv4_channels=64)
