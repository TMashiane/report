import math
import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch=1, embed_dim=256, patch_size=(8, 8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B,C,D,H,W] => [B, E, D',H',W'] => tokens [B, N, E]
        x = self.proj(x)
        B, E, Dp, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, Dp * Hp * Wp, E)
        return x, (Dp, Hp, Wp)


class ViT3DAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                 patch_size=(8, 8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.embed = PatchEmbed3D(in_ch, embed_dim, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embed = None  # Lazily create based on token count
        self.decoder = nn.Linear(embed_dim, int(out_ch * patch_size[0] * patch_size[1] * patch_size[2]))

    def forward(self, x):
        # x: [B,1,D,H,W]
        B, C, D, H, W = x.shape
        x_tok, grid = self.embed(x)  # [B, N, E]
        N = x_tok.shape[1]
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, x_tok.shape[2], device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        z = x_tok + self.pos_embed
        z = self.encoder(z)
        patches = self.decoder(z)  # [B, N, out_ch*pd*ph*pw]
        pd, ph, pw = self.patch_size
        out = patches.view(B, grid[0], grid[1], grid[2], -1, pd, ph, pw)
        out_ch = out.shape[4]
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.view(B, out_ch, grid[0] * pd, grid[1] * ph, grid[2] * pw)
        return out

