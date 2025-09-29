# ------------------ Minimal SETR-style ViT + Decoders ------------------
import torch, torch.nn as nn, torch.nn.functional as F


# Patch embedding via conv: (B, C, H, W) -> (B, N, D)
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: B,C,H,W -> B,D,H',W' -> B,N,D
        x = self.proj(x)  # (B, D, H', W')
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N=H'W', D)
        return x, (Hp, Wp)


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, nheads=12, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.pos = None  # set at runtime based on N
        block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nheads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(block, num_layers=depth)
        self.return_intermediate = True
        self.depth = depth

    def forward(self, x):
        # x: (B,N,D)
        B, N, D = x.shape
        # learnable pos (B,N,D) so we can vary N across images safely
        if (self.pos is None) or (self.pos.shape[1] != N):
            self.pos = nn.Parameter(torch.zeros(1, N, D, device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        z = x + self.pos
        if not self.return_intermediate:
            return self.enc(z), []  # (B,N,D), no intermediates
        # collect a few intermediate layers (MLA)
        inter = []
        for i, layer in enumerate(self.enc.layers):
            z = layer(z)
            if i + 1 in {3, 6, 9, self.depth}:  # take 4 stages
                inter.append(z.clone())  # list of (B,N,D)
        z = self.enc.norm(z) if hasattr(self.enc, "norm") else z
        return z, inter  # final (B,N,D), list[(B,N,D)*4]


# --- Decoders ---
class SETRNaiveDecoder(nn.Module):
    """Reshape tokens -> (B,D,H',W') then a small upsampling head to HxW."""

    def __init__(self, embed_dim=768, num_classes=1, up_stages=2, ch=256):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, ch, 1)
        ups = []
        for _ in range(up_stages):
            ups += [nn.ConvTranspose2d(ch, ch, 2, 2), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.up = nn.Sequential(*ups)
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, tokens, hw):
        B, N, D = tokens.shape
        Hp, Wp = hw
        x = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)  # (B,D,H',W')
        x = self.proj(x)
        x = self.up(x)
        return self.head(x)  # (B,K,H,W) if upscales match


class SETRPUPDecoder(nn.Module):
    """Progressive Upsampling (PUP): interleave upsample + conv refinement."""

    def __init__(self, embed_dim=768, num_classes=1, ch=256, stages=3):
        super().__init__()
        self.inproj = nn.Conv2d(embed_dim, ch, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(True),
                )
                for _ in range(stages)
            ]
        )
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, tokens, hw):
        B, N, D = tokens.shape
        Hp, Wp = hw
        x = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)
        x = self.inproj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class SETRMLADecoder(nn.Module):
    """MLA-lite: fuse 4 intermediate token maps after projecting & aligning."""

    def __init__(self, embed_dim=768, num_classes=1, ch=128):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(embed_dim, ch, 1) for _ in range(4)])
        self.fuse = nn.Conv2d(4 * ch, ch, 3, padding=1)
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inter_list, hw):
        # inter_list: 4 tensors (B,N,D) from early->late layers
        feats = []
        for i, tokens in enumerate(inter_list):
            B, N, D = tokens.shape
            Hp, Wp = hw
            x = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)
            x = self.projs[i](x)
            # progressively upsample deeper features so all match the first's scale
            if i < 3:
                x = F.interpolate(
                    x, scale_factor=2 ** (3 - i), mode="bilinear", align_corners=False
                )
            feats.append(x)
        x = torch.cat(feats, dim=1)  # (B, 4*ch, H, W) at the finest aligned scale
        x = self.fuse(x)
        return self.head(x)


# --- SegFormer-style tiny head: hierarchical encoder omitted; head shown ---
class SegFormerHead(nn.Module):
    """Given multi-scale features F1..F4 (C_i,H_i,W_i): 1x1 -> same ch, upsample & fuse via MLP-ish conv."""

    def __init__(self, in_chs=(64, 128, 320, 512), out_ch=256, num_classes=1):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.fuse = nn.Conv2d(out_ch * 4, out_ch, 1)  # lightweight MLP-ish
        self.head = nn.Conv2d(out_ch, num_classes, 1)

    def forward(self, feats):  # feats=[F1,F2,F3,F4]
        B, _, H, W = feats[0].shape
        up = [
            F.interpolate(
                self.projs[i](f), size=(H, W), mode="bilinear", align_corners=False
            )
            for i, f in enumerate(feats)
        ]
        x = torch.cat(up, dim=1)
        x = self.fuse(x)
        return self.head(x)


# -------------- End-to-end SETR example --------------
class SETR(nn.Module):
    def __init__(
        self, in_ch=3, num_classes=1, patch=16, D=512, depth=12, decoder="pup"
    ):
        super().__init__()
        self.pe = PatchEmbed(in_ch, D, patch)
        self.vit = ViTEncoder(embed_dim=D, depth=depth, nheads=max(4, D // 64))
        if decoder == "naive":
            self.dec = SETRNaiveDecoder(
                embed_dim=D, num_classes=num_classes, up_stages=2, ch=256
            )
        elif decoder == "mla":
            self.dec = SETRMLADecoder(embed_dim=D, num_classes=num_classes, ch=128)
        else:  # 'pup'
            self.dec = SETRPUPDecoder(
                embed_dim=D, num_classes=num_classes, ch=256, stages=3
            )
        self.decoder_kind = decoder

    def forward(self, x):
        tokens, hw = self.pe(x)  # (B,N,D), (H',W')
        z_final, inter = self.vit(tokens)  # final + intermediates
        if self.decoder_kind == "mla":
            return self.dec(inter, hw)
        else:
            return self.dec(z_final, hw)
