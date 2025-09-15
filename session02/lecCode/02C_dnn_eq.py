#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demos for Modern DNN-based Image Enhancement
  1) HDRNet-style Deep Bilateral Learning (local affine in bilateral grid)
  2) ESRGAN-style Perceptual Super-Resolution (RRDB generator)

Usage examples:
  # HDRNet-style: learn to mimic CLAHE tone mapping on one/few images
  python dnn_enhance_demo.py --algo hdrnet --image path/to/photo.jpg --iters 200 --save

  # HDRNet with a folder of images (uses CLAHE as targets by default)
  python dnn_enhance_demo.py --algo hdrnet --data path/to/folder --batch 2 --iters 800 --save --device cuda

  # ESRGAN-style: train on a single HR image by self-supervision (downsampled LR as input)
  python dnn_enhance_demo.py --algo esrgan --image path/to/photo_hr.jpg --iters 400 --patch 128 --save

  # ESRGAN-style on a folder of HR images
  python dnn_enhance_demo.py --algo esrgan --data path/to/HR_folder --batch 2 --iters 2000 --save --device cuda

Notes:
- This is a compact *teaching* implementation: small networks & defaults to keep runtime light.
- For VGG perceptual loss, torchvision will fetch weights if not present.
"""

import argparse
import os
from pathlib import Path
import random
import math

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision as tv
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_bgr(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return bgr

def bgr2rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def rgb2bgr(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def np_to_torch_img(np_rgb_uint8):
    """HWC uint8 [0,255] -> torch float BCHW [0,1]."""
    x = torch.from_numpy(np_rgb_uint8).float()/255.0
    x = x.permute(2,0,1).unsqueeze(0)
    return x

def torch_to_np_uint8(x):
    """BCHW [0,1] -> HWC uint8."""
    x = x.clamp(0,1).detach().cpu()[0].permute(1,2,0).numpy()
    return (x*255.0 + 0.5).astype(np.uint8)

def save_rgb(path, rgb_u8):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8).save(path)

def show_and_save_pair(outdir, stem, a, b, title_a="Input", title_b="Output"):
    """Save side-by-side comparison for slides."""
    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(a); plt.title(title_a); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(b); plt.title(title_b); plt.axis("off")
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(Path(outdir)/f"{stem}_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Simple datasets
# ----------------------------

class FolderRGB(Dataset):
    """Loads RGB images from a folder. Optionally center-crops & resizes for speed."""
    def __init__(self, root, load_max=256, center_crop=None):
        self.paths = sorted([p for p in Path(root).glob("**/*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
        if len(self.paths)==0:
            raise RuntimeError(f"No images found in {root}")
        self.load_max = load_max
        self.center_crop = center_crop

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        bgr = load_bgr(p)
        rgb = bgr2rgb(bgr)
        if self.center_crop is not None:
            h,w,_ = rgb.shape
            s = min(h,w)
            y0=(h-s)//2; x0=(w-s)//2
            rgb = rgb[y0:y0+s, x0:x0+s]
        if self.load_max is not None:
            h,w,_ = rgb.shape
            scale = self.load_max / max(h,w)
            if scale < 1.0:
                new = (int(round(w*scale)), int(round(h*scale)))
                rgb = cv2.resize(rgb, new, interpolation=cv2.INTER_AREA)
        x = np_to_torch_img(rgb) # BCHW
        return x, str(p)


# ----------------------------
# HDRNet-style model
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=k//2)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class HDRNetMini(nn.Module):
    """
    Predicts a bilateral grid of 3x4 affine color coefficients (12 per cell) at low resolution:
      Grid shape: [B, Hs, Ws, D, 12]
    Uses a guidance branch to produce g(x) in [0,1] for edge-aware slicing (trilinear in x/y/g).
    Then applies per-pixel affine: out = A*[R,G,B,1]^T  (A in R^{3x4})
    """
    def __init__(self, D=8, grid_ch=64):
        super().__init__()
        self.D = D
        # Low-res stream to predict bilateral grid
        self.low = nn.Sequential(
            ConvBlock(3, 32, 3, 2),    # /2
            ConvBlock(32, 64, 3, 2),   # /4
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, grid_ch, 3, 1),
        )
        # Head to produce 12*D coefficients per spatial location
        self.head = nn.Conv2d(grid_ch, 12*D, kernel_size=1)

        # Guidance branch (full-res to 1 channel in [0,1])
        self.guidance = nn.Sequential(
            ConvBlock(3, 16, 3, 1),
            ConvBlock(16, 16, 3, 1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _meshgrid(hw, device):
        H,W = hw
        ys = torch.linspace(0, H-1, H, device=device)
        xs = torch.linspace(0, W-1, W, device=device)
        yy,xx = torch.meshgrid(ys,xs, indexing="ij")
        return yy,xx

    def slice_trilinear(self, grid, guidance):
        """
        grid: [B, Hs, Ws, D, 12]
        guidance: [B,1,H,W] in [0,1]
        returns per-pixel coeffs: [B, H, W, 12]
        """
        B,H,W = guidance.shape[0], guidance.shape[2], guidance.shape[3]
        Hs, Ws, D = grid.shape[1], grid.shape[2], grid.shape[3]
        device = guidance.device

        # Map pixel coords to grid coords
        yy, xx = self._meshgrid((H,W), device)  # [H,W]
        ys = yy / (H-1+1e-8) * (Hs-1)
        xs = xx / (W-1+1e-8) * (Ws-1)
        zs = guidance.squeeze(1) * (D-1)  # [B,H,W]

        x0 = xs.floor().clamp(0, Ws-1); x1 = (x0+1).clamp(0, Ws-1)
        y0 = ys.floor().clamp(0, Hs-1); y1 = (y0+1).clamp(0, Hs-1)

        # broadcast for batch
        x0 = x0[None].expand(B,-1,-1).long(); x1 = x1[None].expand_as(x0).long()
        y0 = y0[None].expand(B,-1,-1).long(); y1 = y1[None].expand_as(y0).long()

        z0 = zs.floor().clamp(0, D-1).long()
        z1 = (z0+1).clamp(0, D-1).long()

        wx = (xs - x0.float()); wx0 = 1.0 - wx; wx1 = wx
        wy = (ys - y0.float()); wy0 = 1.0 - wy; wy1 = wy
        wz = (zs - z0.float()); wz0 = 1.0 - wz; wz1 = wz

        def gather(y,x,z):
            # grid[B, Hs, Ws, D, 12] -> take per-pixel corner
            return grid[torch.arange(B)[:,None,None], y, x, z]  # [B,H,W,12]

        c000 = gather(y0, x0, z0); c001 = gather(y0, x0, z1)
        c010 = gather(y0, x1, z0); c011 = gather(y0, x1, z1)
        c100 = gather(y1, x0, z0); c101 = gather(y1, x0, z1)
        c110 = gather(y1, x1, z0); c111 = gather(y1, x1, z1)

        w000 = (wy0*wx0*wz0)[...,None]; w001 = (wy0*wx0*wz1)[...,None]
        w010 = (wy0*wx1*wz0)[...,None]; w011 = (wy0*wx1*wz1)[...,None]
        w100 = (wy1*wx0*wz0)[...,None]; w101 = (wy1*wx0*wz1)[...,None]
        w110 = (wy1*wx1*wz0)[...,None]; w111 = (wy1*wx1*wz1)[...,None]

        coeff = (w000*c000 + w001*c001 + w010*c010 + w011*c011 +
                 w100*c100 + w101*c101 + w110*c110 + w111*c111)  # [B,H,W,12]
        return coeff

    def apply_affine(self, coeff, x):
        """
        coeff: [B,H,W,12] -> reshape to [B,H,W,3,4]
        x:     [B,3,H,W] in [0,1]
        return: [B,3,H,W]
        """
        B,C,H,W = x.shape
        A = coeff.view(B, H, W, 3, 4)                  # [B,H,W,3,4]
        X = torch.cat([x.permute(0,2,3,1), torch.ones(B,H,W,1, device=x.device)], dim=-1)  # [B,H,W,4]
        y = torch.einsum('bhwij,bhwj->bhwi', A, X)     # [B,H,W,3]
        return y.permute(0,3,1,2).clamp(0,1)

    def forward(self, x):
        """
        x: [B,3,H,W] in [0,1]
        """
        B,_,H,W = x.shape
        feat = self.low(F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False))
        Hs, Ws = feat.shape[2], feat.shape[3]
        grid_logits = self.head(feat)  # [B, 12*D, Hs, Ws]
        grid = grid_logits.view(B, 12, self.D, Hs, Ws).permute(0, 3, 4, 2, 1).contiguous() # [B,Hs,Ws,D,12]

        g = self.guidance(x)  # [B,1,H,W] in [0,1]
        coeff = self.slice_trilinear(grid, g)  # [B,H,W,12]
        y = self.apply_affine(coeff, x)        # [B,3,H,W]
        return y


def clahe_like_target(x_bchw):
    """Build a supervision target by applying CLAHE to L channel (LAB), then back to RGB."""
    x = x_bchw.detach().cpu()
    out = []
    for i in range(x.shape[0]):
        rgb = (x[i].permute(1,2,0).numpy()*255).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L,A,B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L2 = clahe.apply(L)
        y = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2RGB)
        out.append(np_to_torch_img(y)[0])
    return torch.stack(out, dim=0).to(x_bchw.device)


# ----------------------------
# ESRGAN-lite: RRDB Generator + Patch Discriminator + Perceptual Loss
# ----------------------------

class ResidualDenseBlock5C(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); self.l1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1); self.l2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1); self.l3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1); self.l4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1)
    def forward(self, x):
        x1 = self.l1(self.conv1(x))
        x2 = self.l2(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.l3(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.l4(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.res_scale

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(nf, gc, res_scale)
        self.rdb2 = ResidualDenseBlock5C(nf, gc, res_scale)
        self.rdb3 = ResidualDenseBlock5C(nf, gc, res_scale)
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2

class RRDBNetSmall(nn.Module):
    """Compact ESRGAN-like generator with 4× upscaling."""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=3, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)], nn.Conv2d(nf, nf, 3, 1, 1))
        # Upsample x2 twice -> x4
        self.up1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3,1,1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
        self.up2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3,1,1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk(fea)
        fea = fea + trunk
        fea = self.up1(fea)
        fea = self.up2(fea)
        out = self.conv_last(fea)
        return out.clamp(0,1)

class PatchDiscriminator(nn.Module):
    """Lightweight discriminator for 4× SR patches."""
    def __init__(self, in_nc=3, nf=64):
        super().__init__()
        layers = []
        def block(ci, co, s):
            layers.extend([nn.Conv2d(ci, co, 3, stride=s, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
        block(in_nc, nf, 1)
        block(nf, nf, 2)
        block(nf, nf*2, 1)
        block(nf*2, nf*2, 2)
        block(nf*2, nf*4, 1)
        block(nf*4, nf*4, 2)
        layers.append(nn.Conv2d(nf*4, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # [B,1,H',W']
        return self.net(x)

class VGGPerceptual(nn.Module):
    def __init__(self, layer="relu5_4"):
        super().__init__()
        vgg = tv.models.vgg19(weights=tv.models.VGG19_Weights.IMAGENET1K_V1).features
        # up to relu5_4 (35 index)
        self.features = vgg[:36].eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x):
        # x in [0,1]
        x = (x - self.mean) / self.std
        return self.features(x)


# ----------------------------
# Training helpers
# ----------------------------

def make_loader(image, data, batch, max_side=512):
    if image:
        x = bgr2rgb(load_bgr(Path(image)))
        # shrink very large inputs for speed
        s = max_side / max(x.shape[:2])
        if s < 1.0:
            x = cv2.resize(x, (int(x.shape[1]*s), int(x.shape[0]*s)), interpolation=cv2.INTER_AREA)
        t = np_to_torch_img(x)
        ds = [(t, image)]
        return DataLoader(ds, batch_size=1, shuffle=True)
    else:
        ds = FolderRGB(data, load_max=max_side, center_crop=True)
        return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

def random_crop_hr(hr, patch):
    # hr: BCHW in [0,1]
    B,_,H,W = hr.shape
    y = random.randint(0, H-patch)
    x = random.randint(0, W-patch)
    return hr[:,:,y:y+patch, x:x+patch]

def downsample_bicubic(x, scale):
    B,C,H,W = x.shape
    h = H//scale; w = W//scale
    return F.interpolate(x, size=(h,w), mode='bicubic', align_corners=False)

# ----------------------------
# HDRNet demo
# ----------------------------

def run_hdrnet(args):
    device = torch.device(args.device)
    net = HDRNetMini(D=8).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    l1  = nn.L1Loss()
    tv_reg = lambda g: sum((p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean() +
                           (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
                           for p in net.low.parameters() if p.ndim==4)*0.0  # very light/no reg for demo

    loader = make_loader(args.image, args.data, args.batch, max_side=args.max_side)

    it = 0
    for epoch in range(99999):
        for x, path in loader:
            it += 1
            x = x.to(device)
            with torch.no_grad():
                y_star = clahe_like_target(x) if not args.target else x  # placeholder; custom targets could be added
            y = net(x)
            loss = l1(y, y_star)  # + tv_reg(net.low[-1].weight)  # optional tiny reg
            opt.zero_grad(); loss.backward(); opt.step()

            if it % 20 == 0:
                print(f"[HDRNet] iter {it:05d}  L1={loss.item():.4f}")

            if it % 100 == 0 or it == args.iters:
                x_np = torch_to_np_uint8(x)
                y_np = torch_to_np_uint8(y)
                ygt  = torch_to_np_uint8(y_star)
                if args.save:
                    outdir = Path(args.outdir or "hdrnet_outputs"); outdir.mkdir(parents=True, exist_ok=True)
                    stem = Path(path if isinstance(path,str) else path[0]).stem
                    save_rgb(outdir/f"{stem}_input.png", x_np)
                    save_rgb(outdir/f"{stem}_hdrnet.png", y_np)
                    save_rgb(outdir/f"{stem}_target.png", ygt)
                    show_and_save_pair(outdir, stem, x_np, y_np, "Input", "HDRNet-like")
                if it >= args.iters: return

# ----------------------------
# ESRGAN demo
# ----------------------------

def gan_losses():
    bce = nn.BCEWithLogitsLoss()
    def d_loss(d_real, d_fake):
        return bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
    def g_loss_adv(d_fake):
        return bce(d_fake, torch.ones_like(d_fake))
    return d_loss, g_loss_adv

def run_esrgan(args):
    device = torch.device(args.device)
    G = RRDBNetSmall(nb=3).to(device)
    D = PatchDiscriminator().to(device)
    vgg = VGGPerceptual().to(device)

    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.9, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.9, 0.99))
    l1   = nn.L1Loss()
    d_loss, g_adv = gan_losses()

    loader = make_loader(args.image, args.data, args.batch, max_side=args.max_side)

    it = 0
    for epoch in range(99999):
        for hr, path in loader:
            it += 1
            hr = hr.to(device)  # HR in [0,1]
            # Random crop HR -> downsample to LR (self-supervised)
            hr_patch = random_crop_hr(hr, args.patch)
            lr = downsample_bicubic(hr_patch, scale=4).clamp(0,1)

            # --- Train D ---
            with torch.no_grad():
                sr = G(lr)
            d_real = D(hr_patch)
            d_fake = D(sr)
            lossD = d_loss(d_real, d_fake)
            optD.zero_grad(); lossD.backward(); optD.step()

            # --- Train G ---
            sr = G(lr)
            d_fake = D(sr)
            # perceptual features
            with torch.no_grad():
                f_hr = vgg(hr_patch)
            f_sr = vgg(sr)
            loss_pix  = l1(sr, hr_patch)
            loss_perc = l1(f_sr, f_hr)
            loss_adv  = g_adv(d_fake)
            lossG = 0.01*loss_pix + 1.0*loss_perc + 5e-3*loss_adv  # modest weights for demo

            optG.zero_grad(); lossG.backward(); optG.step()

            if it % 20 == 0:
                print(f"[ESRGAN] iter {it:05d}  D={lossD.item():.3f}  G={lossG.item():.3f}  (pix={loss_pix.item():.3f}, perc={loss_perc.item():.3f}, adv={loss_adv.item():.3f})")

            if it % 100 == 0 or it == args.iters:
                with torch.no_grad():
                    sr_full = G(downsample_bicubic(hr, 4))
                hr_np = torch_to_np_uint8(hr)
                lr_np = torch_to_np_uint8(downsample_bicubic(hr,4))
                sr_np = torch_to_np_uint8(sr_full)
                if args.save:
                    outdir = Path(args.outdir or "esrgan_outputs"); outdir.mkdir(parents=True, exist_ok=True)
                    stem = Path(path if isinstance(path,str) else path[0]).stem
                    save_rgb(outdir/f"{stem}_HR.png", hr_np)
                    save_rgb(outdir/f"{stem}_LRx4.png", lr_np)
                    save_rgb(outdir/f"{stem}_SRx4.png", sr_np)
                    show_and_save_pair(outdir, stem, lr_np, sr_np, "LR (×4 downsampled)", "SR (RRDB demo)")
                if it >= args.iters: return


# ----------------------------
# Main / CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["hdrnet","esrgan"], required=True)
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--data",  type=str, default=None, help="Folder of images")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--max-side", type=int, default=512, help="Max side when loading for speed")

    # ESRGAN-only
    p.add_argument("--patch", type=int, default=128, help="HR patch size for ESRGAN training")
    # HDRNet-only
    p.add_argument("--target", action="store_true", help="(placeholder) if you have paired targets, wire them in")
    args = p.parse_args()
    if (args.image is None) == (args.data is None):
        p.error("Specify exactly one of --image or --data")
    return args

def main():
    set_seed(0)
    args = parse_args()
    if args.algo == "hdrnet":
        run_hdrnet(args)
    else:
        run_esrgan(args)

if __name__ == "__main__":
    main()
