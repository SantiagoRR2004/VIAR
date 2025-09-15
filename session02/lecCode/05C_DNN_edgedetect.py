#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Edge Detection with DNNs — HED & BDCN-mini (Teaching Demo)

Usage (train HED with pseudo-GT from Canny):
  python dnn_edges.py --algo hed --images data/BSDS/images --epochs 2 --save

Usage (train BDCN-mini with GT edges):
  python dnn_edges.py --algo bdcn --images data/BSDS/images --edges data/BSDS/edges --epochs 3 --save --device cuda

Single-image inference (loads weights, saves side/fused PNGs):
  python dnn_edges.py --algo hed --inference path/to/img.jpg --weights runs/hed/best.pt --save

Notes:
- Compact backbones for classroom runtime; conceptually faithful to HED/BDCN.
- BDCN scale-specific targets are a DIDACTIC approximation via Gaussian blurs.
"""

import argparse, os, math, random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Utilities
# -----------------------

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def imread_rgb(path):  # HWC uint8
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imread_gray(path):  # HxW uint8
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None: raise FileNotFoundError(path)
    return g

def to_tensor(img_u8):  # HWC uint8 -> CHW float [0,1]
    t = torch.from_numpy(img_u8).float()/255.0
    return t.permute(2,0,1)

def mask_to_tensor(mask_u8):  # HxW uint8 -> 1xHxW float {0,1}
    m = (torch.from_numpy(mask_u8)>0).float()
    return m.unsqueeze(0)

def save_gray(path, arr01):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(arr01,0,1)*255+0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)

def save_rgb(path, arr01):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(arr01,0,1)*255+0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


# -----------------------
# Dataset
# -----------------------

class EdgeDataset(Dataset):
    """
    images/..*.jpg (or png)
    edges/..*.png  (binary/soft masks). If edges folder is None, uses Canny on-the-fly.
    """
    def __init__(self, img_dir, edge_dir=None, max_side=512, train=True):
        self.imgs = sorted([p for p in Path(img_dir).glob("**/*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
        if not self.imgs: raise RuntimeError(f"No images in {img_dir}")
        self.edge_dir = Path(edge_dir) if edge_dir else None
        self.max_side = max_side
        self.train = train

    def __len__(self): return len(self.imgs)

    def _resize_pair(self, rgb, edge=None):
        H, W = rgb.shape[:2]
        s = min(1.0, self.max_side / max(H,W))
        if s < 1.0:
            neww, newh = int(W*s), int(H*s)
            rgb = cv2.resize(rgb, (neww,newh), interpolation=cv2.INTER_AREA)
            if edge is not None:
                edge = cv2.resize(edge, (neww,newh), interpolation=cv2.INTER_NEAREST)
        return rgb, edge

    def _random_hflip(self, rgb, edge):
        if self.train and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            if edge is not None: edge = np.ascontiguousarray(edge[:, ::-1])
        return rgb, edge

    def __getitem__(self, i):
        p = self.imgs[i]
        rgb = imread_rgb(p)
        # GT edge:
        edge = None
        if self.edge_dir:
            epath = self.edge_dir / p.relative_to(p.parents[0])
            epath = epath.with_suffix(".png") if not epath.exists() else epath
            if epath.exists(): edge = imread_gray(epath)
        if edge is None:
            # pseudo-GT via Canny (self-supervised)
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            g = cv2.GaussianBlur(g, (0,0), 1.0)
            edge = cv2.Canny(g, 50, 150, L2gradient=True)

        rgb, edge = self._resize_pair(rgb, edge)
        rgb, edge = self._random_hflip(rgb, edge)
        x = to_tensor(rgb)                    # 3xHxW [0,1]
        y = mask_to_tensor(edge)              # 1xHxW {0,1}
        name = p.stem
        return x, y, name


# -----------------------
# HED (mini)
# -----------------------

class ConvBNReLU(nn.Module):
    def __init__(self, ci, co, k=3, s=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, padding=d*(k//2), dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(co)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class HEDMini(nn.Module):
    """
    VGG-ish 4-stage backbone with side outputs at each stage.
    Side heads: 1x1 conv -> upsample to input size -> logits
    Fusion: 1x1 conv over concatenated side logits -> fused logits
    """
    def __init__(self, side_ch=(16,32,64,64)):
        super().__init__()
        c1,c2,c3,c4 = side_ch
        # Stages (keep small)
        self.b1 = nn.Sequential(ConvBNReLU(3, 16), ConvBNReLU(16, 16))
        self.p1 = nn.MaxPool2d(2)
        self.b2 = nn.Sequential(ConvBNReLU(16, 32), ConvBNReLU(32, 32))
        self.p2 = nn.MaxPool2d(2)
        self.b3 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.p3 = nn.MaxPool2d(2)
        self.b4 = nn.Sequential(ConvBNReLU(64, 64), ConvBNReLU(64, 64))

        # Side heads (1x1 -> 1 logit)
        self.side1 = nn.Conv2d(16, 1, 1)
        self.side2 = nn.Conv2d(32, 1, 1)
        self.side3 = nn.Conv2d(64, 1, 1)
        self.side4 = nn.Conv2d(64, 1, 1)

        # Fusion head
        self.fuse = nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, x):
        B,_,H,W = x.shape
        f1 = self.b1(x)
        f2 = self.b2(self.p1(f1))
        f3 = self.b3(self.p2(f2))
        f4 = self.b4(self.p3(f3))

        s1 = F.interpolate(self.side1(f1), size=(H,W), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.side2(f2), size=(H,W), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(f3), size=(H,W), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(f4), size=(H,W), mode='bilinear', align_corners=False)

        concat = torch.cat([s1,s2,s3,s4], dim=1)
        fused  = self.fuse(concat)

        # Return logits; apply sigmoid outside for visualization
        return [s1,s2,s3,s4], fused


# -----------------------
# BDCN-mini (SEM + scale-specific supervision)
# -----------------------

class SEM(nn.Module):
    """Scale Enhancement Module: parallel dilated convs (d=1,2,4) -> concat -> 1x1."""
    def __init__(self, c):
        super().__init__()
        self.d1 = ConvBNReLU(c, c, k=3, d=1)
        self.d2 = ConvBNReLU(c, c, k=3, d=2)
        self.d3 = ConvBNReLU(c, c, k=3, d=4)
        self.mix = nn.Conv2d(3*c, c, 1)
    def forward(self, x):
        y = torch.cat([self.d1(x), self.d2(x), self.d3(x)], dim=1)
        return F.relu(self.mix(y), inplace=True)

class BDCNMini(nn.Module):
    """
    4 stages with SEM; each stage has a side 1x1 head producing a logit map.
    Fusion: 1x1 over concatenated side logits.
    """
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(ConvBNReLU(3, 16), ConvBNReLU(16,16), SEM(16))
        self.p1 = nn.MaxPool2d(2)

        self.s2 = nn.Sequential(ConvBNReLU(16,32), ConvBNReLU(32,32), SEM(32))
        self.p2 = nn.MaxPool2d(2)

        self.s3 = nn.Sequential(ConvBNReLU(32,64), ConvBNReLU(64,64), SEM(64))
        self.p3 = nn.MaxPool2d(2)

        self.s4 = nn.Sequential(ConvBNReLU(64,64), ConvBNReLU(64,64), SEM(64))

        self.head1 = nn.Conv2d(16, 1, 1)
        self.head2 = nn.Conv2d(32, 1, 1)
        self.head3 = nn.Conv2d(64, 1, 1)
        self.head4 = nn.Conv2d(64, 1, 1)
        self.fuse   = nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, x):
        B,_,H,W = x.shape
        f1 = self.s1(x)
        f2 = self.s2(self.p1(f1))
        f3 = self.s3(self.p2(f2))
        f4 = self.s4(self.p3(f3))

        s1 = F.interpolate(self.head1(f1), size=(H,W), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.head2(f2), size=(H,W), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.head3(f3), size=(H,W), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.head4(f4), size=(H,W), mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([s1,s2,s3,s4], dim=1))
        return [s1,s2,s3,s4], fused


# -----------------------
# Losses / Targets
# -----------------------

def class_balanced_bce_logits(logits, target):
    """
    logits: [B,1,H,W], target: [B,1,H,W] in {0,1} or [0,1]
    pos_weight = (N_neg / N_pos) for BCEWithLogitsLoss
    """
    with torch.no_grad():
        p = target.mean().clamp(1e-6, 1-1e-6)
        pos_weight = (1 - p) / p
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return loss

def gaussian_blur_torch(m, sigma):
    """Separable Gaussian blur (approx via OpenCV on CPU for simplicity if sigma>0)."""
    if sigma <= 0: return m
    # to numpy (B,1,H,W) -> HxW
    Ms = []
    for i in range(m.shape[0]):
        u8 = (m[i,0].detach().cpu().numpy()*255.0).astype(np.uint8)
        k = max(1, int(round(3*sigma))*2+1)
        bl = cv2.GaussianBlur(u8, (k,k), sigmaX=sigma)
        Ms.append(torch.from_numpy(bl).float()/255.0)
    return torch.stack(Ms,0)[:,None].to(m.device)

def bdcn_scale_targets(gt01, sigmas=(0.7, 1.0, 1.4, 2.0)):
    """
    Create scale-specific soft targets by Gaussian-blurring the GT with different sigmas.
    gt01: [B,1,H,W] in {0,1}
    returns list of [B,1,H,W] floats in [0,1]
    (Didactic approximation to the paper's cascade supervision.)
    """
    outs = []
    for s in sigmas:
        outs.append(gaussian_blur_torch(gt01, s).clamp(0,1))
    return outs


# -----------------------
# Training / Eval
# -----------------------

def make_loader(img_dir, edge_dir, batch, max_side, train=True):
    ds = EdgeDataset(img_dir, edge_dir, max_side=max_side, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=0, drop_last=train)

def train_one_epoch(model, loader, opt, device, algo, epoch, outdir=None):
    model.train()
    running = 0.0
    for it,(x,y,_) in enumerate(loader,1):
        x = x.to(device); y = y.to(device)
        sides, fused = model(x)  # logits

        if algo == "hed":
            # same GT at all scales + fused
            loss = sum(class_balanced_bce_logits(s, y) for s in sides) / len(sides)
            loss += class_balanced_bce_logits(fused, y)
        else:  # bdcn
            targets = bdcn_scale_targets(y)  # list len=4
            loss = 0.0
            for s,t in zip(sides, targets):
                loss += class_balanced_bce_logits(s, t)
            loss = loss/len(sides) + class_balanced_bce_logits(fused, y)

        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
        if it % 10 == 0:
            print(f"[{algo.upper()}] epoch {epoch} it {it}/{len(loader)}  loss={loss.item():.4f}")
    return running / max(len(loader),1)

@torch.no_grad()
def evaluate_and_save(model, loader, device, algo, outdir):
    model.eval()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    avg_iou = []
    for x,y,name in loader:
        x=x.to(device); y=y.to(device)
        sides, fused = model(x)
        probs = [torch.sigmoid(s) for s in sides] + [torch.sigmoid(fused)]
        # Save side/fused
        B= x.shape[0]
        for b in range(B):
            stem = name[b]
            for i,p in enumerate(probs):
                arr = p[b,0].cpu().numpy()
                save_gray(outdir/f"{stem}_side{i+1 if i<4 else 'fused'}.png", arr)
            # IoU at 0.5
            pred = (probs[-1][b,0] >= 0.5).float()
            gt   = (y[b,0] >= 0.5).float()
            inter = (pred*gt).sum().item()
            union = (pred+gt).clamp(0,1).sum().item()
            iou = inter/max(union,1e-6)
            avg_iou.append(iou)
    miou = float(np.mean(avg_iou)) if avg_iou else 0.0
    print(f"[eval] mean IoU@0.5 = {miou:.3f}")
    return miou


# -----------------------
# CLI / Main
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["hed","bdcn"], required=True)
    ap.add_argument("--images", type=str, help="Training images folder")
    ap.add_argument("--edges",  type=str, default=None, help="GT edges folder (optional; else Canny)")
    ap.add_argument("--val_images", type=str, default=None, help="Optional val images")
    ap.add_argument("--val_edges",  type=str, default=None, help="Optional val edges")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)

    # Inference-only
    ap.add_argument("--inference", type=str, default=None, help="Single image path for inference")
    ap.add_argument("--weights",   type=str, default=None, help="Load model weights for inference/training resume")
    return ap.parse_args()

def build_model(algo):
    if algo == "hed": return HEDMini()
    else: return BDCNMini()

def main():
    set_seed(0)
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(args.algo).to(device)

    # Inference-only
    if args.inference:
        if not args.weights: raise SystemExit("Please provide --weights for inference")
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state["model"])
        img = imread_rgb(Path(args.inference))
        H,W = img.shape[:2]
        s = min(1.0, args.max_side/max(H,W))
        if s < 1.0: img = cv2.resize(img, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        x = to_tensor(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            sides, fused = model(x)
            outs = [torch.sigmoid(s) for s in sides] + [torch.sigmoid(fused)]
        outdir = Path(args.outdir or f"runs/{args.algo}/inference")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.inference).stem
        for i,p in enumerate(outs):
            arr = p[0,0].cpu().numpy()
            save_gray(outdir/f"{stem}_side{i+1 if i<4 else 'fused'}.png", arr)
        print(f"[inference] saved to {outdir.resolve()}")
        return

    # Training path
    if not args.images:
        raise SystemExit("Provide --images for training")
    train_loader = make_loader(args.images, args.edges, args.batch, args.max_side, train=True)
    val_loader = None
    if args.val_images:
        val_loader = make_loader(args.val_images, args.val_edges, batch=1, max_side=args.max_side, train=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1.0
    outdir = Path(args.outdir or f"runs/{args.algo}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state["model"])
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        print(f"[resume] loaded {args.weights}")

    for ep in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, opt, device, args.algo, ep)
        print(f"[train] epoch {ep} avg loss {loss:.4f}")
        miou = 0.0
        if val_loader:
            miou = evaluate_and_save(model, val_loader, device, args.algo, outdir/"val_preds")
        if args.save:
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}
            torch.save(ckpt, outdir/"last.pt")
            if miou >= best:
                best = miou
                torch.save(ckpt, outdir/"best.pt")
                print(f"[save] best updated: mIoU={best:.3f} → {outdir/'best.pt'}")

if __name__ == "__main__":
    main()
