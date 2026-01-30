#!/usr/bin/env python3
"""
Pyramid Stereo Matching Network (PSMNet) – simplified demo
Chang and Chen, CVPR 2018

Key ideas retained:
  - 2D feature extraction at 1/4 resolution
  - Spatial Pyramid Pooling (SPP) for multi-scale context
  - 4D cost volume (D x H x W x C) via concatenated features
  - 3D CNN regularization
  - Soft-argmin disparity regression

Training:
  Supervised L1 loss on synthetic disparity from data_synth/stereo:
    left/*.png, right/*.png, disp/*.png (Kitti-style, /16 scaling)
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Adjust path to your project layout as needed
sys.path.append(str(Path(__file__).parent.parent))

from utils.stereo_utils import disparity_to_depth, soft_argmin
from utils.visualization import colorize_depth, save_depth_visualization

# ================================================================
# Dataset: synthetic stereo (left/right/disp) like your other demos
# ================================================================


class StereoDispDataset(Dataset):
    """
    Stereo disparity dataset:

    root/
      left/*.png
      right/*.png
      disp/*.png    # uint16, disparity * 16 (Kitti-style)

    Optional resize.
    """

    def __init__(self, root: str, resize: tuple[int, int] | None = None):
        self.root = Path(root)
        self.left_root = self.root / "left"
        self.right_root = self.root / "right"
        self.disp_root = self.root / "disp"
        self.resize = resize

        self.left_paths = sorted(self.left_root.glob("*.png"))
        if not self.left_paths:
            raise RuntimeError(f"No PNGs found in {self.left_root}")

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_path = self.left_paths[idx]
        name = left_path.name
        right_path = self.right_root / name
        disp_path = self.disp_root / name

        # Load RGB
        left_bgr = cv2.imread(str(left_path))
        right_bgr = cv2.imread(str(right_path))
        if left_bgr is None or right_bgr is None:
            raise RuntimeError(f"Failed to read stereo pair {left_path}, {right_path}")

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right_rgb = (
            cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )

        # Load disparity (uint16, scaled by 16)
        disp_raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp_raw is None:
            raise RuntimeError(f"Failed to read disparity {disp_path}")
        disp = disp_raw.astype(np.float32) / 16.0  # disparity in pixels

        if self.resize is not None:
            w, h = self.resize
            left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)
            right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)

        left_t = torch.from_numpy(left_rgb).permute(2, 0, 1)  # (3,H,W)
        right_t = torch.from_numpy(right_rgb).permute(2, 0, 1)  # (3,H,W)
        disp_t = torch.from_numpy(disp).unsqueeze(0)  # (1,H,W)

        return left_t, right_t, disp_t


# ================================================================
# PSMNet components
# ================================================================


class SpatialPyramidPooling(nn.Module):
    """
    SPP module using adaptive pooling (close in spirit to PSMNet's SPP):
    pool to 1x1, 2x2, 4x4, 8x8 → 1x1 conv → upsample → concat.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.pool4 = nn.AdaptiveAvgPool2d((8, 8))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv after concatenation of (x + 4 pooled branches)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels + 4 * out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        p1 = self.pool1(x)
        p1 = self.conv1(p1)
        p1 = F.interpolate(p1, size=(H, W), mode="bilinear", align_corners=False)

        p2 = self.pool2(x)
        p2 = self.conv2(p2)
        p2 = F.interpolate(p2, size=(H, W), mode="bilinear", align_corners=False)

        p3 = self.pool3(x)
        p3 = self.conv3(p3)
        p3 = F.interpolate(p3, size=(H, W), mode="bilinear", align_corners=False)

        p4 = self.pool4(x)
        p4 = self.conv4(p4)
        p4 = F.interpolate(p4, size=(H, W), mode="bilinear", align_corners=False)

        out = torch.cat([x, p1, p2, p3, p4], dim=1)
        out = self.fusion(out)
        return out


class PSMNetSimplified(nn.Module):
    """
    Simplified PSMNet-like network:

    - Feature extractor: 1/4 resolution shared 2D CNN.
    - SPP: multi-scale context.
    - Cost volume: concat left/right features across disparities.
    - 3D CNN: regularization over (D, H, W).
    - soft-argmin: disparity regression.
    """

    def __init__(self, max_disp: int = 192):
        super().__init__()
        self.max_disp = max_disp
        # At 1/4 resolution, we sample D' = max_disp / 4 disparity planes
        self.num_disp = max_disp // 4

        # -------- Feature extraction (1/4 resolution) --------
        # conv1: /2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # conv2: /4
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # conv3: keep /4
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # SPP module: in_channels=32 → out_channels=32
        self.spp = SpatialPyramidPooling(32, 32)

        # -------- 3D CNN for cost volume regularization --------
        # Cost volume input channels = 2*C (left+right)
        in_ch_3d = 64
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_ch_3d, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_out = nn.Conv3d(32, 1, kernel_size=3, padding=1, bias=False)

    def feature_extraction(self, x):
        x = self.conv1(x)  # /2
        x = self.conv2(x)  # /4
        x = self.conv3(x)  # /4
        x = self.spp(x)  # /4 with context
        return x  # (B,32,H/4,W/4)

    def build_cost_volume(self, left_feat, right_feat):
        """
        Build 4D cost volume:

        cost[d, x, y] = concat(F_L(x,y), F_R(x-d,y)) at feature resolution.

        left_feat, right_feat: (B,C,Hf,Wf)
        returns: (B,2C,D',Hf,Wf) with D' = max_disp/4
        """
        B, C, H, W = left_feat.shape
        D = self.num_disp

        cost_volume = left_feat.new_zeros(B, 2 * C, D, H, W)

        for d in range(D):
            if d == 0:
                cost_volume[:, :C, d, :, :] = left_feat
                cost_volume[:, C:, d, :, :] = right_feat
            else:
                # Shift right features by d pixels to the right (feature space)
                # Valid range: x >= d
                cost_volume[:, :C, d, :, d:] = left_feat[:, :, :, d:]
                cost_volume[:, C:, d, :, d:] = right_feat[:, :, :, :-d]

        return cost_volume

    def forward(self, left, right):
        """
        left, right: (B,3,H,W) rectified stereo images.

        returns:
            disp_full: (B,1,H,W) disparity in pixels.
        """
        B, _, H, W = left.shape

        # 1) Feature extraction (shared weights)
        left_feat = self.feature_extraction(left)
        right_feat = self.feature_extraction(right)
        _, C, Hf, Wf = left_feat.shape

        # 2) Cost volume (B,2C,D',Hf,Wf)
        cost = self.build_cost_volume(left_feat, right_feat)

        # 3) 3D CNN regularization
        x = self.conv3d_1(cost)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_out(x)  # (B,1,D',Hf,Wf)
        x = x.squeeze(1)  # (B,D',Hf,Wf)

        # 4) Soft-argmin over disparity → [0, D'-1] at feature scale
        # soft_argmin expects cost volume (B,D,H,W)
        disp_feat_idx = soft_argmin(x)  # (B,Hf,Wf) in [0, D'-1]
        disp_feat_idx = disp_feat_idx.unsqueeze(1)  # (B,1,Hf,Wf)

        # Convert disparity indices → pixels at feature scale (each step = 4px)
        disp_feat = disp_feat_idx * 4.0  # (B,1,Hf,Wf)

        # 5) Upsample to full resolution (H,W)
        disp_full = F.interpolate(
            disp_feat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B,1,H,W)

        return disp_full


# ================================================================
# Training loop
# ================================================================


def train_psmnet(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    resize: tuple[int, int] = (192, 128),
    max_disp: int = 192,
    output_dir: str = "psmnet_runs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StereoDispDataset(data_root, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PSMNetSimplified(max_disp=max_disp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for left_t, right_t, disp_gt in dataloader:
            left_t = left_t.to(device)  # (B,3,H,W)
            right_t = right_t.to(device)  # (B,3,H,W)
            disp_gt = disp_gt.to(device)  # (B,1,H,W)

            optimizer.zero_grad()

            disp_pred = model(left_t, right_t)  # (B,1,H,W)

            # Mask invalid disparities (<=0)
            mask = disp_gt > 0
            if mask.sum() == 0:
                continue

            loss = F.l1_loss(disp_pred[mask], disp_gt[mask])

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}/{epochs}] L1 disparity loss: {avg_loss:.4f}")

    # Save model
    model_path = out_dir / "psmnet_simplified.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save training curve
    epochs_axis = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training L1 Loss")
    plt.title("PSMNet-simplified training loss")
    plt.grid(True)
    plt.tight_layout()
    loss_fig = out_dir / "train_loss.png"
    plt.savefig(loss_fig)
    plt.close()
    print(f"Saved loss curve to {loss_fig}")

    return model_path


# ================================================================
# Inference: predict disparity + depth on a single stereo pair
# ================================================================


def infer_psmnet(
    left_path: str,
    right_path: str,
    model_path: str,
    max_disp: int = 192,
    focal_length: float = 721.5,
    baseline: float = 0.54,
    output: str = "psmnet_output.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading images: {left_path}, {right_path}")

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    if left_img is None or right_img is None:
        raise ValueError("Could not load stereo images")

    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    H, W, _ = left_rgb.shape

    left_t = torch.from_numpy(left_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    right_t = torch.from_numpy(right_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    model = PSMNetSimplified(max_disp=max_disp).to(device)
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        disp_pred = model(left_t, right_t)  # (1,1,H,W)
        disp_np = disp_pred.squeeze().cpu().numpy()  # (H,W)

    # Convert disparity → depth
    depth = disparity_to_depth(
        torch.from_numpy(disp_np), focal_length, baseline
    ).numpy()

    # Some stats
    valid = disp_np > 0
    if valid.any():
        print(
            f"Disparity range: [{disp_np[valid].min():.2f}, {disp_np[valid].max():.2f}] px"
        )
        print(f"Depth range: [{depth[valid].min():.2f}, {depth[valid].max():.2f}] m")
    else:
        print("No valid disparities > 0 found.")

    # Visualization (same helper as your other demos)
    save_depth_visualization(depth, output, left_rgb * 255.0)
    print(f"Saved PSMNet visualization to {output}")


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PSMNet-style stereo demo (simplified)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---- Train ----
    p_train = subparsers.add_parser(
        "train", help="Train PSMNet on synthetic stereo data"
    )
    p_train.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of stereo data (with left/right/disp)",
    )
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--resize_w", type=int, default=192)
    p_train.add_argument("--resize_h", type=int, default=128)
    p_train.add_argument("--max_disp", type=int, default=192)
    p_train.add_argument("--output_dir", type=str, default="psmnet_runs")

    # ---- Infer ----
    p_infer = subparsers.add_parser("infer", help="Run inference with trained PSMNet")
    p_infer.add_argument("--left", type=str, required=True, help="Left image path")
    p_infer.add_argument("--right", type=str, required=True, help="Right image path")
    p_infer.add_argument(
        "--model", type=str, required=True, help="Path to psmnet_simplified.pth"
    )
    p_infer.add_argument("--max_disp", type=int, default=192)
    p_infer.add_argument("--focal_length", type=float, default=721.5)
    p_infer.add_argument("--baseline", type=float, default=0.54)
    p_infer.add_argument("--output", type=str, default="psmnet_output.png")

    args = parser.parse_args()

    if args.mode == "train":
        resize = (args.resize_w, args.resize_h)
        train_psmnet(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resize=resize,
            max_disp=args.max_disp,
            output_dir=args.output_dir,
        )
    elif args.mode == "infer":
        infer_psmnet(
            left_path=args.left,
            right_path=args.right,
            model_path=args.model,
            max_disp=args.max_disp,
            focal_length=args.focal_length,
            baseline=args.baseline,
            output=args.output,
        )


if __name__ == "__main__":
    main()


"""
python generate_synthetic_stereo_depth.py \
    --out_root data_synth \
    --num_scenes 100 \
    --width 192 --height 128

    
python psmnet_demo.py train \
    --data_root data_synth/stereo \
    --epochs 20 \
    --batch_size 4 \
    --resize_w 192 --resize_h 128 \
    --max_disp 64 \
    --output_dir psmnet_runs


    
python psmnet_demo.py infer \
    --left  data_synth/stereo/left/0000.png \
    --right data_synth/stereo/right/0000.png \
    --model psmnet_runs/psmnet_simplified.pth \
    --max_disp 64 \
    --output psmnet_pred_0000.png

"""
