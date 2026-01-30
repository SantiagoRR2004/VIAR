#!/usr/bin/env python3
"""
GC-Net-style stereo demo on synthetic data.

- Feature extractor: 2D CNN on left/right images.
- Cost volume: (B, 2C, D', H', W') built at 1/4 resolution.
- 3D CNN: regularizes cost over (D', H', W').
- Soft-argmin: differentiable disparity estimation.
- Training + inference CLI, similar to dispnet_demo.py.

This is a simplified version of GC-Net but keeps the core ideas.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================================================================
# Dataset (same convention as DispNet synthetic data)
# ================================================================


class StereoDispDataset(Dataset):
    """
    For synthetic stereo data from generate_synthetic_stereo_depth.py:

    data_synth/
        stereo/
            left/*.png    # uint8 BGR RGB
            right/*.png
            disp/*.png    # uint16 disparity * 16  (fixed-point)

    We return:
        left:  (3,H,W) float32 in [0,1]
        right: (3,H,W) float32 in [0,1]
        disp:  (1,H,W) float32 disparity in pixels
    """

    def __init__(self, root: str, resize: tuple[int, int] | None = None):
        self.root = Path(root)
        self.left_paths = sorted((self.root / "left").glob("*.png"))
        self.right_root = self.root / "right"
        self.disp_root = self.root / "disp"
        self.resize = resize

        if not self.left_paths:
            raise RuntimeError(f"No PNGs found in {self.root / 'left'}")

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_path = self.left_paths[idx]
        name = left_path.name
        right_path = self.right_root / name
        disp_path = self.disp_root / name

        # Load left/right RGB
        left_bgr = cv2.imread(str(left_path))
        right_bgr = cv2.imread(str(right_path))
        if left_bgr is None or right_bgr is None:
            raise RuntimeError(f"Failed to read stereo pair for {name}")

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right_rgb = (
            cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )

        # Load disparity (uint16) -> float32 pixels
        disp_raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp_raw is None:
            raise RuntimeError(f"Failed to read disparity {disp_path}")

        # We stored disp_fixed = disp * 16 -> undo that
        disp = disp_raw.astype(np.float32) / 16.0  # disparity in pixels

        if self.resize is not None:
            w, h = self.resize  # (width, height)
            left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)
            right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        left_t = torch.from_numpy(left_rgb).permute(2, 0, 1)  # (3,H,W)
        right_t = torch.from_numpy(right_rgb).permute(2, 0, 1)  # (3,H,W)
        disp_t = torch.from_numpy(disp).unsqueeze(0)  # (1,H,W)

        return left_t, right_t, disp_t


# ================================================================
# GC-Net-style model
# ================================================================


class FeatureExtractor2D(nn.Module):
    """
    Simple 2D CNN feature extractor for GC-Net.

    Input: (B,3,H,W)
    Output: (B,C,H/4,W/4)   with C = base_channels * 4.

    This is much simpler than the original ResNet-like unary tower,
    but keeps the idea: shared 2D CNN applied to both views.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        C = base_channels

        # conv1: /2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        # conv2: /4
        self.conv2 = nn.Sequential(
            nn.Conv2d(C, 2 * C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * C),
            nn.ReLU(inplace=True),
        )
        # conv3: keep resolution (/4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * C, 4 * C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * C),
            nn.ReLU(inplace=True),
        )

        self.out_channels = 4 * C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # (B,C,H/2,W/2)
        x = self.conv2(x)  # (B,2C,H/4,W/4)
        x = self.conv3(x)  # (B,4C,H/4,W/4)
        return x


class GCNet(nn.Module):
    """
    Simplified GC-Net:

    - Shared 2D feature extractor on left/right → F_L, F_R (B,C,H',W').
    - Cost volume: concat(F_L(x,y), F_R(x-d,y)) for disparities d=0..D'-1,
      at 1/4 resolution → (B, 2C, D', H', W').
    - 3D CNN: regularize over (D',H',W') → cost_agg (B,1,D',H',W').
    - Soft-argmin along disparity dimension → disparity at 1/4 resolution.
    - Upsample to full resolution and multiply by scale factor (e.g. 4).

    max_disp is in full-resolution pixels; we build a smaller D' = max_disp // disp_down.
    """

    def __init__(self, max_disp: int = 64, base_channels: int = 16, disp_down: int = 4):
        super().__init__()
        assert max_disp % disp_down == 0, "max_disp must be divisible by disp_down"
        self.max_disp = max_disp
        self.disp_down = disp_down
        self.num_disp = max_disp // disp_down  # disparity samples at feature scale

        # Shared 2D feature extractor
        self.unary = FeatureExtractor2D(base_channels=base_channels)
        C = self.unary.out_channels

        # 3D CNN cost regularization
        # Input: (B,2C,D',H',W'), so channels_in = 2*C
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(2 * C, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3d_4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        # Output after conv3d_4: (B,1,D',H',W') = cost volume

    def build_cost_volume(
        self, featL: torch.Tensor, featR: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the concatenation-based cost volume:

        featL, featR: (B,C,H',W')
        Returns: (B,2C,D',H',W')
        """
        B, C, H, W = featL.shape
        D = self.num_disp

        # Initialize cost volume with zeros
        cost = featL.new_zeros((B, 2 * C, D, H, W))

        # For each disparity d, shift right features and stack
        # At disparity d, valid matches exist for x >= d; we simply
        # leave invalid positions as zeros.
        for d in range(D):
            if d > 0:
                # Left features at x >= d
                cost[:, :C, d, :, d:] = featL[:, :, :, d:]
                # Right features at x < W-d (shifted left)
                cost[:, C:, d, :, d:] = featR[:, :, :, :-d]
            else:
                cost[:, :C, d, :, :] = featL
                cost[:, C:, d, :, :] = featR

        return cost

    def soft_argmin(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        cost_volume: (B,1,D',H',W')  (lower is better)
        Returns disparity at feature scale: (B,1,H',W') in [0, D'-1]
        """
        B, _, D, H, W = cost_volume.shape

        # Convert costs to probabilities along disparity dim:
        # lower cost → higher probability.
        cost = cost_volume.squeeze(1)  # (B,D,H,W)
        prob = F.softmax(-cost, dim=1)  # (B,D,H,W)

        # Disparity values 0..D-1
        disp_values = torch.arange(
            D, device=cost_volume.device, dtype=cost_volume.dtype
        ).view(1, D, 1, 1)
        disp_map = torch.sum(prob * disp_values, dim=1, keepdim=True)  # (B,1,H,W)
        return disp_map

    def forward(self, imgL: torch.Tensor, imgR: torch.Tensor) -> torch.Tensor:
        """
        imgL, imgR: (B,3,H,W) rectified stereo images.

        Returns:
            disp_full: (B,1,H,W) full-resolution disparity in pixels.
        """
        B, _, H, W = imgL.shape

        # 1. Unary 2D features (shared weights)
        featL = self.unary(imgL)  # (B,C,H/4,W/4)
        featR = self.unary(imgR)  # (B,C,H/4,W/4)
        _, C, Hf, Wf = featL.shape

        # 2. Cost volume construction at feature resolution
        cost = self.build_cost_volume(featL, featR)  # (B,2C,D',Hf,Wf)

        # 3. 3D CNN regularization
        x = self.conv3d_1(cost)  # (B,32,D',Hf,Wf)
        x = self.conv3d_2(x)  # (B,32,D',Hf,Wf)
        x = self.conv3d_3(x)  # (B,32,D',Hf,Wf)
        x = self.conv3d_4(x)  # (B,1,D',Hf,Wf)   cost_agg

        # 4. Soft-argmin -> disparity at feature scale (in "disparity steps")
        disp_feat = self.soft_argmin(x)  # (B,1,Hf,Wf) in [0, D'-1]

        # 5. Convert to full-resolution pixels:
        # each disparity step corresponds to 'disp_down' pixels.
        disp_feat = disp_feat * float(self.disp_down)

        # 6. Upsample disparity to full resolution
        disp_full = F.interpolate(
            disp_feat, size=(H, W), mode="bilinear", align_corners=False
        )
        return disp_full


# ================================================================
# Visualization helpers
# ================================================================


def colorize_disparity(disp: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    Colorize a disparity map (H,W) into RGB using a Jet colormap.
    """
    valid = disp > 0
    if vmin is None:
        vmin = float(disp[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(disp[valid].max()) if valid.any() else 1.0

    disp_norm = (disp - vmin) / (vmax - vmin + 1e-8)
    disp_norm = np.clip(disp_norm, 0, 1)
    disp_uint8 = (disp_norm * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)
    return disp_color


# ================================================================
# Training loop
# ================================================================


def train_gcnet(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    resize: tuple[int, int] | None = (192, 128),
    max_disp: int = 64,
    output_dir: str = "gcnet_runs",
):
    """
    Train GCNet-style model on synthetic stereo data.

    data_root: e.g. "data_synth/stereo"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StereoDispDataset(data_root, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GCNet(max_disp=max_disp, base_channels=16, disp_down=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for left_t, right_t, disp_gt in dataloader:
            left_t = left_t.to(device)
            right_t = right_t.to(device)
            disp_gt = disp_gt.to(device)

            optimizer.zero_grad()

            disp_pred = model(left_t, right_t)  # (B,1,H,W)
            loss = F.l1_loss(disp_pred, disp_gt)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}/{epochs}] training loss: {avg_loss:.4f}")

    # Save model and loss curve
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "gcnet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    epochs_axis = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("GCNet-style Training on Synthetic Stereo")
    plt.grid(True)
    loss_plot_path = out_dir / "train_loss.png"
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    return model_path


# ================================================================
# Inference
# ================================================================


def run_gcnet_inference(
    left_path: str,
    right_path: str,
    model_path: str,
    output: str = "gcnet_pred.png",
    resize: tuple[int, int] | None = (192, 128),
    max_disp: int = 64,
):
    """
    Run inference with a trained GCNet-style model.

    - Loads left/right RGB images.
    - Resizes to 'resize' if given.
    - Predicts disparity.
    - Saves side-by-side visualization: [left | disparity].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    print(f"Loading stereo pair:\n  {left_path}\n  {right_path}")

    left_bgr = cv2.imread(left_path)
    right_bgr = cv2.imread(right_path)
    if left_bgr is None or right_bgr is None:
        raise ValueError("Could not load stereo images")

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    H_orig, W_orig, _ = left_rgb.shape
    print(f"Original size: {H_orig} x {W_orig}")

    if resize is not None:
        w, h = resize
        left_resized = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        left_resized = left_rgb
        right_resized = right_rgb

    # Preprocess to tensors
    left = left_resized.astype(np.float32) / 255.0
    right = right_resized.astype(np.float32) / 255.0
    left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # Build and load model
    model = GCNet(max_disp=max_disp, base_channels=16, disp_down=4).to(device)
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        left_t = left_t.to(device)
        right_t = right_t.to(device)
        disp_pred = model(left_t, right_t)  # (1,1,H,W)
        disp_np = disp_pred.squeeze().cpu().numpy()

    disp_np = np.clip(disp_np, 0.0, max_disp)
    # Adaptive vmin/vmax so things don't look all dark-blue
    valid = disp_np > 0
    if valid.any():
        vmin = float(np.percentile(disp_np[valid], 5))
        vmax = float(np.percentile(disp_np[valid], 95))
    else:
        vmin, vmax = 0.0, max_disp

    disp_color = colorize_disparity(disp_np, vmin=vmin, vmax=vmax)

    # Resize disp to original size for easier comparison
    disp_color_full = cv2.resize(
        disp_color, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )

    # Side-by-side: left original + disparity
    vis = np.concatenate([left_rgb, disp_color_full], axis=1)

    cv2.imwrite(output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output}")


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="GC-Net-style stereo depth demo")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----- Train mode -----
    p_train = subparsers.add_parser(
        "train", help="Train GCNet on synthetic stereo data"
    )
    p_train.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of stereo dataset (e.g., data_synth/stereo)",
    )
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--resize_w", type=int, default=192)
    p_train.add_argument("--resize_h", type=int, default=128)
    p_train.add_argument("--max_disp", type=int, default=64)
    p_train.add_argument("--output_dir", type=str, default="gcnet_runs")

    # ----- Inference mode -----
    p_infer = subparsers.add_parser("infer", help="Run inference with trained GCNet")
    p_infer.add_argument("--left", type=str, required=True, help="Path to left image")
    p_infer.add_argument("--right", type=str, required=True, help="Path to right image")
    p_infer.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g. gcnet_runs/gcnet.pth)",
    )
    p_infer.add_argument(
        "--output",
        type=str,
        default="gcnet_pred.png",
        help="Path to save visualization",
    )
    p_infer.add_argument("--resize_w", type=int, default=192)
    p_infer.add_argument("--resize_h", type=int, default=128)
    p_infer.add_argument("--max_disp", type=int, default=64)

    args = parser.parse_args()

    if args.mode == "train":
        resize = (args.resize_w, args.resize_h)
        train_gcnet(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resize=resize,
            max_disp=args.max_disp,
            output_dir=args.output_dir,
        )
    elif args.mode == "infer":
        resize = (args.resize_w, args.resize_h)
        run_gcnet_inference(
            left_path=args.left,
            right_path=args.right,
            model_path=args.model,
            output=args.output,
            resize=resize,
            max_disp=args.max_disp,
        )


if __name__ == "__main__":
    main()


"""
# Train on your existing synthetic set
python gcnet_demo.py train \
    --data_root data_synth/stereo \
    --epochs 20 \
    --batch_size 4 \
    --resize_w 192 --resize_h 128 \
    --max_disp 64 \
    --output_dir gcnet_runs

# Inference on one pair
python gcnet_demo.py infer \
    --left data_synth/stereo/left/0000.png \
    --right data_synth/stereo/right/0000.png \
    --model gcnet_runs/gcnet.pth \
    --resize_w 192 --resize_h 128 \
    --max_disp 64 \
    --output gcnet_pred_0000.png

"""
