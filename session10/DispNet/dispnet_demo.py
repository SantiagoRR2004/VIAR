import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Small helper blocks ----------


def conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
    """Standard conv + ReLU."""
    return nn.Sequential(
        nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.ReLU(inplace=True),
    )


def deconv(in_ch, out_ch, kernel_size=4, stride=2, padding=1):
    """Transposed conv (upconv) + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.ReLU(inplace=True),
    )


class DispNet(nn.Module):
    """
    DispNet-style network (Mayer et al. 2016, simplified but faithful):

    - Input: concatenated left & right RGB images (B,6,H,W).
    - Encoder: conv1 .. conv6b (downsampling).
    - Decoder: upconv6..1, iconv6..1, disparity predictions pr6..1.
    - Skip connections from encoder to decoder.
    - Multi-scale disparities; pr1 is the final output.

    NOTE: You can train either:
      - only on pr1 (full-res), or
      - with a weighted multi-scale loss over pr1..pr6.
    """

    def __init__(self, max_disp: float = 192.0, scale_factor: float = 0.3):
        super().__init__()
        self.max_disp = max_disp
        self.scale_factor = scale_factor  # scale for raw disparity outputs

        # ---------- Encoder (contracting path) ----------
        # These kernel sizes/strides follow the FlowNet/DispNet spirit.
        self.conv1 = conv(6, 64, 7, stride=2, padding=3)  # /2
        self.conv2 = conv(64, 128, 5, stride=2, padding=2)  # /4
        self.conv3a = conv(128, 256, 5, stride=2, padding=2)  # /8
        self.conv3b = conv(256, 256, 3, stride=1, padding=1)
        self.conv4a = conv(256, 512, 3, stride=2, padding=1)  # /16
        self.conv4b = conv(512, 512, 3, stride=1, padding=1)
        self.conv5a = conv(512, 512, 3, stride=2, padding=1)  # /32
        self.conv5b = conv(512, 512, 3, stride=1, padding=1)
        self.conv6a = conv(512, 1024, 3, stride=2, padding=1)  # /64
        self.conv6b = conv(1024, 1024, 3, stride=1, padding=1)

        # ---------- Decoder (expanding path) ----------
        # Upconvs
        self.upconv5 = deconv(1024, 512)  # /32
        self.upconv4 = deconv(512, 256)  # /16
        self.upconv3 = deconv(256, 128)  # /8
        self.upconv2 = deconv(128, 64)  # /4
        self.upconv1 = deconv(64, 32)  # /2
        self.upconv0 = deconv(32, 16)  # /1 (optional final upsample)

        # Iconvs (after concatenation of upconv + skip + upsampled disparity)
        # Channel counts mirror the original DispNet table (approx).
        # iconv5: upconv5 (512) + conv5b (512) + pr6_up (1) = 1025
        self.iconv5 = conv(512 + 512 + 1, 512, 3, 1, 1)
        # iconv4: upconv4 (256) + conv4b (512) + pr5_up (1) = 769
        self.iconv4 = conv(256 + 512 + 1, 256, 3, 1, 1)
        # iconv3: upconv3 (128) + conv3b (256) + pr4_up (1) = 385
        self.iconv3 = conv(128 + 256 + 1, 128, 3, 1, 1)
        # iconv2: upconv2 (64) + conv2 (128) + pr3_up (1) = 193
        self.iconv2 = conv(64 + 128 + 1, 64, 3, 1, 1)
        # iconv1: upconv1 (32) + conv1 (64) + pr2_up (1) = 97
        self.iconv1 = conv(32 + 64 + 1, 32, 3, 1, 1)
        # iconv0: upconv0 (16) + pr1_up (1) = 17  (this one was already correct)
        self.iconv0 = conv(16 + 1, 16, 3, 1, 1)

        # Disparity prediction heads at each scale
        self.pr6 = nn.Conv2d(1024, 1, 3, 1, 1)
        self.pr5 = nn.Conv2d(512, 1, 3, 1, 1)
        self.pr4 = nn.Conv2d(256, 1, 3, 1, 1)
        self.pr3 = nn.Conv2d(128, 1, 3, 1, 1)
        self.pr2 = nn.Conv2d(64, 1, 3, 1, 1)
        self.pr1 = nn.Conv2d(32, 1, 3, 1, 1)
        self.pr0 = nn.Conv2d(16, 1, 3, 1, 1)  # final full-res pred

    def disp_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure non-negative disparities and keep them in a reasonable range.

        Following many FlowNet/DispNet implementations:
        disparity = scale_factor * ReLU(raw).
        """
        x = self.scale_factor * F.relu(x)
        if self.max_disp is not None:
            x = torch.clamp(x, 0.0, self.max_disp)
        return x

    def forward(
        self, imgL: torch.Tensor, imgR: torch.Tensor, return_pyramid: bool = False
    ):
        """
        imgL, imgR: (B,3,H,W) rectified stereo images.
        return_pyramid:
            - False: return only final full-res disparity.
            - True:  also return [disp0, disp1, ..., disp6].

        Returns:
            disp0: (B,1,H,W) final disparity.
            (optionally) list of all multi-scale disparities.
        """
        # ---------- Encoder ----------
        x = torch.cat([imgL, imgR], dim=1)  # (B,6,H,W)

        conv1 = self.conv1(x)  # /2
        conv2 = self.conv2(conv1)  # /4
        conv3a = self.conv3a(conv2)  # /8
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)  # /16
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)  # /32
        conv5b = self.conv5b(conv5a)
        conv6a = self.conv6a(conv5b)  # /64
        conv6b = self.conv6b(conv6a)

        # ---------- Decoder with multi-scale predictions ----------

        # Scale 6 (coarsest)
        pr6 = self.disp_activation(self.pr6(conv6b))  # (B,1,H/64,W/64)
        pr6_up = F.interpolate(
            pr6, scale_factor=2, mode="bilinear", align_corners=False
        )

        upconv5 = self.upconv5(conv6b)  # (B,512,H/32,W/32)
        # concat upsampled disparity + skip connection + upconv
        iconv5_in = torch.cat([upconv5, conv5b, pr6_up], dim=1)
        iconv5 = self.iconv5(iconv5_in)
        pr5 = self.disp_activation(self.pr5(iconv5))
        pr5_up = F.interpolate(
            pr5, scale_factor=2, mode="bilinear", align_corners=False
        )

        upconv4 = self.upconv4(iconv5)  # (B,256,H/16,W/16)
        iconv4_in = torch.cat([upconv4, conv4b, pr5_up], dim=1)
        iconv4 = self.iconv4(iconv4_in)
        pr4 = self.disp_activation(self.pr4(iconv4))
        pr4_up = F.interpolate(
            pr4, scale_factor=2, mode="bilinear", align_corners=False
        )

        upconv3 = self.upconv3(iconv4)  # (B,128,H/8,W/8)
        iconv3_in = torch.cat([upconv3, conv3b, pr4_up], dim=1)
        iconv3 = self.iconv3(iconv3_in)
        pr3 = self.disp_activation(self.pr3(iconv3))
        pr3_up = F.interpolate(
            pr3, scale_factor=2, mode="bilinear", align_corners=False
        )

        upconv2 = self.upconv2(iconv3)  # (B,64,H/4,W/4)
        iconv2_in = torch.cat([upconv2, conv2, pr3_up], dim=1)
        iconv2 = self.iconv2(iconv2_in)
        pr2 = self.disp_activation(self.pr2(iconv2))
        pr2_up = F.interpolate(
            pr2, scale_factor=2, mode="bilinear", align_corners=False
        )

        upconv1 = self.upconv1(iconv2)  # (B,32,H/2,W/2)
        iconv1_in = torch.cat([upconv1, conv1, pr2_up], dim=1)
        iconv1 = self.iconv1(iconv1_in)
        pr1 = self.disp_activation(self.pr1(iconv1))
        pr1_up = F.interpolate(
            pr1, scale_factor=2, mode="bilinear", align_corners=False
        )

        # Optional final refinement at full resolution
        upconv0 = self.upconv0(iconv1)  # (B,16,H,W)
        iconv0_in = torch.cat([upconv0, pr1_up], dim=1)  # (B,17,H,W)
        iconv0 = self.iconv0(iconv0_in)
        pr0 = self.disp_activation(self.pr0(iconv0))  # final (B,1,H,W)

        if return_pyramid:
            # From coarsest to finest, or vice versa as you prefer
            return pr0, [pr0, pr1, pr2, pr3, pr4, pr5, pr6]
        else:
            return pr0


import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================================================================
# Stereo synthetic dataset (matches generate_synthetic_stereo_depth.py)
# ==================================================================


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


# ==================================================================
# Multi-scale disparity loss (optional, in DispNet spirit)
# ==================================================================


def multiscale_disp_loss(
    disp_pyramid: list[torch.Tensor],
    disp_gt: torch.Tensor,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """
    Multi-scale L1 loss for disparity.

    disp_pyramid: [pr0, pr1, pr2, pr3, pr4, pr5, pr6]
      each prX is (B,1,Hx,Wx)
    disp_gt: (B,1,H,W) ground-truth disparity at full resolution
    weights: 7 weights, from finest to coarsest.
    """
    if weights is None:
        # Higher weight for fine scales, lower for coarse scales
        weights = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    loss = 0.0
    for pr, w in zip(disp_pyramid, weights):
        # Resize GT disparity to the current prediction size
        gt_resized = F.interpolate(disp_gt, size=pr.shape[2:], mode="nearest")
        loss = loss + w * F.l1_loss(pr, gt_resized)
    return loss


# ==================================================================
# Simple colorization for disparity
# ==================================================================


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


# ==================================================================
# Training loop for DispNet
# ==================================================================


def train_dispnet(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    resize: tuple[int, int] | None = (192, 128),
    max_disp: float = 192.0,
    use_multiscale: bool = True,
    output_dir: str = "dispnet_runs",
):
    """
    Train DispNet on synthetic stereo data.

    data_root: e.g. "data_synth/stereo"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StereoDispDataset(data_root, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DispNet(max_disp=max_disp).to(device)
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

            if use_multiscale:
                disp0, pyramid = model(left_t, right_t, return_pyramid=True)
                loss = multiscale_disp_loss(pyramid, disp_gt)
            else:
                disp0 = model(left_t, right_t)  # (B,1,H,W)
                loss = F.l1_loss(disp0, disp_gt)

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

    model_path = out_dir / "dispnet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    epochs_axis = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("DispNet Training on Synthetic Stereo")
    plt.grid(True)
    loss_plot_path = out_dir / "train_loss.png"
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    return model_path


# ==================================================================
# Inference for DispNet: left+right → disparity + visualization
# ==================================================================


def run_dispnet_inference(
    left_path: str,
    right_path: str,
    model_path: str,
    output: str = "dispnet_pred.png",
    resize: tuple[int, int] | None = (192, 128),
    max_disp: float = 192.0,
):
    """
    Run inference with a trained DispNet model.

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
    model = DispNet(max_disp=max_disp).to(device)
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        left_t = left_t.to(device)
        right_t = right_t.to(device)
        disp0 = model(left_t, right_t)  # (1,1,H,W)
        disp_np = disp0.squeeze().cpu().numpy()

    # Colorize disparity
    disp_np = np.clip(disp_np, 0.0, max_disp)
    disp_color = colorize_disparity(disp_np, vmin=0.0, vmax=max_disp)

    # Resize disp to original size for easier comparison
    disp_color_full = cv2.resize(
        disp_color, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )

    # Side-by-side: left original + disparity
    vis = np.concatenate([left_rgb, disp_color_full], axis=1)

    cv2.imwrite(output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output}")


# ==================================================================
# CLI (train + infer)
# ==================================================================


def main():
    parser = argparse.ArgumentParser(description="DispNet-style stereo depth demo")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----- Train mode -----
    p_train = subparsers.add_parser(
        "train", help="Train DispNet on synthetic stereo data"
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
    p_train.add_argument("--max_disp", type=float, default=192.0)
    p_train.add_argument(
        "--no_multiscale",
        action="store_true",
        help="Disable multi-scale loss (use only final disparity)",
    )
    p_train.add_argument("--output_dir", type=str, default="dispnet_runs")

    # ----- Inference mode -----
    p_infer = subparsers.add_parser("infer", help="Run inference with trained DispNet")
    p_infer.add_argument("--left", type=str, required=True, help="Path to left image")
    p_infer.add_argument("--right", type=str, required=True, help="Path to right image")
    p_infer.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g. dispnet_runs/dispnet.pth)",
    )
    p_infer.add_argument(
        "--output",
        type=str,
        default="dispnet_pred.png",
        help="Path to save visualization",
    )
    p_infer.add_argument("--resize_w", type=int, default=192)
    p_infer.add_argument("--resize_h", type=int, default=128)
    p_infer.add_argument("--max_disp", type=float, default=192.0)

    args = parser.parse_args()

    if args.mode == "train":
        resize = (args.resize_w, args.resize_h)
        use_multiscale = not args.no_multiscale

        train_dispnet(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resize=resize,
            max_disp=args.max_disp,
            use_multiscale=use_multiscale,
            output_dir=args.output_dir,
        )
    elif args.mode == "infer":
        resize = (args.resize_w, args.resize_h)
        run_dispnet_inference(
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
python generate_synthetic_stereo_depth.py \
    --out_root data_synth \
    --num_scenes 100 \
    --width 192 --height 128


    
Training: 

python dispnet_demo.py train \
    --data_root data_synth/stereo \
    --epochs 20 \
    --batch_size 4 \
    --resize_w 192 --resize_h 128 \
    --output_dir dispnet_runs



Inference:

ls data_synth/stereo/left
# 0000.png  0001.png  ...

python dispnet_demo.py infer \
    --left data_synth/stereo/left/0000.png \
    --right data_synth/stereo/right/0000.png \
    --model dispnet_runs/dispnet.pth \
    --resize_w 192 --resize_h 128 \
    --output dispnet_pred_0000.png


"""
