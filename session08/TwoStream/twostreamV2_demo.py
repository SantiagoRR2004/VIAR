# two_stream_demo.py
"""
Two-Stream Convolutional Networks (Working Demo)
- Spatial stream: RGB frame (VGG16 ImageNet)
- Temporal stream: Stacked TV-L1 optical flow (20 channels)
- Late fusion: average logits
NOTE: Uses ImageNet labels for demonstration. For true action classes, fine-tune on UCF101/Kinetics.
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

# -----------------------------
# Utilities: labels & transforms
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_imagenet_labels():
    # Try to read from weights meta; fallback to generic names
    try:
        w = VGG16_Weights.IMAGENET1K_V1
        meta = getattr(w, "meta", {})
        cats = meta.get("categories", None)
        if cats and len(cats) == 1000:
            return cats
    except Exception:
        pass
    # Fallback: readable placeholders
    return [f"class_{i}" for i in range(1000)]


def build_rgb_transform(size: int = 224):
    # Use the canonical ImageNet preprocessing (stable across versions)
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def center_crop_nd(arr: np.ndarray, size: int = 224) -> np.ndarray:
    # arr: HxWxC
    h, w, _ = arr.shape
    ch, cw = (h - size) // 2, (w - size) // 2
    return arr[ch : ch + size, cw : cw + size, :]


# -----------------------------
# Optical Flow (TV-L1)
# -----------------------------
# -----------------------------
# Optical Flow (TV-L1 or Farnebäck fallback)
# -----------------------------
def _farneback(prev, nxt):
    # Stronger params for sports clips (larger winsize, more iters)
    return cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )  # HxWx2 float32


def compute_flow_stack(
    frames_gray: list,
    num_pairs: int = 10,
    out_size: int = 224,
    method: str = "farneback",
) -> tuple[np.ndarray, dict]:
    """
    Returns:
      flow_stack: HxWx(2*num_pairs) float32
      stats: dict with per-pair magnitudes and global stats
    """
    assert len(frames_gray) >= num_pairs + 1, "Need at least num_pairs+1 frames."

    # ---- choose a centered window around the middle of the clip ----
    T = len(frames_gray)
    mid = T // 2
    half = num_pairs // 2
    start = max(0, min(T - (num_pairs + 1), mid - half))
    window = frames_gray[start : start + num_pairs + 1]

    flows = []
    pair_means, pair_maxes = [], []

    for i in range(num_pairs):
        prev = window[i]
        nxt = window[i + 1]
        # mild denoise helps Farneback
        prev_b = cv2.GaussianBlur(prev, (5, 5), 0)
        nxt_b = cv2.GaussianBlur(nxt, (5, 5), 0)

        flow = _farneback(prev_b, nxt_b)  # HxWx2
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        pair_means.append(float(mag.mean()))
        pair_maxes.append(float(mag.max()))
        flows.append(flow)

    # Stack flows along channel: (u1,v1,...,uK,vK)
    flow_stack = np.concatenate(flows, axis=2)  # HxWx(2K)

    # Resize shortest side to 256 then center crop to out_size
    H, W = flow_stack.shape[:2]
    scale = 256.0 / min(H, W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    flow_resized = cv2.resize(flow_stack, (newW, newH), interpolation=cv2.INTER_LINEAR)

    ch, cw = (newH - out_size) // 2, (newW - out_size) // 2
    flow_cropped = flow_resized[ch : ch + out_size, cw : cw + out_size, :].astype(
        np.float32
    )

    stats = {
        "pair_mean_mag": pair_means,
        "pair_max_mag": pair_maxes,
        "global_mean": float(np.mean(pair_means)),
        "global_max": float(np.max(pair_maxes)),
        "window_start": start,
    }
    return flow_cropped, stats


def flow_magnitude_stats(flow_stack_hw20: np.ndarray) -> tuple[float, float]:
    # flow_stack: HxWx(2K); compute mean & max magnitude over all K
    H, W, C = flow_stack_hw20.shape
    assert C % 2 == 0
    uvs = flow_stack_hw20.reshape(H, W, -1, 2)  # HxWxKx2
    mag = np.sqrt((uvs[..., 0] ** 2 + uvs[..., 1] ** 2))  # HxWxK
    return float(mag.mean()), float(mag.max())


def save_flowviz(flow_stack_hw20: np.ndarray, path: str):
    # Save a single-frame visualization: average magnitude normalized to 0..255
    H, W, C = flow_stack_hw20.shape
    uvs = flow_stack_hw20.reshape(H, W, -1, 2)
    mag = np.sqrt((uvs[..., 0] ** 2 + uvs[..., 1] ** 2)).mean(axis=2)  # HxW
    m = np.clip((mag + 1.0) / 2.0, 0, 1)  # our flows are in [-1,1] approx
    img = (m * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)


# -----------------------------
# Flow visualizations
# -----------------------------
def save_flow_hsv(flow_uv: np.ndarray, path: str):
    """
    flow_uv: HxWx2 (u,v) single-pair flow
    Saves classic HSV wheel viz (H=angle, S=1, V=normalized magnitude).
    """
    u, v = flow_uv[..., 0], flow_uv[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)  # ang in [0,360)
    # Normalize magnitude to [0,1] by per-image max (avoid gray)
    m = mag / (mag.max() + 1e-6)

    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV H: 0..180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(m * 255, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, bgr)


def save_flow_stack_preview(flow_stack_hw2k: np.ndarray, path_prefix: str):
    """
    Saves first, middle, and last flow pairs as HSV images.
    """
    H, W, C = flow_stack_hw2k.shape
    K = C // 2
    idxs = [0, K // 2, K - 1] if K >= 3 else list(range(K))
    for j, k in enumerate(idxs):
        uv = flow_stack_hw2k[..., 2 * k : 2 * k + 2]
        save_flow_hsv(uv, f"{path_prefix}_pair{k:02d}.png")


# -----------------------------
# Two-Stream Model
# -----------------------------
class VGGStream(nn.Module):
    """Wrap torchvision VGG16 into a 'stream' with customizable first conv."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        inflate_from_rgb: bool = False,
    ):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Replace first conv if needed
        if in_channels != 3:
            old_conv: nn.Conv2d = base.features[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if inflate_from_rgb:
                # Inflate RGB weights to N channels by repeating/averaging
                with torch.no_grad():
                    w = old_conv.weight  # [64,3,3,3]
                    # Average across RGB and replicate to in_channels
                    w_avg = w.mean(dim=1, keepdim=True)  # [64,1,3,3]
                    w_rep = w_avg.repeat(1, in_channels, 1, 1) * (
                        3.0 / float(in_channels)
                    )
                    new_conv.weight.copy_(w_rep)
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
            base.features[0] = new_conv

        # Replace classifier final layer with correct num_classes (keep pretrained up to penultimate)
        if num_classes != 1000:
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)

        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # logits


class TwoStreamNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.spatial = VGGStream(in_channels=3, num_classes=num_classes)
        self.temporal = VGGStream(
            in_channels=20, num_classes=num_classes, inflate_from_rgb=True
        )

    @torch.no_grad()
    def forward(
        self, rgb: torch.Tensor, flow20: torch.Tensor, fusion: str = "late"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rgb:  [B,3,224,224]
        flow20: [B,20,224,224]  (u1,v1,...,u10,v10)
        returns: (spatial_logits, temporal_logits, fused_logits)
        """
        self.eval()
        s = self.spatial(rgb)
        t = self.temporal(flow20)
        if fusion == "late":
            fused = (s + t) / 2.0
        else:
            raise ValueError("Only 'late' fusion is implemented in this demo.")
        return s, t, fused


# -----------------------------
# Video I/O and Sampling
# -----------------------------
def read_video_frames(path: str, max_frames: int = 64) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if len(frames) < 12:
        raise RuntimeError(
            "Video too short for a clean demo (need at least ~12 frames)."
        )
    return frames


def prepare_rgb_tensor(frame_bgr: np.ndarray, tfm) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return tfm(rgb)  # 3x224x224 (tensor)


def prepare_flow_tensor(flow_224x224x20: np.ndarray) -> torch.Tensor:
    # Convert HxWxC (float32 in [-1,1]) to CxHxW torch tensor (no normalize)
    chw = np.transpose(flow_224x224x20, (2, 0, 1))  # 20x224x224
    return torch.from_numpy(chw)


# -----------------------------
# Inference helpers
# -----------------------------
def topk_readable(
    logits: torch.Tensor, labels: List[str], k: int = 5
) -> List[Tuple[str, float]]:
    probs = F.softmax(logits, dim=-1)
    vals, idxs = probs.topk(k)
    return [(labels[i.item()], float(v.item())) for i, v in zip(idxs[0], vals[0])]


def print_topk(title: str, pairs: List[Tuple[str, float]]):
    print(f"\n{title}")
    for i, (name, p) in enumerate(pairs, 1):
        print(f"  {i}. {name:<30} {p*100:5.2f}%")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-Stream (RGB + Optical Flow) Demo")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to an input video file"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_frames", type=int, default=96, help="Max frames to read")
    parser.add_argument(
        "--num_flow_pairs",
        type=int,
        default=10,
        help="TV-L1 flow pairs (u,v) -> 20 channels",
    )
    parser.add_argument("--size", type=int, default=224, help="Input crop size")
    # in argparse section
    parser.add_argument(
        "--flow",
        type=str,
        default="auto",
        choices=["auto", "tvl1", "farneback"],
        help="Optical flow backend: TV-L1 (contrib), Farnebäck (fallback), or auto",
    )

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print("=" * 80)
    print("Two-Stream Networks Demo (RGB + TV-L1 Flow, Late Fusion)")
    print("Backbone: VGG16 (ImageNet pretrained)")
    print(f"Device: {device}")
    print("=" * 80)

    # Read frames
    frames_bgr = read_video_frames(args.video, max_frames=args.max_frames)
    print(f"Loaded video: {args.video} ({len(frames_bgr)} frames)")

    # Build grayscale list for flow
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]

    # Transform for RGB
    rgb_tfm = build_rgb_transform(args.size)

    # Choose a representative RGB frame (middle)
    mid = len(frames_bgr) // 2
    rgb_tensor = prepare_rgb_tensor(frames_bgr[mid], rgb_tfm).unsqueeze(
        0
    )  # [1,3,224,224]

    # Build 20-channel optical flow stack from last num_pairs+1 frames (or any window)
    # flow_stack = compute_tvl1_flow_stack(
    #     frames_gray=frames_gray,
    #     num_pairs=args.num_flow_pairs,
    #     out_size=args.size,
    # )

    flow_stack, stats = compute_flow_stack(
        frames_gray=frames_gray,
        num_pairs=args.num_flow_pairs,
        out_size=args.size,
        method="farneback",
    )
    print(
        f"Flow window starts at frame {stats['window_start']} "
        f"| mean mag per-pair: {[f'{m:.3f}' for m in stats['pair_mean_mag']]}"
    )
    print(
        f"Global mean mag: {stats['global_mean']:.3f} | global max: {stats['global_max']:.3f}"
    )

    # Save a few HSV previews so you can inspect motion
    save_flow_stack_preview(flow_stack, "flow_preview")
    print("Saved flow previews: flow_preview_pair00.png / pairXX.png")

    # mean_mag, max_mag = flow_magnitude_stats(flow_stack)
    # print(f"Flow magnitude: mean={mean_mag:.3f}, max={max_mag:.3f}")
    # save_flowviz(flow_stack, "flow_mag.png")
    # print("Saved flow magnitude viz -> flow_mag.png")

    flow_tensor = prepare_flow_tensor(flow_stack).unsqueeze(0)  # [1,20,224,224]

    # Model
    model = TwoStreamNet(num_classes=1000).to(device)
    labels = load_imagenet_labels()

    # Inference
    rgb_tensor = rgb_tensor.to(device)
    flow_tensor = flow_tensor.to(device)

    with torch.no_grad():
        s_logits, t_logits, f_logits = model(rgb_tensor, flow_tensor, fusion="late")

    # Report
    s_top5 = topk_readable(s_logits.cpu(), labels, k=5)
    t_top5 = topk_readable(t_logits.cpu(), labels, k=5)
    f_top5 = topk_readable(f_logits.cpu(), labels, k=5)

    print_topk("Spatial stream (RGB) – Top-5", s_top5)
    print_topk("Temporal stream (Flow) – Top-5", t_top5)
    print_topk("Fused (late) – Top-5", f_top5)

    print("\nNotes:")
    print(
        "- This demo shows the mechanics of Two-Stream inference with real optical flow."
    )
    print(
        "- For *action* labels, fine-tune on UCF101/Kinetics and set num_classes accordingly."
    )
    print(
        "- Temporal first conv is inflated from RGB conv weights to accept 20 channels (flow stack)."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
