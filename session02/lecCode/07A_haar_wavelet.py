#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haar-like Features & Integral Image (Viola–Jones style) — Teaching Demo

What it does:
  • Builds integral image S (with leading zero row/col).
  • Evaluates Haar features with 4-lookups per rectangle:
      - edge_h, edge_v, line_h, line_v, checker2x2, center_surround
  • Slides a window, outputs response heatmaps + top detections.
  • (Optional) 1-level 2D Haar DWT bands (LL, LH, HL, HH) for wavelet intuition.

Usage
  python haar_vj_demo.py --image path/to/img.jpg --save
  python haar_vj_demo.py --synthetic --save --stride 2

Arguments
  --win  (WxH) Haar window size, default 24x24 (canonical VJ size).
  --stride step for sliding window (>=1). Larger = faster.
  --topk how many top detections to draw per feature.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------


def imread_gray(path: Path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g


def save_img(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(path)


def normalize01(x: np.ndarray):
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


# ---------------- Synthetic canvas (for class) ----------------


def synthetic_canvas(h=256, w=320):
    img = np.full((h, w), 220, np.uint8)
    # Rect + hole
    cv2.rectangle(img, (20, 40), (120, 140), 50, -1)
    cv2.circle(img, (70, 90), 18, 220, -1)
    # Lines
    cv2.line(img, (160, 30), (300, 30), 50, 2)
    cv2.line(img, (160, 32), (300, 80), 50, 2)
    # Dots
    rng = np.random.default_rng(0)
    for _ in range(100):
        y = int(rng.uniform(0, h))
        x = int(rng.uniform(0, w))
        img[y, x] = 50
    return img


# ---------------- Integral image ----------------


def integral_image_u32(gray: np.ndarray) -> np.ndarray:
    """
    Build integral image S with shape (H+1, W+1), S[0,*]=S[*,0]=0.
    Using 64-bit to avoid overflow on larger images.
    """
    S = np.cumsum(np.cumsum(gray.astype(np.int64), axis=0), axis=1)
    S = np.pad(S, ((1, 0), (1, 0)), mode="constant")  # leading zeros
    return S


def rect_sum(S: np.ndarray, y1: int, x1: int, y2: int, x2: int) -> int:
    """
    Sum over rectangle [y1:y2, x1:x2) (top-left inclusive, bottom-right exclusive).
    S is integral image with leading zeros.
    """
    return int(S[y2, x2] - S[y1, x2] - S[y2, x1] + S[y1, x1])


# ---------------- Haar-like features ----------------


def feature_response(S: np.ndarray, y: int, x: int, w: int, h: int, ftype: str) -> int:
    """
    Compute Haar response at window top-left (y,x) of size (w,h).
    Positive sums are "white", negative are "black"; response = sum(white) - sum(black).
    """

    # Helper to get absolute coords inside the window:
    def rs(y1, x1, y2, x2):  # rectangle sum in local coords (0..h, 0..w)
        return rect_sum(S, y + y1, x + x1, y + y2, x + x2)

    if ftype == "edge_h":  # vertical edge: left - right
        mid = x + w // 2
        return rs(0, 0, h, w // 2) - rs(0, w // 2, h, w)

    if ftype == "edge_v":  # horizontal edge: top - bottom
        mid = y + h // 2
        return rs(0, 0, h // 2, w) - rs(h // 2, 0, h, w)

    if ftype == "line_h":  # three vertical stripes: + - +
        t1, t2 = w // 3, 2 * w // 3
        return rs(0, 0, h, t1) - rs(0, t1, h, t2) + rs(0, t2, h, w)

    if ftype == "line_v":  # three horizontal stripes: + - +
        t1, t2 = h // 3, 2 * h // 3
        return rs(0, 0, t1, w) - rs(t1, 0, t2, w) + rs(t2, 0, h, w)

    if ftype == "checker2x2":  # 2x2 checker: (+ -; - +)
        hx, hy = w // 2, h // 2
        return rs(0, 0, hy, hx) - rs(0, hx, hy, w) - rs(hy, 0, h, hx) + rs(hy, hx, h, w)

    if ftype == "center_surround":
        # Outer minus inner (ring negative, center positive): resp = 2*center - outer
        # Choose inner rect ~ 1/3 of width/height centered.
        cx, cy = w // 3, h // 3
        x1 = (w - cx) // 2
        y1 = (h - cy) // 2
        outer = rs(0, 0, h, w)
        center = rs(y1, x1, y1 + cy, x1 + cx)
        return 2 * center - outer

    raise ValueError(f"Unknown feature type: {ftype}")


def response_map(
    S: np.ndarray, w: int, h: int, ftype: str, stride: int = 1
) -> np.ndarray:
    H, W = S.shape[0] - 1, S.shape[1] - 1
    outH = (H - h) // stride + 1
    outW = (W - w) // stride + 1
    R = np.zeros((outH, outW), dtype=np.int64)
    oi = 0
    for y in range(0, H - h + 1, stride):
        oj = 0
        for x in range(0, W - w + 1, stride):
            R[oi, oj] = feature_response(S, y, x, w, h, ftype)
            oj += 1
        oi += 1
    return R


# ---------------- Visualization helpers ----------------


def heatmap_to_u8(R: np.ndarray) -> np.ndarray:
    # symmetric normalization around zero to highlight +/- responses
    Rf = R.astype(np.float64)
    m = np.max(np.abs(Rf)) if Rf.size else 1.0
    H = (Rf / (m + 1e-9)) * 0.5 + 0.5  # [-1,1] -> [0,1]
    u8 = (H * 255 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def draw_topk_boxes(
    gray: np.ndarray, R: np.ndarray, w: int, h: int, stride: int, k: int, pos=True
):
    """
    Draw k top responses (pos=True for maxima, False for minima).
    """
    Hmap, Wmap = R.shape
    flat_idx = np.argpartition((-R if pos else R).ravel(), kth=min(k - 1, R.size - 1))[
        :k
    ]
    ys, xs = np.unravel_index(flat_idx, R.shape)
    color = (0, 220, 0) if pos else (0, 0, 255)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).copy()
    for y, x in zip(ys, xs):
        y0, x0 = int(y * stride), int(x * stride)
        cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), color, 2, cv2.LINE_AA)
    return vis


# ---------------- Optional: 1-level 2D Haar DWT (for intuition) ----------------


def dwt_haar_level1(gray: np.ndarray):
    """
    Separable 1-level Haar: low = [1,1]/sqrt(2), high = [1,-1]/sqrt(2).
    Downsample by 2 after filtering. Returns LL, LH, HL, HH (as float64).
    """
    g = gray.astype(np.float64)
    # 1D filters
    h0 = np.array([1.0, 1.0]) / np.sqrt(2.0)  # low-pass
    h1 = np.array([1.0, -1.0]) / np.sqrt(2.0)  # high-pass

    # Convolve rows then downsample columns
    L = cv2.sepFilter2D(g, -1, h0, np.array([1.0]))[:, ::2]
    H = cv2.sepFilter2D(g, -1, h1, np.array([1.0]))[:, ::2]
    # Convolve columns then downsample rows
    LL = cv2.sepFilter2D(L, -1, np.array([1.0]), h0)[::2, :]
    LH = cv2.sepFilter2D(L, -1, np.array([1.0]), h1)[::2, :]
    HL = cv2.sepFilter2D(H, -1, np.array([1.0]), h0)[::2, :]
    HH = cv2.sepFilter2D(H, -1, np.array([1.0]), h1)[::2, :]
    return LL, LH, HL, HH


def dwt_panel(gray):
    LL, LH, HL, HH = dwt_haar_level1(gray)

    # Normalize each for display, upsample to original size for a clean mosaic
    def vis(a):
        a = normalize01(a)
        return (a * 255 + 0.5).astype(np.uint8)

    LLu, LHu, HLu, HHu = map(vis, (LL, LH, HL, HH))
    # Put bands in a 2x2 grid at their natural size
    top = cv2.hconcat([LLu, LHu])
    bot = cv2.hconcat([HLu, HHu])
    grid = cv2.vconcat([top, bot])
    return grid


# ---------------- CLI ----------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Haar-like features (Viola–Jones) + integral image"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to grayscale/RGB image")
    src.add_argument(
        "--synthetic", action="store_true", help="Use synthetic test canvas"
    )
    ap.add_argument(
        "--win",
        type=int,
        nargs=2,
        default=[24, 24],
        metavar=("W", "H"),
        help="Haar window size (w h)",
    )
    ap.add_argument("--stride", type=int, default=1, help="Sliding step (>=1)")
    ap.add_argument(
        "--topk", type=int, default=5, help="Draw top-k detections (max responses)"
    )
    ap.add_argument(
        "--save", action="store_true", help="Save outputs to ./haar_outputs"
    )
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument(
        "--show-dwt",
        action="store_true",
        help="Also render 1-level 2D Haar (LL/LH/HL/HH)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir or "haar_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load image (grayscale)
    if args.synthetic:
        gray = synthetic_canvas()
        name = "synthetic"
    else:
        img = imread_gray(Path(args.image))
        gray = img
        name = Path(args.image).stem

    H, W = gray.shape
    w, h = args.win
    stride = max(1, int(args.stride))

    # Integral image
    S = integral_image_u32(gray)

    # Evaluate features
    features = ["edge_h", "edge_v", "line_h", "line_v", "checker2x2", "center_surround"]
    heatmaps = {}
    vis_boxes = {}

    for f in features:
        R = response_map(S, w, h, f, stride=stride)
        heatmaps[f] = R
        pos_boxes = draw_topk_boxes(gray, R, w, h, stride, k=args.topk, pos=True)
        vis_boxes[f] = pos_boxes

    # Save/Show per-feature results
    for f in features:
        R = heatmaps[f]
        Hmap = heatmap_to_u8(R)
        # Resize heatmap to image for nice overlay/reference
        Hres = cv2.resize(Hmap, (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.6, Hres, 0.8, 0
        )

        if args.save:
            save_img(outdir / f"{name}_{f}_heat.png", Hres)
            save_img(outdir / f"{name}_{f}_overlay.png", overlay)
            save_img(outdir / f"{name}_{f}_topk.png", vis_boxes[f])

    # Make a quick multi-panel preview figure
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()
    for i, f in enumerate(features):
        axs[i].imshow(cv2.cvtColor(heatmap_to_u8(heatmaps[f]), cv2.COLOR_BGR2RGB))
        axs[i].set_title(f)
        axs[i].axis("off")
    fig.suptitle(f"Haar-like feature responses (win={w}x{h}, stride={stride})")
    fig.tight_layout()
    if args.save:
        fig.savefig(
            outdir / f"{name}_haar_heatmaps_grid.png", dpi=180, bbox_inches="tight"
        )
    plt.show(block=False)

    # Optional Haar DWT panel
    if args.show_dwt:
        dwt_grid = dwt_panel(gray)
        if args.save:
            save_img(outdir / f"{name}_haar_dwt_level1.png", dwt_grid)
        cv2.imshow("Haar DWT (LL | LH; HL | HH)", dwt_grid)
        cv2.waitKey(1)

    print(f"[done] Results in {outdir.resolve()} (set --save to write images).")


if __name__ == "__main__":
    main()
