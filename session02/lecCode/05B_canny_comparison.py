#!/usr/bin/env python3
"""
Canny Edge Detection Demo:
- Manual pipeline (Gaussian -> Sobel -> NMS -> Double Threshold -> Hysteresis)
- OpenCV Canny (L2gradient=True)
- XOR disagreement heatmap, simple metrics (TP/FP/FN/IoU)
- Optional parameter sweep

Usage:
  python canny_compare.py --image path/to/img.jpg --sigma 1.4 --lowf 0.10 --highf 0.25 --save
  python canny_compare.py --image path/to/img.jpg --sweep --save

Dependencies: opencv-python, numpy, pillow (only for robust saving on some systems)
"""

import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Utilities
# -------------------------------


def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def save_image(path: Path, arr_bgr_or_gray: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr_bgr_or_gray.ndim == 2:
        Image.fromarray(arr_bgr_or_gray).save(path)
    else:
        # BGR->RGB for PIL saving
        rgb = cv2.cvtColor(arr_bgr_or_gray, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path)


# -------------------------------
# Manual Canny pieces
# -------------------------------


def sobel_gradients(blur: np.ndarray):
    """Sobel x/y gradients, magnitude, and angle in degrees [0,180)."""
    Ix = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(Ix, Iy)
    ang = (np.degrees(np.arctan2(Iy, Ix)) + 180.0) % 180.0  # [0,180)
    return mag, ang


def angle_sector(ang_deg: np.ndarray) -> np.ndarray:
    """
    Quantize gradient angle into 4 sectors (0,45,90,135).
    sector = round(ang/45) mod 4
    """
    return ((ang_deg + 22.5) // 45).astype(np.uint8) % 4


def nonmax_suppression(mag: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    """
    Thin edges by keeping only local maxima along gradient direction.
    Returns float magnitudes with non-maxima set to 0.
    """
    H, W = mag.shape
    sec = angle_sector(ang_deg)

    # Pad with -inf so comparisons at borders work
    M = np.pad(mag, 1, mode="constant", constant_values=-np.inf)

    out = np.zeros_like(mag, dtype=np.float32)

    # Neighbour slices (shifted views into padded M)
    def sl(y, x):  # convenience
        return M[1 + y : H + 1 + y, 1 + x : W + 1 + x]

    # For each sector, pick the two neighbours along the gradient direction
    # sec 0: compare left/right (E-W)
    m0a, m0b = sl(0, -1), sl(0, +1)
    # sec 1: compare NE/SW
    m1a, m1b = sl(-1, +1), sl(+1, -1)
    # sec 2: compare up/down (N-S)
    m2a, m2b = sl(-1, 0), sl(+1, 0)
    # sec 3: compare NW/SE
    m3a, m3b = sl(-1, -1), sl(+1, +1)

    keep0 = (sec == 0) & (mag >= m0a) & (mag >= m0b)
    keep1 = (sec == 1) & (mag >= m1a) & (mag >= m1b)
    keep2 = (sec == 2) & (mag >= m2a) & (mag >= m2b)
    keep3 = (sec == 3) & (mag >= m3a) & (mag >= m3b)

    out[keep0 | keep1 | keep2 | keep3] = mag[keep0 | keep1 | keep2 | keep3]
    return out


def double_threshold_and_hysteresis(
    M: np.ndarray, low: float, high: float
) -> np.ndarray:
    """
    M: thinned magnitudes (float)
    low, high: thresholds in absolute magnitude units
    Returns uint8 edge map in {0,255}.
    """
    strong = M >= high
    weak = (M >= low) & ~strong

    H, W = M.shape
    edges = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)

    # Seed stack with strong pixels
    ys, xs = np.where(strong)
    stack = deque(zip(ys, xs))
    edges[ys, xs] = 255
    visited[ys, xs] = True

    # 8-connected flood from strong through weak
    nbr8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while stack:
        y, x = stack.pop()
        for dy, dx in nbr8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and not visited[yy, xx]:
                if weak[yy, xx]:
                    edges[yy, xx] = 255
                    stack.append((yy, xx))
                visited[yy, xx] = True

    return edges


def canny_manual(gray_u8: np.ndarray, sigma: float, low_f: float, high_f: float):
    """
    Manual Canny using fraction-of-max thresholds.
    Returns (E, blur, mag, M_thin) where E is uint8 {0,255}.
    """
    blur = cv2.GaussianBlur(gray_u8, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    mag, ang = sobel_gradients(blur)
    M_thin = nonmax_suppression(mag, ang)
    Tl = low_f * (M_thin.max() if M_thin.size else 0.0)
    Th = high_f * (M_thin.max() if M_thin.size else 0.0)
    E = double_threshold_and_hysteresis(M_thin, Tl, Th)
    return E, blur, mag, M_thin


# -------------------------------
# Comparison utilities
# -------------------------------


def cv2_canny(gray_u8: np.ndarray, sigma: float, low_f: float, high_f: float):
    """OpenCV Canny run on the same blurred image, thresholds mapped to 0..255 scale."""
    blur = cv2.GaussianBlur(gray_u8, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    t1 = int(round(low_f * 255))
    t2 = int(round(high_f * 255))
    E = cv2.Canny(blur, t1, t2, L2gradient=True)
    return E, blur


def disagreement_overlay(gray_u8: np.ndarray, E_manual: np.ndarray, E_cv: np.ndarray):
    """Create color overlay of XOR disagreements on top of the grayscale image."""
    xor = cv2.bitwise_xor(E_manual, E_cv)  # 0/255
    heat = cv2.applyColorMap(xor, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heat, 0.8, 0)
    return overlay, xor


def edge_metrics(E_manual: np.ndarray, E_cv: np.ndarray):
    tp = np.logical_and(E_manual == 255, E_cv == 255).sum()
    fp = np.logical_and(E_manual == 255, E_cv == 0).sum()
    fn = np.logical_and(E_manual == 0, E_cv == 255).sum()
    iou = tp / max(tp + fp + fn, 1)
    return tp, fp, fn, iou


def side_by_side(Em, Ecv, overlay):
    """Build a comparison panel."""
    if Em.ndim == 2:
        Em = cv2.cvtColor(Em, cv2.COLOR_GRAY2BGR)
    if Ecv.ndim == 2:
        Ecv = cv2.cvtColor(Ecv, cv2.COLOR_GRAY2BGR)
    return np.hstack([Em, Ecv, overlay])


# -------------------------------
# Main routines
# -------------------------------


def single_run(
    image_path: Path,
    sigma: float,
    lowf: float,
    highf: float,
    save: bool,
    outdir: Path | None,
):
    I = imread_gray(image_path)

    Em, blur_m, _, _ = canny_manual(I, sigma, lowf, highf)
    Ecv, blur_c = cv2_canny(I, sigma, lowf, highf)
    overlay, xor = disagreement_overlay(I, Em, Ecv)
    tp, fp, fn, iou = edge_metrics(Em, Ecv)

    print(
        f"[single] sigma={sigma:.2f} lowf={lowf:.2f} highf={highf:.2f} "
        f"TP={tp} FP={fp} FN={fn} IoU={iou:.3f} "
        f"#E(man)={int(Em.sum()/255)} #E(cv2)={int(Ecv.sum()/255)}"
    )

    panel = side_by_side(Em, Ecv, overlay)

    if save:
        outdir = outdir or (image_path.parent / "canny_outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        save_image(outdir / f"{stem}_manual.png", Em)
        save_image(outdir / f"{stem}_cv2.png", Ecv)
        save_image(outdir / f"{stem}_xor_overlay.png", panel)
        print(f"[saved] {outdir.resolve()}/{stem}_*.png")


def sweep(image_path: Path, save: bool, outdir: Path | None):
    I = imread_gray(image_path)
    configs = [
        (0.8, 0.10, 0.25),
        (1.4, 0.10, 0.25),
        (2.0, 0.10, 0.25),
        (1.4, 0.05, 0.15),
        (1.4, 0.15, 0.30),
    ]
    print("sigma  lowf  highf   #E(man)  #E(cv2)   IoU")
    rows = []
    for sigma, lowf, highf in configs:
        Em, _, _, _ = canny_manual(I, sigma, lowf, highf)
        Ecv, _ = cv2_canny(I, sigma, lowf, highf)
        tp, fp, fn, iou = edge_metrics(Em, Ecv)
        nm, nc = int(Em.sum() / 255), int(Ecv.sum() / 255)
        print(f"{sigma:4.1f}  {lowf:4.2f}  {highf:4.2f}   {nm:7d}  {nc:7d}  {iou:6.3f}")
        rows.append((sigma, lowf, highf, nm, nc, iou))

    if save:
        outdir = outdir or (image_path.parent / "canny_outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        # Save a visual for the last config
        Em, _, _, _ = canny_manual(I, configs[-1][0], configs[-1][1], configs[-1][2])
        Ecv, _ = cv2_canny(I, configs[-1][0], configs[-1][1], configs[-1][2])
        overlay, _ = disagreement_overlay(I, Em, Ecv)
        panel = side_by_side(Em, Ecv, overlay)
        save_image(outdir / f"{image_path.stem}_sweep_last_panel.png", panel)

        # Save CSV of the sweep
        csv_path = outdir / f"{image_path.stem}_sweep.csv"
        with open(csv_path, "w") as f:
            f.write("sigma,lowf,highf,edges_manual,edges_cv2,IoU\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")
        print(f"[saved] sweep panel + CSV at {outdir.resolve()}")


# -------------------------------
# CLI
# -------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Manual Canny vs OpenCV Canny comparison")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=1.4, help="Gaussian sigma")
    p.add_argument(
        "--lowf",
        type=float,
        default=0.10,
        help="low threshold fraction of max gradient",
    )
    p.add_argument(
        "--highf",
        type=float,
        default=0.25,
        help="high threshold fraction of max gradient",
    )
    p.add_argument("--sweep", action="store_true", help="run preset parameter sweep")
    p.add_argument("--save", action="store_true", help="save outputs (PNG/CSV)")
    p.add_argument("--outdir", type=Path, default=None, help="custom output directory")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    if args.sweep:
        sweep(args.image, args.save, args.outdir)
    else:
        single_run(
            args.image, args.sigma, args.lowf, args.highf, args.save, args.outdir
        )


if __name__ == "__main__":
    main()
