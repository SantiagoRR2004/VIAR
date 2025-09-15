#!/usr/bin/env python3
"""
CLAHE demo (OpenCV):
- Grayscale CLAHE vs global histogram equalization
- Color-safe CLAHE on LAB L-channel (preserves chroma)
- Side-by-side matplotlib figures + optional saving

Usage:
  python clahe_demo.py --image path/to/img.jpg --clip 2.0 --grid 8 8 --save

Dependencies: opencv-python, numpy, matplotlib
  pip install opencv-python numpy matplotlib
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_global_hist_eq_gray(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalization (for comparison)."""
    return cv2.equalizeHist(gray)


def apply_clahe_gray(gray: np.ndarray, clip: float = 2.0, grid=(8, 8)) -> np.ndarray:
    """CLAHE on a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)


def apply_clahe_lab(bgr: np.ndarray, clip: float = 2.0, grid=(8, 8)) -> np.ndarray:
    """
    Color-safe CLAHE: operate only on LAB L-channel, then convert back to BGR.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    bgr_out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return bgr_out


def imread_gray_bgr(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, bgr


def fig_side_by_side_color(original_bgr, clahe_bgr, title_left="Original", title_right="LAB-CLAHE"):
    """Return a Matplotlib figure comparing color images side by side."""
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(bgr_to_rgb(original_bgr))
    plt.title(title_left)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(bgr_to_rgb(clahe_bgr))
    plt.title(title_right)
    plt.axis("off")
    fig.tight_layout()
    return fig


def fig_gray_comparison(gray, he, clahe, title="Grayscale: Original vs Global HE vs CLAHE"):
    """Return a Matplotlib figure showing grayscale original, global HE, and CLAHE."""
    fig = plt.figure(figsize=(15, 4.5))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(he, cmap="gray", vmin=0, vmax=255)
    plt.title("Global HE (equalizeHist)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clahe, cmap="gray", vmin=0, vmax=255)
    plt.title("CLAHE")
    plt.axis("off")

    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    return fig


def histogram_panel_gray(gray, he, clahe):
    """Optional: histogram panel to show distribution changes."""
    fig = plt.figure(figsize=(12, 4))
    bins = 256
    plt.subplot(1, 3, 1)
    plt.hist(gray.ravel(), bins=bins, range=(0, 255))
    plt.title("Original Gray Histogram")

    plt.subplot(1, 3, 2)
    plt.hist(he.ravel(), bins=bins, range=(0, 255))
    plt.title("Global HE Histogram")

    plt.subplot(1, 3, 3)
    plt.hist(clahe.ravel(), bins=bins, range=(0, 255))
    plt.title("CLAHE Histogram")

    fig.tight_layout()
    return fig


def process_image(image_path: Path, clip: float, grid: tuple[int, int], save: bool, show_hist: bool, outdir: Path | None):
    gray, bgr = imread_gray_bgr(image_path)

    he_gray = apply_global_hist_eq_gray(gray)
    clahe_gray = apply_clahe_gray(gray, clip=clip, grid=grid)
    clahe_bgr = apply_clahe_lab(bgr, clip=clip, grid=grid)

    # Build figures
    fig1 = fig_gray_comparison(gray, he_gray, clahe_gray,
                               title=f"Grayscale Comparison (clip={clip}, grid={grid})")
    fig2 = fig_side_by_side_color(bgr, clahe_bgr,
                                  title_left="Original (BGR→RGB)",
                                  title_right=f"LAB-CLAHE (clip={clip}, grid={grid})")

    if show_hist:
        fig3 = histogram_panel_gray(gray, he_gray, clahe_gray)
    else:
        fig3 = None

    # Save outputs if requested
    if save:
        outdir = outdir or image_path.parent / "clahe_outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        # Save processed images
        cv2.imwrite(str(outdir / f"{stem}_gray.png"), gray)
        cv2.imwrite(str(outdir / f"{stem}_he_gray.png"), he_gray)
        cv2.imwrite(str(outdir / f"{stem}_clahe_gray_clip{clip}_grid{grid[0]}x{grid[1]}.png"), clahe_gray)
        cv2.imwrite(str(outdir / f"{stem}_clahe_color_clip{clip}_grid{grid[0]}x{grid[1]}.png"), clahe_bgr)

        # Save figures (RGB space via matplotlib)
        fig1.savefig(outdir / f"{stem}_gray_comparison.png", dpi=200, bbox_inches="tight")
        fig2.savefig(outdir / f"{stem}_color_comparison.png", dpi=200, bbox_inches="tight")
        if fig3 is not None:
            fig3.savefig(outdir / f"{stem}_histograms.png", dpi=200, bbox_inches="tight")

        print(f"[Saved] Outputs to: {outdir.resolve()}")

    # Show figures interactively
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="CLAHE demo (OpenCV): grayscale & LAB color")
    p.add_argument("--image", type=Path, required=True, help="Path to input image")
    p.add_argument("--clip", type=float, default=2.0, help="CLAHE clipLimit (typical 2–3)")
    p.add_argument("--grid", type=int, nargs=2, default=[8, 8], metavar=("TILES_X", "TILES_Y"),
                   help="CLAHE tileGridSize (e.g., 8 8 or 16 16)")
    p.add_argument("--save", action="store_true", help="Save processed images and figures")
    p.add_argument("--hist", action="store_true", help="Also draw histogram comparison panel")
    p.add_argument("--outdir", type=Path, default=None, help="Custom output directory (used with --save)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    grid = (int(args.grid[0]), int(args.grid[1]))
    process_image(args.image, clip=args.clip, grid=grid, save=args.save, show_hist=args.hist, outdir=args.outdir)


if __name__ == "__main__":
    main()
