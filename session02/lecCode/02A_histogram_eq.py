#!/usr/bin/env python3
"""
Histogram Equalization (Global HE) — Teaching Script

Implements:
  • Basic HE (textbook):  T[k] = round( cdf[k] / N * (L-1) )
  • OpenCV-style HE with c_min: T[k] = round( (cdf[k]-c_min) / (N-c_min) * (L-1) )
  • Color-safe HE via LAB L-channel
  • Comparison against cv2.equalizeHist (optional)
  • Side-by-side figs for slides

Usage:
  python histeq_demo.py --image path/to/img.jpg --save --check
  python histeq_demo.py --image path/to/img.jpg --method basic
  python histeq_demo.py --image path/to/img.jpg --method cv --grid  # (grid is ignored; kept for CLAHE parity)

Dependencies: opencv-python, numpy, matplotlib
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Core helpers ----------


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def hist_and_cdf(gray: np.ndarray, L: int = 256):
    """Return (hist, cdf) for 8-bit grayscale."""
    hist = np.bincount(gray.ravel(), minlength=L).astype(np.int64)
    cdf = np.cumsum(hist)
    return hist, cdf


def histeq_basic(gray: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Textbook HE:
      T[k] = round( cdf[k] / N * (L-1) )
    """
    _, cdf = hist_and_cdf(gray, L=L)
    N = gray.size
    lut = np.round(cdf / max(N, 1) * (L - 1))
    lut = np.clip(lut, 0, L - 1).astype(np.uint8)
    return lut[gray]


def histeq_cvstyle(gray: np.ndarray, L: int = 256) -> np.ndarray:
    """
    OpenCV-style HE (equalizeHist):
      c_min = first non-zero entry of CDF
      T[k]  = round( (cdf[k] - c_min) / (N - c_min) * (L - 1) )
    """
    _, cdf = hist_and_cdf(gray, L=L)
    nz = np.flatnonzero(cdf)
    if len(nz) == 0:
        return gray.copy()  # empty image fallback
    cmin = cdf[nz[0]]
    N = gray.size
    denom = max(N - cmin, 1)
    lut = np.round((cdf - cmin) / denom * (L - 1))
    lut = np.clip(lut, 0, L - 1).astype(np.uint8)
    return lut[gray]


def histeq_lab_color(bgr: np.ndarray, method: str = "cv") -> np.ndarray:
    """
    Apply HE to LAB L-channel only; preserve chroma.
    method ∈ {"basic", "cv"} selects mapping for L.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    if method == "basic":
        L2 = histeq_basic(L)
    else:
        L2 = histeq_cvstyle(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# ---------- Visualization ----------


def fig_gray_triplet(
    gray, basic, cvstyle, title="Global Histogram Equalization (Gray)"
):
    fig = plt.figure(figsize=(14, 4.5))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(basic, cmap="gray", vmin=0, vmax=255)
    plt.title("Basic HE")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cvstyle, cmap="gray", vmin=0, vmax=255)
    plt.title("OpenCV-style HE")
    plt.axis("off")

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def fig_color_pair(bgr, bgr_eq, title_right="LAB L-channel HE"):
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB))
    plt.title(title_right)
    plt.axis("off")

    fig.tight_layout()
    return fig


def fig_histograms(gray, basic, cvstyle):
    fig = plt.figure(figsize=(13, 4))
    for idx, (name, im) in enumerate(
        [("Original", gray), ("Basic HE", basic), ("OpenCV-style", cvstyle)], start=1
    ):
        plt.subplot(1, 3, idx)
        plt.hist(im.ravel(), bins=256, range=(0, 255))
        plt.title(f"{name} Histogram")
    fig.tight_layout()
    return fig


# ---------- CLI / Main ----------


def parse_args():
    p = argparse.ArgumentParser(
        description="Global Histogram Equalization demo (basic & OpenCV-style)"
    )
    p.add_argument("--image", type=Path, required=True, help="Path to input image")
    p.add_argument(
        "--method",
        choices=["basic", "cv"],
        default="cv",
        help="Which mapping to highlight for color demo (gray shows both)",
    )
    p.add_argument(
        "--save", action="store_true", help="Save processed images and figures"
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Compare cv-style result to cv2.equalizeHist",
    )
    p.add_argument(
        "--outdir", type=Path, default=None, help="Directory to save outputs"
    )
    return p.parse_args()


def main():
    args = parse_args()
    img_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    gray = to_gray(img_bgr)

    # Process (grayscale)
    gray_basic = histeq_basic(gray)
    gray_cv = histeq_cvstyle(gray)

    # Optional check vs OpenCV
    if args.check:
        gray_cv_ref = cv2.equalizeHist(gray)
        diff = np.abs(gray_cv.astype(np.int16) - gray_cv_ref.astype(np.int16))
        print(f"[check] max|cvstyle - cv2.equalizeHist| = {diff.max()}")
        if diff.max() <= 1:
            print("[check] OK: Matches OpenCV within rounding.")
        else:
            print("[check] Warning: Larger deviation than expected.")

    # Color-safe HE on LAB L-channel
    bgr_eq = histeq_lab_color(img_bgr, method=args.method)

    # Make figures
    fig1 = fig_gray_triplet(
        gray,
        gray_basic,
        gray_cv,
        title="Grayscale: Original vs Basic HE vs OpenCV-style HE",
    )
    fig2 = fig_color_pair(
        img_bgr,
        bgr_eq,
        title_right=f"LAB L-channel HE ({'Basic' if args.method=='basic' else 'OpenCV-style'})",
    )
    fig3 = fig_histograms(gray, gray_basic, gray_cv)

    # Save if requested
    if args.save:
        outdir = args.outdir or (args.image.parent / "histeq_outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = args.image.stem

        # images
        cv2.imwrite(str(outdir / f"{stem}_gray.png"), gray)
        cv2.imwrite(str(outdir / f"{stem}_gray_basicHE.png"), gray_basic)
        cv2.imwrite(str(outdir / f"{stem}_gray_cvHE.png"), gray_cv)
        cv2.imwrite(str(outdir / f"{stem}_color_labHE_{args.method}.png"), bgr_eq)

        # figures
        fig1.savefig(outdir / f"{stem}_gray_triplet.png", dpi=200, bbox_inches="tight")
        fig2.savefig(outdir / f"{stem}_color_pair.png", dpi=200, bbox_inches="tight")
        fig3.savefig(outdir / f"{stem}_histograms.png", dpi=200, bbox_inches="tight")

        print(f"[Saved] Outputs → {outdir.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
