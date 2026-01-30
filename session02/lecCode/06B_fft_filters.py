#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT Filtering Demo (2D DFT):
- Design frequency masks: Gaussian / Ideal / Butterworth
  (low-pass, high-pass, band-pass, band-stop) + Laplacian HP
- Apply in frequency domain: F * H  ->  ifft2
- Optional zero-padding to reduce circular convolution artifacts
- Grayscale or per-channel color
- Saves spectrum/mask/results for slides and compares to spatial Gaussian

Usage:
  python fft_filter_demo.py --image path/to/img.jpg --mode gray \
      --filter gaussian --type lowpass --cutoff 0.10 --save

  python fft_filter_demo.py --image path/to/img.jpg --mode color \
      --filter butter --type bandpass --cutoff 0.05 --cutoff2 0.18 --order 2 --save

  python fft_filter_demo.py --image path/to/img.jpg --filter laplacian --save

  # With zero-padding (to next power of 2)
  python fft_filter_demo.py --image path/to/img.jpg --pad --save
"""

import argparse
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --------------------- I/O helpers ---------------------


def imread_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_gray(path, arr01):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(arr01, 0, 1)
    Image.fromarray((u8 * 255 + 0.5).astype(np.uint8)).save(path)


def save_rgb(path, arr01):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(arr01, 0, 1)
    Image.fromarray((u8 * 255 + 0.5).astype(np.uint8)).save(path)


def normalize01(x):
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# --------------------- FFT utils ---------------------


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def pad_to_pow2(img):
    H, W = img.shape[:2]
    H2, W2 = next_pow2(H), next_pow2(W)
    pad_h, pad_w = H2 - H, W2 - W
    # reflect padding reduces boundary discontinuities
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)


def crop_to(img, H, W):
    return img[:H, :W] if img.ndim == 2 else img[:H, :W, :]


def fft2c(x):  # centered FFT
    return np.fft.fftshift(np.fft.fft2(x))
    # note: input should be float64


def ifft2c(X):  # inverse from centered FFT
    return np.fft.ifft2(np.fft.ifftshift(X))


def mag_spectrum(X, eps=1e-9):
    return np.log1p(np.abs(X) + eps)


def freq_grid(H, W):
    """Return centered frequency grid in cycles/pixel."""
    fy = np.fft.fftshift(np.fft.fftfreq(H))  # [-0.5,0.5)
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    Fy, Fx = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(Fx * Fx + Fy * Fy)  # radius in cycles/pixel
    return Fx, Fy, R


# --------------------- Masks ---------------------


def mask_gaussian_lp(H, W, cutoff):
    # H(r) = exp(-(r^2)/(2*c^2)), cutoff is std in cycles/pixel
    _, _, R = freq_grid(H, W)
    c = max(cutoff, 1e-8)
    return np.exp(-(R * R) / (2.0 * c * c))


def mask_ideal_lp(H, W, cutoff):
    _, _, R = freq_grid(H, W)
    return (R <= cutoff).astype(np.float64)


def mask_butter_lp(H, W, cutoff, order):
    _, _, R = freq_grid(H, W)
    c = max(cutoff, 1e-8)
    n = max(int(order), 1)
    return 1.0 / (1.0 + (R / c) ** (2 * n))


def mask_from_spec(H, W, filt, ftype, cutoff, cutoff2=None, order=2):
    """
    filt: 'gaussian' | 'ideal' | 'butter' | 'laplacian'
    ftype: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop' (ignored for laplacian)
    cutoff, cutoff2: normalized radii in cycles/pixel (0..0.5). For band* you need both.
    """
    if filt == "laplacian":
        # H(u,v) = -4*pi^2*(fx^2 + fy^2)  (continuous approximation)
        Fx, Fy, _ = freq_grid(H, W)
        Hlap = -4.0 * (math.pi**2) * (Fx * Fx + Fy * Fy)
        # normalize to [0,1] magnitude for visualization (actual scaling affects amplitude)
        return Hlap
    # Base LP
    if filt == "gaussian":
        LP = mask_gaussian_lp(H, W, cutoff)
    elif filt == "ideal":
        LP = mask_ideal_lp(H, W, cutoff)
    elif filt == "butter":
        LP = mask_butter_lp(H, W, cutoff, order)
    else:
        raise ValueError("Unknown filter type")
    if ftype == "lowpass":
        return LP
    if ftype == "highpass":
        return 1.0 - LP
    if ftype in ("bandpass", "bandstop"):
        if cutoff2 is None:
            raise ValueError("Provide --cutoff2 for band filters")
        lo, hi = sorted((cutoff, cutoff2))
        LP_lo = mask_from_spec(H, W, filt, "lowpass", lo, order=order)
        LP_hi = mask_from_spec(H, W, filt, "lowpass", hi, order=order)
        BP = LP_hi - LP_lo  # pass ring between lo..hi
        return BP if ftype == "bandpass" else (1.0 - BP)
    raise ValueError("Unknown pass type")


# --------------------- Apply filter ---------------------


def apply_fft_filter(img_f64, Hmask):
    """
    img_f64: HxW (float64, 0..1)
    Hmask:   HxW (float64); may be non-binary
    """
    F = fft2c(img_f64)
    Ff = F * Hmask
    out = np.real(ifft2c(Ff))
    return out, F, Ff


def process_gray(gray01, filt, ftype, cutoff, cutoff2, order, do_pad):
    H, W = gray01.shape
    work = pad_to_pow2(gray01) if do_pad else gray01
    HH, WW = work.shape
    Hmask = mask_from_spec(HH, WW, filt, ftype, cutoff, cutoff2, order)
    if filt == "laplacian":
        # Laplacian response (edge map); scale for visibility
        out, F, Ff = apply_fft_filter(work, Hmask)
        out = normalize01(out)  # for visualization only
    else:
        out, F, Ff = apply_fft_filter(work, Hmask)
        out = np.clip(out, 0.0, 1.0)
    if do_pad:
        out = crop_to(out, H, W)
        F = F[:H, :W]
        Ff = Ff[:H, :W]
        Hmask = Hmask[:H, :W]
    return out, Hmask, F, Ff


def process_color(rgb01, **kwargs):
    chans = []
    for c in range(3):
        out_c, Hmask, F, Ff = process_gray(rgb01[..., c], **kwargs)
        chans.append(out_c[..., None])
    return np.concatenate(chans, axis=-1), Hmask, F, Ff


# --------------------- Spatial vs frequency check ---------------------


def gaussian_spatial(gray01, sigma_px):
    # Compare with frequency Gaussian LP (cutoff ~ sigma_f ~ 1/(2π*sigma_px))
    k = max(1, int(round(sigma_px * 6)) | 1)  # odd kernel
    g = cv2.GaussianBlur((gray01 * 255).astype(np.uint8), (k, k), sigmaX=sigma_px)
    return g.astype(np.float64) / 255.0


def approx_sigma_f_from_sigma_px(sigma_px):
    # Rough relation: sigma_f [cycles/pixel] ≈ 1 / (2π * sigma_px)
    return 1.0 / (2.0 * math.pi * max(sigma_px, 1e-6))


# --------------------- Panels ---------------------


def panel_spectrum_and_mask(F, Hmask, Ff, title=""):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(mag_spectrum(F), cmap="gray")
    ax1.set_title("Spectrum |F| (log)")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(normalize01(Hmask), cmap="gray")
    ax2.set_title("Mask H(u,v)")
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(mag_spectrum(Ff), cmap="gray")
    ax3.set_title("|F·H| (log)")
    ax3.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def side_by_side(orig01, out01, title_left="Original", title_right="Filtered"):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(
        orig01 if orig01.ndim == 3 else orig01,
        cmap=None if orig01.ndim == 3 else "gray",
    )
    ax1.set_title(title_left)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(
        out01 if out01.ndim == 3 else out01, cmap=None if out01.ndim == 3 else "gray"
    )
    ax2.set_title(title_right)
    ax2.axis("off")
    fig.tight_layout()
    return fig


# --------------------- CLI ---------------------


def parse_args():
    p = argparse.ArgumentParser(description="FFT-based image filtering demo")
    p.add_argument("--image", type=str, required=True, help="Path to image")
    p.add_argument("--mode", choices=["gray", "color"], default="gray")
    p.add_argument(
        "--filter",
        choices=["gaussian", "ideal", "butter", "laplacian"],
        default="gaussian",
    )
    p.add_argument(
        "--type",
        choices=["lowpass", "highpass", "bandpass", "bandstop"],
        default="lowpass",
        help="Ignored for laplacian",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=0.10,
        help="Normalized radius (cycles/pixel), e.g., 0.10",
    )
    p.add_argument(
        "--cutoff2", type=float, default=None, help="Second cutoff for band filters"
    )
    p.add_argument("--order", type=int, default=2, help="Butterworth order")
    p.add_argument(
        "--pad",
        action="store_true",
        help="Zero-pad to next power of 2 (with reflection boundary)",
    )
    p.add_argument(
        "--compare-spatial",
        action="store_true",
        help="Compare Gaussian LP (freq vs spatial)",
    )
    p.add_argument(
        "--sigma-px",
        type=float,
        default=2.0,
        help="Spatial Gaussian sigma (pixels) for comparison",
    )
    p.add_argument("--save", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.image)
    rgb = imread_rgb(path)
    outdir = Path(args.outdir or "fft_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "gray":
        gray01 = normalize01(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64))
        out01, Hmask, F, Ff = process_gray(
            gray01,
            filt=args.filter,
            ftype=args.type,
            cutoff=args.cutoff,
            cutoff2=args.cutoff2,
            order=args.order,
            do_pad=args.pad,
        )
        fig1 = side_by_side(
            gray01, out01, "Original (gray)", f"{args.filter}-{args.type}"
        )
        fig2 = panel_spectrum_and_mask(F, Hmask, Ff, f"{args.filter}-{args.type}")
        if args.save:
            save_gray(outdir / (path.stem + "_orig_gray.png"), gray01)
            save_gray(outdir / (path.stem + f"_{args.filter}_{args.type}.png"), out01)
            fig1.savefig(
                outdir / (path.stem + f"_{args.filter}_{args.type}_pair.png"),
                dpi=180,
                bbox_inches="tight",
            )
            fig2.savefig(
                outdir / (path.stem + f"_{args.filter}_{args.type}_spectra.png"),
                dpi=180,
                bbox_inches="tight",
            )
        plt.show(block=False)

        # Optional: compare frequency Gaussian LP with spatial blur
        if (
            args.compare_spatial
            and args.filter == "gaussian"
            and args.type == "lowpass"
        ):
            sigma_f = approx_sigma_f_from_sigma_px(args.sigma_px)
            print(
                f"[compare] sigma_px={args.sigma_px:.2f} -> approx cutoff (σ_f)={sigma_f:.4f} cycles/pixel"
            )
            out_freq, _, _, _ = process_gray(
                gray01, "gaussian", "lowpass", sigma_f, None, args.order, args.pad
            )
            out_spat = gaussian_spatial(gray01, args.sigma_px)
            fig3 = side_by_side(
                out_spat, out_freq, "Spatial Gaussian", "Freq Gaussian (σ_f≈1/2πσ)"
            )
            if args.save:
                save_gray(outdir / (path.stem + "_gauss_spatial.png"), out_spat)
                save_gray(outdir / (path.stem + "_gauss_freq.png"), out_freq)
                fig3.savefig(
                    outdir / (path.stem + "_gauss_compare.png"),
                    dpi=180,
                    bbox_inches="tight",
                )
            plt.show()

    else:  # color
        rgb01 = normalize01(rgb.astype(np.float64))
        out01, Hmask, F, Ff = process_color(
            rgb01,
            filt=args.filter,
            ftype=args.type,
            cutoff=args.cutoff,
            cutoff2=args.cutoff2,
            order=args.order,
            do_pad=args.pad,
        )
        fig1 = side_by_side(
            rgb01, out01, "Original (RGB)", f"{args.filter}-{args.type} (per-channel)"
        )
        # For spectra, show luminance channel for readability
        gray01 = normalize01(
            cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
                np.float64
            )
        )
        _, Hmask_g, Fg, Ffg = process_gray(
            gray01,
            filt=args.filter,
            ftype=args.type,
            cutoff=args.cutoff,
            cutoff2=args.cutoff2,
            order=args.order,
            do_pad=args.pad,
        )
        fig2 = panel_spectrum_and_mask(
            Fg, Hmask_g, Ffg, f"{args.filter}-{args.type} (luma view)"
        )
        if args.save:
            save_rgb(outdir / (path.stem + "_orig_rgb.png"), rgb01)
            save_rgb(
                outdir / (path.stem + f"_{args.filter}_{args.type}_rgb.png"), out01
            )
            fig1.savefig(
                outdir / (path.stem + f"_{args.filter}_{args.type}_rgb_pair.png"),
                dpi=180,
                bbox_inches="tight",
            )
            fig2.savefig(
                outdir / (path.stem + f"_{args.filter}_{args.type}_rgb_spectra.png"),
                dpi=180,
                bbox_inches="tight",
            )
        plt.show()


if __name__ == "__main__":
    main()
