#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV DFT Demo (matches slide code):
- Part 1: DFT, centered magnitude (log) and phase visualizations
- Part 2: Ideal low-pass and notch-reject filtering in the frequency domain

Outputs (into ./figs):
  fft_mag.png, fft_phase.png, fft_lowpass.png, fft_notch.png
  (plus optional montage)

Usage:
  python opencv_dft_filters.py --input path/to/image.jpg --R 40 --notch "0,70,8" "0,-70,8" --montage
"""

import argparse, os
from pathlib import Path
import numpy as np
import cv2


def ensure_figs():
    outdir = Path("figs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def read_gray(path: Path) -> np.ndarray:
    I0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if I0 is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return I0


def optimal_pad_gray(I0: np.ndarray) -> np.ndarray:
    M = cv2.getOptimalDFTSize(I0.shape[0])
    N = cv2.getOptimalDFTSize(I0.shape[1])
    I = np.zeros((M, N), np.float32)
    I[: I0.shape[0], : I0.shape[1]] = I0
    return I


def forward_dft_centered(I: np.ndarray) -> np.ndarray:
    # I float32, single-channel -> complex 2-channel (real, imag)
    F = cv2.dft(I, flags=cv2.DFT_COMPLEX_OUTPUT)  # (M,N,2)
    Fshift = np.fft.fftshift(F, axes=(0, 1))  # center DC
    return Fshift


def mag_phase_visuals(Fshift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = cv2.magnitude(Fshift[..., 0], Fshift[..., 1])
    mag_log = np.log1p(mag)
    mag_vis = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    phase = np.arctan2(Fshift[..., 1], Fshift[..., 0])  # [-pi, pi]
    phase_vis = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_vis, phase_vis


def ideal_lowpass_mask(shape, R: int) -> np.ndarray:
    M, N = shape
    cy, cx = M // 2, N // 2
    yy, xx = np.ogrid[:M, :N]
    LP = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (R * R)
    return LP.astype(np.float32)  # HxW (0/1)


def apply_mask_and_idft(
    Fshift: np.ndarray, mask_hw: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    # Duplicate mask across (real, imag)
    M2 = np.dstack([mask_hw, mask_hw]).astype(np.float32)  # HxWx2
    Ff = Fshift * M2
    # Inverse: unshift -> idft -> real output
    out = cv2.idft(np.fft.ifftshift(Ff), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    out = out[:out_h, :out_w]
    out_vis = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out_vis


def notch_reject_mask(shape, notches: list[tuple[int, int, int]]) -> np.ndarray:
    M, N = shape
    cy, cx = M // 2, N // 2
    yy, xx = np.ogrid[:M, :N]
    NR = np.ones((M, N), np.float32)
    for dy, dx, rr in notches:
        NR[((yy - (cy + dy)) ** 2 + (xx - (cx + dx)) ** 2) <= rr * rr] = 0.0
    return NR


def parse_notches(items: list[str]) -> list[tuple[int, int, int]]:
    out = []
    for s in items:
        dy, dx, rr = map(int, s.split(","))
        out.append((dy, dx, rr))
    return out


def montage_three_rows(a, b, c):
    # ensure same size via resizing to width of the smallest
    def to3(x):
        return x if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    a, b, c = to3(a), to3(b), to3(c)
    w = min(a.shape[1], b.shape[1], c.shape[1])

    def rs(im):
        return cv2.resize(
            im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA
        )

    return cv2.vconcat([rs(a), rs(b), rs(c)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (grayscale will be used)",
    )
    ap.add_argument(
        "--R",
        type=int,
        default=40,
        help="Ideal low-pass radius (pixels in padded spectrum)",
    )
    ap.add_argument(
        "--notch",
        type=str,
        nargs="*",
        default=["0,70,8", "0,-70,8"],
        help='Notch triplets "dy,dx,rr" in pixels (centered coords)',
    )
    ap.add_argument(
        "--montage", action="store_true", help="Also save a quick montage grid"
    )
    args = ap.parse_args()

    figs = ensure_figs()
    # Part 1: DFT + mag/phase
    I0 = read_gray(Path(args.input))
    I = optimal_pad_gray(I0)
    Fshift = forward_dft_centered(I)
    mag_vis, phase_vis = mag_phase_visuals(Fshift)
    cv2.imwrite(str(figs / "fft_mag.png"), mag_vis)
    cv2.imwrite(str(figs / "fft_phase.png"), phase_vis)

    # Part 2: Ideal Low-Pass
    LP = ideal_lowpass_mask(I.shape, args.R)
    Ilp_vis = apply_mask_and_idft(Fshift, LP, I0.shape[0], I0.shape[1])
    cv2.imwrite(str(figs / "fft_lowpass.png"), Ilp_vis)

    # Part 2: Notch Reject
    notches = parse_notches(args.notch)
    NR = notch_reject_mask(I.shape, notches)
    Inr_vis = apply_mask_and_idft(Fshift, NR, I0.shape[0], I0.shape[1])
    cv2.imwrite(str(figs / "fft_notch.png"), Inr_vis)

    # Optional montage (top: input + mag + phase, bottom: low-pass + notch + blank)
    if args.montage:
        top = cv2.hconcat([I0, mag_vis, phase_vis])
        blank = 255 * np.ones_like(I0)
        bottom = cv2.hconcat([Ilp_vis, Inr_vis, blank])
        grid = cv2.vconcat([top, bottom])
        cv2.imwrite(str(figs / "fft_montage.png"), grid)

    print(
        f"[done] Wrote: {figs/'fft_mag.png'}, {figs/'fft_phase.png'}, {figs/'fft_lowpass.png'}, {figs/'fft_notch.png'}"
        + (f", {figs/'fft_montage.png'}" if args.montage else "")
    )


if __name__ == "__main__":
    main()
