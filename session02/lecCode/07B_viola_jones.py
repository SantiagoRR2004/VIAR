#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viola–Jones Face Detection (Haar + AdaBoost + Cascade) — Demo

Features
- Image or webcam/video input
- Classic detectMultiScale with tunable scaleFactor/minNeighbors/minSize
- Optional detectMultiScale3 introspection (reject levels + weights)
- Optional histogram equalization (often helps)
- Optional simple NMS and colored boxes by score
- Saves annotated frames for slides

Usage
  # Image
  python viola_jones_demo.py --input path/to/img.jpg --save

  # Webcam
  python viola_jones_demo.py --webcam 0

  # Show cascade stage/score info (if supported)
  python viola_jones_demo.py --input path.jpg --stages --save

  # Tweak params
  python viola_jones_demo.py --input path.jpg --scale 1.1 --neighbors 3 --minsize 24 24 --equalize
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np


# ---------- Helpers ----------
def load_cascade(name: str) -> cv2.CascadeClassifier:
    # Common cascade names (from cv2.data.haarcascades)
    aliases = {
        "frontal": "haarcascade_frontalface_default.xml",
        "frontal_alt2": "haarcascade_frontalface_alt2.xml",
        "profile": "haarcascade_profileface.xml",
        "eye": "haarcascade_eye.xml",
    }
    filename = aliases.get(name, name)  # allow raw filename too
    path = Path(cv2.data.haarcascades) / filename
    clf = cv2.CascadeClassifier(str(path))
    if clf.empty():
        raise FileNotFoundError(f"Could not load cascade: {path}")
    return clf


def to_gray(img_bgr, equalize=False):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(g) if equalize else g


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
    return out


def nms(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores if scores is not None else np.ones(len(boxes)))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return boxes[keep].tolist(), scores[keep].tolist()


def color_by_score(score, smin, smax):
    """Map score to color (blue->green->red)."""
    t = 0.0 if smax == smin else (score - smin) / (smax - smin + 1e-9)
    # simple jet-ish: B->G->R
    r = int(255 * max(0, min(1, 2 * t - 0.0)))
    g = int(255 * (1 - abs(2 * t - 1)))
    b = int(255 * max(0, min(1, 2 * (1 - t))))
    return (b, g, r)


# ---------- Detection ----------
def detect_on_frame(frame_bgr, clf, args):
    gray = to_gray(frame_bgr, equalize=args.equalize)
    t0 = time.time()
    if args.stages:
        # detectMultiScale3 gives reject levels & weights (OpenCV 3.4+)
        try:
            rects, rejectLevels, levelWeights = clf.detectMultiScale3(
                gray,
                scaleFactor=args.scale,
                minNeighbors=args.neighbors,
                minSize=(args.minsize[0], args.minsize[1]),
                outputRejectLevels=True,
            )
            dt = (time.time() - t0) * 1000
            rects = rects if rects is not None else []
            rejectLevels = rejectLevels.flatten().tolist() if len(rects) > 0 else []
            levelWeights = levelWeights.flatten().tolist() if len(rects) > 0 else []
            # Optional NMS on rects with weights as scores
            if args.nms and len(rects) > 0:
                rects, levelWeights = nms(rects, levelWeights, iou_thresh=0.3)
            # Draw with colors by weight (higher=warmer)
            out = frame_bgr.copy()
            if len(rects) > 0:
                smin, smax = min(levelWeights), max(levelWeights)
                for box, wgt, rj in zip(
                    rects, levelWeights, rejectLevels[: len(rects)]
                ):
                    color = color_by_score(wgt, smin, smax)
                    x, y, w, h = box
                    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
                    cv2.putText(
                        out,
                        f"{wgt:.2f} | stg~{rj}",
                        (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            info = f"{len(rects)} faces | {dt:.1f} ms"
            cv2.putText(
                out,
                info,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )
            return out, rects, dt
        except Exception as e:
            # Fallback to classic API if 3-API unsupported in user install
            print(f"[warn] detectMultiScale3 not available ({e}); falling back.")
            args.stages = False  # so next frames use classic
    # Classic detectMultiScale
    rects = clf.detectMultiScale(
        gray,
        scaleFactor=args.scale,
        minNeighbors=args.neighbors,
        minSize=(args.minsize[0], args.minsize[1]),
    )
    dt = (time.time() - t0) * 1000
    rects = rects if rects is not None else []
    if args.nms and len(rects) > 0:
        rects, _ = nms(rects, None, iou_thresh=0.3)
    out = draw_boxes(frame_bgr, rects, (0, 255, 0), 2)
    cv2.putText(
        out,
        f"{len(rects)} faces | {dt:.1f} ms",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 220, 30),
        2,
        cv2.LINE_AA,
    )
    return out, rects, dt


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Viola–Jones Haar Cascade Demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to image or video")
    src.add_argument("--webcam", type=int, help="Webcam index (e.g., 0)")

    ap.add_argument(
        "--cascade",
        type=str,
        default="frontal",
        help="Cascade alias or filename (frontal|frontal_alt2|profile|eye|*.xml)",
    )
    ap.add_argument(
        "--scale", type=float, default=1.1, help="Image pyramid scale factor (>1.0)"
    )
    ap.add_argument("--neighbors", type=int, default=3, help="minNeighbors (grouping)")
    ap.add_argument(
        "--minsize",
        type=int,
        nargs=2,
        default=[24, 24],
        metavar=("W", "H"),
        help="Minimum window size",
    )
    ap.add_argument(
        "--equalize", action="store_true", help="Histogram equalization on grayscale"
    )
    ap.add_argument(
        "--stages",
        action="store_true",
        help="Show reject levels / weights (detectMultiScale3)",
    )
    ap.add_argument("--nms", action="store_true", help="Apply simple NMS to detections")
    ap.add_argument("--save", action="store_true", help="Save annotated output")
    ap.add_argument(
        "--out",
        type=str,
        default="vj_output.png",
        help="Save path (image) or output video",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    clf = load_cascade(args.cascade)

    # Image or video/webcam?
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise SystemExit("Could not open webcam")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if args.save else None
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, rects, dt = detect_on_frame(frame, clf, args)
            cv2.imshow("Viola–Jones (Haar) — press q to quit", out)
            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(args.out, fourcc, 25, (w, h))
                writer.write(out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        return

    # File input: decide image vs video
    p = Path(args.input)
    cap = cv2.VideoCapture(str(p))
    is_video = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 1
    if not is_video:
        # Treat as image
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read: {p}")
        out, rects, dt = detect_on_frame(img, clf, args)
        cv2.imshow("Viola–Jones (Haar)", out)
        cv2.waitKey(1)
        if args.save:
            cv2.imwrite(args.out, out)
            print(f"[saved] {args.out}  ({len(rects)} faces, {dt:.1f} ms)")
        cv2.destroyAllWindows()
    else:
        # Video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if args.save else None
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, rects, dt = detect_on_frame(frame, clf, args)
            cv2.imshow("Viola–Jones (Haar) — press q to quit", out)
            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(
                        args.out, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25, (w, h)
                    )
                writer.write(out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
