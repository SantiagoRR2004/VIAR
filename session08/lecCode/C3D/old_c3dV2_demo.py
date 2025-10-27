# c3d_demo.py
# -----------------------------------------------------------------------------
# C3D demo: load video -> sample 16-frame clips (112x112) -> run C3D -> avg logits
# Optional: --weights path/to/c3d_checkpoint.pth (state_dict with model.fc8 = num_classes)
# Optional: --labels path/to/classInd.txt (UCF101 format) for readable names
# -----------------------------------------------------------------------------

import argparse, os, math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model: C3D (Tran et al. 2015)
# -----------------------------
class C3D(nn.Module):
    def __init__(self, num_classes: int = 101, dropout: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))
        # For 16x112x112 input, pool5 output is [B,512,1,3,3] -> 512*1*3*3=4608
        self.fc6 = nn.Linear(512 * 1 * 3 * 3, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # x: [B,3,T,112,112] with T=16
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.drop(self.relu(self.fc6(x)))
        x = self.drop(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


# -----------------------------
# Video -> clip sampling utils
# -----------------------------
def read_video_rgb(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
    finally:
        cap.release()
    if len(frames) < 16:
        raise RuntimeError("Video too short (<16 frames).")
    return frames


def resize_short_side(frame: np.ndarray, short: int = 128) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) == short:
        return frame
    scale = short / min(h, w)
    newH, newW = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(frame, (newW, newH), interpolation=cv2.INTER_AREA)


def center_crop(frame: np.ndarray, size: int = 112) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = (h - size) // 2, (w - size) // 2
    return frame[ch : ch + size, cw : cw + size, :]


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # Convert to float in [0,1], normalize to mean=0.5, std=0.5 (symmetric)
    x = frame.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x


def make_clips(
    frames: List[np.ndarray],
    clip_len: int = 16,
    stride: int = 16,
    max_clips: int | None = None,
) -> List[np.ndarray]:
    clips = []
    T = len(frames)
    for start in range(0, T - clip_len + 1, stride):
        buf = []
        for t in range(start, start + clip_len):
            f = resize_short_side(frames[t], short=128)
            f = center_crop(f, size=112)
            f = preprocess_frame(f)
            buf.append(f)
        clip = np.stack(buf, axis=0)  # [T,112,112,3]
        clips.append(clip)
        if max_clips and len(clips) >= max_clips:
            break
    if not clips:  # fallback: last 16
        buf = []
        for t in range(T - clip_len, T):
            f = resize_short_side(frames[t], short=128)
            f = center_crop(f, size=112)
            f = preprocess_frame(f)
            buf.append(f)
        clips = [np.stack(buf, axis=0)]
    return clips


def clips_to_tensor(clips: List[np.ndarray]) -> torch.Tensor:
    # list of [T,H,W,3] -> [N,3,T,H,W]
    chwts = []
    for c in clips:
        c = np.transpose(c, (3, 0, 1, 2))  # 3,T,H,W
        chwts.append(torch.from_numpy(c))
    return torch.stack(chwts, dim=0)


# -----------------------------
# Labels
# -----------------------------
def load_ucf101_labels(classInd_path: str | None) -> list[str]:
    if not classInd_path or not os.path.isfile(classInd_path):
        return [f"class_{i}" for i in range(101)]
    names = [None] * 101
    with open(classInd_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split()
            names[int(idx) - 1] = name
    return names


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str, help="Path to input video")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--weights", default=None, type=str, help="Optional C3D checkpoint (.pth)"
    )
    ap.add_argument(
        "--labels", default=None, type=str, help="Optional classInd.txt (UCF101)"
    )
    ap.add_argument("--num_classes", default=101, type=int)
    ap.add_argument("--stride", default=16, type=int, help="clip stride")
    ap.add_argument("--max_clips", default=None, type=int, help="limit number of clips")
    args = ap.parse_args()

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    labels = load_ucf101_labels(args.labels)

    print("=" * 80)
    print("C3D Demo (Tran et al. 2015) — video inference with clip averaging")
    print(f"Video: {args.video}")
    print(f"Device: {dev}")
    print("=" * 80)

    frames = read_video_rgb(args.video)
    clips = make_clips(
        frames, clip_len=16, stride=args.stride, max_clips=args.max_clips
    )
    x = clips_to_tensor(clips).to(dev)  # [N,3,T,H,W]
    print(f"Prepared {len(clips)} clip(s) -> tensor {tuple(x.shape)}")

    model = C3D(num_classes=args.num_classes).to(dev)

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {args.weights}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)
    else:
        print(
            "No weights provided: using ImageNet-like init (random) — predictions will be untrained."
        )

    model.eval()
    with torch.no_grad():
        logits_all = []
        for i in range(x.shape[0]):
            logits = model(x[i : i + 1])
            logits_all.append(logits)
        logits_avg = torch.stack(logits_all, 0).mean(0)  # [1,num_classes]
        probs = F.softmax(logits_avg, dim=1)[0]
        topk = torch.topk(probs, k=5)

    print("\nTop-5 (avg over clips):")
    for r, (p, idx) in enumerate(zip(topk.values, topk.indices), 1):
        name = (
            labels[idx.item()]
            if 0 <= idx.item() < len(labels)
            else f"class_{idx.item()}"
        )
        print(f"  {r}. {name:<30} {float(p)*100:5.2f}%")

    print("=" * 80)
    print(
        "Tip: supply --weights (fine-tuned on UCF101) and --labels (classInd.txt) for meaningful actions."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

"""


python c3d_demo.py --video ../dataset/golf.mp4 --device cuda --stride 8 --max_clips 8


"""
