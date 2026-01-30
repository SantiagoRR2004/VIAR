#!/usr/bin/env python3
import argparse, os, time, pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Model: C3D Sports-1M layout
# pool5 padding=(0,1,1) -> fc6 in_features=8192
# ----------------------------
class C3D_Sports1M_8192(nn.Module):
    def __init__(self, num_classes: int = 101, dropout: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

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
        self.pool5 = nn.MaxPool3d((2, 2, 2), (2, 2, 2), padding=(0, 1, 1))  # critical

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # x: [B,3,16,112,112]
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        h = h.view(h.size(0), -1)  # 8192
        h = self.drop(self.relu(self.fc6(h)))
        h = self.drop(self.relu(self.fc7(h)))
        return self.fc8(h)


# ----------------------------
# Robust loader for Sports-1M pickle/pth
# ----------------------------
def load_c3d_pickle_into_model(model, ckpt_path, drop_fc8_if_mismatch: bool = True):
    """
    Safely load Sports-1M C3D weights (.pickle/.pth) into model.
    - Tries torch.load(..., weights_only=True) first, falls back gracefully.
    - Strips 'module.' prefixes.
    - Drops fc8.* if class count mismatches (e.g., 487 -> 101).
    Returns (missing_keys, unexpected_keys).
    """
    ckpt = None
    try:
        ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )  # torch>=2.4
    except TypeError:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f, encoding="latin1")

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    else:
        state = ckpt  # many files are a raw state_dict

    # strip DataParallel prefix
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    # drop fc8 if needed
    if drop_fc8_if_mismatch and "fc8.weight" in state:
        want = model.fc8.weight.shape[0]
        have = state["fc8.weight"].shape[0]
        if want != have:
            state.pop("fc8.weight", None)
            state.pop("fc8.bias", None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


# ----------------------------
# UCF101 list parsing
# ----------------------------
def parse_lists(
    root: Path, split: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    listdir = root / "ucfTrainTestlist"
    cls_map = {
        ln.split()[1]: int(ln.split()[0]) - 1
        for ln in (listdir / "classInd.txt").read_text().splitlines()
        if ln.strip()
    }
    train, test = [], []
    # trainlist0X.txt: "ClassName/v_ClassName_g01_c01.avi 1"
    for line in (listdir / f"trainlist0{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        rel, idx = line.strip().split()
        train.append((root / "UCF-101" / rel, int(idx) - 1))
    # testlist0X.txt: "ClassName/v_ClassName_g01_c01.avi"
    for line in (listdir / f"testlist0{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        rel = line.strip()
        cls = Path(rel).parts[0]
        test.append((root / "UCF-101" / rel, cls_map[cls]))
    return train, test


# ----------------------------
# Video reading & preprocessing (Sports-1M style)
# ----------------------------
def read_video_rgb(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def preprocess_frame_sports1m(f: np.ndarray) -> np.ndarray:
    # Resize to (H=112, W=200), center-crop to 112x112, no mean/std
    f = cv2.resize(f, (200, 112), interpolation=cv2.INTER_AREA)  # (W,H)
    x0 = (200 - 112) // 2
    f = f[:, x0 : x0 + 112, :].astype(np.float32)  # (112,112,3)
    return f


def make_clip(frames: List[np.ndarray], center: int) -> torch.Tensor:
    # 16 frames around center (clamped), Sports-1M preproc
    half = 8
    T = len(frames)
    if T < 16:
        raise RuntimeError("Video too short (<16 frames).")
    start = max(0, min(T - 16, center - half))
    seq = frames[start : start + 16]
    buf = [preprocess_frame_sports1m(f) for f in seq]  # [T,112,112,3]
    x = np.transpose(np.stack(buf, 0), (3, 0, 1, 2))  # [3,16,112,112]
    return torch.from_numpy(x)


# ----------------------------
# Dataset: K clips per video
# ----------------------------
class UCFClipsC3D(Dataset):
    def __init__(self, items, segments=3, train=True):
        self.items = items
        self.segments = segments
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        frames = read_video_rgb(path)
        T = len(frames)
        # evenly-spaced centers (avoid edges)
        centers = np.linspace(8, max(8, T - 8), self.segments).astype(int)
        # basic jitter for train
        if self.train and self.segments > 1:
            jitter = np.random.randint(-2, 3, size=self.segments)  # +/-2 frames
            centers = np.clip(centers + jitter, 8, max(8, T - 8))
        clips = [make_clip(frames, c) for c in centers]  # K x [3,16,112,112]
        x = torch.stack(clips, 0)  # [K,3,16,112,112]
        return x, torch.tensor(y, dtype=torch.long)


# ----------------------------
# Train / Eval
# ----------------------------
def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0


def train_one_epoch(model, loader, opt, dev, log_interval=25, amp=False):
    model.train()
    ce = nn.CrossEntropyLoss()
    n_batches = len(loader)
    total, loss_sum, acc_sum = 0, 0.0, 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    t0 = time.time()
    for b, (xK, y) in enumerate(loader, start=1):
        B, K = xK.size(0), xK.size(1)
        y = y.to(dev, non_blocking=True)
        logits_sum = torch.zeros(B, model.fc8.out_features, device=dev)
        for k in range(K):
            x = xK[:, k].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
            logits_sum += logits
        logits_mean = logits_sum / K
        with torch.cuda.amp.autocast(enabled=amp):
            loss = ce(logits_mean, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            acc = accuracy_top1(logits_mean, y)
        loss_sum += loss.item() * B
        acc_sum += acc * B
        total += B

        if b == 1 or b % log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / b * (n_batches - b)
            print(
                f"[Train] {b:4d}/{n_batches:4d} | loss {loss.item():.4f} | "
                f"avg_loss {(loss_sum/total):.4f} | avg_acc {(acc_sum/total):5.2f}% | "
                f"eta {eta/60:.1f} min"
            )
    return loss_sum / total, acc_sum / total


@torch.no_grad()
def evaluate(model, loader, dev, log_interval=100, amp=False):
    model.eval()
    total, acc_sum = 0, 0.0
    n_batches = len(loader)
    t0 = time.time()
    for b, (xK, y) in enumerate(loader, start=1):
        B, K = xK.size(0), xK.size(1)
        y = y.to(dev, non_blocking=True)
        logits_sum = torch.zeros(B, model.fc8.out_features, device=dev)
        for k in range(K):
            x = xK[:, k].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits_sum += model(x)
        logits_mean = logits_sum / K
        acc = accuracy_top1(logits_mean, y)
        acc_sum += acc * B
        total += B

        if b == 1 or b % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"[Val]   {b:4d}/{n_batches:4d} | avg_acc_so_far {(acc_sum/total):5.2f}% | elapsed {elapsed:.1f}s"
            )
    return acc_sum / total


# ----------------------------
# Freeze policy
# ----------------------------
def set_freeze_policy(model: nn.Module, policy: str):
    """
    policy ∈ {'none','fc','conv5','all'}
    - none  : train everything
    - fc    : train fc6/fc7/fc8 (classic light FT)
    - conv5 : train fc6/7/8 + last conv block (conv5a/b)
    - all   : train full network
    """
    for p in model.parameters():
        p.requires_grad = True  # start unfrozen

    if policy == "all":
        return

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    if policy == "fc":
        for p in (
            list(model.fc6.parameters())
            + list(model.fc7.parameters())
            + list(model.fc8.parameters())
        ):
            p.requires_grad = True
    elif policy == "conv5":
        for p in (
            list(model.fc6.parameters())
            + list(model.fc7.parameters())
            + list(model.fc8.parameters())
        ):
            p.requires_grad = True
        for p in list(model.conv5a.parameters()) + list(model.conv5b.parameters()):
            p.requires_grad = True
    elif policy == "none":
        # unfreeze everything
        for p in model.parameters():
            p.requires_grad = True
    elif policy == "all":
        pass
    else:
        raise ValueError(f"Unknown freeze policy: {policy}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf_root", required=True, type=str)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Sports-1M C3D checkpoint (.pickle/.pth)",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segments", type=int, default=3, help="clips per video")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument(
        "--freeze", type=str, default="fc", choices=["none", "fc", "conv5", "all"]
    )
    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints_c3d")
    ap.add_argument("--log_interval", type=int, default=25)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA)")
    args = ap.parse_args()

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    root = Path(args.ucf_root)

    # Data
    train_items, val_items = parse_lists(root, args.split)
    train_ds = UCFClipsC3D(train_items, segments=args.segments, train=True)
    val_ds = UCFClipsC3D(val_items, segments=args.segments, train=False)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model (101-way head for UCF101)
    model = C3D_Sports1M_8192(num_classes=101).to(dev)

    # Load Sports-1M backbone (drops fc8 automatically)
    missing, unexpected = load_c3d_pickle_into_model(
        model, args.weights, drop_fc8_if_mismatch=True
    )
    print(f"[Load] From: {args.weights}")
    if missing:
        print("  missing:", missing)
    if unexpected:
        print("  unexpected:", unexpected)

    # Freeze policy
    set_freeze_policy(model, args.freeze)

    # Optimizer on trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(
        f"[Setup] Train {len(train_ds)} | Val {len(val_ds)} | seg={args.segments} | bs={args.batch_size} "
        f"| freeze={args.freeze} | amp={args.amp}"
    )

    best = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_ld, opt, dev, log_interval=args.log_interval, amp=args.amp
        )
        va_acc = evaluate(
            model, val_ld, dev, log_interval=args.log_interval * 2, amp=args.amp
        )
        print(
            f"[Epoch] {ep:02d}/{args.epochs} | Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | Val acc {va_acc:.2f}%"
        )

        ckpt = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"c3d_epoch{ep:02d}.pth"))
        if va_acc > best:
            best = va_acc
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))

    print(f"[Best] Val Top-1: {best:.2f}%")


if __name__ == "__main__":
    main()


"""
#  Quick sanity fine-tune (light: fc only, 3 segments)
python -u finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101 \
  --split 1 \
  --weights /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/c3d_sports1m.pickle \
  --epochs 3 \
  --batch_size 8 \
  --segments 3 \
  --device cuda \
  --freeze fc \
  --checkpoint_dir ./checkpoints_c3d \
  --log_interval 25


  

###  Deeper fine-tune (unfreeze last conv block)
python -u finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101 \
  --split 1 \
  --weights /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/c3d_sports1m.pickle \
  --epochs 5 \
  --batch_size 8 \
  --segments 5 \
  --device cuda \
  --freeze conv5 \
  --checkpoint_dir ./checkpoints_c3d \
  --log_interval 25

  

### Full fine-tune (unfreeze everything)
python -u finetune_c3d_ucf101_from_sports1m.py \
  --ucf_root /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101 \
  --split 1 \
  --weights /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101/c3d_sports1m.pickle \
  --epochs 10 \
  --batch_size 8 \
  --segments 5 \
  --device cuda \
  --freeze none \
  --checkpoint_dir ./checkpoints_c3d \
  --log_interval 25


  
## Then run  demo on the fine-tuned model
python c3d_demo_pretrained.py \
  --video ../dataset/vgolf2.mpg \
  --weights ./checkpoints_c3d/best.pth \
  --labels /media/david/.../UCF101/ucfTrainTestlist/classInd.txt \
  --device cuda \
  --num_classes 101 \
  --stride 8 --max_clips 8

"""
