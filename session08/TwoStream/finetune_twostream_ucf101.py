# finetune_twostream_ucf101.py
# ---------------------------------------------------------------------
# Light fine-tune of a Two-Stream (VGG16) model on UCF101
# - Temporal stream trained by default (20-ch flow), spatial frozen
# - Farnebäck flow (fallback) or TV-L1 if opencv-contrib is available
# - Multi-segment sampling; late fusion at eval
# ---------------------------------------------------------------------
import argparse, os, math, time, json, hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

# -----------------------
# Globals / constants
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def load_class_index(classInd_path: Path) -> Dict[str, int]:
    # classInd.txt has lines: "1 ApplyEyeMakeup"
    mp = {}
    for line in read_lines(classInd_path):
        idx, name = line.split()
        mp[name] = int(idx) - 1  # 0-based
    return mp


def parse_train_test_lists(
    root: Path, split: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[str]]:
    lists = root / "ucfTrainTestlist"
    class_map = load_class_index(lists / "classInd.txt")
    # trainlist01.txt lines: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1"
    # testlist01.txt  lines: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    tr = []
    for line in read_lines(lists / f"trainlist0{split}.txt"):
        rel, idx = line.split()
        label = int(idx) - 1
        tr.append((root / "UCF-101" / rel, label))
    te = []
    for line in read_lines(lists / f"testlist0{split}.txt"):
        rel = line
        # label from folder name:
        cls = Path(rel).parts[0]
        label = class_map[cls]
        te.append((root / "UCF-101" / rel, label))
    # names ordered by index
    names = [None] * len(class_map)
    for k, v in class_map.items():
        names[v] = k
    return tr, te, names


# -----------------------
# Flow utilities
# -----------------------
def tvl1_available():
    return hasattr(cv2, "optflow") and hasattr(
        cv2.optflow, "DualTVL1OpticalFlow_create"
    )


def compute_flow_pair(
    prev_gray: np.ndarray, next_gray: np.ndarray, method: str = "auto"
) -> np.ndarray:
    if method == "tvl1" or (method == "auto" and tvl1_available()):
        of = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = of.calc(prev_gray, next_gray, None)
    else:
        # Farnebäck (stronger params)
        prev_blr = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        next_blr = cv2.GaussianBlur(next_gray, (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(
            prev_blr,
            next_blr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
    return flow.astype(np.float32)  # HxWx2


def flow_stack_from_frames(
    frames_gray: List[np.ndarray],
    center: int,
    num_pairs: int,
    out_size: int = 224,
    method: str = "auto",
) -> np.ndarray:
    # build window around center
    half = num_pairs // 2
    start = max(0, min(len(frames_gray) - (num_pairs + 1), center - half))
    wnd = frames_gray[start : start + num_pairs + 1]
    flows = []
    for i in range(num_pairs):
        flow = compute_flow_pair(wnd[i], wnd[i + 1], method=method)  # HxWx2
        flows.append(flow)
    flow_stack = np.concatenate(flows, axis=2)  # HxWx(2*num_pairs)

    H, W = flow_stack.shape[:2]
    scale = 256 / min(H, W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    fs = cv2.resize(flow_stack, (newW, newH), interpolation=cv2.INTER_LINEAR)
    ch, cw = (newH - out_size) // 2, (newW - out_size) // 2
    fs = fs[ch : ch + out_size, cw : cw + out_size, :]
    # Normalize flow roughly to [-1,1] by robust clamping (95th percentile)
    mag = np.sqrt(fs[..., 0] ** 2 + fs[..., 1] ** 2)
    q = np.percentile(mag, 95) + 1e-6
    fs = np.clip(fs, -q, q) / q
    return fs.astype(np.float32)  # 224x224x(2K)


# Simple disk cache for flow stacks per video center index
def flow_cache_key(
    video_path: Path, center: int, num_pairs: int, size: int, method: str
) -> str:
    s = f"{video_path}:{center}:{num_pairs}:{size}:{method}"
    return hashlib.md5(s.encode()).hexdigest() + ".npy"


# -----------------------
# Two-Stream model
# -----------------------
class VGGStream(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 101,
        inflate_from_rgb: bool = False,
    ):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            old: nn.Conv2d = base.features[0]
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )
            with torch.no_grad():
                w = old.weight  # [64,3,3,3]
                w_avg = w.mean(1, keepdim=True)  # [64,1,3,3]
                w_rep = w_avg.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
                new.weight.copy_(w_rep)
                if old.bias is not None and new.bias is not None:
                    new.bias.copy_(old.bias)
            base.features[0] = new
        # replace classifier out
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_classes)
        self.backbone = base

    def forward(self, x):  # logits
        return self.backbone(x)


class TwoStreamNet(nn.Module):
    def __init__(self, num_classes: int = 101):
        super().__init__()
        self.spatial = VGGStream(in_channels=3, num_classes=num_classes)
        self.temporal = VGGStream(
            in_channels=20, num_classes=num_classes, inflate_from_rgb=True
        )

    def forward(self, rgb, flow, fusion="late"):
        s = self.spatial(rgb) if rgb is not None else None
        t = self.temporal(flow) if flow is not None else None
        if fusion == "late":
            if s is not None and t is not None:
                return s, t, (s + t) / 2
            return s, t, s if t is None else t
        raise ValueError("fusion must be 'late'")


# -----------------------
# Dataset
# -----------------------
class UCFTwoStreamDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, int]],
        class_names: List[str],
        segments: int = 3,
        num_pairs: int = 10,
        size: int = 224,
        split: str = "train",
        flow_method: str = "auto",
        cache_dir: Path | None = None,
        spatial_on: bool = False,
    ):
        """
        segments: number of temporal centers sampled per video
        spatial_on: whether to return RGB frames; if False, returns None for rgb tensor
        """
        self.items = items
        self.names = class_names
        self.segments = segments
        self.num_pairs = num_pairs
        self.size = size
        self.split = split
        self.flow_method = flow_method
        self.cache_dir = cache_dir
        self.spatial_on = spatial_on

        self.rgb_tfm_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.rgb_tfm_eval = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.items)

    def _read_video_grayscale(self, path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        frames = []
        try:
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                frames.append(g)
        finally:
            cap.release()
        if len(frames) < self.num_pairs + 1:
            raise RuntimeError(f"Video too short: {path}")
        return frames

    def _read_video_rgb(self, path: Path) -> List[np.ndarray]:
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

    def _sample_centers(self, T: int) -> np.ndarray:
        # evenly spaced centers
        idxs = np.linspace(
            self.num_pairs // 2, T - self.num_pairs // 2 - 1, num=self.segments
        )
        return np.clip(np.round(idxs).astype(int), 0, T - 1)

    def __getitem__(self, i):
        vpath, label = self.items[i]
        # Load frames once (gray for flow); RGB only if requested
        frames_gray = self._read_video_grayscale(vpath)
        T = len(frames_gray)
        centers = self._sample_centers(T)

        rgb_frames = None
        if self.spatial_on:
            rgb_frames = self._read_video_rgb(vpath)

        # Build K flow stacks (+ optional K rgb crops)
        flow_tensors = []
        rgb_tensors = []
        for c in centers:
            # caching
            flow_arr = None
            if self.cache_dir is not None:
                ensure_dir(self.cache_dir)
                key = flow_cache_key(
                    vpath, int(c), self.num_pairs, self.size, self.flow_method
                )
                cache_file = self.cache_dir / key
                if cache_file.exists():
                    flow_arr = np.load(cache_file)
            if flow_arr is None:
                flow_arr = flow_stack_from_frames(
                    frames_gray, int(c), self.num_pairs, self.size, self.flow_method
                )
                if self.cache_dir is not None:
                    np.save(cache_file, flow_arr)

            # HxWx(2K) -> CxHxW
            flow_chw = torch.from_numpy(np.transpose(flow_arr, (2, 0, 1)))
            flow_tensors.append(flow_chw)

            if self.spatial_on:
                # prepare RGB centered at c
                frame = rgb_frames[int(c)]
                tfm = self.rgb_tfm_train if self.split == "train" else self.rgb_tfm_eval
                rgb_tensors.append(tfm(frame))

        flow_batch = torch.stack(flow_tensors, dim=0)  # [K,20,224,224]
        rgb_batch = (
            torch.stack(rgb_tensors, dim=0) if self.spatial_on else torch.empty(0)
        )
        return rgb_batch, flow_batch, torch.tensor(label, dtype=torch.long)


# -----------------------
# Training / evaluation
# -----------------------
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(1)
    return (pred == target).float().mean().item() * 100.0


def train_one_epoch(model: TwoStreamNet, loader: DataLoader, opt, dev, args):
    model.train()
    ce = nn.CrossEntropyLoss()

    total_seen = 0
    loss_cum = 0.0
    acc_cum = 0.0
    t_epoch0 = time.time()
    t_last = t_epoch0

    for bidx, (rgbK, flowK, y) in enumerate(loader, start=1):
        B = y.size(0)
        total_seen += B

        flowK = flowK.to(dev)  # [B,K,20,224,224]
        rgbK = rgbK.to(dev) if args.spatial and rgbK.nelement() else None
        y = y.to(dev)

        # average logits across segments
        logits_sum = torch.zeros(B, args.num_classes, device=dev)
        for k in range(flowK.size(1)):
            rgb_in = rgbK[:, k] if args.spatial and rgbK is not None else None
            flow_in = flowK[:, k]
            s, t, f = model(rgb_in, flow_in, fusion="late")
            logits = f if args.fuse else (t if args.temporal_only else s)
            logits_sum += logits
        logits_mean = logits_sum / flowK.size(1)

        loss = ce(logits_mean, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # running metrics
        with torch.no_grad():
            pred = logits_mean.argmax(1)
            acc = (pred == y).float().mean().item() * 100.0

        loss_cum += loss.item() * B
        acc_cum += acc * B

        # periodic print
        if bidx % args.log_interval == 0 or bidx == 1:
            now = time.time()
            dt = now - t_last
            t_last = now
            # samples processed in the last window (approx)
            window_samples = args.log_interval * B if bidx != 1 else B
            sps = window_samples / max(dt, 1e-6)
            elapsed = now - t_epoch0
            # rough ETA
            est_total = (elapsed / bidx) * len(loader)
            eta = est_total - elapsed
            print(
                f"[Train] batch {bidx:4d}/{len(loader):4d} | "
                f"loss {loss.item():.4f} | acc {acc:5.2f}% | "
                f"avg_loss {(loss_cum/total_seen):.4f} | avg_acc {(acc_cum/total_seen):5.2f}% | "
                f"{sps:6.1f} samp/s | eta {eta/60:.1f} min"
            )

    epoch_time = time.time() - t_epoch0
    return loss_cum / total_seen, acc_cum / total_seen, epoch_time


@torch.no_grad()
def evaluate(model: TwoStreamNet, loader: DataLoader, dev, args):
    model.eval()
    total, acc = 0, 0.0
    t0 = time.time()
    for bidx, (rgbK, flowK, y) in enumerate(loader, start=1):
        B = y.size(0)
        flowK = flowK.to(dev)
        rgbK = rgbK.to(dev) if args.spatial and rgbK.nelement() else None

        logits_sum = torch.zeros(B, args.num_classes, device=dev)
        for k in range(flowK.size(1)):
            rgb_in = rgbK[:, k] if args.spatial and rgbK is not None else None
            flow_in = flowK[:, k]
            s, t, f = model(rgb_in, flow_in, fusion="late")
            logits = f if args.fuse else (t if args.temporal_only else s)
            logits_sum += logits
        logits_mean = logits_sum / flowK.size(1)

        y = y.to(dev)
        total += B
        acc += accuracy_top1(logits_mean, y) * B

        if bidx % max(1, (args.log_interval * 2)) == 0 or bidx == 1:
            elapsed = time.time() - t0
            print(
                f"[Val]   batch {bidx:4d}/{len(loader):4d} | "
                f"avg_acc_so_far {(acc/total):5.2f}% | elapsed {elapsed:.1f}s"
            )
    return acc / total


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucf_root", type=str, required=True)
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--segments", type=int, default=3, help="segments per video")
    parser.add_argument(
        "--num_pairs", type=int, default=10, help="(u,v) pairs per flow stack"
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--flow_method", type=str, default="auto", choices=["auto", "tvl1", "farneback"]
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_twostream")
    parser.add_argument("--cache_dir", type=str, default="./flow_cache")
    # Training mode switches
    parser.add_argument(
        "--temporal_only",
        action="store_true",
        help="train only temporal stream (default if set)",
    )
    parser.add_argument(
        "--spatial", action="store_true", help="also include spatial stream data"
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="optimize fused logits (requires spatial+temporal)",
    )
    parser.add_argument(
        "--unfreeze_spatial", action="store_true", help="train spatial stream too"
    )
    parser.add_argument(
        "--unfreeze_temporal_backbone",
        action="store_true",
        help="unfreeze more of temporal backbone",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Print training status every N batches",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    args.num_classes = 101

    # Data
    root = Path(args.ucf_root)
    train_items, test_items, class_names = parse_train_test_lists(root, args.split)

    # Datasets
    cache_dir = Path(args.cache_dir)
    train_ds = UCFTwoStreamDataset(
        train_items,
        class_names,
        segments=args.segments,
        num_pairs=args.num_pairs,
        size=args.size,
        split="train",
        flow_method=args.flow_method,
        cache_dir=cache_dir,
        spatial_on=args.spatial,
    )
    val_ds = UCFTwoStreamDataset(
        test_items,
        class_names,
        segments=args.segments,
        num_pairs=args.num_pairs,
        size=args.size,
        split="val",
        flow_method=args.flow_method,
        cache_dir=cache_dir,
        spatial_on=args.spatial,
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"[Setup] Train videos: {len(train_ds)} | Val videos: {len(val_ds)} "
        f"| segments={args.segments} | pairs={args.num_pairs} | batch_size={args.batch_size}"
    )
    # Model
    model = TwoStreamNet(num_classes=args.num_classes).to(dev)

    # Freeze / unfreeze
    for p in model.parameters():
        p.requires_grad = False
    # Always train temporal classifier head
    for p in model.temporal.backbone.classifier.parameters():
        p.requires_grad = True
    # Optionally unfreeze some temporal backbone (first conv often helps)
    if args.unfreeze_temporal_backbone:
        for n, p in model.temporal.backbone.features.named_parameters():
            if n.startswith("0"):  # first conv block
                p.requires_grad = True
    # Spatial?
    if args.unfreeze_spatial:
        for p in model.spatial.backbone.classifier.parameters():
            p.requires_grad = True

    # Which logits do we optimize?
    if args.fuse and not args.spatial:
        print("`--fuse` requires `--spatial`; falling back to temporal_only.")
        args.fuse = False
        args.temporal_only = True
    if not args.temporal_only and not args.spatial:
        print("No stream selected for training; enabling temporal_only.")
        args.temporal_only = True

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    ensure_dir(args.checkpoint_dir)
    best_acc = 0.0

    print(
        f"Train videos: {len(train_ds)} | Val videos: {len(val_ds)} | Temporal-only: {args.temporal_only} | Spatial used: {args.spatial} | Fuse: {args.fuse}"
    )
    print(
        f"Flow: {args.flow_method} | segments: {args.segments} | pairs: {args.num_pairs}"
    )

    print(
        "[Info] Starting training … computing optical flow & building batches (first epoch may be slow)."
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, ep_time = train_one_epoch(model, train_ld, opt, dev, args)
        va_acc = evaluate(model, val_ld, dev, args)
        print(
            f"[Epoch] {epoch:02d}/{args.epochs} | {ep_time:.1f}s | "
            f"Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | Val acc {va_acc:.2f}%"
        )
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | {dt:.1f}s | Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | Val acc {va_acc:.2f}%"
        )

        # checkpoint
        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(
            ckpt, os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch:02d}.pth")
        )
        if is_best:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
    print(f"Best Val Top-1: {best_acc:.2f}%")


if __name__ == "__main__":
    main()


"""

python -u finetune_twostream_ucf101.py \
  --ucf_root /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101 \
  --split 1 \
  --epochs 1 \
  --batch_size 4 \
  --segments 2 \
  --num_pairs 10 \            
  --flow_method farneback \
  --device cuda \
  --temporal_only \
  --unfreeze_temporal_backbone \
  --cache_dir /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/tmp/flow_cache_ucf101 \
  --checkpoint_dir ./checkpoints_twostream \
  --log_interval 10 | tee train_sanity.log


3) Longer run with more segments (after sanity check)
python -u finetune_twostream_ucf101.py \
  --ucf_root /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/Asignaturas/UCF101 \
  --split 1 \
  --epochs 5 \
  --batch_size 8 \
  --segments 5 \
  --num_pairs 10 \
  --flow_method farneback \
  --device cuda \
  --temporal_only \
  --unfreeze_temporal_backbone \
  --cache_dir /media/david/3e2b3869-3ed0-4dd2-93a2-c4dfb7005301/tmp/flow_cache_ucf101 \
  --checkpoint_dir ./checkpoints_twostream \
  --log_interval 25 | tee train_long.log
"""
