# ed_demo.py
import argparse, os, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# ----------------- Data -----------------
def draw_circle(h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    return mask.astype(np.float32)


class BlobsDataset(Dataset):
    def __init__(self, n=512, h=128, w=128, max_blobs=4, noise=0.15, seed=0):
        rng = np.random.RandomState(seed)
        self.samples = []
        for _ in range(n):
            img = np.zeros((h, w), np.float32)
            mask = np.zeros((h, w), np.float32)
            b = rng.randint(1, max_blobs + 1)
            for _ in range(b):
                r = rng.randint(8, 22)
                cx = rng.randint(r, w - r)
                cy = rng.randint(r, h - r)
                m = draw_circle(h, w, cx, cy, r)
                mask = np.maximum(mask, m)
                img += m * (0.6 + 0.4 * rng.rand())  # brighter inside blobs
            img += noise * rng.randn(h, w)
            img = np.clip(img, 0, 1)
            self.samples.append((img, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, m = self.samples[i]
        # to tensors: (1,H,W)
        return torch.from_numpy(img)[None, ...], torch.from_numpy(m)[None, ...]


# ----------------- Model -----------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncDec(nn.Module):
    def __init__(self, in_ch=1, mid=32, out_ch=1):
        super().__init__()
        # Encoder (strided conv downsamples)
        self.e1 = ConvBlock(in_ch, mid)  # B x mid x H x W
        self.d1 = nn.Conv2d(mid, mid, 3, stride=2, padding=1)  # H/2
        self.e2 = ConvBlock(mid, mid * 2)  # B x 2mid x H/2 x W/2
        self.d2 = nn.Conv2d(mid * 2, mid * 2, 3, stride=2, padding=1)  # H/4

        # Bottleneck
        self.bott = ConvBlock(mid * 2, mid * 4)  # B x 4mid x H/4 x W/4

        # Decoder (upsample + conv)
        self.u2 = nn.ConvTranspose2d(mid * 4, mid * 2, 2, stride=2)  # H/2
        self.dec2 = ConvBlock(mid * 2, mid * 2)
        self.u1 = nn.ConvTranspose2d(mid * 2, mid, 2, stride=2)  # H
        self.dec1 = ConvBlock(mid, mid)

        self.head = nn.Conv2d(mid, out_ch, 1)  # logits

    def forward(self, x):
        x = self.e1(x)
        x = self.d1(x)
        x = self.e2(x)
        x = self.d2(x)
        x = self.bott(x)
        x = self.u2(x)
        x = self.dec2(x)
        x = self.u1(x)
        x = self.dec1(x)
        return self.head(x)


class EncDecWithSkips(nn.Module):
    def __init__(self, in_ch=1, mid=32, out_ch=1):
        super().__init__()
        self.e1 = ConvBlock(in_ch, mid)
        self.d1 = nn.Conv2d(mid, mid, 3, 2, 1)  # H/2
        self.e2 = ConvBlock(mid, mid * 2)
        self.d2 = nn.Conv2d(mid * 2, mid * 2, 3, 2, 1)  # H/4
        self.bott = ConvBlock(mid * 2, mid * 4)

        self.u2 = nn.ConvTranspose2d(mid * 4, mid * 2, 2, 2)
        self.m2 = nn.Conv2d(mid * 2 + mid * 2, mid * 2, 3, padding=1)
        self.dec2 = ConvBlock(mid * 2, mid * 2)

        self.u1 = nn.ConvTranspose2d(mid * 2, mid, 2, 2)
        self.m1 = nn.Conv2d(mid + mid, mid, 3, padding=1)
        self.dec1 = ConvBlock(mid, mid)

        self.head = nn.Conv2d(mid, out_ch, 1)

    def forward(self, x):
        s1 = self.e1(x)
        x = self.d1(s1)
        s2 = self.e2(x)
        x = self.d2(s2)
        x = self.bott(x)
        x = self.u2(x)
        x = self.m2(torch.cat([x, s2], dim=1))
        x = self.dec2(x)
        x = self.u1(x)
        x = self.m1(torch.cat([x, s1], dim=1))
        x = self.dec1(x)
        return self.head(x)


# ----------------- Train / Eval -----------------
def train_one_epoch(model, loader, opt, lossf, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = lossf(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def sample_preds(model, device, out_path, n=8, h=128, w=128):
    model.eval()
    ds = BlobsDataset(n=n, h=h, w=w, seed=999)

    xs, ys, ps = [], [], []
    for i in range(n):
        x, y = ds[i]  # x,y: (1,H,W) tensors (CPU)
        x_dev = x.to(device).unsqueeze(0)  # (1,1,H,W)
        logits = model(x_dev)
        pred = torch.sigmoid(logits).cpu().squeeze(0)  # (1,H,W)

        xs.append(x)  # (1,H,W), CPU
        ys.append(y)  # (1,H,W), CPU
        ps.append(pred)  # (1,H,W), CPU

    # --- Plot: 3 rows (Input / GT / Pred) x n columns ---
    fig, axes = plt.subplots(
        nrows=3, ncols=n, figsize=(n * 1.6, 4.8), constrained_layout=True
    )

    row_labels = ["Input Image", "Ground Truth", "Prediction"]
    rows = [xs, ys, ps]

    for r in range(3):
        for c in range(n):
            ax = axes[r, c] if n > 1 else axes[r]
            img = rows[r][c].squeeze(0).numpy()  # (H,W)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            # Put a thin frame to mimic image grid look
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

    # Add row labels on the left
    for r in range(3):
        ax = axes[r, 0] if n > 1 else axes[r]
        ax.set_ylabel(row_labels[r], fontsize=10, rotation=90, labelpad=10)

    # Optional column header for readability
    for c in range(n):
        axes[0, c].set_title(f"#{c+1}", fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--model",
        type=str,
        default="skips",
        choices=["noskip", "skips"],
        help="Choose 'noskip' (EncDec) or 'skips' (EncDecWithSkips)",
    )
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    train_ds = BlobsDataset(n=1024, h=args.h, w=args.w, seed=42)
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False
    )

    if args.model == "noskip":
        model = EncDec(in_ch=1, out_ch=1).to(device)
        print("=> Training EncDec (no skips)")
    else:
        model = EncDecWithSkips(in_ch=1, out_ch=1).to(device)
        print("=> Training EncDecWithSkips (lightweight skips)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, opt, lossf, device)
        print(f"Epoch {ep}/{args.epochs}  loss={loss:.4f}")

    sample_preds(
        model,
        device,
        out_path=f"outputs/sample_preds_{args.model}.png",
        n=8,
        h=args.h,
        w=args.w,
    )
    print(f"Saved predictions grid to outputs/sample_preds_{args.model}.png")


if __name__ == "__main__":
    main()


"""
python ed_demo.py --epochs 5
# Check outputs/sample_preds.png (rows: inputs | targets | predictions)


python ed_demo.py --model noskip --epochs 5
python ed_demo.py --model skips  --epochs 5
"""
