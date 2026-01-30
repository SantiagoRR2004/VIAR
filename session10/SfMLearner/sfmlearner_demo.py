#!/usr/bin/env python3
"""
SfMLearner-style demo (Zhou et al. 2017, simplified)

Unsupervised learning of:
  - Depth for a target frame
  - Relative pose between target and neighboring frames

Loss:
  - Photometric reprojection loss (target vs warped sources)
  - Edge-aware depth smoothness

Assumed data layout (monocular sequences):

data_synth/
  mono_seq/
    seq_000/
      0000.png
      0001.png
      0002.png
      ...
    seq_001/
      0000.png
      0001.png
      ...

Training uses triplets (t-1, t, t+1); you can extend to more context if you like.
"""

import argparse
from pathlib import Path
import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================================================================
# Dataset: monocular sequences → (target, sources)
# ================================================================


class MonoSequenceDataset(Dataset):
    """
    Monocular sequence dataset for SfMLearner-style training.

    root/
      seq_000/
        0000.png
        0001.png
        0002.png
        ...
      seq_001/
        ...

    We form 3-frame snippets: (t-1, t, t+1).
    Returned:
        target:  (3,H,W)
        sources: (N_src,3,H,W)  with N_src=2 here (previous + next)
    """

    def __init__(self, root: str, resize: tuple[int, int] | None = (192, 128)):
        self.root = Path(root)
        self.resize = resize

        # Collect all sequences
        self.samples = []  # list of (seq_dir, center_index)
        seq_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        for seq in seq_dirs:
            img_paths = sorted(glob.glob(str(seq / "*.png")))
            n = len(img_paths)
            # We need at least 3 frames to form (t-1, t, t+1)
            if n < 3:
                continue
            # Valid centers are 1..n-2 (0-based)
            for idx in range(1, n - 1):
                self.samples.append((seq, idx))
        if not self.samples:
            raise RuntimeError(f"No valid 3-frame snippets found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.resize is not None:
            w, h = self.resize
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
        return img_rgb

    def __getitem__(self, idx):
        seq_dir, center_idx = self.samples[idx]
        img_paths = sorted(glob.glob(str(seq_dir / "*.png")))

        # Paths for t-1, t, t+1
        path_prev = Path(img_paths[center_idx - 1])
        path_t = Path(img_paths[center_idx])
        path_next = Path(img_paths[center_idx + 1])

        img_prev = self._load_image(path_prev)
        img_t = self._load_image(path_t)
        img_next = self._load_image(path_next)

        # Convert to tensors (C,H,W)
        tgt = torch.from_numpy(img_t).permute(2, 0, 1)  # (3,H,W)
        src_prev = torch.from_numpy(img_prev).permute(2, 0, 1)  # (3,H,W)
        src_next = torch.from_numpy(img_next).permute(2, 0, 1)

        sources = torch.stack([src_prev, src_next], dim=0)  # (2,3,H,W)
        return tgt, sources


# ================================================================
# Model: DepthNet + PoseNet
# ================================================================


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def upconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthNet(nn.Module):
    """
    Simple UNet-style depth network (predicts inverse depth / disparity-like).

    Input:  (B,3,H,W)
    Output: (B,1,H,W) positive depth (via sigmoid + scaling)
    """

    def __init__(self, base_channels=32, min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        C = base_channels

        # Encoder
        self.conv1 = conv_block(3, C, 7, stride=2, padding=3)  # H/2
        self.conv2 = conv_block(C, 2 * C, 5, stride=2, padding=2)  # H/4
        self.conv3 = conv_block(2 * C, 4 * C, 3, stride=2, padding=1)  # H/8
        self.conv4 = conv_block(4 * C, 8 * C, 3, stride=2, padding=1)  # H/16
        self.conv5 = conv_block(8 * C, 8 * C, 3, stride=2, padding=1)  # H/32

        # Decoder with skip connections
        self.up4 = upconv_block(8 * C, 8 * C)  # H/16
        self.iconv4 = conv_block(8 * C + 8 * C, 8 * C)

        self.up3 = upconv_block(8 * C, 4 * C)  # H/8
        self.iconv3 = conv_block(4 * C + 4 * C, 4 * C)

        self.up2 = upconv_block(4 * C, 2 * C)  # H/4
        self.iconv2 = conv_block(2 * C + 2 * C, 2 * C)

        self.up1 = upconv_block(2 * C, C)  # H/2
        self.iconv1 = conv_block(C + C, C)

        self.up0 = upconv_block(C, C)  # H
        self.iconv0 = conv_block(C, C)

        # Final depth head
        self.depth_head = nn.Conv2d(C, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)  # (B,C,H/2,W/2)
        conv2 = self.conv2(conv1)  # (B,2C,H/4,W/4)
        conv3 = self.conv3(conv2)  # (B,4C,H/8,W/8)
        conv4 = self.conv4(conv3)  # (B,8C,H/16,W/16)
        conv5 = self.conv5(conv4)  # (B,8C,H/32,W/32)

        # Decoder with skips
        up4 = self.up4(conv5)  # (B,8C,H/16)
        iconv4 = self.iconv4(torch.cat([up4, conv4], dim=1))

        up3 = self.up3(iconv4)  # (B,4C,H/8)
        iconv3 = self.iconv3(torch.cat([up3, conv3], dim=1))

        up2 = self.up2(iconv3)  # (B,2C,H/4)
        iconv2 = self.iconv2(torch.cat([up2, conv2], dim=1))

        up1 = self.up1(iconv2)  # (B,C,H/2)
        iconv1 = self.iconv1(torch.cat([up1, conv1], dim=1))

        up0 = self.up0(iconv1)  # (B,C,H)
        iconv0 = self.iconv0(up0)

        depth_logits = self.depth_head(iconv0)  # (B,1,H,W)
        # Convert to positive depth via sigmoid: d in (min_depth, max_depth)
        depth = self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(
            depth_logits
        )
        return depth


class PoseNet(nn.Module):
    """
    Pose network: predicts 6-DoF pose for each source frame w.r.t target.

    Input:  concatenated [target, sources] along channel dimension.
            e.g., for 3 frames: (B, 3*3, H, W) = (B,9,H,W).
    Output: (B, N_src, 6)   [tx, ty, tz, rx, ry, rz] per source.
    """

    def __init__(self, n_frames: int = 3, base_channels: int = 16):
        super().__init__()
        self.n_frames = n_frames
        in_ch = 3 * n_frames  # target + (n_frames-1) sources

        C = base_channels
        self.conv1 = conv_block(in_ch, C, 7, stride=2, padding=3)
        self.conv2 = conv_block(C, 2 * C, 5, stride=2, padding=2)
        self.conv3 = conv_block(2 * C, 4 * C, 3, stride=2, padding=1)
        self.conv4 = conv_block(4 * C, 8 * C, 3, stride=2, padding=1)

        self.conv5 = conv_block(8 * C, 8 * C, 3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pose_fc = nn.Linear(8 * C, 6 * (n_frames - 1))

    def forward(self, target: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        """
        target:  (B,3,H,W)
        sources: (B,N_src,3,H,W)
        Returns:
            poses: (B,N_src,6)
        """
        B, N_src, _, H, W = sources.shape
        assert N_src == self.n_frames - 1

        # Concatenate [target, source_0, source_1, ...] along channels
        imgs = [target]
        for i in range(N_src):
            imgs.append(sources[:, i])
        x = torch.cat(imgs, dim=1)  # (B,3*n_frames,H,W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)  # (B,8C,1,1)
        x = x.view(B, -1)
        pose_vec = self.pose_fc(x)  # (B,6*(n_frames-1))
        pose_vec = pose_vec.view(B, N_src, 6)

        # Small initialization: scale translation
        pose_vec[..., :3] = 0.01 * pose_vec[..., :3]
        return pose_vec  # (B,N_src,6)


class SfMLearner(nn.Module):
    """
    Full model: DepthNet + PoseNet
    """

    def __init__(self, min_depth=0.1, max_depth=100.0, n_frames=3):
        super().__init__()
        self.depth_net = DepthNet(min_depth=min_depth, max_depth=max_depth)
        self.pose_net = PoseNet(n_frames=n_frames)

    def forward(self, target, sources):
        depth = self.depth_net(target)  # (B,1,H,W)
        pose_vec = self.pose_net(target, sources)  # (B,N_src,6)
        return depth, pose_vec


# ================================================================
# Geometry utilities: SE3, backproject, project, warping
# ================================================================


def euler_to_matrix(rx, ry, rz):
    """
    Convert small-angle rotations (radians) to 3x3 rotation matrices.
    rx,ry,rz: (...,) tensors
    """
    # Compute cos/sin
    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

    # Rotation matrices around x,y,z
    Rx = torch.stack(
        [
            torch.stack(
                [torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)],
                dim=-1,
            ),
            torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
        ],
        dim=-2,
    )

    Ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack(
                [torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)],
                dim=-1,
            ),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
        ],
        dim=-2,
    )

    Rz = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
            torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
            torch.stack(
                [torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    # R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def pose_vec_to_mat(pose_vec):
    """
    pose_vec: (B,N_src,6)  [tx,ty,tz, rx,ry,rz]
    Returns:  (B,N_src,4,4) SE3 matrices T_{t->s}
    """
    t = pose_vec[..., :3]  # (B,N,3)
    r = pose_vec[..., 3:]  # (B,N,3)
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]

    R = euler_to_matrix(rx, ry, rz)  # (B,N,3,3)
    B, N = t.shape[:2]

    T = torch.zeros(B, N, 4, 4, device=pose_vec.device, dtype=pose_vec.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# def create_pixel_grid(B, H, W, device):
#     """
#     Create homogeneous pixel coordinates grid (B,3,H,W):
#     [u, v, 1]^T with u in [0,W-1], v in [0,H-1].
#     """
#     u = torch.arange(0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
#     v = torch.arange(0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
#     ones = torch.ones_like(u)
#     grid = torch.cat([u, v, ones], dim=1)  # (B,3,H,W)
#     return grid


def create_pixel_grid(B, H, W, device, dtype=torch.float32):
    """
    Create homogeneous pixel coordinates grid (B,3,H,W):
    [u, v, 1]^T with u in [0,W-1], v in [0,H-1].
    """
    u = (
        torch.arange(0, W, device=device, dtype=dtype)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )
    v = (
        torch.arange(0, H, device=device, dtype=dtype)
        .view(1, 1, H, 1)
        .expand(B, 1, H, W)
    )
    ones = torch.ones_like(u)
    grid = torch.cat([u, v, ones], dim=1)  # (B,3,H,W)
    return grid


# def backproject(depth, K_inv, pixel_grid):
#     """
#     Back-project pixels from target camera to 3D points in target frame.

#     depth:      (B,1,H,W)
#     K_inv:      (B,3,3)
#     pixel_grid: (B,3,H,W) homogeneous pixel coords [u,v,1]^T

#     Returns:    (B,3,H,W) 3D points in camera coordinates.
#     """
#     B, _, H, W = depth.shape
#     grid = pixel_grid.view(B, 3, -1)          # (B,3,HW)
#     cam = K_inv @ grid                        # (B,3,HW)
#     cam = cam.view(B, 3, H, W)
#     X = cam * depth                           # (B,3,H,W)
#     return X


def backproject(depth, K_inv, pixel_grid):
    """
    Back-project pixels from target camera to 3D points in target frame.

    depth:      (B,1,H,W)
    K_inv:      (B,3,3)
    pixel_grid: (B,3,H,W) homogeneous pixel coords [u,v,1]^T

    Returns:    (B,3,H,W) 3D points in camera coordinates.
    """
    B, _, H, W = depth.shape

    # Flatten and cast to same dtype as K_inv / depth
    grid = pixel_grid.view(B, 3, -1).to(depth.dtype)  # (B,3,HW)

    # Batched matrix multiply: (B,3,3) x (B,3,HW) -> (B,3,HW)
    cam = torch.bmm(K_inv, grid)

    cam = cam.view(B, 3, H, W)
    X = cam * depth  # broadcast depth (B,1,H,W) over 3 channels
    return X


def project(points, K, T):
    """
    Project 3D points (in target frame) into source camera frame.

    points: (B,3,H,W)
    K:      (B,3,3)
    T:      (B,4,4)   T_{t->s}
    Returns:
        x_norm: (B,H,W,2) normalized to [-1,1] for grid_sample.
    """
    B, _, H, W = points.shape
    # Flatten
    X = points.view(B, 3, -1)  # (B,3,HW)

    # Transform to source frame
    R = T[:, :3, :3]  # (B,3,3)
    t = T[:, :3, 3].view(B, 3, 1)  # (B,3,1)
    Xs = R @ X + t  # (B,3,HW)

    # Project
    Xs_cam = K @ Xs  # (B,3,HW)
    x = Xs_cam[:, 0, :] / (Xs_cam[:, 2, :] + 1e-8)
    y = Xs_cam[:, 1, :] / (Xs_cam[:, 2, :] + 1e-8)

    # Normalize to [-1,1] range for grid_sample
    x_norm = 2.0 * (x / (W - 1)) - 1.0
    y_norm = 2.0 * (y / (H - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,HW,2)
    grid = grid.view(B, H, W, 2)
    return grid


def warp_image(source, depth_t, pose_vec, K, K_inv):
    """
    Warp source image into target frame using predicted depth and pose.

    source:  (B,3,H,W)
    depth_t: (B,1,H,W)   depth of target frame
    pose_vec:(B,6)       pose of source relative to target
    K, K_inv:(B,3,3)

    Returns:
        warped: (B,3,H,W)
    """
    B, _, H, W = source.shape
    T = pose_vec_to_mat(pose_vec.unsqueeze(1))[:, 0]  # (B,4,4)

    # pixel_grid = create_pixel_grid(B, H, W, depth_t.device)
    pixel_grid = create_pixel_grid(B, H, W, depth_t.device, dtype=depth_t.dtype)

    cam_points = backproject(depth_t, K_inv, pixel_grid)  # (B,3,H,W)

    proj_grid = project(cam_points, K, T)  # (B,H,W,2)

    warped = F.grid_sample(source, proj_grid, padding_mode="border", align_corners=True)
    return warped


# ================================================================
# Losses: photometric + smoothness
# ================================================================


def ssim(x, y):
    """
    SSIM over 3-channel images, windowed implicitly by avg pooling.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = SSIM_n / (SSIM_d + 1e-8)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def photometric_loss(target, recon):
    """
    Combination of L1 and SSIM, as in many depth papers.
    target, recon: (B,3,H,W)
    """
    l1 = (target - recon).abs().mean(1, keepdim=True)  # (B,1,H,W)
    s = ssim(target, recon).mean(1, keepdim=True)  # (B,1,H,W)
    loss = 0.15 * s + 0.85 * l1
    return loss


def smoothness_loss(depth, image):
    """
    Edge-aware depth smoothness:
    encourage depth gradients to be small where image is smooth.
    depth: (B,1,H,W)
    image: (B,3,H,W)
    """
    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    img_dx = torch.mean(
        torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, keepdim=True
    )
    img_dy = torch.mean(
        torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, keepdim=True
    )

    depth_dx *= torch.exp(-img_dx)
    depth_dy *= torch.exp(-img_dy)

    return depth_dx.mean() + depth_dy.mean()


# ================================================================
# Training
# ================================================================


def train_sfmlearner(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    resize: tuple[int, int] = (192, 128),
    fx: float | None = None,
    fy: float | None = None,
    output_dir: str = "sfmlearner_runs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MonoSequenceDataset(data_root, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SfMLearner(min_depth=0.1, max_depth=20.0, n_frames=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = resize[1], resize[0]
    # Intrinsics (assume pinhole, roughly centered principal point)
    if fx is None:
        fx = 0.8 * W
    if fy is None:
        fy = 0.8 * H
    cx = W / 2.0
    cy = H / 2.0

    # Build K, K_inv (same for whole batch; broadcast later)
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device
    ).unsqueeze(
        0
    )  # (1,3,3)
    K_inv = torch.inverse(K)

    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for tgt, srcs in dataloader:
            tgt = tgt.to(device)  # (B,3,H,W)
            srcs = srcs.to(device)  # (B,2,3,H,W)
            B = tgt.shape[0]
            N_src = srcs.shape[1]

            optimizer.zero_grad()

            depth_t, pose_vec = model(
                tgt, srcs
            )  # depth_t: (B,1,H,W); pose_vec: (B,2,6)

            # Photometric reconstruction for each source frame
            photometric_terms = []

            # Expand K, K_inv per batch
            K_b = K.expand(B, -1, -1)
            K_inv_b = K_inv.expand(B, -1, -1)

            for i in range(N_src):
                src_i = srcs[:, i]  # (B,3,H,W)
                pose_i = pose_vec[:, i]  # (B,6)

                recon_i = warp_image(src_i, depth_t, pose_i, K_b, K_inv_b)  # (B,3,H,W)
                photometric_terms.append(photometric_loss(tgt, recon_i))  # (B,1,H,W)

            # Combine photometric losses from all sources (average)
            photometric_combined = torch.stack(photometric_terms, dim=0).mean(
                0
            )  # (B,1,H,W)
            photometric_loss_val = photometric_combined.mean()

            # Smoothness loss on depth
            smooth_loss_val = smoothness_loss(depth_t, tgt)

            loss = photometric_loss_val + 0.1 * smooth_loss_val

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}/{epochs}] loss: {avg_loss:.4f}")

    # Save model + loss curve
    model_path = out_dir / "sfmlearner.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    epochs_axis = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SfMLearner-style training loss")
    plt.grid(True)
    loss_plot_path = out_dir / "train_loss.png"
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    return model_path


# ================================================================
# Inference: depth prediction for a single image
# ================================================================


def colorize_depth(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    valid = depth > 0
    if vmin is None:
        vmin = float(depth[valid].min()) if valid.any() else depth.min()
    if vmax is None:
        vmax = float(depth[valid].max()) if valid.any() else depth.max()
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return depth_color


def run_depth_inference(
    img_path: str,
    model_path: str,
    output: str = "sfm_depth_pred.png",
    resize: tuple[int, int] = (192, 128),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    print(f"Loading image {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig, _ = img_rgb.shape

    if resize is not None:
        w, h = resize
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb

    img = img_resized.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    model = SfMLearner(min_depth=0.1, max_depth=20.0, n_frames=3).to(device)
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        img_t = img_t.to(device)
        depth_pred, _ = (
            model.depth_net(img_t),
            None,
        )  # or depth_pred, _ = model(img_t, dummy_sources)
        depth_np = depth_pred.squeeze().cpu().numpy()

    depth_color = colorize_depth(depth_np)
    depth_color_full = cv2.resize(
        depth_color, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )

    vis = np.concatenate([img_rgb, depth_color_full], axis=1)
    cv2.imwrite(output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output}")


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SfMLearner-style demo (unsupervised depth + pose)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Train ---
    p_train = subparsers.add_parser("train", help="Train SfMLearner")
    p_train.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of monocular sequences (e.g., data_synth/mono_seq)",
    )
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--resize_w", type=int, default=192)
    p_train.add_argument("--resize_h", type=int, default=128)
    p_train.add_argument(
        "--fx", type=float, default=None, help="Focal length fx (pixels); default 0.8*W"
    )
    p_train.add_argument(
        "--fy", type=float, default=None, help="Focal length fy (pixels); default 0.8*H"
    )
    p_train.add_argument("--output_dir", type=str, default="sfmlearner_runs")

    # --- Infer depth ---
    p_infer = subparsers.add_parser(
        "infer_depth", help="Depth inference using trained model"
    )
    p_infer.add_argument("--image", type=str, required=True, help="Path to RGB image")
    p_infer.add_argument(
        "--model", type=str, required=True, help="Path to sfmlearner.pth"
    )
    p_infer.add_argument("--output", type=str, default="sfm_depth_pred.png")
    p_infer.add_argument("--resize_w", type=int, default=192)
    p_infer.add_argument("--resize_h", type=int, default=128)

    args = parser.parse_args()

    if args.mode == "train":
        resize = (args.resize_w, args.resize_h)
        train_sfmlearner(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resize=resize,
            fx=args.fx,
            fy=args.fy,
            output_dir=args.output_dir,
        )
    elif args.mode == "infer_depth":
        resize = (args.resize_w, args.resize_h)
        run_depth_inference(
            img_path=args.image,
            model_path=args.model,
            output=args.output,
            resize=resize,
        )


if __name__ == "__main__":
    main()


"""
python sfmlearner_demo.py train \
    --data_root data_synth/mono_seq \
    --epochs 20 \
    --batch_size 4 \
    --resize_w 192 --resize_h 128 \
    --output_dir sfmlearner_runs

    

python sfmlearner_demo.py infer_depth \
    --image data_synth/mono_seq/seq_000/0002.png \
    --model sfmlearner_runs/sfmlearner.pth \
    --resize_w 192 --resize_h 128 \
    --output sfm_depth_seq000_0002.png


"""
