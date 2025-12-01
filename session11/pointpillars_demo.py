"""
PointPillars Demo for 3D Object Detection
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements PointPillars for detecting 3D objects from LiDAR point clouds.
Key concepts:
- Pillar-based point cloud encoding
- PointNet feature extraction per pillar
- 2D convolution on bird's eye view representation
- 3D bounding box prediction with orientation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm


# ============================================================================
# Pillar Feature Network
# ============================================================================


class PillarFeatureNet(nn.Module):
    """
    Extract features from points in each pillar using PointNet.

    For each pillar:
    1. Apply PointNet to points within the pillar
    2. Max-pool to get pillar feature
    3. Create pseudo-image by scattering pillar features to BEV grid
    """

    def __init__(self, in_channels=9, feat_channels=64):
        super(PillarFeatureNet, self).__init__()

        # Shared MLP (like PointNet)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, feat_channels, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(feat_channels)

    def forward(self, pillar_points, pillar_coords):
        """
        Args:
            pillar_points: [P, N, C] - P pillars, N points per pillar, C features
            pillar_coords: [P, 2] - (x_idx, y_idx) grid coordinates of each pillar
        Returns:
            pseudo_image: [B, C, H, W] - Bird's eye view feature map
        """
        # Transpose for conv1d: [P, C, N]
        x = pillar_points.transpose(2, 1)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Max pooling across points in each pillar: [P, C]
        pillar_features = torch.max(x, dim=2)[0]

        return pillar_features

    def create_bev_map(self, pillar_features, pillar_coords, H, W):
        """
        Scatter pillar features to create bird's eye view pseudo-image.

        Args:
            pillar_features: [P, C] - Features for each pillar
            pillar_coords: [P, 2] - Grid coordinates
            H, W: Grid height and width
        Returns:
            bev_map: [1, C, H, W]
        """
        C = pillar_features.size(1)
        bev_map = torch.zeros((1, C, H, W), device=pillar_features.device)

        # Scatter features to grid
        x_idx = pillar_coords[:, 0].long()
        y_idx = pillar_coords[:, 1].long()

        # Ensure indices are within bounds
        valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        features = pillar_features[valid_mask]

        # Scatter (handle multiple pillars at same location by taking max)
        for i in range(features.size(0)):
            bev_map[0, :, y_idx[i], x_idx[i]] = torch.max(
                bev_map[0, :, y_idx[i], x_idx[i]], features[i]
            )

        return bev_map


# ============================================================================
# 2D Backbone Network
# ============================================================================


class Backbone2D(nn.Module):
    """
    2D CNN backbone for processing bird's eye view feature map.
    Similar to standard image detection backbones.
    """

    def __init__(self, in_channels=64, out_channels=256):
        super(Backbone2D, self).__init__()

        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Upsampling to recover resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - BEV feature map
        Returns:
            [B, C_out, H, W] - Multi-scale features
        """
        # Encoder
        x1 = self.block1(x)  # [B, 64, H, W]
        x2 = self.block2(x1)  # [B, 128, H/2, W/2]
        x3 = self.block3(x2)  # [B, 256, H/4, W/4]

        # Decoder with skip connections
        x = self.up1(x3)  # [B, 128, H/2, W/2]
        x = x + x2  # Skip connection

        x = self.up2(x)  # [B, 64, H, W]
        x = x + x1  # Skip connection

        x = self.final(x)  # [B, 256, H, W]

        return x


# ============================================================================
# Detection Head
# ============================================================================


class DetectionHead(nn.Module):
    """
    Detection head for predicting 3D bounding boxes.

    For each anchor location, predict:
    - Classification score (1 per class)
    - 3D box parameters: (x, y, z, w, l, h, yaw)
    - Direction classification (for angle disambiguation)
    """

    def __init__(self, in_channels=256, num_classes=3, num_anchors=2):
        super(DetectionHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1),
        )

        # Regression head (7 parameters per box)
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * 7, 1),
        )

        # Direction head (2 bins: 0-180, 180-360)
        self.dir_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Feature map
        Returns:
            cls_preds: [B, H, W, num_anchors, num_classes]
            box_preds: [B, H, W, num_anchors, 7]
            dir_preds: [B, H, W, num_anchors, 2]
        """
        B, _, H, W = x.size()

        # Predictions
        cls = self.cls_head(x)  # [B, num_anchors*num_classes, H, W]
        reg = self.reg_head(x)  # [B, num_anchors*7, H, W]
        dir = self.dir_head(x)  # [B, num_anchors*2, H, W]

        # Reshape
        cls = cls.permute(0, 2, 3, 1).reshape(
            B, H, W, self.num_anchors, self.num_classes
        )
        reg = reg.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, 7)
        dir = dir.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, 2)

        return cls, reg, dir


# ============================================================================
# Complete PointPillars Model
# ============================================================================


class PointPillars(nn.Module):
    """
    Complete PointPillars model for 3D object detection.
    """

    def __init__(self, num_classes=3, feat_channels=64):
        super(PointPillars, self).__init__()

        self.pillar_net = PillarFeatureNet(
            in_channels=9,  # x, y, z, r, x_c, y_c, z_c, x_p, y_p
            feat_channels=feat_channels,
        )

        self.backbone = Backbone2D(in_channels=feat_channels, out_channels=256)

        self.head = DetectionHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=2,  # Two rotations per location
        )

    def forward(self, pillar_points, pillar_coords, grid_size):
        """
        Args:
            pillar_points: [P, N, C] - Points in pillars
            pillar_coords: [P, 2] - Grid coordinates
            grid_size: (H, W) - Grid dimensions
        Returns:
            cls_preds, box_preds, dir_preds
        """
        H, W = grid_size

        # Extract pillar features
        pillar_features = self.pillar_net(pillar_points, pillar_coords)

        # Create BEV pseudo-image
        bev_map = self.pillar_net.create_bev_map(pillar_features, pillar_coords, H, W)

        # 2D backbone
        features = self.backbone(bev_map)

        # Detection head
        cls_preds, box_preds, dir_preds = self.head(features)

        return cls_preds, box_preds, dir_preds


# ============================================================================
# Point Cloud to Pillars
# ============================================================================


def points_to_pillars(points, voxel_size, point_cloud_range, max_points_per_pillar=100):
    """
    Convert point cloud to pillar representation.

    Args:
        points: [N, 4] - Point cloud (x, y, z, intensity)
        voxel_size: [vx, vy] - Pillar size in meters
        point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        max_points_per_pillar: Maximum points to keep per pillar
    Returns:
        pillar_points: [P, max_points, C] - Points in each pillar with features
        pillar_coords: [P, 2] - Grid coordinates of pillars
    """
    xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
    vx, vy = voxel_size

    # Compute grid dimensions
    nx = int((xmax - xmin) / vx)
    ny = int((ymax - ymin) / vy)

    # Filter points within range
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] < xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] < ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] < zmax)
    )
    points = points[mask]

    # Compute pillar indices
    x_idx = ((points[:, 0] - xmin) / vx).long()
    y_idx = ((points[:, 1] - ymin) / vy).long()

    # Group points by pillar
    pillar_dict = {}
    for i in range(points.size(0)):
        key = (x_idx[i].item(), y_idx[i].item())
        if key not in pillar_dict:
            pillar_dict[key] = []
        if len(pillar_dict[key]) < max_points_per_pillar:
            pillar_dict[key].append(points[i])

    # Convert to tensors
    pillar_list = []
    coord_list = []

    for (x, y), pts in pillar_dict.items():
        if len(pts) == 0:
            continue

        pts = torch.stack(pts)  # [num_points, 4]

        # Compute pillar center
        x_c = (x + 0.5) * vx + xmin
        y_c = (y + 0.5) * vy + ymin
        z_c = pts[:, 2].mean()

        # Augment points with offsets from pillar center
        x_offset = pts[:, 0] - x_c
        y_offset = pts[:, 1] - y_c
        z_offset = pts[:, 2] - z_c

        # Features: [x, y, z, intensity, x_c, y_c, z_c, x_offset, y_offset]
        features = torch.cat(
            [
                pts,  # x, y, z, intensity
                torch.full((pts.size(0), 3), torch.tensor([x_c, y_c, z_c])),
                torch.stack([x_offset, y_offset], dim=1),
            ],
            dim=1,
        )

        # Pad or truncate to max_points_per_pillar
        if features.size(0) < max_points_per_pillar:
            padding = torch.zeros(
                (max_points_per_pillar - features.size(0), features.size(1))
            )
            features = torch.cat([features, padding], dim=0)
        else:
            features = features[:max_points_per_pillar]

        pillar_list.append(features)
        coord_list.append(torch.tensor([x, y], dtype=torch.float32))

    if len(pillar_list) == 0:
        return None, None

    pillar_points = torch.stack(pillar_list)  # [P, max_points, 9]
    pillar_coords = torch.stack(coord_list)  # [P, 2]

    return pillar_points, pillar_coords


# ============================================================================
# Synthetic KITTI-like Dataset
# ============================================================================


class KITTIDataset(Dataset):
    """
    Synthetic KITTI-like dataset for demonstration.
    In practice, load actual KITTI data.
    """

    def __init__(self, num_samples=1000, split="train"):
        self.num_samples = num_samples
        self.split = split

        # Point cloud range (meters)
        self.pc_range = [0, -40, -3, 70.4, 40, 1]
        self.voxel_size = [0.16, 0.16]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic point cloud
        num_points = np.random.randint(10000, 20000)

        # Random points in range
        x = np.random.uniform(self.pc_range[0], self.pc_range[3], num_points)
        y = np.random.uniform(self.pc_range[1], self.pc_range[4], num_points)
        z = np.random.uniform(self.pc_range[2], self.pc_range[5], num_points)
        intensity = np.random.uniform(0, 1, num_points)

        points = torch.tensor(
            np.stack([x, y, z, intensity], axis=1), dtype=torch.float32
        )

        # Generate synthetic bounding boxes (3 cars)
        boxes = []
        for _ in range(3):
            cx = np.random.uniform(10, 60)
            cy = np.random.uniform(-30, 30)
            cz = 0.0
            w, l, h = 2.0, 4.5, 1.6  # Car dimensions
            yaw = np.random.uniform(0, np.pi)
            boxes.append([cx, cy, cz, w, l, h, yaw])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.zeros(3, dtype=torch.long)  # Class 0: car

        return points, boxes, labels


# ============================================================================
# Training and Evaluation
# ============================================================================


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for points, boxes, labels in pbar:
        # Convert points to pillars
        pillar_points, pillar_coords = points_to_pillars(
            points[0],  # Batch size 1 for simplicity
            dataloader.dataset.voxel_size,
            dataloader.dataset.pc_range,
        )

        if pillar_points is None:
            continue

        pillar_points = pillar_points.to(device)
        pillar_coords = pillar_coords.to(device)

        # Grid size
        xmin, ymin, _, xmax, ymax, _ = dataloader.dataset.pc_range
        vx, vy = dataloader.dataset.voxel_size
        grid_h = int((ymax - ymin) / vy)
        grid_w = int((xmax - xmin) / vx)

        # Forward
        cls_preds, box_preds, dir_preds = model(
            pillar_points, pillar_coords, (grid_h, grid_w)
        )

        # Simple dummy loss (in practice, use proper detection loss)
        loss = cls_preds.mean() + box_preds.mean() + dir_preds.mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    return total_loss / len(dataloader)


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("PointPillars Demo for 3D Object Detection")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Dataset
    print("\nLoading KITTI dataset...")
    train_dataset = KITTIDataset(num_samples=args.num_samples, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one point cloud at a time
        shuffle=True,
        num_workers=0,
    )

    print(f"Training samples: {len(train_dataset)}")

    # Model
    print("\nInitializing PointPillars...")
    model = PointPillars(num_classes=3, feat_channels=64).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:.1f}M")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print("\nTraining...")
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "pointpillars_model.pth",
    )
    print("\nModel saved to pointpillars_model.pth")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointPillars 3D Object Detection")

    parser.add_argument(
        "--data_path", type=str, default="./kitti", help="Path to KITTI dataset"
    )
    parser.add_argument(
        "--class", type=str, default="car", help="Object class to detect"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of training samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    main(args)
