"""
PointNet Demo for 3D Shape Classification (Synthetic Data)
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements a PointNet-style architecture for classifying
synthetic 3D point clouds.

Key concepts:
- Permutation-invariant point cloud processing (shared MLP + max-pooling)
- Transformation networks (T-Net) for canonical alignment
- Max-pooling for global feature aggregation

NOTE:
This uses a TOY synthetic dataset (5 classes of deformed spheres),
not the real ModelNet40. The goal is to show the architecture and
training dynamics quickly in class, without heavy data dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import random

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# ============================================================================
# T-Net: Spatial Transformation Network
# ============================================================================


class TNet(nn.Module):
    """
    Transformation network that learns a canonical alignment.

    For k=3: input coordinates (3xN) -> 3x3 matrix.
    For k=64: feature vectors (64xN) -> 64x64 matrix.
    """

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        # Shared MLP (Conv1d over points)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x: [B, k, N]
        Returns:
            trans: [B, k, k] transformation matrices
        """
        B = x.size(0)

        # Shared MLP over points
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]

        # Max pool over N points -> [B, 1024]
        x = torch.max(x, 2, keepdim=False)[0]

        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [B, k*k]

        # Initialize close to identity
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + identity.repeat(B, 1)
        x = x.view(B, self.k, self.k)
        return x


# ============================================================================
# PointNet Classification Model
# ============================================================================


class PointNet(nn.Module):
    """
    PointNet architecture for 3D shape classification (synthetic data).

    Components:
    1. Input T-Net (3x3) for spatial alignment.
    2. Shared MLP on points (Conv1d).
    3. Feature T-Net (64x64) for feature-space alignment.
    4. Further shared MLP.
    5. Global max-pooling for permutation invariance.
    6. FC head for classification.
    """

    def __init__(self, num_classes=5, dropout=0.3):
        super(PointNet, self).__init__()

        # Input transform
        self.input_transform = TNet(k=3)

        # Shared MLP (first stage)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Feature transform
        self.feature_transform = TNet(k=64)

        # Shared MLP (second stage)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        # BatchNorms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, 3]
        Returns:
            logits: [B, num_classes]
            trans_feat: [B, 64, 64] feature transform (for regularization)
        """
        B, N, _ = x.size()
        # Transpose to [B, 3, N] for Conv1d
        x = x.transpose(2, 1)

        # Input transform
        trans = self.input_transform(x)  # [B, 3, 3]
        x = torch.bmm(trans, x)  # apply transform to [B, 3, N]

        # Shared MLP (3 -> 64 -> 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        trans_feat = self.feature_transform(x)  # [B, 64, 64]
        x = torch.bmm(trans_feat, x)

        # Further feature extraction
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 1024, N]

        # Global max-pooling over points
        x = torch.max(x, 2, keepdim=False)[0]  # [B, 1024]

        # Classification head
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # [B, num_classes]

        return x, trans_feat


# ============================================================================
# Regularization Loss for Feature Transform
# ============================================================================


def feature_transform_regularizer(trans):
    """
    Regularization to keep feature transform close to orthogonal.

    L_reg = || I - A A^T ||_F
    """
    B, K, _ = trans.size()
    I = torch.eye(K, device=trans.device).unsqueeze(0).repeat(B, 1, 1)
    # Frobenius norm over each batch element, then mean
    diff = I - torch.bmm(trans, trans.transpose(2, 1))
    loss = torch.mean(torch.norm(diff, dim=(1, 2)))
    return loss


# ============================================================================
# Synthetic Point Cloud Dataset (5 classes)
# ============================================================================


class SyntheticPointCloudDataset(Dataset):
    """
    Synthetic dataset for 3D point cloud classification.

    We generate 5 classes of deformed spheres. Each class has
    a different radial deformation pattern.

    This is purely for demonstration of PointNet training dynamics.
    """

    def __init__(self, num_points=1024, num_samples=2000, num_classes=5, split="train"):
        self.num_points = num_points
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Choose a class label
        class_id = np.random.randint(0, self.num_classes)

        # Sample random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi, self.num_points)
        phi = np.random.uniform(0, np.pi, self.num_points)

        # Define class-specific radial deformations
        # Base radius
        r0 = 1.0
        if class_id == 0:
            # Almost perfect sphere
            r = r0 * np.ones_like(theta)
        elif class_id == 1:
            # Slight "pumpkin" lobes around theta
            r = r0 + 0.2 * np.sin(3 * theta)
        elif class_id == 2:
            # Variation with phi (poles vs equator)
            r = r0 + 0.3 * np.cos(2 * phi)
        elif class_id == 3:
            # Mix of theta and phi
            r = r0 + 0.25 * np.sin(theta) * np.sin(phi)
        else:  # class_id == 4
            # More complex multi-frequency bump
            r = r0 + 0.2 * np.sin(4 * theta) * np.cos(2 * phi)

        # Convert spherical to Cartesian
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points = np.stack([x, y, z], axis=1).astype(np.float32)

        # Add small noise (train only)
        if self.split == "train":
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        # Random rotation augmentation (train only)
        if self.split == "train":
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
            )
            points = points @ R.T

        return points, class_id


# ============================================================================
# Training and Evaluation Functions
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device, reg_weight=0.001):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for points, labels in pbar:
        points = points.to(device)  # [B, N, 3]
        labels = labels.to(device)  # [B]

        optimizer.zero_grad()

        outputs, trans_feat = model(points)
        cls_loss = criterion(outputs, labels)
        reg_loss = feature_transform_regularizer(trans_feat)

        loss = cls_loss + reg_weight * reg_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        pbar.set_postfix(
            {
                "loss": f"{running_loss / (pbar.n + 1):.3f}",
                "acc": f"{100.0 * correct / total:.1f}%",
            }
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for points, labels in pbar:
            points = points.to(device)
            labels = labels.to(device)

            outputs, _ = model(points)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (pbar.n + 1):.3f}",
                    "acc": f"{100.0 * correct / total:.1f}%",
                }
            )

    return running_loss / len(dataloader), 100.0 * correct / total


# ============================================================================
# Main Training Script
# ============================================================================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("PointNet Demo for 3D Shape Classification (Synthetic Dataset)")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)
    print(f"Using device: {device}\n")

    # Datasets
    print("Creating synthetic datasets...")
    train_dataset = SyntheticPointCloudDataset(
        num_points=args.num_points,
        num_samples=args.num_train,
        num_classes=args.num_classes,
        split="train",
    )
    val_dataset = SyntheticPointCloudDataset(
        num_points=args.num_points,
        num_samples=args.num_val,
        num_classes=args.num_classes,
        split="val",
    )
    test_dataset = SyntheticPointCloudDataset(
        num_points=args.num_points,
        num_samples=args.num_test,
        num_classes=args.num_classes,
        split="test",
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val   samples: {len(val_dataset)}")
    print(f"  Test  samples: {len(test_dataset)}\n")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # safer for student laptops / Windows
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model
    model = PointNet(num_classes=args.num_classes, dropout=args.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.\n")

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    print("Training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, desc="Validation"
        )

        print(f"  Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.3f}, Val   Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "pointnet_synthetic_best.pth")
            print(f"  Saved best model (val acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

    # Test evaluation
    print("\nEvaluating on test set (best model)...")
    model.load_state_dict(
        torch.load("pointnet_synthetic_best.pth", map_location=device)
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet Synthetic Demo")

    # Dataset parameters
    parser.add_argument(
        "--num_points", type=int, default=1024, help="Number of points per point cloud"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of synthetic classes (fixed at 5 in this demo)",
    )
    parser.add_argument(
        "--num_train", type=int, default=2000, help="Number of training samples"
    )
    parser.add_argument(
        "--num_val", type=int, default=500, help="Number of validation samples"
    )
    parser.add_argument(
        "--num_test", type=int, default=500, help="Number of test samples"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )

    args = parser.parse_args()
    main(args)
