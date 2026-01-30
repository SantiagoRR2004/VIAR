"""
PointNet Demo for 3D Shape Classification
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements PointNet for classifying 3D shapes from the ModelNet40 dataset.
Key concepts:
- Permutation-invariant point cloud processing
- Transformation networks (T-Net) for canonical alignment
- Max-pooling for global feature aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm

# ============================================================================
# T-Net: Spatial Transformation Network
# ============================================================================


class TNet(nn.Module):
    """
    Transformation network that learns a canonical alignment for points.
    Predicts a rotation matrix to transform input points.
    """

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x: [B, k, N] - B batches, k features, N points
        Returns:
            [B, k, k] - Transformation matrix
        """
        batch_size = x.size(0)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling across points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        identity = identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


# ============================================================================
# PointNet Classification Model
# ============================================================================


class PointNet(nn.Module):
    """
    PointNet architecture for 3D shape classification.

    Key components:
    1. Input transform (T-Net) for spatial alignment
    2. Shared MLP for per-point features
    3. Feature transform (T-Net) for feature space alignment
    4. Max-pooling for global features
    5. Classification head
    """

    def __init__(self, num_classes=40, dropout=0.3):
        super(PointNet, self).__init__()

        # Input transformation
        self.input_transform = TNet(k=3)

        # Shared MLP for point features
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Feature transformation
        self.feature_transform = TNet(k=64)

        # Further feature extraction
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        # Batch normalization
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
            x: [B, N, 3] - B batches, N points, 3 coordinates (x,y,z)
        Returns:
            logits: [B, num_classes]
            trans_feat: [B, 64, 64] - Feature transformation matrix
        """
        B, N, _ = x.size()

        # Transpose for conv1d: [B, 3, N]
        x = x.transpose(2, 1)

        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = torch.bmm(trans_feat, x)

        # Further feature extraction
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Classification head
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, trans_feat


# ============================================================================
# Regularization Loss for Feature Transform
# ============================================================================


def feature_transform_regularizer(trans):
    """
    Regularization loss to keep feature transform close to orthogonal.

    L_reg = ||I - AA^T||_F^2

    Args:
        trans: [B, K, K] - Transformation matrices
    Returns:
        loss: scalar
    """
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    I = I.repeat(trans.size(0), 1, 1)
    loss = torch.mean(
        torch.norm(I - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2))
    )
    return loss


# ============================================================================
# Synthetic ModelNet40 Dataset
# ============================================================================


class ModelNet40Dataset(Dataset):
    """
    Simplified synthetic ModelNet40 dataset for demonstration.
    In practice, you would load actual .off or .h5 files.
    """

    def __init__(
        self, num_points=1024, num_samples=1000, num_classes=40, split="train"
    ):
        self.num_points = num_points
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random point cloud (in practice, load from file)
        # Here we create simple geometric shapes
        class_id = np.random.randint(0, self.num_classes)

        # Simple sphere-like point cloud
        theta = np.random.uniform(0, 2 * np.pi, self.num_points)
        phi = np.random.uniform(0, np.pi, self.num_points)

        # Add class-specific deformations
        r = 1.0 + 0.3 * np.sin(class_id * theta / 10)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points = np.stack([x, y, z], axis=1).astype(np.float32)

        # Add noise
        if self.split == "train":
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        # Random rotation augmentation (training only)
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
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, trans_feat = model(points)

        # Classification loss
        cls_loss = criterion(outputs, labels)

        # Regularization loss on feature transform
        reg_loss = feature_transform_regularizer(trans_feat)

        # Total loss
        loss = cls_loss + reg_weight * reg_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for points, labels in pbar:
            points, labels = points.to(device), labels.to(device)

            outputs, _ = model(points)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
            )

    return running_loss / len(dataloader), 100.0 * correct / total


# ============================================================================
# Main Training Script
# ============================================================================


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print("\nLoading ModelNet40 dataset...")
    train_dataset = ModelNet40Dataset(
        num_points=args.num_points, num_samples=9840, num_classes=40, split="train"
    )
    val_dataset = ModelNet40Dataset(
        num_points=args.num_points, num_samples=2468, num_classes=40, split="val"
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("\nInitializing PointNet...")
    model = PointNet(num_classes=40, dropout=args.dropout).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    print("\nTraining PointNet...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "pointnet_best.pth",
            )
            print(f"  Saved best model with val acc: {val_acc:.2f}%")

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Test evaluation
    print("\nTesting on test set...")
    test_dataset = ModelNet40Dataset(
        num_points=args.num_points, num_samples=2468, num_classes=40, split="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Load best model
    checkpoint = torch.load("pointnet_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet for 3D Shape Classification")

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="modelnet40", help="Dataset name"
    )
    parser.add_argument(
        "--num_points", type=int, default=1024, help="Number of points to sample"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PointNet Demo for 3D Shape Classification")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    main(args)
