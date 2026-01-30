"""
3D ShapeNets Demo (Wu et al., 2015)
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements 3D ShapeNets for 3D shape classification and completion.
Key concepts:
- Voxel-based 3D representation
- 3D Convolutional Neural Networks
- Probabilistic 3D shape completion
- Deep Belief Network (DBN) inspired architecture

Reference: Wu et al., "3D ShapeNets: A Deep Representation for Volumetric Shapes", CVPR 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# 3D ShapeNets Architecture
# ============================================================================


class ShapeNets3D(nn.Module):
    """
    3D ShapeNets architecture using 3D convolutions.

    Architecture follows the paper:
    - Input: 32x32x32 voxel grid
    - 3D Conv layers with decreasing spatial dimensions
    - Fully connected layers for classification
    - Can be extended for generative completion

    The original paper used Deep Belief Networks (DBN), but we use
    a more standard CNN architecture that's easier to train.
    """

    def __init__(self, num_classes=10, input_size=32, use_dropout=True):
        super(ShapeNets3D, self).__init__()

        self.input_size = input_size

        # 3D Convolutional layers
        # Conv1: 32x32x32 -> 16x16x16
        self.conv1 = nn.Conv3d(1, 48, kernel_size=6, stride=2, padding=2)
        self.bn1 = nn.BatchNorm3d(48)

        # Conv2: 16x16x16 -> 8x8x8
        self.conv2 = nn.Conv3d(48, 160, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm3d(160)

        # Conv3: 8x8x8 -> 4x4x4
        self.conv3 = nn.Conv3d(160, 512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(512)

        # Calculate flattened size
        self.flat_size = 512 * 4 * 4 * 4  # 32768

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] - Batch of voxel grids
        Returns:
            [B, num_classes] - Classification logits
        """
        # 3D Convolutions with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def extract_features(self, x):
        """Extract deep features for visualization or other tasks."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# ============================================================================
# 3D Shape Completion Network
# ============================================================================


class ShapeCompletion3D(nn.Module):
    """
    3D shape completion network (encoder-decoder architecture).

    Takes partial voxel grid and predicts complete shape.
    This is one of the key capabilities of 3D ShapeNets.
    """

    def __init__(self, input_size=32):
        super(ShapeCompletion3D, self).__init__()

        # Encoder (same as classification network)
        self.enc_conv1 = nn.Conv3d(1, 48, kernel_size=6, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm3d(48)

        self.enc_conv2 = nn.Conv3d(48, 160, kernel_size=5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm3d(160)

        self.enc_conv3 = nn.Conv3d(160, 512, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm3d(512)

        # Decoder (transposed convolutions)
        self.dec_conv1 = nn.ConvTranspose3d(
            512, 160, kernel_size=4, stride=2, padding=1
        )
        self.dec_bn1 = nn.BatchNorm3d(160)

        self.dec_conv2 = nn.ConvTranspose3d(
            160, 48, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.dec_bn2 = nn.BatchNorm3d(48)

        self.dec_conv3 = nn.ConvTranspose3d(
            48, 1, kernel_size=6, stride=2, padding=2, output_padding=0
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] - Partial voxel grid
        Returns:
            [B, 1, D, H, W] - Completed voxel grid (probabilities)
        """
        # Encoder
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))

        # Decoder
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = torch.sigmoid(self.dec_conv3(x))

        return x


# ============================================================================
# Voxel Utilities
# ============================================================================


def mesh_to_voxels(vertices, faces, resolution=32):
    """
    Convert mesh to voxel grid (simplified voxelization).

    In practice, you would use libraries like trimesh or pyvoxel.
    Here we use a simple point-in-bounding-box approach.
    """
    # Normalize vertices to [0, resolution-1]
    vertices = vertices - vertices.min(axis=0)
    vertices = vertices / vertices.max() * (resolution - 1)

    # Create voxel grid
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Simple voxelization: mark voxels containing vertices
    for v in vertices:
        i, j, k = v.astype(int)
        if 0 <= i < resolution and 0 <= j < resolution and 0 <= k < resolution:
            voxels[i, j, k] = 1.0

    # Fill interior (simple flood fill from faces)
    # For demo purposes, we'll dilate to fill some interior
    from scipy.ndimage import binary_dilation

    voxels = binary_dilation(voxels, iterations=2).astype(np.float32)

    return voxels


def create_synthetic_voxel(shape_type="cube", resolution=32, class_id=0):
    """
    Create synthetic voxel shapes for demonstration.

    Args:
        shape_type: 'cube', 'sphere', 'cylinder', 'pyramid', etc.
        resolution: Voxel grid resolution
        class_id: Class identifier
    Returns:
        voxels: [resolution, resolution, resolution] binary voxel grid
    """
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    center = resolution // 2

    if shape_type == "cube" or class_id == 0:
        # Cube
        size = resolution // 3
        voxels[
            center - size : center + size,
            center - size : center + size,
            center - size : center + size,
        ] = 1.0

    elif shape_type == "sphere" or class_id == 1:
        # Sphere
        radius = resolution // 3
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    dist = np.sqrt(
                        (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
                    )
                    if dist < radius:
                        voxels[i, j, k] = 1.0

    elif shape_type == "cylinder" or class_id == 2:
        # Cylinder (vertical)
        radius = resolution // 4
        height = resolution // 2
        for i in range(resolution):
            for j in range(resolution):
                for k in range(center - height, center + height):
                    dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                    if dist < radius and 0 <= k < resolution:
                        voxels[i, j, k] = 1.0

    elif shape_type == "pyramid" or class_id == 3:
        # Pyramid
        base_size = resolution // 2
        height = resolution // 2
        for k in range(height):
            size = base_size * (height - k) // height
            voxels[
                center - size : center + size, center - size : center + size, center + k
            ] = 1.0

    elif shape_type == "torus" or class_id == 4:
        # Torus
        R = resolution // 4  # Major radius
        r = resolution // 8  # Minor radius
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    x, y, z = i - center, j - center, k - center
                    dist_to_center_ring = np.sqrt(x**2 + y**2)
                    dist = np.sqrt((dist_to_center_ring - R) ** 2 + z**2)
                    if dist < r:
                        voxels[i, j, k] = 1.0

    else:
        # Random blob for other classes
        voxels[
            center - 5 : center + 5, center - 5 : center + 5, center - 5 : center + 5
        ] = 1.0
        from scipy.ndimage import gaussian_filter

        voxels = gaussian_filter(voxels, sigma=2)
        voxels = (voxels > 0.3).astype(np.float32)

    return voxels


def create_partial_voxel(complete_voxel, occlusion_ratio=0.3):
    """
    Create partial voxel grid by randomly removing voxels.
    This simulates incomplete 3D scans.

    Args:
        complete_voxel: Complete voxel grid
        occlusion_ratio: Fraction of voxels to remove
    Returns:
        Partial voxel grid
    """
    partial = complete_voxel.copy()
    mask = np.random.rand(*complete_voxel.shape) > occlusion_ratio
    partial = partial * mask
    return partial


# ============================================================================
# Dataset
# ============================================================================


class VoxelDataset(Dataset):
    """
    Synthetic voxel dataset for 3D ShapeNets.

    Generates random voxelized 3D shapes on-the-fly.
    In practice, you would load actual ModelNet or ShapeNet data.
    """

    def __init__(
        self,
        num_samples=1000,
        resolution=32,
        num_classes=10,
        task="classification",
        occlusion_ratio=0.3,
    ):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_classes = num_classes
        self.task = task
        self.occlusion_ratio = occlusion_ratio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random class
        class_id = np.random.randint(0, self.num_classes)

        # Generate voxel shape
        voxel = create_synthetic_voxel(
            shape_type=None, resolution=self.resolution, class_id=class_id
        )

        # Add noise
        noise = np.random.randn(*voxel.shape) * 0.05
        voxel = np.clip(voxel + noise, 0, 1).astype(np.float32)

        # Convert to tensor [1, D, H, W]
        voxel_tensor = torch.from_numpy(voxel).unsqueeze(0)

        if self.task == "classification":
            return voxel_tensor, class_id

        elif self.task == "completion":
            # Create partial version
            partial_voxel = create_partial_voxel(voxel, self.occlusion_ratio)
            partial_tensor = torch.from_numpy(partial_voxel).unsqueeze(0)
            return partial_tensor, voxel_tensor  # Input: partial, Target: complete

        else:
            raise ValueError(f"Unknown task: {self.task}")


# ============================================================================
# Training Functions
# ============================================================================


def train_classification(model, dataloader, optimizer, criterion, device):
    """Train classification model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training Classification")
    for voxels, labels in pbar:
        voxels, labels = voxels.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(voxels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    return running_loss / len(dataloader), 100.0 * correct / total


def train_completion(model, dataloader, optimizer, criterion, device):
    """Train completion model for one epoch."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training Completion")
    for partial, complete in pbar:
        partial, complete = partial.to(device), complete.to(device)

        optimizer.zero_grad()
        outputs = model(partial)

        # Binary cross-entropy loss for voxel occupancy
        loss = criterion(outputs, complete)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_classification(model, dataloader, criterion, device):
    """Evaluate classification model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for voxels, labels in dataloader:
        voxels, labels = voxels.to(device), labels.to(device)
        outputs = model(voxels)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate_completion(model, dataloader, criterion, device):
    """Evaluate completion model."""
    model.eval()
    running_loss = 0.0

    for partial, complete in dataloader:
        partial, complete = partial.to(device), complete.to(device)
        outputs = model(partial)
        loss = criterion(outputs, complete)
        running_loss += loss.item()

    return running_loss / len(dataloader)


# ============================================================================
# Visualization
# ============================================================================


def visualize_voxels(voxels, title="Voxel Grid", threshold=0.5):
    """
    Visualize 3D voxel grid.

    Args:
        voxels: [D, H, W] numpy array
        title: Plot title
        threshold: Threshold for occupancy
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get occupied voxel coordinates
    occupied = voxels > threshold
    x, y, z = np.where(occupied)

    # Plot
    ax.scatter(x, y, z, c="blue", marker="s", s=20, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = max(voxels.shape)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    return fig


def visualize_completion_results(
    partial, complete, predicted, save_path="completion_results.png"
):
    """Visualize shape completion results."""
    fig = plt.figure(figsize=(15, 5))

    # Partial input
    ax1 = fig.add_subplot(131, projection="3d")
    x, y, z = np.where(partial > 0.5)
    ax1.scatter(x, y, z, c="red", marker="s", s=20, alpha=0.6)
    ax1.set_title("Partial Input")
    ax1.set_xlim([0, partial.shape[0]])
    ax1.set_ylim([0, partial.shape[1]])
    ax1.set_zlim([0, partial.shape[2]])

    # Ground truth complete
    ax2 = fig.add_subplot(132, projection="3d")
    x, y, z = np.where(complete > 0.5)
    ax2.scatter(x, y, z, c="green", marker="s", s=20, alpha=0.6)
    ax2.set_title("Ground Truth Complete")
    ax2.set_xlim([0, complete.shape[0]])
    ax2.set_ylim([0, complete.shape[1]])
    ax2.set_zlim([0, complete.shape[2]])

    # Predicted complete
    ax3 = fig.add_subplot(133, projection="3d")
    x, y, z = np.where(predicted > 0.5)
    ax3.scatter(x, y, z, c="blue", marker="s", s=20, alpha=0.6)
    ax3.set_title("Predicted Complete")
    ax3.set_xlim([0, predicted.shape[0]])
    ax3.set_ylim([0, predicted.shape[1]])
    ax3.set_zlim([0, predicted.shape[2]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")


# ============================================================================
# Main Training Script
# ============================================================================


def main(args):
    print("=" * 80)
    print("3D ShapeNets Demo (Wu et al., 2015)")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Task selection
    if args.task == "classification":
        print(f"\nTask: 3D Shape Classification")
        print(f"Resolution: {args.resolution}³ voxels")
        print(f"Number of classes: {args.num_classes}")

        # Create datasets
        train_dataset = VoxelDataset(
            num_samples=args.num_train,
            resolution=args.resolution,
            num_classes=args.num_classes,
            task="classification",
        )
        val_dataset = VoxelDataset(
            num_samples=args.num_val,
            resolution=args.resolution,
            num_classes=args.num_classes,
            task="classification",
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Model
        model = ShapeNets3D(
            num_classes=args.num_classes, input_size=args.resolution, use_dropout=True
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params/1e6:.2f}M")

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training loop
        print("\nTraining...")
        best_val_acc = 0.0

        for epoch in range(args.epochs):
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")

            train_loss, train_acc = train_classification(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate_classification(
                model, val_loader, criterion, device
            )

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "shapenets_classifier.pth")
                print(f"  Saved best model (val acc: {val_acc:.2f}%)")

        print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")

        # Visualize some examples
        model.eval()
        with torch.no_grad():
            voxels, labels = next(iter(val_loader))
            voxels = voxels.to(device)
            outputs = model(voxels)
            _, predicted = outputs.max(1)

            # Show first example
            voxel_np = voxels[0, 0].cpu().numpy()
            fig = visualize_voxels(
                voxel_np, f"Class: {labels[0].item()}, Predicted: {predicted[0].item()}"
            )
            plt.savefig("shapenets_example.png", dpi=150, bbox_inches="tight")
            print("Saved example visualization to shapenets_example.png")

    elif args.task == "completion":
        print(f"\nTask: 3D Shape Completion")
        print(f"Resolution: {args.resolution}³ voxels")
        print(f"Occlusion ratio: {args.occlusion_ratio}")

        # Create datasets
        train_dataset = VoxelDataset(
            num_samples=args.num_train,
            resolution=args.resolution,
            num_classes=args.num_classes,
            task="completion",
            occlusion_ratio=args.occlusion_ratio,
        )
        val_dataset = VoxelDataset(
            num_samples=args.num_val,
            resolution=args.resolution,
            num_classes=args.num_classes,
            task="completion",
            occlusion_ratio=args.occlusion_ratio,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Model
        model = ShapeCompletion3D(input_size=args.resolution).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params/1e6:.2f}M")

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()  # Binary cross-entropy for voxel occupancy
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training loop
        print("\nTraining...")
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")

            train_loss = train_completion(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = evaluate_completion(model, val_loader, criterion, device)

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "shapenets_completion.pth")
                print(f"  Saved best model (val loss: {val_loss:.4f})")

        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

        # Visualize completion results
        model.eval()
        with torch.no_grad():
            partial, complete = next(iter(val_loader))
            partial, complete = partial.to(device), complete.to(device)
            predicted = model(partial)

            # Show first example
            partial_np = partial[0, 0].cpu().numpy()
            complete_np = complete[0, 0].cpu().numpy()
            predicted_np = predicted[0, 0].cpu().numpy()

            visualize_completion_results(partial_np, complete_np, predicted_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D ShapeNets Demo")

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "completion"],
        help="Task to perform",
    )

    # Data
    parser.add_argument(
        "--resolution", type=int, default=32, help="Voxel grid resolution"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of shape classes"
    )
    parser.add_argument(
        "--num_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--num_val", type=int, default=200, help="Number of validation samples"
    )
    parser.add_argument(
        "--occlusion_ratio",
        type=float,
        default=0.3,
        help="Occlusion ratio for completion task",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    main(args)
