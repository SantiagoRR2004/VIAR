"""
Multi-View CNN Demo (Su et al., 2015)
Lecture 11: 3D Scene Understanding and Neural Rendering

This script implements Multi-View CNN for 3D shape recognition.
Key concepts:
- Rendering 3D objects from multiple viewpoints
- Extracting 2D CNN features from each view
- View-pooling aggregation for 3D representation
- Leveraging pre-trained 2D CNNs

Reference: Su et al., "Multi-view Convolutional Neural Networks for 3D Shape Recognition", ICCV 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================================
# Simple 3D Object Rendering
# ============================================================================


def create_simple_mesh(shape_type="cube"):
    """
    Create simple 3D mesh for rendering.
    Returns vertices and faces.
    """
    if shape_type == "cube":
        # Cube vertices
        vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float32,
        )

        # Cube faces (triangles)
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom
                [4, 5, 6],
                [4, 6, 7],  # top
                [0, 1, 5],
                [0, 5, 4],  # front
                [2, 3, 7],
                [2, 7, 6],  # back
                [0, 3, 7],
                [0, 7, 4],  # left
                [1, 2, 6],
                [1, 6, 5],  # right
            ],
            dtype=np.int32,
        )

    elif shape_type == "pyramid":
        vertices = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 2]],  # base  # apex
            dtype=np.float32,
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # base
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],  # sides
            ],
            dtype=np.int32,
        )

    elif shape_type == "octahedron":
        vertices = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float32,
        )

        faces = np.array(
            [
                [0, 2, 4],
                [0, 4, 3],
                [0, 3, 5],
                [0, 5, 2],
                [1, 4, 2],
                [1, 3, 4],
                [1, 5, 3],
                [1, 2, 5],
            ],
            dtype=np.int32,
        )

    else:
        # Default to cube
        return create_simple_mesh("cube")

    return vertices, faces


def simple_perspective_projection(
    vertices, camera_pos, focal_length=2.0, image_size=224
):
    """
    Simple perspective projection of 3D vertices to 2D image plane.

    Args:
        vertices: [N, 3] array of 3D vertices
        camera_pos: [3] camera position
        focal_length: Focal length
        image_size: Output image size
    Returns:
        [N, 2] array of 2D pixel coordinates
    """
    # Translate vertices relative to camera
    vertices_cam = vertices - camera_pos

    # Perspective projection
    x = vertices_cam[:, 0] * focal_length / (vertices_cam[:, 2] + 1e-6)
    y = vertices_cam[:, 1] * focal_length / (vertices_cam[:, 2] + 1e-6)

    # Convert to image coordinates
    x_img = (x + 1) * image_size / 4 + image_size / 2
    y_img = (y + 1) * image_size / 4 + image_size / 2

    return np.stack([x_img, y_img], axis=1)


def render_mesh_simple(vertices, faces, camera_pos, image_size=224):
    """
    Simple wireframe rendering of mesh.

    Args:
        vertices: [N, 3] vertices
        faces: [M, 3] faces (triangle indices)
        camera_pos: [3] camera position
        image_size: Output image size
    Returns:
        [image_size, image_size, 3] rendered image
    """
    # Project vertices to 2D
    vertices_2d = simple_perspective_projection(
        vertices, camera_pos, image_size=image_size
    )

    # Create image
    image = np.ones((image_size, image_size, 3), dtype=np.float32)

    # Draw edges
    for face in faces:
        for i in range(3):
            v1 = vertices_2d[face[i]]
            v2 = vertices_2d[face[(i + 1) % 3]]

            # Simple line drawing
            draw_line(image, v1, v2, color=[0, 0, 0])

    # Fill faces (simple z-buffer)
    for face in faces:
        pts = vertices_2d[face]
        if pts.min() >= 0 and pts.max() < image_size:
            # Compute center depth
            depth = vertices[face, 2].mean()
            if depth > 0:  # Only render faces facing camera
                fill_triangle(image, pts, color=[0.7, 0.7, 0.7])

    return image


def draw_line(image, p1, p2, color=[0, 0, 0], thickness=2):
    """Draw line using Bresenham's algorithm."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
            for i in range(-thickness, thickness + 1):
                for j in range(-thickness, thickness + 1):
                    xi, yj = x1 + i, y1 + j
                    if 0 <= xi < image.shape[1] and 0 <= yj < image.shape[0]:
                        image[yj, xi] = color

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def fill_triangle(image, pts, color=[0.7, 0.7, 0.7]):
    """Fill triangle using barycentric coordinates."""
    pts = pts.astype(int)

    # Bounding box
    min_x = max(0, pts[:, 0].min())
    max_x = min(image.shape[1] - 1, pts[:, 0].max())
    min_y = max(0, pts[:, 1].min())
    max_y = min(image.shape[0] - 1, pts[:, 1].max())

    # Fill
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_triangle([x, y], pts):
                image[y, x] = color


def point_in_triangle(p, triangle):
    """Check if point is inside triangle using barycentric coordinates."""

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, triangle[0], triangle[1])
    d2 = sign(p, triangle[1], triangle[2])
    d3 = sign(p, triangle[2], triangle[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def get_camera_positions(num_views=12, radius=5.0):
    """
    Generate camera positions on a sphere.

    Following the MVCNN paper, we use 12 views by default:
    - Views on the equator at different azimuths
    - Views at different elevations

    Args:
        num_views: Number of views
        radius: Distance from object center
    Returns:
        [num_views, 3] camera positions
    """
    cameras = []

    # Generate views on multiple elevation levels
    elevations = [0, 30, -30]  # degrees
    num_views_per_elevation = num_views // len(elevations)

    for elev in elevations:
        for i in range(num_views_per_elevation):
            azimuth = 360.0 * i / num_views_per_elevation

            # Convert to radians
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elev)

            # Spherical to Cartesian
            x = radius * np.cos(el_rad) * np.cos(az_rad)
            y = radius * np.cos(el_rad) * np.sin(az_rad)
            z = radius * np.sin(el_rad)

            cameras.append([x, y, z])

    return np.array(cameras, dtype=np.float32)


def render_multiview(vertices, faces, num_views=12, image_size=224):
    """
    Render object from multiple viewpoints.

    Args:
        vertices: [N, 3] vertices
        faces: [M, 3] faces
        num_views: Number of views to render
        image_size: Size of each rendered image
    Returns:
        [num_views, 3, image_size, image_size] tensor of rendered images
    """
    camera_positions = get_camera_positions(num_views)

    images = []
    for cam_pos in camera_positions:
        img = render_mesh_simple(vertices, faces, cam_pos, image_size)

        # Convert to tensor [3, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img_tensor)

    return torch.stack(images)


# ============================================================================
# Multi-View CNN Architecture
# ============================================================================


class MultiViewCNN(nn.Module):
    """
    Multi-View CNN for 3D shape recognition.

    Architecture:
    1. Use pre-trained 2D CNN (e.g., ResNet) as view-specific feature extractor
    2. Extract features from each view independently
    3. Aggregate features using view-pooling
    4. Final classification layer

    The key insight: 2D CNNs are very good at extracting features from images,
    so we render 3D objects from multiple views and leverage 2D CNNs.
    """

    def __init__(
        self,
        num_classes=10,
        num_views=12,
        pretrained=True,
        pooling_type="max",
        dropout=0.5,
    ):
        super(MultiViewCNN, self).__init__()

        self.num_views = num_views
        self.pooling_type = pooling_type

        # Pre-trained 2D CNN backbone (ResNet-18)
        base_model = models.resnet18(pretrained=pretrained)

        # Remove final FC layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Feature dimension
        self.feature_dim = 512  # ResNet-18 final layer

        # View pooling layer (optional learnable weights)
        if pooling_type == "learned":
            self.view_attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 4, 1),
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Freeze early layers if using pretrained model
        if pretrained:
            for param in list(self.feature_extractor.parameters())[:-10]:
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, V, 3, H, W] - Batch of multi-view images
               B: batch size
               V: number of views
               3, H, W: RGB image
        Returns:
            [B, num_classes] - Classification logits
        """
        batch_size, num_views, C, H, W = x.size()

        # Reshape to process all views in parallel
        x = x.view(batch_size * num_views, C, H, W)

        # Extract features from each view
        features = self.feature_extractor(x)  # [B*V, feature_dim, 1, 1]
        features = features.view(batch_size, num_views, -1)  # [B, V, feature_dim]

        # View pooling
        if self.pooling_type == "max":
            # Max pooling across views
            pooled_features, _ = torch.max(features, dim=1)  # [B, feature_dim]

        elif self.pooling_type == "mean":
            # Average pooling across views
            pooled_features = torch.mean(features, dim=1)  # [B, feature_dim]

        elif self.pooling_type == "learned":
            # Learned attention-based pooling
            attention_scores = self.view_attention(features)  # [B, V, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, V, 1]
            pooled_features = torch.sum(
                features * attention_weights, dim=1
            )  # [B, feature_dim]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Classification
        output = self.classifier(pooled_features)

        return output

    def extract_view_features(self, x):
        """Extract per-view features (for visualization)."""
        batch_size, num_views, C, H, W = x.size()
        x = x.view(batch_size * num_views, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_views, -1)
        return features


# ============================================================================
# Dataset
# ============================================================================


class MultiViewDataset(Dataset):
    """
    Multi-view dataset for 3D shape classification.

    Generates synthetic 3D shapes and renders them from multiple views.
    In practice, you would use ModelNet or ShapeNet with pre-rendered views.
    """

    def __init__(
        self,
        num_samples=1000,
        num_views=12,
        num_classes=10,
        image_size=224,
        split="train",
    ):
        self.num_samples = num_samples
        self.num_views = num_views
        self.num_classes = num_classes
        self.image_size = image_size
        self.split = split

        # Define shape types
        self.shape_types = ["cube", "pyramid", "octahedron"] + [
            f"shape_{i}" for i in range(7)
        ]

        # ImageNet normalization (for pretrained models)
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random class
        class_id = np.random.randint(0, self.num_classes)

        # Get shape type
        shape_type = self.shape_types[class_id % len(self.shape_types)]

        # Create mesh
        vertices, faces = create_simple_mesh(shape_type)

        # Add random rotation and scaling (data augmentation)
        if self.split == "train":
            # Random rotation
            angle_x = np.random.uniform(0, 2 * np.pi)
            angle_y = np.random.uniform(0, 2 * np.pi)
            angle_z = np.random.uniform(0, 2 * np.pi)

            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)],
                ]
            )
            Rz = np.array(
                [
                    [np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1],
                ]
            )

            R = Rz @ Ry @ Rx
            vertices = vertices @ R.T

            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            vertices = vertices * scale

        # Render from multiple views
        views = render_multiview(vertices, faces, self.num_views, self.image_size)

        # Apply normalization
        views = self.transform(views)

        return views, class_id


# ============================================================================
# Training and Evaluation
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for views, labels in pbar:
        views, labels = views.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(views)
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


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for views, labels in dataloader:
        views, labels = views.to(device), labels.to(device)
        outputs = model(views)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


# ============================================================================
# Visualization
# ============================================================================


def visualize_views(views, label, predicted=None, save_path="multiview_example.png"):
    """Visualize multi-view images."""
    num_views = views.shape[0]

    # Create grid
    n_cols = 4
    n_rows = (num_views + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(num_views):
        # Denormalize
        img = views[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"View {i+1}")
        axes[i].axis("off")

    # Hide empty subplots
    for i in range(num_views, len(axes)):
        axes[i].axis("off")

    title = f"Multi-View Images - True Label: {label}"
    if predicted is not None:
        title += f", Predicted: {predicted}"

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("Multi-View CNN Demo (Su et al., 2015)")
    print("Lecture 11: 3D Scene Understanding and Neural Rendering")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Number of views: {args.num_views}")
    print(f"Pooling type: {args.pooling}")
    print(f"Pretrained: {args.pretrained}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiViewDataset(
        num_samples=args.num_train,
        num_views=args.num_views,
        num_classes=args.num_classes,
        image_size=args.image_size,
        split="train",
    )
    val_dataset = MultiViewDataset(
        num_samples=args.num_val,
        num_views=args.num_views,
        num_classes=args.num_classes,
        image_size=args.image_size,
        split="val",
    )

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nInitializing Multi-View CNN...")
    model = MultiViewCNN(
        num_classes=args.num_classes,
        num_views=args.num_views,
        pretrained=args.pretrained,
        pooling_type=args.pooling,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f}M")

    # Optimizer and loss
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "multiview_cnn_best.pth",
            )
            print(f"  Saved best model (val acc: {val_acc:.2f}%)")

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")

    # Visualize some examples
    print("\nGenerating visualizations...")
    model.eval()
    with torch.no_grad():
        views, labels = next(iter(val_loader))
        views = views.to(device)
        outputs = model(views)
        _, predicted = outputs.max(1)

        # Show first example
        visualize_views(views[0], labels[0].item(), predicted[0].item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-View CNN Demo")

    # Data
    parser.add_argument(
        "--num_views", type=int, default=12, help="Number of views per object"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of shape classes"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of rendered images"
    )
    parser.add_argument(
        "--num_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--num_val", type=int, default=200, help="Number of validation samples"
    )

    # Model
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained ResNet"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean", "learned"],
        help="View pooling method",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    main(args)
