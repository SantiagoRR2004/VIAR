"""
Lab 4: Encoder-Decoder Architectures for Dense Prediction
Student Version - Complete the TODOs

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import Utils
import os

warnings.filterwarnings("ignore")

# Set device
device = torch.device(Utils.canUseGPU())
print(f"Using device: {device}")

currentDirectory = os.path.dirname(os.path.abspath(__file__))

# ================== Part 1: U-Net Architecture ==================


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO Task 1.1: Implement double convolution block
        # Hint: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        # Use kernel_size=3, padding=1 to maintain spatial dimensions
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with double convolution and pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO Task 1.1: Initialize layers
        # You need: DoubleConv and MaxPool2d
        self.conv = DoubleConv(in_channels, out_channels)  # TODO
        self.pool = nn.MaxPool2d(kernel_size=2)  # TODO

    def forward(self, x):
        # TODO Task 1.1: Implement forward pass
        # Return both: features before pooling (for skip connection) and after pooling
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and double convolution"""

    def __init__(
        self, in_channels, skip_channels, out_channels, upsampling="transpose"
    ):
        super().__init__()
        self.upsampling = upsampling

        # TODO Task 1.2: Initialize upsampling layer
        if upsampling == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:  # bilinear
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )

        # TODO Task 1.2: Initialize double convolution
        # Note: Input will be concatenated features (in_channels + skip_channels)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_features):
        # TODO Task 1.2: Implement forward pass
        # 1. Upsample x
        # 2. Handle dimension mismatch if necessary (crop or pad)
        # 3. Concatenate with skip_features
        # 4. Apply double convolution
        x = self.up(x)
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x, skip_features], dim=1))


class UNet(nn.Module):
    """Complete U-Net architecture"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # TODO Task 1.3: Build encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # TODO: Create encoder blocks
        # Hint: First block takes in_channels, others take features[i-1]
        self.encoders.append(EncoderBlock(in_channels, features[0]))
        for i in range(1, len(features)):
            self.encoders.append(EncoderBlock(features[i - 1], features[i]))

        # TODO Task 1.3: Bottleneck (deepest part)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # TODO Task 1.3: Build decoder path
        self.decoders = nn.ModuleList()

        # TODO: Create decoder blocks (in reverse order)
        for i in range(len(features) - 1, 0, -1):
            pass
            # self.decoders.append(DecoderBlock(features[i]*2, features[i-1], features[i-1])) Esto aún no estoy seguro
        # self.decoders.append(DecoderBlock(features[0]*2, in_channels, features[0]))

        # TODO Task 1.4: Final output layer
        self.final_conv = None  # TODO: Conv2d to get desired output channels

    def forward(self, x):
        # TODO Task 1.4: Connect everything together
        skip_connections = []

        # Encoder path
        # TODO: Process through encoders, save skip connections

        # Bottleneck
        # TODO: Process through bottleneck

        # Decoder path
        # TODO: Process through decoders with skip connections

        # Final layer
        # TODO: Apply final convolution

        return x  # TODO: Return final output


# ================== Part 2: Skip Connection Strategies ==================


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""

    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        # TODO Task 2.3: Implement attention gate
        # You need: Conv2d layers for gating signal, skip features, and psi
        self.W_g = None  # TODO: Conv2d for gating signal
        self.W_x = None  # TODO: Conv2d for skip features
        self.psi = None  # TODO: Conv2d for final attention coefficients
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # TODO Task 2.3: Implement attention mechanism
        # 1. Process gate and skip through respective convolutions
        # 2. Add and apply ReLU
        # 3. Apply psi convolution and sigmoid
        # 4. Multiply with skip features
        attention = None  # TODO
        return skip * attention


class FlexibleSkipConnection(nn.Module):
    """Flexible skip connection with different strategies"""

    def __init__(self, decoder_channels, skip_channels, mode="concat"):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            # TODO Task 2.1: Setup for concatenation
            # Output conv to handle concatenated channels
            self.conv = None  # TODO

        elif mode == "add":
            # TODO Task 2.2: Setup for addition
            # May need 1x1 conv to match channels
            self.proj = None  # TODO: Handle channel mismatch

        elif mode == "attention":
            # TODO Task 2.3: Setup attention gate
            self.attention_gate = None  # TODO
            self.conv = None  # TODO: Output conv after attention

    def forward(self, decoder_features, skip_features):
        # TODO: Implement forward pass based on mode
        if self.mode == "concat":
            # TODO Task 2.1
            pass
        elif self.mode == "add":
            # TODO Task 2.2
            pass
        elif self.mode == "attention":
            # TODO Task 2.3
            pass
        return None  # TODO


# ================== Part 3: Loss Functions ==================


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # TODO Task 3.1: Implement Dice loss
        # 1. Apply sigmoid to predictions
        # 2. Flatten both pred and target
        # 3. Compute intersection and union
        # 4. Calculate Dice coefficient
        # 5. Return 1 - dice

        dice = None  # TODO
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # TODO Task 3.2: Implement Focal loss
        # 1. Apply sigmoid and compute BCE
        # 2. Calculate p_t (probability of correct class)
        # 3. Apply focal term: (1-p_t)^gamma
        # 4. Apply alpha weighting

        loss = None  # TODO
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function"""

    def __init__(self, weights={"ce": 0.5, "dice": 0.5, "focal": 0.0}):
        super().__init__()
        # TODO Task 3.3: Initialize component losses
        self.ce_loss = None  # TODO
        self.dice_loss = None  # TODO
        self.focal_loss = None  # TODO
        self.weights = weights

    def forward(self, pred, target):
        # TODO Task 3.3: Compute weighted combination of losses
        total_loss = 0

        # TODO: Add each loss component with its weight

        return total_loss


# ================== Training and Evaluation Functions ==================


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    # TODO Task 3.4: Implement IoU calculation
    # 1. Threshold predictions
    # 2. Calculate intersection and union
    # 3. Return IoU score

    iou = None  # TODO
    return iou


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0

    # TODO: Complete training loop
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        # TODO: Forward pass
        # TODO: Calculate loss
        # TODO: Backward pass
        # TODO: Update weights
        # TODO: Calculate metrics

        pass  # Remove this after implementation

    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        # TODO: Complete validation loop
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)

            # TODO: Forward pass
            # TODO: Calculate loss and metrics

            pass  # Remove this after implementation

    return total_loss / len(dataloader), total_iou / len(dataloader)


def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()

    # TODO: Implement visualization
    # 1. Get a batch of images and masks
    # 2. Generate predictions
    # 3. Create subplot showing: input, ground truth, prediction

    pass


# ================== Main Training Script ==================


def main():
    # Hyperparameters
    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 50,
        "image_size": 128,
        "skip_mode": "concat",  # Try: 'concat', 'add', 'attention'
    }

    # Setup data transforms
    transform = transforms.Compose(
        [
            # Add necessary transforms
            # Resize, ToTensor, Normalize
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load dataset
    # Use OxfordIIITPet or a simple synthetic dataset for testing
    dataset = OxfordIIITPet(
        root=os.path.join(currentDirectory, "images"),
        download=True,
        transform=transform,
    )

    trainSize = int(0.8 * len(dataset))
    valSize = len(dataset) - trainSize
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    # Create data loaders
    trainLoader = DataLoader(
        trainDataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    valLoader = DataLoader(
        valDataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # TODO: Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = DiceLoss()

    # Training loop
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    print("Starting training...")
    for epoch in range(config["epochs"]):
        # TODO: Train and validate
        # TODO: Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # TODO: Print progress
        print(
            f"Epoch [{epoch+1}/{config['epochs']}], "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
        )

        # TODO: Save best model

    # TODO: Plot training curves

    # TODO: Visualize final predictions

    print("Training complete!")


# ================== Analysis Functions ==================


def analyze_skip_connections():
    """Compare different skip connection strategies"""
    # TODO Task 2.4: Implement comparison
    # 1. Train models with different skip modes
    # 2. Compare gradient flow
    # 3. Compare memory usage
    # 4. Compare final performance

    results = {}

    # TODO: Run experiments for each mode
    for mode in ["concat", "add", "attention"]:
        # TODO: Train model
        # TODO: Collect metrics
        pass

    # TODO: Create comparison table/plot

    return results


def ablation_study():
    """Perform ablation study on U-Net components"""
    # TODO: Implement ablation study
    # Test: no skip connections, no batch norm, different depths

    ablation_results = {}

    # TODO: Run different configurations

    return ablation_results


if __name__ == "__main__":
    # Run main training
    main()

    # Run analysis (optional)
    # analyze_skip_connections()
    # ablation_study()
