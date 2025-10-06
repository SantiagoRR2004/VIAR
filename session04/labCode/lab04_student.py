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
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import Utils

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
        # Task 1.1: Implement double convolution block
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
        # Task 1.1: Initialize layers
        # You need: DoubleConv and MaxPool2d
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # Task 1.1: Implement forward pass
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

        # Task 1.2: Initialize upsampling layer
        if upsampling == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:  # bilinear
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )

        # Task 1.2: Initialize double convolution
        # Note: Input will be concatenated features (in_channels + skip_channels)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_features):
        # Task 1.2: Implement forward pass
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

        # Task 1.3: Build encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # : Create encoder blocks
        # Hint: First block takes in_channels, others take features[i-1]
        self.encoders.append(EncoderBlock(in_channels, features[0]))
        for i in range(1, len(features)):
            self.encoders.append(EncoderBlock(features[i - 1], features[i]))

        # Task 1.3: Bottleneck (deepest part)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Task 1.3: Build decoder path
        self.decoders = nn.ModuleList()

        # Create decoder blocks (in reverse order)
        self.decoders.append(
            DecoderBlock(features[-1] * 2, features[-1], features[-1])
        )  # 1024 -> 512
        for i in range(len(features) - 1, 0, -1):
            self.decoders.append(
                DecoderBlock(features[i], features[i - 1], features[i - 1])
            )

        # Task 1.4: Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Task 1.4: Connect everything together
        skip_connections = []

        # Encoder path
        # Process through encoders, save skip connections
        for encoder in self.encoders:
            features, x = encoder(x)
            skip_connections.append(features)

        # Bottleneck
        # Process through bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoding
        # Decoder path
        # Process through decoders with skip connections
        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[idx])

        # Final layer
        # Apply final convolution
        x = self.final_conv(x)

        return x  # Return final output


# ================== Part 2: Skip Connection Strategies ==================


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""

    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        # Task 2.3: Implement attention gate
        # You need: Conv2d layers for gating signal, skip features, and psi
        inter = gate_channels // 2

        # Conv2d for gating signal
        self.W_g = nn.Conv2d(gate_channels, inter, 1)

        # Conv2d for skip features
        self.W_x = nn.Conv2d(skip_channels, inter, 1)

        # Conv2d for final attention coefficients
        self.psi = nn.Conv2d(inter, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # Task 2.3: Implement attention mechanism

        # 1. Process gate and skip through respective convolutions
        g1, x1 = self.W_g(gate), self.W_x(skip)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=True
            )

        # 2. Add and apply ReLU
        attention = self.relu(g1 + x1)

        # 3. Apply psi convolution and sigmoid
        attention = torch.sigmoid(self.psi(attention))

        # 4. Multiply with skip features
        return skip * attention


class FlexibleSkipConnection(nn.Module):
    """Flexible skip connection with different strategies"""

    def __init__(self, decoder_channels, skip_channels, mode="concat"):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            # Task 2.1: Setup for concatenation
            # Output conv to handle concatenated channels
            self.conv = nn.Conv2d(
                decoder_channels + skip_channels, decoder_channels, 3, padding=1
            )

        elif mode == "add":
            # Task 2.2: Setup for addition
            # May need 1x1 conv to match channels
            self.proj = (
                nn.Conv2d(skip_channels, decoder_channels, 1)
                if skip_channels != decoder_channels
                else nn.Identity()
            )

        elif mode == "attention":
            # Task 2.3: Setup attention gate
            self.attention_gate = AttentionGate(decoder_channels, skip_channels)
            self.conv = nn.Conv2d(
                decoder_channels + skip_channels, decoder_channels, 3, padding=1
            )

    def forward(self, decoder_features, skip_features):
        # Implement forward pass based on mode
        if self.mode == "concat":
            return self.conv(torch.cat([decoder_features, skip_features], dim=1))

        elif self.mode == "add":
            return decoder_features + self.proj(skip_features)

        elif self.mode == "attention":
            return self.conv(
                torch.cat(
                    [
                        decoder_features,
                        self.attention_gate(decoder_features, skip_features),
                    ],
                    dim=1,
                )
            )


# ================== Part 3: Loss Functions ==================


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Task 3.1: Implement Dice loss
        # 1. Apply sigmoid to predictions
        pred = torch.sigmoid(pred)

        # 2. Flatten both pred and target
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 3. Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # 4. Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 5. Return 1 - dice
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target):
        # Task 3.2: Implement Focal loss
        # 1. Apply sigmoid and compute BCE
        pred = torch.sigmoid(pred)
        bce = self.ce(pred, target)

        # 2. Calculate p_t (probability of correct class)
        p_t = torch.exp(-bce)

        # 3. Apply focal term: (1-p_t)^gamma
        focalTerm = (1 - p_t) ** self.gamma

        # 4. Apply alpha weighting
        loss = self.alpha * focalTerm * bce
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function"""

    def __init__(self, weights={"ce": 0.5, "dice": 0.5, "focal": 0.0}):
        super().__init__()
        # Task 3.3: Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

    def forward(self, pred, target):
        # Task 3.3: Compute weighted combination of losses
        total_loss = 0

        # Add each loss component with its weight
        total_loss += self.weights.get("ce", 0) * self.ce_loss(pred, target)
        total_loss += self.weights.get("dice", 0) * self.dice_loss(pred, target)
        total_loss += self.weights.get("focal", 0) * self.focal_loss(pred, target)

        return total_loss


# ================== Training and Evaluation Functions ==================


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    # Task 3.4: Implement IoU calculation
    # 1. Threshold predictions
    pred = (pred > threshold).float()

    # 2. Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    # 3. Return IoU score
    iou = intersection / union if union > 0 else 0

    return iou


def train_epoch(model, dataloader, optimizer, criterion: nn.Module, device) -> tuple:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0

    # Complete training loop
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        # Calculate loss
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        total_iou += iou

    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            total_iou += iou

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
    imageTransform = transforms.Compose(
        [
            # Add necessary transforms
            # Resize, ToTensor, Normalize
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    targetTransform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    # Load dataset
    # Use OxfordIIITPet or a simple synthetic dataset for testing
    dataset = OxfordIIITPet(
        root=os.path.join(currentDirectory, "images"),
        download=True,
        target_types="segmentation",
        transform=imageTransform,
        target_transform=targetTransform,
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

    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = DiceLoss()

    # Training loop
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_iou = 0.0

    print("Starting training...")
    for epoch in range(config["epochs"]):
        # Train and validate
        train_loss, train_iou = train_epoch(
            model, trainLoader, optimizer, criterion, device
        )
        val_loss, val_iou = validate(model, valLoader, criterion, device)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # Print progress
        print(
            f"Epoch [{epoch+1}/{config['epochs']}], "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Best model saved!")

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
