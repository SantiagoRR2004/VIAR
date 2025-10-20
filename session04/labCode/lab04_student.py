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
import time
from prettytable import PrettyTable
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
        self,
        in_channels,
        out_channels,
        skipType: "FlexibleSkipConnection",
        upsampling="transpose",
    ):
        super().__init__()
        self.upsampling = upsampling
        self.skipType = skipType

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
        self.conv = DoubleConv(skipType.out_channels, out_channels)

    def forward(self, x, skip_features):
        # Task 1.2: Implement forward pass
        # 1. Upsample x
        # 2. Handle dimension mismatch if necessary (crop or pad)
        # 3. Concatenate with skip_features
        # 4. Apply double convolution
        x = self.up(x)

        x = self.skipType(x, skip_features)

        return self.conv(x)


class UNet(nn.Module):
    """Complete U-Net architecture"""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        skipMode: str = "concat",
    ):
        super().__init__()
        self.skipMode = skipMode

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
        self.skipps: list[FlexibleSkipConnection] = nn.ModuleList()

        # Create decoder blocks (in reverse order)
        self.skipps.append(
            FlexibleSkipConnection(features[-1], features[-1], mode=skipMode)
        )
        self.decoders.append(
            DecoderBlock(features[-1] * 2, features[-1], skipType=self.skipps[-1])
        )  # 1024 -> 512
        for i in range(len(features) - 1, 0, -1):
            self.skipps.append(
                FlexibleSkipConnection(features[i - 1], features[i - 1], mode=skipMode)
            )
            self.decoders.append(
                DecoderBlock(features[i], features[i - 1], skipType=self.skipps[-1])
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
        self.out_channels = decoder_channels

        if mode == "concat":
            # Task 2.1: Setup for concatenation
            # Output conv to handle concatenated channels

            # Concat features from decoder and skip (encoder)
            # Preserve all information, higher computational cost
            # Use a conv layer 3 x 3 to reduce channels back to decoder_channels
            self.conv = nn.Conv2d(
                decoder_channels + skip_channels, decoder_channels, 3, padding=1
            )

        elif mode == "add":
            # Task 2.2: Setup for addition
            # May need 1x1 conv to match channels; to cast skip to decoder channels
            # Element-wise addition, lower computational cost, may lose some information
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

        diffY = skip_features.size()[2] - decoder_features.size()[2]
        diffX = skip_features.size()[3] - decoder_features.size()[3]

        decoder_features = F.pad(
            decoder_features,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

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

    # Implement visualization
    # 1. Get a batch of images and masks
    # 2. Generate predictions
    # 3. Create subplot showing: input, ground truth, prediction

    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

    # Plotting
    plt.figure(figsize=(12, 4))
    for j in range(min(4, images.size(0))):
        plt.subplot(3, 4, j + 1)
        plt.imshow(images[j].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(3, 4, j + 5)
        plt.imshow(masks[j].cpu().squeeze(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(3, 4, j + 9)
        plt.imshow(preds[j].cpu().squeeze(), cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()


def plot_training_curves(train_losses, val_losses, train_ious, val_ious):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ious, label="Train IoU")
    plt.plot(epochs, val_ious, label="Val IoU")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU")
    plt.legend()

    plt.tight_layout()


def get_dataloaders(config: dict):
    """Prepare data loaders for training and validation"""
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

    return trainLoader, valLoader


# ================== Main Training Script ==================


def main(
    batch_size: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 50,
    image_size: int = 128,
    skip_mode: str = "concat",
):
    # Hyperparameters
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "image_size": image_size,
        "skip_mode": skip_mode,  # Try: 'concat', 'add', 'attention'
    }
    # Get data loaders
    trainLoader, valLoader = get_dataloaders(config)

    # Initialize model
    # if we have best_unet_model.pth use it instead training:
    if os.path.exists(
        os.path.join(currentDirectory, config["skip_mode"], "best_unet_model.pth")
    ):
        model = UNet(in_channels=3, out_channels=1, skipMode=config["skip_mode"]).to(
            device
        )
        model.load_state_dict(
            torch.load(
                os.path.join(
                    currentDirectory, config["skip_mode"], "best_unet_model.pth"
                ),
                map_location=device,
            )
        )
        if os.path.exists(
            os.path.join(currentDirectory, config["skip_mode"], "train_losses.npy")
        ):
            train_losses = np.load(
                os.path.join(currentDirectory, config["skip_mode"], "train_losses.npy")
            )
            val_losses = np.load(
                os.path.join(currentDirectory, config["skip_mode"], "val_losses.npy")
            )
            train_ious = np.load(
                os.path.join(currentDirectory, config["skip_mode"], "train_ious.npy")
            )
            val_ious = np.load(
                os.path.join(currentDirectory, config["skip_mode"], "val_ious.npy")
            )

    else:
        model = UNet(in_channels=3, out_channels=1, skipMode=config["skip_mode"]).to(
            device
        )

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
        start_time = time.time()
        for epoch in range(config["epochs"]):
            # Train and validate
            train_loss, train_iou = train_epoch(
                model, trainLoader, optimizer, criterion, device
            )
            train_iou = train_iou.cpu()
            val_loss, val_iou = validate(model, valLoader, criterion, device)
            val_iou = val_iou.cpu()

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
                torch.save(
                    model.state_dict(),
                    os.path.join(currentDirectory, "best_unet_model.pth"),
                )
                print("Best model saved!")

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Collect gradient flow and memory usage
        gradient_norms = {}
        model.train()
        images, masks = next(iter(trainLoader))
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()

        # Collect gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = param.grad.norm().item()

        avg_gradient = np.mean(list(gradient_norms.values()))

        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model.eval()
            with torch.no_grad():
                _ = model(images)
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_usage = 0.0

        np.save(
            os.path.join(currentDirectory, config["skip_mode"], "train_losses.npy"),
            np.array(train_losses),
        )
        np.save(
            os.path.join(currentDirectory, config["skip_mode"], "val_losses.npy"),
            np.array(val_losses),
        )
        np.save(
            os.path.join(currentDirectory, config["skip_mode"], "train_ious.npy"),
            np.array(train_ious),
        )
        np.save(
            os.path.join(currentDirectory, config["skip_mode"], "val_ious.npy"),
            np.array(val_ious),
        )

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_ious, val_ious)

    # Visualize final predictions
    visualize_predictions(model, valLoader, device)

    print("Training complete!")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ious": train_ious,
        "val_ious": val_ious,
        "training_time": training_time,
        "avg_gradient": avg_gradient,
        "memory_usage": memory_usage,
    }


# ================== Analysis Functions ==================


def analyze_skip_connections():
    """Compare different skip connection strategies"""
    # Task 2.4: Implement comparison
    # 1. Train models with different skip modes
    # 2. Compare gradient flow
    # 3. Compare memory usage
    # 4. Compare final performance

    results = {}

    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 50,
        "image_size": 128,
    }

    # Run experiments for each mode
    for mode in ["concat", "add", "attention"]:
        results[mode] = main(skip_mode=mode, **config)

    # Create comparison table/plot
    table = PrettyTable()
    table.field_names = [
        "Skip Mode",
        "Final Val Losses",
        "Final Val IoU",
        "Avg Gradient Norm",
        "Memory Usage (MB)",
        "Training Time (s)",
    ]
    for mode, metrics in results.items():
        table.add_row(
            [
                mode,
                metrics["val_losses"][-1],
                metrics["val_ious"][-1],
                metrics["avg_gradient"],
                metrics["memory_usage"],
                metrics["training_time"],
            ]
        )
    print(table)

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
    # main()

    # Run analysis (optional)
    analyze_skip_connections()
    # ablation_study()

    plt.show()
