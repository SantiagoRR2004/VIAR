"""
Lab 4: Encoder-Decoder Architectures for Dense Prediction
Student Version - Complete the TODOs

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from prettytable import PrettyTable
import warnings
import losses
import UNet
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import Utils

warnings.filterwarnings("ignore")

# Set device
device = torch.device(Utils.canUseGPU())
print(f"Using device: {device}")

currentDirectory = os.path.dirname(os.path.abspath(__file__))

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


def visualize_predictions(models: dict, dataloader, device):
    """Visualize model predictions"""
    for model in models.values():
        model.eval()

    # Implement visualization
    # 1. Get a batch of images and masks
    # 2. Generate predictions
    # 3. Create subplot showing: input, ground truth, prediction

    # Get one batch and take the first sample
    images, masks = next(iter(dataloader))
    image, mask = images[0].unsqueeze(0).to(device), masks[0].unsqueeze(0).to(device)

    # Collect predictions from each model
    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            output = model(image)
            pred = torch.sigmoid(output)  # > 0.5
            preds[name] = pred.cpu().squeeze().numpy()

    cols = max(2, len(models))

    fig = plt.figure(figsize=(4 * cols, 6))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], figure=fig)

    # --- Top row: its own 1x2 grid
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0])

    ax_img = fig.add_subplot(top_gs[0, 0])
    ax_img.imshow(image[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
    ax_img.set_title("Input Image")
    ax_img.axis("off")

    ax_mask = fig.add_subplot(top_gs[0, 1])
    ax_mask.imshow(mask[0].cpu().squeeze(), cmap="gray")
    ax_mask.set_title("Ground Truth")
    ax_mask.axis("off")

    # --- Bottom row: model predictions
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=outer_gs[1])

    for i, (name, pred) in enumerate(preds.items()):
        ax = fig.add_subplot(bottom_gs[0, i])
        ax.imshow(pred, cmap="gray")
        ax.set_title(f"{name.capitalize()}")
        ax.axis("off")

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

    os.makedirs(os.path.join(currentDirectory, config["skip_mode"]), exist_ok=True)

    # Setup loss
    criterion = losses.DiceLoss()

    # Initialize model
    # if we have best_unet_model.pth use it instead training:
    if os.path.exists(
        os.path.join(currentDirectory, config["skip_mode"], "best_unet_model.pth")
    ):
        model = UNet.UNet(
            in_channels=3, out_channels=1, skipMode=config["skip_mode"]
        ).to(device)
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

        training_time = 0

    else:
        model = UNet.UNet(
            in_channels=3, out_channels=1, skipMode=config["skip_mode"]
        ).to(device)

        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

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
                    os.path.join(
                        currentDirectory, config["skip_mode"], "best_unet_model.pth"
                    ),
                )
                print("Best model saved!")

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

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

        # Clear optimizer gradients
        optimizer.zero_grad()

    # Collect gradient flow and memory usage
    gradient_norms = {}
    model.train()
    images, masks = next(iter(trainLoader))
    images, masks = images.to(device), masks.to(device)

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

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_ious, val_ious)

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

    models = {}

    # Run experiments for each mode
    for mode in ["concat", "add", "attention"]:
        results[mode] = main(skip_mode=mode, **config)
        models[mode] = UNet.UNet(in_channels=3, out_channels=1, skipMode=mode).to(
            device
        )
        models[mode].load_state_dict(
            torch.load(
                os.path.join(currentDirectory, mode, "best_unet_model.pth"),
                map_location=device,
            )
        )

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

    visualize_predictions(models, get_dataloaders(config)[1], device)

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
