"""
Lab 3: Architecture Implementation Mastery - Student Template
VIAR25/26 - Artificial Vision - UVigo
Prof. David Olivieri

This template provides the structure for implementing:
1. ResNet architectures (Basic and Bottleneck blocks)
2. Attention mechanisms (SE and CBAM)
3. Vision Transformer components
4. Benchmarking and comparison tools

TODO: Complete all functions marked with "TODO" comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings

import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import time

import os

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Use the current working directory in Colab
currentDirectory = os.getcwd()


warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: RESNET IMPLEMENTATION
# ============================================================================


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions
    Used in ResNet-18 and ResNet-34
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()

        # Implement the basic block
        # Hint: You need:
        # - First 3x3 conv with given stride
        # - BatchNorm + ReLU
        # - Second 3x3 conv with stride=1
        # - BatchNorm (no ReLU here)
        # - Store downsample for skip connection

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement forward pass
        # Remember: output = F(x) + x (or F(x) + downsample(x))

        identity = x.clone()

        # Apply first conv + bn + relu
        out = self.bn1(self.conv1(x))

        # Apply second conv + bn (no relu)
        out = self.bn2(self.conv2(out))

        # Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Add identity and apply final ReLU
        out += identity
        out = F.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with 1x1 -> 3x3 -> 1x1 convolutions
    Used in ResNet-50, ResNet-101, ResNet-152
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BottleneckBlock, self).__init__()

        # Implement the bottleneck block
        # Hint: You need:
        # - 1x1 conv to reduce channels (in_channels -> out_channels)
        # - 3x3 conv with given stride (out_channels -> out_channels)
        # - 1x1 conv to expand channels (out_channels -> out_channels * expansion)
        # - BatchNorm + ReLU after each conv (except last)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement forward pass for bottleneck block
        identity = x

        # Apply conv1 + bn1 + relu
        out = self.relu(self.bn1(self.conv1(x)))

        # Apply conv2 + bn2 + relu
        out = self.relu(self.bn2(self.conv2(out)))

        # Apply conv3 + bn3 (no relu)
        out = self.bn3(self.conv3(out))

        # Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)

            # Add identity and apply final ReLU
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture that can be configured for different depths
    """

    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 10,
        input_channels: int = 3,
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Implement initial layers
        # Hint: Start with 7x7 conv, bn, relu, maxpool for ImageNet
        # For CIFAR: use 3x3 conv, bn, relu (no maxpool)

        self.conv1 = nn.Conv2d(
            input_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,  # Changed stride from 2 to 1 for CIFAR
            padding=1,  # Changed padding from 3 to 1 for CIFAR
            bias=False,
        )
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True)
        )
        self.maxpool = None  # Removed maxpool for CIFAR

        # Implement residual layers
        # Hint: Use _make_layer helper function
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: nn.Module, out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks
        """
        downsample = None

        # Create downsample layer if needed
        # Hint: Need downsample when stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # Add first block with potential downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion

        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement forward pass through entire network

        # Initial layers
        x = self.bn1(self.conv1(x))
        if self.maxpool is not None:
            x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Use flatten instead of squeeze
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 10) -> ResNet:
    """Create ResNet-18"""
    # Return ResNet with BasicBlock and [2, 2, 2, 2] layers
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 10) -> ResNet:
    """Create ResNet-34"""
    # Return ResNet with BasicBlock and [3, 4, 6, 3] layers
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 10) -> ResNet:
    """Create ResNet-50"""
    # Return ResNet with BottleneckBlock and [3, 4, 6, 3] layers
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes: int = 10) -> ResNet:
    """Create ResNet-101"""
    # Return ResNet with BottleneckBlock and [3, 4, 23, 3] layers
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)


class ModelProfiler:
    """
    Tool for profiling model performance
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        # TODO: Implement parameter counting
        return 0

    def measure_flops(self, input_size: Tuple[int, ...]) -> int:
        """Estimate FLOPs for forward pass"""
        # TODO: Implement FLOP counting
        # Hint: You can use torchprofile library or implement manually
        return 0

    def measure_memory(
        self, batch_size: int, input_size: Tuple[int, ...]
    ) -> Dict[str, float]:
        """Measure memory usage during forward/backward pass"""
        # TODO: Implement memory profiling
        # Hint: Use torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated()
        return {"forward": 0.0, "backward": 0.0, "peak": 0.0}

    def measure_latency(
        self, input_size: Tuple[int, ...], batch_size: int = 1, num_runs: int = 100
    ) -> float:
        """Measure inference latency"""
        # TODO: Implement latency measurement
        # Hint: Average over multiple runs, use torch.cuda.synchronize()
        return 0.0

    def benchmark_training(
        self, dataloader: DataLoader, epochs: int = 1
    ) -> Dict[str, float]:
        """Benchmark training performance"""
        # TODO: Implement training benchmark
        # Hint: Measure time per epoch, convergence rate
        return {"time_per_epoch": 0.0, "samples_per_second": 0.0}


class ArchitectureComparator:
    """
    Compare different architectures on the same task
    """

    def __init__(self, models: Dict[str, nn.Module], device: torch.device):
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        self.results = {}

    def compare_accuracy(self, test_loader: DataLoader) -> Dict[str, float]:
        """Compare test accuracy of all models"""
        # TODO: Implement accuracy comparison
        # Hint: Evaluate each model on test set
        accuracies = {}
        for name, model in self.models.items():
            # TODO: Evaluate model and store accuracy
            accuracies[name] = 0.0
        return accuracies

    def compare_efficiency(
        self, input_size: Tuple[int, ...]
    ) -> Dict[str, Dict[str, float]]:
        """Compare computational efficiency of all models"""
        # TODO: Implement efficiency comparison
        # Hint: Use ModelProfiler for each model
        efficiency = {}
        for name, model in self.models.items():
            profiler = ModelProfiler(model, self.device)
            # TODO: Profile each model
            efficiency[name] = {
                "parameters": 0,
                "flops": 0,
                "latency": 0.0,
                "memory": 0.0,
            }
        return efficiency

    def plot_comparison(self, save_path: Optional[str] = None):
        """Create visualization comparing all models"""
        # TODO: Implement comparison plots
        # Hint: Create subplots for accuracy, efficiency metrics
        pass


def download_cifar_dataset():
    """Download CIFAR-10 dataset using torchvision"""
    print("Downloading CIFAR-10 dataset...")
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Download training data
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Download test data
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    print("CIFAR-10 dataset downloaded.")


def create_data_loaders(
    dataset_name: str = "CIFAR10",
    batch_size: int = 128,
    num_batches: int = 5,
    train_split: float = 0.9,
    subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders"""
    # TODO: Implement data loader creation
    # Hint: Support CIFAR-10, CIFAR-100, and subset functionality

    # TODO: Define transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # TODO: Load datasets
    if dataset_name == "CIFAR10":
        # Load the dataset using torchvision
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=False, transform=transform_test
        )

        # Split training data into training and validation sets
        train_size = int(train_split * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            trainset, [train_size, val_size]
        )

        # Use the original testset for testing
        test_dataset = testset

    elif dataset_name == "CIFAR100":
        train_dataset = None
        val_dataset = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # TODO: Create subset if specified
    if subset_size is not None:
        # Create a subset of the training dataset
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, subset_indices)

    # TODO: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    save_model=False,
    model_name="resnet",
) -> Dict[str, List[float]]:
    """Train a model and return training history"""
    # TODO: Implement training loop
    # Hint: Include validation, learning rate scheduling, early stopping

    model.to(device)

    # TODO: Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = None  # Optional: learning rate scheduler

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_time": 0.0,
    }
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)  # Mean loss times batch size
            _, predicted = torch.max(
                outputs.data, 1
            )  # (max_value, index of class with max value)
            train_total += target.size(0)  # Total number of samples
            train_correct += (predicted == target).sum().item()

        train_acc = 100 * train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_acc = 100 * val_correct / val_total
        history["train_loss"].append(train_loss / train_total)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/train_total:.2f}, Train_Accuracy: {train_acc:.2f}%, Val_Accuracy: {val_acc:.2f}%"
        )
        if scheduler is not None:
            scheduler.step()

        # Save the best model
        if save_model and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

    history["train_time"] = time.time() - start_time

    return history


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================


def test_resnet_implementation(models):
    """Test ResNet implementation"""
    print("Testing ResNet implementation...")

    test_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 input size

    for name, model in models.items():
        if model is not None:
            try:
                output = model(test_input)
                print(f"✓ {name}: Output shape {output.shape}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
        else:
            print(f"✗ {name}: Not implemented")

    # Load CIFAR-10 data
    train_loader, val_loader, test_loader = create_data_loaders()

    for name, model in models.items():
        if model is not None:

            match name:
                case "ResNet-18":
                    num_epochs = 20
                case "ResNet-34":
                    num_epochs = 40
                case "ResNet-50":
                    num_epochs = 60
                case "ResNet-101":
                    num_epochs = 80

            print(f"\nTraining {name} on CIFAR-10...")
            history = train_model(
                model,
                train_loader,
                val_loader,
                epochs=num_epochs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_model=True,
                model_name=name,
            )
            print(
                f"Training complete for {name}. Final Val Accuracy: {history['val_acc'][-1]:.2f}%"
            )

            # save history to file
            with open(f"{name}_history.pkl", "wb") as f:
                pickle.dump(history, f)


def plot_history_from_pkl(pkl_file, output_image, title, log_scale=False):
    # Load the training history from the pickle file
    with open(pkl_file, "rb") as f:
        history = pickle.load(f)

    print(history.keys())

    # Extract accuracy and loss values using correct keys
    acc = history["train_acc"]
    val_acc = history["val_acc"]
    loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if log_scale:
        plt.yscale("log")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if log_scale:
        plt.yscale("log")

    # Save the plots to a file
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()


def evaluate_model(model, test_loader, device, name):
    model.load_state_dict(torch.load(f"{name}_best.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the model {model} on the test images: {accuracy:.2f}%")
    return accuracy


def confusion_matrix(model, test_loader, device, name, num_classes=10):
    model.load_state_dict(torch.load(f"{name}_best.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    print("saving confusion matrix as confusion_matrix.png")
    plt.savefig(f"confusion_matrix_{name}.png")


def main():
    """Main function to run tests and example experiments"""
    print("Lab 3: Architecture Implementation Mastery")
    print("=" * 50)

    # download cifar dataset
    download_cifar_dataset()

    # # TODO: Create test cases for different ResNet variants
    # models = {
    #     "ResNet-18": resnet18(num_classes=10),
    #     "ResNet-34": resnet34(num_classes=10),
    # }

    # # Test implementations
    # test_resnet_implementation(models)

    # for model in models:
    #     print(f"Testing {model} implementation...")
    #     test_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 input size
    #     output = models[model](test_input)
    #     print(f"✓ {model}: Output shape {output.shape}")

    #     # call conf matrix and eval
    #     train_loader, val_loader, test_loader = create_data_loaders()
    #     evaluate_model(models[model], test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), model)
    #     confusion_matrix(models[model], test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), model)

    # bigger models
    models = {
        "ResNet-50": resnet50(num_classes=10),
        "ResNet-101": resnet101(num_classes=10),
    }

    test_resnet_implementation(models)

    for model in models:
        print(f"Testing {model} implementation...")
        test_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 input size
        output = models[model](test_input)
        print(f"✓ {model}: Output shape {output.shape}")

        # call conf matrix and eval
        train_loader, val_loader, test_loader = create_data_loaders()
        evaluate_model(
            models[model],
            test_loader,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model,
        )
        confusion_matrix(
            models[model],
            test_loader,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model,
        )

    print("\n" + "=" * 50)
    print("Implementation tests complete!")
    print("TODO: Complete the implementations and run your experiments")

    # TODO: Add example experiments here
    # Example:
    # 1. Train ResNet-18 vs ResNet-50 on CIFAR-10
    # 2. Compare with and without attention mechanisms
    # 3. Train ViT and compare with CNNs
    # 4. Create visualizations and analysis


if __name__ == "__main__":
    main()
