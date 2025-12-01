"""
Era II Methods Comparison Script
Lecture 11: 3D Scene Understanding and Neural Rendering

This script compares 3D ShapeNets vs Multi-View CNN on the same synthetic dataset.
Useful for understanding the trade-offs between the two approaches.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time

# Import from our demos
import sys

sys.path.append(".")

# We'll create simplified versions for comparison


# ============================================================================
# Comparison Metrics
# ============================================================================


def measure_inference_time(model, dataloader, device, num_batches=10):
    """Measure inference time."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            data = data.to(device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            _ = model(data)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()

            times.append(end - start)

    return np.mean(times), np.std(times)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory_usage(model, input_shape, device):
    """Measure memory usage (approximate)."""
    model.eval()

    # Create dummy input
    if len(input_shape) == 5:  # Multi-view: [B, V, C, H, W]
        dummy_input = torch.randn(*input_shape).to(device)
    else:  # Voxel: [B, C, D, H, W]
        dummy_input = torch.randn(*input_shape).to(device)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0  # Can't measure on CPU easily

    return memory_mb


# ============================================================================
# Comparison Results Visualization
# ============================================================================


def plot_comparison_results(results, save_path="comparison_results.png"):
    """
    Plot comparison between 3D ShapeNets and Multi-View CNN.

    Args:
        results: Dictionary with metrics for both methods
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    methods = list(results.keys())
    colors = ["steelblue", "coral"]

    # 1. Accuracy Comparison
    ax = axes[0, 0]
    train_accs = [results[m]["train_acc"][-1] for m in methods]
    val_accs = [results[m]["val_acc"][-1] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, train_accs, width, label="Train", color=colors)
    ax.bar(x + width / 2, val_accs, width, label="Val", alpha=0.7, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training Curves
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        epochs = range(1, len(results[method]["train_acc"]) + 1)
        ax.plot(
            epochs,
            results[method]["train_acc"],
            label=f"{method} (train)",
            color=colors[i],
            linestyle="-",
        )
        ax.plot(
            epochs,
            results[method]["val_acc"],
            label=f"{method} (val)",
            color=colors[i],
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Model Size (Parameters)
    ax = axes[0, 2]
    params = [results[m]["num_params"] / 1e6 for m in methods]
    ax.bar(methods, params, color=colors)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.grid(True, alpha=0.3)

    # 4. Inference Time
    ax = axes[1, 0]
    inf_times = [results[m]["inference_time"] * 1000 for m in methods]  # Convert to ms
    inf_stds = [results[m]["inference_std"] * 1000 for m in methods]
    ax.bar(methods, inf_times, yerr=inf_stds, color=colors, capsize=5)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time per Batch")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.grid(True, alpha=0.3)

    # 5. Memory Usage
    ax = axes[1, 1]
    memory = [results[m]["memory_mb"] for m in methods]
    ax.bar(methods, memory, color=colors)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.grid(True, alpha=0.3)

    # 6. Summary Table
    ax = axes[1, 2]
    ax.axis("off")

    # Create summary text
    summary_text = "Summary Comparison\n" + "=" * 40 + "\n\n"

    for method in methods:
        summary_text += f"{method}:\n"
        summary_text += f"  Val Acc: {results[method]['val_acc'][-1]:.2f}%\n"
        summary_text += f"  Params: {results[method]['num_params']/1e6:.2f}M\n"
        summary_text += f"  Time: {results[method]['inference_time']*1000:.2f}ms\n"
        summary_text += f"  Memory: {results[method]['memory_mb']:.1f}MB\n\n"

    # Determine winner
    best_acc_idx = np.argmax([results[m]["val_acc"][-1] for m in methods])
    best_speed_idx = np.argmin([results[m]["inference_time"] for m in methods])
    best_memory_idx = np.argmin([results[m]["memory_mb"] for m in methods])

    summary_text += "Winners:\n"
    summary_text += f"  Accuracy: {methods[best_acc_idx]}\n"
    summary_text += f"  Speed: {methods[best_speed_idx]}\n"
    summary_text += f"  Memory: {methods[best_memory_idx]}\n"

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {save_path}")


def print_comparison_table(results):
    """Print comparison table to console."""
    methods = list(results.keys())

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<25} {methods[0]:<20} {methods[1]:<20}")
    print("-" * 80)

    # Accuracy
    print(
        f"{'Train Accuracy (%)':<25} "
        f"{results[methods[0]]['train_acc'][-1]:<20.2f} "
        f"{results[methods[1]]['train_acc'][-1]:<20.2f}"
    )
    print(
        f"{'Val Accuracy (%)':<25} "
        f"{results[methods[0]]['val_acc'][-1]:<20.2f} "
        f"{results[methods[1]]['val_acc'][-1]:<20.2f}"
    )

    # Model size
    print(
        f"{'Parameters (M)':<25} "
        f"{results[methods[0]]['num_params']/1e6:<20.2f} "
        f"{results[methods[1]]['num_params']/1e6:<20.2f}"
    )

    # Speed
    print(
        f"{'Inference Time (ms)':<25} "
        f"{results[methods[0]]['inference_time']*1000:<20.2f} "
        f"{results[methods[1]]['inference_time']*1000:<20.2f}"
    )

    # Memory
    print(
        f"{'GPU Memory (MB)':<25} "
        f"{results[methods[0]]['memory_mb']:<20.1f} "
        f"{results[methods[1]]['memory_mb']:<20.1f}"
    )

    print("=" * 80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 80)

    best_acc = max(methods, key=lambda m: results[m]["val_acc"][-1])
    best_speed = min(methods, key=lambda m: results[m]["inference_time"])
    best_memory = min(methods, key=lambda m: results[m]["memory_mb"])

    print(f"• For best accuracy: {best_acc}")
    print(f"• For fastest inference: {best_speed}")
    print(f"• For lowest memory: {best_memory}")

    print("\nKEY INSIGHTS:")
    print("-" * 80)

    if "3D ShapeNets" in methods[0]:
        print("• 3D ShapeNets: Direct 3D processing, better for shape completion")
        print("• Multi-View CNN: Better accuracy, leverages 2D pre-training")

    print(
        "• Multi-View CNN typically achieves higher accuracy due to ImageNet pre-training"
    )
    print("• 3D ShapeNets provides explicit 3D structure, useful for generation tasks")
    print("• Trade-off: accuracy vs. interpretability vs. computational cost")

    print("=" * 80 + "\n")


# ============================================================================
# Main Comparison
# ============================================================================


def main(args):
    print("=" * 80)
    print("Era II Methods Comparison")
    print("3D ShapeNets vs Multi-View CNN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # For actual comparison, you would:
    # 1. Train both models on the same dataset
    # 2. Collect metrics during training
    # 3. Measure inference time and memory
    # 4. Generate comparison plots

    # Here's a template for the results dictionary:
    results = {
        "3D ShapeNets": {
            "train_acc": [],  # Will be filled during training
            "val_acc": [],
            "num_params": 0,
            "inference_time": 0,
            "inference_std": 0,
            "memory_mb": 0,
        },
        "Multi-View CNN": {
            "train_acc": [],
            "val_acc": [],
            "num_params": 0,
            "inference_time": 0,
            "inference_std": 0,
            "memory_mb": 0,
        },
    }

    print("\nTo run a full comparison:")
    print("1. Train 3D ShapeNets:")
    print("   python 3d_shapenets_demo.py --task classification --epochs 30")
    print("\n2. Train Multi-View CNN:")
    print("   python multiview_cnn_demo.py --epochs 30")
    print("\n3. This script will load both models and compare them")

    # Example: Create dummy results for visualization
    if args.demo:
        print("\n[DEMO MODE] Generating example comparison...")

        # Synthetic training curves
        epochs = 30
        for epoch in range(epochs):
            # 3D ShapeNets (lower accuracy, more overfitting)
            results["3D ShapeNets"]["train_acc"].append(
                min(95, 50 + epoch * 1.5 + np.random.randn() * 2)
            )
            results["3D ShapeNets"]["val_acc"].append(
                min(85, 45 + epoch * 1.3 + np.random.randn() * 2)
            )

            # Multi-View CNN (higher accuracy, better generalization)
            results["Multi-View CNN"]["train_acc"].append(
                min(98, 60 + epoch * 1.3 + np.random.randn() * 2)
            )
            results["Multi-View CNN"]["val_acc"].append(
                min(92, 55 + epoch * 1.2 + np.random.randn() * 2)
            )

        # Other metrics
        results["3D ShapeNets"]["num_params"] = 5.2e6
        results["3D ShapeNets"]["inference_time"] = 0.025
        results["3D ShapeNets"]["inference_std"] = 0.003
        results["3D ShapeNets"]["memory_mb"] = 1200

        results["Multi-View CNN"]["num_params"] = 11.2e6
        results["Multi-View CNN"]["inference_time"] = 0.035
        results["Multi-View CNN"]["inference_std"] = 0.004
        results["Multi-View CNN"]["memory_mb"] = 1800

        # Generate plots
        plot_comparison_results(results)
        print_comparison_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare 3D ShapeNets vs Multi-View CNN"
    )

    parser.add_argument(
        "--demo", action="store_true", help="Run in demo mode with synthetic results"
    )
    parser.add_argument(
        "--shapenets_model",
        type=str,
        default="shapenets_classifier.pth",
        help="Path to trained 3D ShapeNets model",
    )
    parser.add_argument(
        "--multiview_model",
        type=str,
        default="multiview_cnn_best.pth",
        help="Path to trained Multi-View CNN model",
    )

    args = parser.parse_args()

    main(args)
