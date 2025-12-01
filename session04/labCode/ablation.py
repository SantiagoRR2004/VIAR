import matplotlib.pyplot as plt
import lab04_student
import numpy as np
import random
import torch
import json
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))
graphFolder = os.path.join(currentDirectory, "graphs")

# Make sure the graph folder exists
os.makedirs(graphFolder, exist_ok=True)


def plotIndividually(
    blueAll, orangeAll, avgBlue, avgOrange, title, xlabel, ylabel, saveName
):
    """
    Helper function to plot individual figures for training and validation metrics.

    It won't show the plot but will save it directly to the specified file.

    Args:
        - blueAll (list): List of all blue metric runs (e.g., training).
        - orangeAll (list): List of all orange metric runs (e.g., validation).
        - avgBlue (list): Average of blue metric across runs.
        - avgOrange (list): Average of orange metric across runs.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - saveName (str): Name of the file to save the figure.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Replot on individual figure
    for i, values in enumerate(blueAll):
        ax.plot(values, color="blue", alpha=0.2, linewidth=0.8)
    for i, values in enumerate(orangeAll):
        ax.plot(values, color="orange", alpha=0.2, linewidth=0.8)
    ax.plot(avgBlue, label=f"Avg Train", color="blue", linewidth=2)
    ax.plot(avgOrange, label=f"Avg Val", color="orange", linewidth=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(graphFolder, saveName))
    plt.close(fig)


def ablationSkipConnections(addNew: bool, createGraph: bool) -> dict:
    """
    Function to run ablation study on different skip connection variants.

    Saves results in JSON files in the ablation/skipConnections directory.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - dict: Collected data from the experiments if createGraph is True, else None.
    """
    dataFolder = os.path.join(currentDirectory, "ablation", "skipConnections")

    # Make sure the data folder exists
    os.makedirs(dataFolder, exist_ok=True)

    files = os.listdir(dataFolder)

    variants = ["concat", "add", "attention", "none"]

    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 1000,
        "image_size": 128,
        "storeData": False,
        "features": [64, 128, 256, 512],
    }

    if addNew:
        for v in variants:
            # Get a seed
            seed = random.randint(100, 999)

            while f"{v}_{seed}.json" in files:
                seed = random.randint(100, 999)

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            generator = torch.Generator().manual_seed(seed)

            data = lab04_student.main(skip_mode=v, generator=generator, **config)

            # Save data to json
            with open(os.path.join(dataFolder, f"{v}_{seed}.json"), "w") as outfile:
                json.dump(data, outfile)

            # Reset the seed
            random.seed()
            np.random.seed()
            torch.manual_seed(torch.initial_seed())

    if createGraph:
        data = {v: [] for v in variants}

        for file in files:
            variant = file.split("_")[0]

            if variant in variants:
                try:
                    with open(os.path.join(dataFolder, file), "r") as infile:
                        runData = json.load(infile)
                    data[variant].append(runData)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Could not load {file}: {e}")
                    continue

        # Check if we have any data to plot
        if not any(data.values()):
            print(
                "No valid data found to create graphs. Please run with addNew=True first."
            )
            return

        # Show the average train_losses and val_losses - one figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, v in enumerate(variants):
            ax = axes[idx]

            allTrainLosses = [run["train_losses"] for run in data[v]]
            allValLosses = [run["val_losses"] for run in data[v]]

            # Plot individual runs with transparency
            for i, losses in enumerate(allTrainLosses):
                ax.plot(losses, color="blue", alpha=0.2, linewidth=0.8)
            for i, losses in enumerate(allValLosses):
                ax.plot(losses, color="orange", alpha=0.2, linewidth=0.8)

            # Find the maximum number of epochs across all runs
            maxEpochsTrain = max(len(losses) for losses in allTrainLosses)
            maxEpochsVal = max(len(losses) for losses in allValLosses)

            # Calculate mean for each epoch, considering only runs that have that epoch
            avgTrainLosses = []
            for epoch in range(maxEpochsTrain):
                epochLosses = [
                    losses[epoch] for losses in allTrainLosses if len(losses) > epoch
                ]
                avgTrainLosses.append(np.mean(epochLosses))

            avgValLosses = []
            for epoch in range(maxEpochsVal):
                epochLosses = [
                    losses[epoch] for losses in allValLosses if len(losses) > epoch
                ]
                avgValLosses.append(np.mean(epochLosses))

            # Plot averages with solid lines
            ax.plot(avgTrainLosses, label=f"Avg Train", color="blue", linewidth=2)
            ax.plot(avgValLosses, label=f"Avg Val", color="orange", linewidth=2)

            ax.set_title(f"{v.capitalize()} Variant", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save each subplot as an individual image
            plotIndividually(
                allTrainLosses,
                allValLosses,
                avgTrainLosses,
                avgValLosses,
                title=f"{v.capitalize()} Variant",
                xlabel="Epochs",
                ylabel="Loss",
                saveName=f"Variant{v}Loss.png",
            )

        fig.suptitle(
            "Training and Validation Loss - Skip Connection Variants",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save the combined figure
        fig.savefig(os.path.join(graphFolder, "SkipConnectionVariantsLoss.png"))

        # Show the average train_ious and val_ious - one figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, v in enumerate(variants):
            ax = axes[idx]

            allTrainIous = [run["train_ious"] for run in data[v]]
            allValIous = [run["val_ious"] for run in data[v]]

            # Plot individual runs with transparency
            for i, ious in enumerate(allTrainIous):
                ax.plot(ious, color="blue", alpha=0.2, linewidth=0.8)
            for i, ious in enumerate(allValIous):
                ax.plot(ious, color="orange", alpha=0.2, linewidth=0.8)

            # Find the maximum number of epochs across all runs
            maxEpochsTrain = max(len(ious) for ious in allTrainIous)
            maxEpochsVal = max(len(ious) for ious in allValIous)

            # Calculate mean for each epoch, considering only runs that have that epoch
            avgTrainIous = []
            for epoch in range(maxEpochsTrain):
                epochIous = [ious[epoch] for ious in allTrainIous if len(ious) > epoch]
                avgTrainIous.append(np.mean(epochIous))

            avgValIous = []
            for epoch in range(maxEpochsVal):
                epochIous = [ious[epoch] for ious in allValIous if len(ious) > epoch]
                avgValIous.append(np.mean(epochIous))

            # Plot averages with solid lines
            ax.plot(avgTrainIous, label=f"Avg Train", color="blue", linewidth=2)
            ax.plot(avgValIous, label=f"Avg Val", color="orange", linewidth=2)

            ax.set_title(f"{v.capitalize()} Variant", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("IoU")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save each subplot as an individual image
            plotIndividually(
                allTrainIous,
                allValIous,
                avgTrainIous,
                avgValIous,
                title=f"{v.capitalize()} Variant",
                xlabel="Epochs",
                ylabel="IoU",
                saveName=f"Variant{v}IoU.png",
            )

        fig.suptitle(
            "Training and Validation IoU - Skip Connection Variants",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save the combined figure
        fig.savefig(os.path.join(graphFolder, "SkipConnectionVariantsIoU.png"))

        return data


def ablationSkipConnectionsBigger(addNew: bool, createGraph: bool) -> dict:
    """
    Function to run ablation study on different skip connection variants.

    Saves results in JSON files in the ablation/skipConnectionsBigger directory.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - dict: Collected data from the experiments if createGraph is True, else None.
    """
    dataFolder = os.path.join(currentDirectory, "ablation", "skipConnectionsBigger")

    # Make sure the data folder exists
    os.makedirs(dataFolder, exist_ok=True)

    files = os.listdir(dataFolder)

    variants = ["concat", "add", "attention", "none"]

    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 1000,
        "image_size": 128,
        "storeData": False,
        "features": [64, 64, 128, 128, 256, 256, 512, 512],
    }

    if addNew:
        for v in variants:
            # Get a seed
            seed = random.randint(100, 999)

            while f"{v}_{seed}.json" in files:
                seed = random.randint(100, 999)

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            generator = torch.Generator().manual_seed(seed)

            data = lab04_student.main(skip_mode=v, generator=generator, **config)

            # Save data to json
            with open(os.path.join(dataFolder, f"{v}_{seed}.json"), "w") as outfile:
                json.dump(data, outfile)

            # Reset the seed
            random.seed()
            np.random.seed()
            torch.manual_seed(torch.initial_seed())

    if createGraph:
        data = {v: [] for v in variants}

        for file in files:
            variant = file.split("_")[0]

            if variant in variants:
                try:
                    with open(os.path.join(dataFolder, file), "r") as infile:
                        runData = json.load(infile)
                    data[variant].append(runData)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Could not load {file}: {e}")
                    continue

        # Check if we have any data to plot
        if not any(data.values()):
            print(
                "No valid data found to create graphs. Please run with addNew=True first."
            )
            return

        # Show the average train_losses and val_losses - one figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, v in enumerate(variants):
            ax = axes[idx]

            allTrainLosses = [run["train_losses"] for run in data[v]]
            allValLosses = [run["val_losses"] for run in data[v]]

            # Plot individual runs with transparency
            for i, losses in enumerate(allTrainLosses):
                ax.plot(losses, color="blue", alpha=0.2, linewidth=0.8)
            for i, losses in enumerate(allValLosses):
                ax.plot(losses, color="orange", alpha=0.2, linewidth=0.8)

            # Find the maximum number of epochs across all runs
            maxEpochsTrain = max(len(losses) for losses in allTrainLosses)
            maxEpochsVal = max(len(losses) for losses in allValLosses)

            # Calculate mean for each epoch, considering only runs that have that epoch
            avgTrainLosses = []
            for epoch in range(maxEpochsTrain):
                epochLosses = [
                    losses[epoch] for losses in allTrainLosses if len(losses) > epoch
                ]
                avgTrainLosses.append(np.mean(epochLosses))

            avgValLosses = []
            for epoch in range(maxEpochsVal):
                epochLosses = [
                    losses[epoch] for losses in allValLosses if len(losses) > epoch
                ]
                avgValLosses.append(np.mean(epochLosses))

            # Plot averages with solid lines
            ax.plot(avgTrainLosses, label=f"Avg Train", color="blue", linewidth=2)
            ax.plot(avgValLosses, label=f"Avg Val", color="orange", linewidth=2)

            ax.set_title(f"{v.capitalize()} Variant", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save each subplot as an individual image
            plotIndividually(
                allTrainLosses,
                allValLosses,
                avgTrainLosses,
                avgValLosses,
                title=f"{v.capitalize()} Variant",
                xlabel="Epochs",
                ylabel="Loss",
                saveName=f"Variant{v}LossBigger.png",
            )

        fig.suptitle(
            "Training and Validation Loss (Bigger) - Skip Connection Variants",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save the combined figure
        fig.savefig(os.path.join(graphFolder, "SkipConnectionVariantsLossBigger.png"))

        # Show the average train_ious and val_ious - one figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, v in enumerate(variants):
            ax = axes[idx]

            allTrainIous = [run["train_ious"] for run in data[v]]
            allValIous = [run["val_ious"] for run in data[v]]

            # Plot individual runs with transparency
            for i, ious in enumerate(allTrainIous):
                ax.plot(ious, color="blue", alpha=0.2, linewidth=0.8)
            for i, ious in enumerate(allValIous):
                ax.plot(ious, color="orange", alpha=0.2, linewidth=0.8)

            # Find the maximum number of epochs across all runs
            maxEpochsTrain = max(len(ious) for ious in allTrainIous)
            maxEpochsVal = max(len(ious) for ious in allValIous)

            # Calculate mean for each epoch, considering only runs that have that epoch
            avgTrainIous = []
            for epoch in range(maxEpochsTrain):
                epochIous = [ious[epoch] for ious in allTrainIous if len(ious) > epoch]
                avgTrainIous.append(np.mean(epochIous))

            avgValIous = []
            for epoch in range(maxEpochsVal):
                epochIous = [ious[epoch] for ious in allValIous if len(ious) > epoch]
                avgValIous.append(np.mean(epochIous))

            # Plot averages with solid lines
            ax.plot(avgTrainIous, label=f"Avg Train", color="blue", linewidth=2)
            ax.plot(avgValIous, label=f"Avg Val", color="orange", linewidth=2)

            ax.set_title(f"{v.capitalize()} Variant", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("IoU")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save each subplot as an individual image
            plotIndividually(
                allTrainIous,
                allValIous,
                avgTrainIous,
                avgValIous,
                title=f"{v.capitalize()} Variant",
                xlabel="Epochs",
                ylabel="IoU",
                saveName=f"Variant{v}IoUBigger.png",
            )

        fig.suptitle(
            "Training and Validation IoU (Bigger) - Skip Connection Variants",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save the combined figure
        fig.savefig(os.path.join(graphFolder, "SkipConnectionVariantsIoUBigger.png"))

        return data


def oneSeedEpochGraph(data: dict) -> None:
    """
    Create box plots for training_time, avg_gradient, and memory_usage
    for different skip connection variants.

    Args:
        - data (dict): Collected data from the experiments.

    Returns:
        - None
    """
    variants = list(data.keys())
    legendVars = [v.title() for v in variants]

    # Create a box plot for training_time
    plt.figure(figsize=(10, 6))
    training_times = [[run["training_time"] for run in data[v]] for v in variants]
    plt.boxplot(training_times, labels=legendVars)
    plt.title("Training Time by Skip Connection Variant")
    plt.ylabel("Training Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphFolder, "TrainingTime.png"))

    # Create a box plot for avg_gradient
    plt.figure(figsize=(10, 6))
    avg_gradients = [[run["avg_gradient"] for run in data[v]] for v in variants]
    plt.boxplot(avg_gradients, labels=legendVars)
    plt.title("Average Gradient by Skip Connection Variant")
    plt.ylabel("Average Gradient")
    plt.tight_layout()
    plt.savefig(os.path.join(graphFolder, "AvgGradient.png"))

    # Create a box plot for memory_usage
    plt.figure(figsize=(10, 6))
    memory_usages = [[run["memory_usage"] for run in data[v]] for v in variants]
    plt.boxplot(memory_usages, labels=legendVars)
    plt.title("Memory Usage by Skip Connection Variant")
    plt.ylabel("Memory Usage (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphFolder, "MemoryUsage.png"))


def main(addNew: bool = False, createGraph: bool = True) -> None:
    """
    Main function to run ablation studies.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - None
    """
    data = ablationSkipConnections(addNew, createGraph)
    dataBigger = ablationSkipConnectionsBigger(addNew, createGraph)

    if createGraph:
        for variant in dataBigger.keys():
            data[f"{variant} Bigger"] = dataBigger[variant]

        oneSeedEpochGraph(data)


if __name__ == "__main__":
    while True:
        main(addNew=True, createGraph=False)
