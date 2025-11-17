import lab04_student
import numpy as np
import random
import torch
import json
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))


def ablationSkipConnections(addNew: bool, createGraph: bool) -> None:
    """
    Function to run ablation study on different skip connection variants.

    Saves results in JSON files in the ablation/skipConnections directory.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - None
    """
    dataFolder = os.path.join(currentDirectory, "ablation", "skipConnections")

    # Make sure the data folder exists
    os.makedirs(dataFolder, exist_ok=True)

    files = os.listdir(dataFolder)

    variants = ["concat", "add", "attention", "none"]

    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 50,
        "image_size": 128,
        "storeData": False,
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

            data = lab04_student.main(skip_mode=v, **config)

            # Save data to json
            with open(os.path.join(dataFolder, f"{v}_{seed}.json"), "w") as outfile:
                json.dump(data, outfile)

            # Reset the seed
            random.seed()
            np.random.seed()
            torch.manual_seed(torch.initial_seed())


def main(addNew: bool = False, createGraph: bool = True) -> None:
    """
    Main function to run ablation studies.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - None
    """
    ablationSkipConnections(addNew, createGraph)


if __name__ == "__main__":
    while True:
        main(addNew=True, createGraph=False)
