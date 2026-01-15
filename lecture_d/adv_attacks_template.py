"""
Adversarial Attacks on CIFAR10 - Assignment D Task 3

This script computes adversarial examples for the pre-trained SimpleCIFARNet classifier.

TODO: Complete the implementation as described in the assignment.
"""
import os
import sys
import pathlib as pl
from datetime import datetime
from argparse import ArgumentParser

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from cifar10_simple import SimpleCIFARNet, CLASS_NAMES, get_cifar10_dataloaders
from cifar10_simple.model import load_pretrained
from cifar10_simple.utils import get_wandb_key

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def get_attack_loader(val_dataset, source_class: str, n_samples: int) -> DataLoader:
    """
    Create a DataLoader containing n_samples images from the source_class.

    Args:
        val_dataset: CIFAR10 validation dataset
        source_class: Name of the source class (e.g., "airplane")
        n_samples: Number of samples to extract

    Returns:
        DataLoader with the selected samples (batch_size=1)
    """
    # TODO: Implement this function
    # 1. Find the numeric label for source_class using CLASS_NAMES
    # 2. Iterate through val_dataset to find n_samples images with that label
    # 3. Create a dataset for these samples
    # 4. Return a DataLoader with batch_size=1
    raise NotImplementedError("TODO: Implement get_attack_loader")


def compute_adversarial(
    model: SimpleCIFARNet,
    image: torch.Tensor,
    target_class: int,
    lr: float,
    max_iter: int,
    prob_threshold: float,
) -> tuple[torch.Tensor, float]:
    """
    Compute an adversarial example that fools the model into predicting target_class.

    Uses Adam optimizer to maximize the probability of the target class.

    Args:
        model: The classifier model
        image: Input image tensor (1, C, H, W)
        target_class: Target class index to fool the model towards
        lr: Learning rate for Adam optimizer
        max_iter: Maximum number of optimization iterations
        prob_threshold: Stop when target class probability exceeds this

    Returns:
        Tuple of (adversarial_image, final_target_probability)
    """
    # TODO: Implement this function
    # 1. Clone the image and enable gradient computation
    # 2. Create an Adam optimizer with maximize=True
    # 3. Iterate up to max_iter times:
    #    a. Compute model output and softmax probabilities
    #    b. Get probability of target_class
    #    c. If probability > prob_threshold, stop
    #    d. Backpropagate and update
    # 4. Return the adversarial image and final probability
    raise NotImplementedError("TODO: Implement compute_adversarial")


def main(args):
    torch.manual_seed(0xDEADBEEF)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."

    # Initialize WandB
    wandb.login(key=get_wandb_key())
    wandb.init(
        project="ms-in-dnns-cifar10-adv-attacks",
        config=vars(args),
        name=args.run_name,
        dir=wandb_save_dir,
    )

    # Load data and model
    _, val_loader = get_cifar10_dataloaders(args.data_root, batch_size=1)
    val_dataset = val_loader.dataset

    model = load_pretrained(args.ckpt_path, device=device)
    model.eval()

    # Get samples from source class
    attack_loader = get_attack_loader(val_dataset, args.source_class, args.n_samples)

    # Create results table
    result_tbl = wandb.Table(
        columns=["source_image", "gt_class", "target_class", "adversary", "diff", "target_prob"]
    )

    # TODO: Complete the main attack loop
    # For each sample in attack_loader:
    #   For each target_class in range(10):
    #     1. Compute adversarial example using compute_adversarial()
    #     2. Compute the difference image (rescaled for visualization)
    #     3. Add row to result_tbl with:
    #        - source_image: original image as wandb.Image
    #        - gt_class: args.source_class
    #        - target_class: CLASS_NAMES[target_class]
    #        - adversary: adversarial image as wandb.Image
    #        - diff: difference image as wandb.Image
    #        - target_prob: final probability

    # Log results
    wandb.log({"adversaries": result_tbl})
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        data_root = str(pl.PurePosixPath("/gcs", bucket_name, "cifar10_data"))
    else:
        data_root = str(pl.PurePath("..", "data", "cifar10"))
    parser.add_argument("--data-root", type=str, default=data_root)

    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)

    parser.add_argument(
        "--ckpt-path", type=str, required=True, help="Path to pre-trained SimpleCIFARNet checkpoint"
    )
    parser.add_argument(
        "--source-class",
        type=str,
        choices=CLASS_NAMES,
        required=True,
        help="Class of samples to attack",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5, help="Number of samples to start attack from"
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Maximum optimization iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.99,
        help="Target probability threshold to stop optimization",
    )

    args = parser.parse_args()
    main(args)
