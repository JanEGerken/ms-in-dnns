"""Utility functions for medmnist_equivariant package."""
import os
import json
import pathlib as pl


def get_wandb_key():
    """
    Get WandB API key from file or environment variable.

    Looks for wandb_key.json in parent directory first, then checks WANDB_KEY env var.

    Returns:
        WandB API key string
    """
    json_file = str(pl.PurePath("..", "wandb_key.json"))
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def args_to_flat_dict(args):
    """Convert argparse namespace to flat dict for logging."""
    args_dict = vars(args)
    for key in args_dict.keys():
        if args_dict[key] is None:
            args_dict[key] = "None"
    return args_dict
