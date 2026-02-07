import os
import json
import pathlib as pl

import torch


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


def load_pretrained(model, ckpt_path):
    """Load pretrained weights into model."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
