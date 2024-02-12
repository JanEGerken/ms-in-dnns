import os
import json
import pathlib as pl


def get_wandb_key():
    json_file = str(pl.PurePath("..", "wandb_key.json"))
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def args_to_flat_dict(args):
    args_dict = vars(args.as_flat())
    for key in args_dict.keys():
        if args_dict[key] is None:
            args_dict[key] = "None"
    return args_dict
