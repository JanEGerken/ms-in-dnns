import os
import sys
import argparse

import pytorch_lightning as pl

from hello_world import utils

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def main(args):
    print("Hello World")
    print(f"Found Lightning version {pl.__version__}")
    print("Got the following texts from the command line")
    print(args.text1)
    print(args.text2)
    utils.print_cuda_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text1", type=str, required=True)
    parser.add_argument("--text2", type=str, required=True)
    args = parser.parse_args()
    main(args)
