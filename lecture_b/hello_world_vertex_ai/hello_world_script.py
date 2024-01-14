import os
import sys
import argparse

import torch
import pytorch_lightning as pl

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def print_cuda_status():
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        print(f"found {n_devices}")
        for i in range(n_devices):
            print(f"device {i}:\t{torch.cuda.get_device_properties(i)}")
    else:
        print("no cuda devices found")


def main(args):
    print("Hello World")
    print(f"Found Lightning version {pl.__version__}")
    print("Got the following texts from the command line")
    print(args.text1)
    print(args.text2)
    print_cuda_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text1", type=str, required=True)
    parser.add_argument("--text2", type=str, required=True)
    args = parser.parse_args()
    main(args)
