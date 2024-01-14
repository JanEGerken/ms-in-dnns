import torch


def print_cuda_status():
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        print(f"found {n_devices}")
        for i in range(n_devices):
            print(f"device {i}:\t{torch.cuda.get_device_properties(i)}")
    else:
        print("no cuda devices found")
