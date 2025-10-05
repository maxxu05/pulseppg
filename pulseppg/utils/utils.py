import os
import numpy as np
import torch
import random
from datetime import datetime
from threading import Timer
import time
from tqdm import tqdm


def printlog(line, path, type="a", dontwrite=False):
    line = f"{datetime.now().strftime('%y/%m/%d %H:%M')} | " + line
    # we have to use tqdm.write() to not interfere with tqdm code...
    tqdm.write(line)
    if not dontwrite:
        with open(os.path.join(path, "log.txt"), type) as file:
            file.write(line + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_dl_program(config, device_name, seed=42, use_tf32=False, max_threads=None):
    set_seed(seed)

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl

            mkl.set_num_threads(max_threads)
        except:
            pass

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == "cuda":
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
    devices.reverse()

    # # unfortunately, with dilated convolutions these are too slow to be enabled
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    config.set_device(devices if len(devices) > 1 else devices[0])


# # remove pretty table functionality for our codebase
# from prettytable import PrettyTable
def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    # for name, param in model.named_parameters():
    #     print(f"{name} requires_grad: {param.requires_grad}")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    # return table, total_params
    return None, total_params
