from typing import Union
import torch
from torch import nn
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
):
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(nn.Identity())

    mlp = nn.Sequential(*layers)
    mlp.to(device)
    return mlp


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)