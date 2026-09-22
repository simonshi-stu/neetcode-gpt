import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / (fan_in + fan_out)) ** 0.5
        W = torch.randn(fan_out, fan_in) * std
        return torch.round(W, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / fan_in) ** 0.5
        W = torch.randn(fan_out, fan_in) * std
        return torch.round(W, decimals = 4).tolist() 

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        biases = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            if init_type == 'xavier':
                std = (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':
                std = (2.0 / fan_in) ** 0.5
            else:  # 'random'
                std = 1.0
            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)
            biases.append(torch.zeros(fan_out))
            
        x = torch.randn(1, input_dim)
        stds = []
        for i in range(num_layers):
            W = weights[i]
            b = biases[i]
            x = x @ W.T + b
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
        return stds


