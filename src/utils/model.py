import os
import sys
import torch
from torch import nn, optim
from typing import List
import torch.nn.functional as F

sys.path.append('..')
from src.config import *


class Network(nn.Module):
    def __init__(self, base_model, dropout: float, output_dims: List[int]) -> None:
        super().__init__()

        self.base_model = base_model
        input_dim: int = base_model.classifier[1].in_features

        layers: List[nn.Module] = []
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, NUM_CLASSES))
        layers.append(nn.Softmax(dim=-1))

        self.base_model.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)