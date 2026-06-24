"""DNN (Deep Neural Network) — 완전연결 피드포워드 네트워크."""

from typing import List

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("dnn")
class DNN(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 **kwargs):
        super().__init__(input_dim, **kwargs)
        hidden_dims = hidden_dims or [128, 64]
        self._output_dim = hidden_dims[-1]

        # Linear + ReLU 레이어 구성
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3D 입력 (batch, window_size, features) → 2D로 flatten
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.layers(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
