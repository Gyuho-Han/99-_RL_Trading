"""CNN 네트워크 — 1D 컨볼루션으로 시계열 특성 추출."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("cnn")
class CNNNetwork(BaseNetwork):

    def __init__(self, input_dim: int, num_filters: int = 64,
                 kernel_size: int = 3, seq_len: int = 20,
                 dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.num_filters = num_filters
        self.seq_len = seq_len
        self._output_dim = num_filters

        self.features_per_step = input_dim // seq_len

        # Three Conv1d layers with increasing filters, BatchNorm, ReLU
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(self.features_per_step, num_filters // 2, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Conv1d(num_filters // 2, num_filters, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Conv1d(num_filters, num_filters, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, flat_input_dim) or (batch, seq_len, features)
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, features_per_step, seq_len)

        x = self.conv_layers(x)  # (batch, num_filters, seq_len)

        # Global average pooling over the time dimension
        x = x.mean(dim=2)  # (batch, num_filters)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
