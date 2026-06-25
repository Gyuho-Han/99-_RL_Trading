"""LSTM 네트워크 — 시계열 패턴 학습용."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("lstm")
class LSTMNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 seq_len: int = 20, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        # input_dim is flat (seq_len * n_features); recover feature dim
        self.features_per_step = input_dim // seq_len

        self.lstm = nn.LSTM(
            input_size=self.features_per_step,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, flat_input_dim) or (batch, seq_len, features)
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.layer_norm(out)
        return out

    @property
    def output_dim(self) -> int:
        return self._output_dim
