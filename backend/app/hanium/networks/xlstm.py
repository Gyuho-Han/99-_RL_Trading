"""xLSTM (Extended LSTM) — 지수 게이팅 + 행렬 메모리.

2026년 벤치마크 2위 (하락장 방어 최고).
기존 LSTM의 기억 용량과 기울기 흐름을 대폭 개선한 차세대 순환 네트워크.

두 가지 변형:
  - sLSTM: 스칼라 메모리 + 지수 게이트 → 빠르고 가벼움
  - mLSTM: 행렬 메모리 → 대용량 연상 기억, 더 높은 성능

구조:
  입력 → [지수 게이팅] → [행렬 메모리 셀] → [정규화] → 출력
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from .registry import NetworkRegistry


class MLSTMCell(nn.Module):
    """mLSTM cell with matrix memory and exponential gating."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        # Exponential gates: input gate (i) and forget gate (f)
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)

        # Output gate (standard sigmoid)
        self.W_o = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_size)
        Returns: (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Initialize matrix memory C and normalizer n
        C = torch.zeros(batch, self.hidden_size, self.hidden_size, device=device)
        n = torch.zeros(batch, self.hidden_size, 1, device=device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            q = self.W_q(x_t)  # (batch, hidden_size)
            k = self.W_k(x_t)  # (batch, hidden_size)
            v = self.W_v(x_t)  # (batch, hidden_size)

            # Exponential gating (clamped for numerical stability)
            i_tilde = self.W_i(x_t)  # (batch, 1)
            f_tilde = self.W_f(x_t)  # (batch, 1)

            i_t = torch.exp(i_tilde.clamp(-10, 10))  # (batch, 1)
            f_t = torch.exp(f_tilde.clamp(-10, 10))  # (batch, 1)

            # Output gate (sigmoid)
            o_t = torch.sigmoid(self.W_o(x_t))  # (batch, hidden_size)

            # Update matrix memory: C = f * C + i * (v outer k)
            f_t_2d = f_t.unsqueeze(-1)  # (batch, 1, 1)
            i_t_2d = i_t.unsqueeze(-1)  # (batch, 1, 1)
            vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))  # (batch, H, H)
            C = f_t_2d * C + i_t_2d * vk

            # Update normalizer: n = f * n + i * k
            n = f_t.unsqueeze(-1) * n + i_t.unsqueeze(-1) * k.unsqueeze(2)

            # Retrieve from memory: h = o * (C @ q) / max(|n^T q|, 1)
            Cq = torch.bmm(C, q.unsqueeze(2)).squeeze(2)  # (batch, H)
            nq = torch.bmm(n.transpose(1, 2), q.unsqueeze(2)).squeeze(2)  # (batch, 1)
            denom = torch.clamp(nq.abs(), min=1.0)
            h_t = o_t * (Cq / denom)  # (batch, hidden_size)

            outputs.append(h_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)


@NetworkRegistry.register("xlstm")
class XLSTMNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, variant: str = "mlstm",
                 dropout: float = 0.1, seq_len: int = 20, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.variant = variant
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        self.features_per_step = input_dim // seq_len

        # Input projection
        self.input_proj = nn.Linear(self.features_per_step, hidden_dim)

        # Stack mLSTM layers with LayerNorm between
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(MLSTMCell(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        # Project input features to hidden_dim
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Pass through stacked mLSTM layers with residual connections
        for cell, norm, drop in zip(self.cells, self.norms, self.dropouts):
            residual = x
            x = cell(x)  # (batch, seq_len, hidden_dim)
            x = drop(x)
            x = norm(x + residual)  # residual + layer norm

        # Return last time step
        out = self.output_norm(x[:, -1, :])
        return out

    @property
    def output_dim(self) -> int:
        return self._output_dim
