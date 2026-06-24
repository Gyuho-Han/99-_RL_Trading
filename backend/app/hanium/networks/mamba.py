"""Mamba (Selective State Space Model) — 선형 시간 복잡도의 시퀀스 모델.

2026년 벤치마크 4위 (긴 시계열 / 고빈도 최고).
Transformer의 O(n^2) 어텐션을 O(n) 선형 재귀로 대체.
입력에 따라 선택적으로 정보를 기억/망각하는 SSM(State Space Model).

구조:
  입력 → [선형 투영] → [Conv1D] → [선택적 SSM] → [게이팅] → 출력
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from .registry import NetworkRegistry


class MambaBlock(nn.Module):
    """Single Mamba block: projection -> Conv1d -> selective SSM -> gating."""

    def __init__(self, d_model: int, state_dim: int = 16,
                 conv_kernel: int = 4, expand_factor: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        inner_dim = d_model * expand_factor

        # Project to 2x inner_dim: one for SSM path (x), one for gate (z)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)

        # Causal Conv1d on the SSM path
        self.conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=inner_dim,
        )

        # SSM parameter projections (input-dependent = selective)
        self.x_proj = nn.Linear(inner_dim, state_dim * 2 + 1)  # B, C, delta

        # Learnable SSM matrix A (log-space for stability)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(inner_dim, -1))
        )  # (inner_dim, state_dim)

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(inner_dim))

        # Output projection
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Simplified selective scan (for loop over time steps).

        Args:
            x: (batch, seq_len, inner_dim)
            delta: (batch, seq_len, inner_dim) - discretization step
            B: (batch, seq_len, state_dim) - input matrix
            C: (batch, seq_len, state_dim) - output matrix
        Returns:
            y: (batch, seq_len, inner_dim)
        """
        batch, seq_len, inner_dim = x.shape
        state_dim = B.shape[-1]
        device = x.device

        # A: (inner_dim, state_dim)
        A = -torch.exp(self.A_log)

        # State: (batch, inner_dim, state_dim)
        h = torch.zeros(batch, inner_dim, state_dim, device=device)

        outputs = []
        for t in range(seq_len):
            # Discretize: A_bar = exp(delta * A), B_bar = delta * B
            dt = delta[:, t, :].unsqueeze(-1)  # (batch, inner_dim, 1)
            A_bar = torch.exp(dt * A.unsqueeze(0))  # (batch, inner_dim, state_dim)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, state_dim)
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, inner_dim, 1)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + (dt * B_t) * x_t

            # Output: y = C^T @ h + D * x
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, state_dim)
            y_t = (h * C_t).sum(dim=-1)  # (batch, inner_dim)
            y_t = y_t + self.D * x[:, t, :]

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        residual = x
        batch, seq_len, _ = x.shape

        # Project and split into SSM path and gate path
        xz = self.in_proj(x)  # (batch, seq_len, inner_dim * 2)
        inner_dim = xz.shape[-1] // 2
        x_ssm, z = xz.split(inner_dim, dim=-1)

        # Causal Conv1d
        x_ssm = x_ssm.transpose(1, 2)  # (batch, inner_dim, seq_len)
        x_ssm = self.conv(x_ssm)[:, :, :seq_len]  # causal: trim future
        x_ssm = x_ssm.transpose(1, 2)  # (batch, seq_len, inner_dim)
        x_ssm = F.silu(x_ssm)

        # Generate input-dependent SSM parameters
        ssm_params = self.x_proj(x_ssm)  # (batch, seq_len, state_dim*2 + 1)
        state_dim = self.state_dim
        B = ssm_params[:, :, :state_dim]
        C = ssm_params[:, :, state_dim:state_dim * 2]
        delta = F.softplus(ssm_params[:, :, -1:].expand(-1, -1, inner_dim))

        # Selective scan
        y = self.selective_scan(x_ssm, delta, B, C)

        # Gating with SiLU
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual + LayerNorm
        return self.norm(y + residual)


@NetworkRegistry.register("mamba")
class MambaNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 state_dim: int = 16, num_layers: int = 2,
                 conv_kernel: int = 4, expand_factor: int = 2,
                 dropout: float = 0.1, seq_len: int = 20, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        self.features_per_step = input_dim // seq_len

        # Input projection
        self.input_proj = nn.Linear(self.features_per_step, hidden_dim)

        # Stack of Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, state_dim, conv_kernel, expand_factor, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Return last time step
        out = self.output_norm(x[:, -1, :])
        return out

    @property
    def output_dim(self) -> int:
        return self._output_dim
