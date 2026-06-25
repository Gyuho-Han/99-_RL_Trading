"""iTransformer (Inverted Transformer) — 변수별 시계열을 토큰으로 취급.

2026년 벤치마크 5위 (다변량 예측 정확도).
기존 Transformer: 시점(time step)이 토큰
iTransformer: 변수(variate)가 토큰 → 변수 간 상관관계를 더 잘 포착.

구조:
  [변수1의 20일] [변수2의 20일] ... → 각각을 토큰으로 임베딩 → Attention → 출력
  (종가 시계열)  (RSI 시계열)       (변수 간 관계 학습)
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("itransformer")
class ITransformerNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 seq_len: int = 20, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        self.features_per_step = input_dim // seq_len
        n_features = self.features_per_step

        # Variate Embedding: each variable's full time series -> one token
        # Linear(seq_len -> hidden_dim) applied per variable
        self.variate_embed = nn.Linear(seq_len, hidden_dim)

        # Learnable variate type embedding (optional, helps distinguish variables)
        self.variate_pos = nn.Parameter(
            torch.randn(1, n_features, hidden_dim) * 0.02
        )

        # Transformer Encoder: attention over variable tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        # x: (batch, seq_len, n_features)

        # 1. Transpose: each variable's time series becomes a row
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)

        # 2. Variate Embedding: project each variable's time series to hidden_dim
        x = self.variate_embed(x)  # (batch, n_features, hidden_dim)

        # Add variate positional encoding
        x = x + self.variate_pos

        # 3. Transformer Encoder: attention over variable tokens
        x = self.transformer(x)  # (batch, n_features, hidden_dim)

        # 4. Mean pooling over variable tokens
        x = self.norm(x.mean(dim=1))  # (batch, hidden_dim)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
