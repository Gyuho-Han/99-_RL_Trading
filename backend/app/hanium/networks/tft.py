"""TFT (Temporal Fusion Transformer) — LSTM + Attention + 변수 선택.

2026년 벤치마크 1위 (샤프 비율 2.27).
어떤 변수가 중요한지 자동으로 선택하고 해석 가능한 예측을 제공한다.

구조:
  입력 → [Variable Selection Network] → [LSTM Encoder] → [Multi-Head Attention] → 출력
         (어떤 지표가 중요한지)         (시간 순서 학습)    (중요 시점 집중)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("tft")
class TFTNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, seq_len: int = 20, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        self.features_per_step = input_dim // seq_len
        n_features = self.features_per_step

        # --- Variable Selection Network ---
        # Per-feature linear transforms
        self.feature_transforms = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_features)
        ])
        # Softmax weights over features (context-dependent)
        self.variable_gate = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        # --- LSTM Encoder ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Gated Layer Normalization ---
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # --- Multi-Head Attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # --- Feed Forward ---
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.features_per_step)

        batch, seq_len, n_feat = x.shape

        # 1. Variable Selection
        # Compute variable weights from the mean across time
        feat_mean = x.mean(dim=1)  # (batch, n_feat)
        var_weights = F.softmax(self.variable_gate(feat_mean), dim=-1)  # (batch, n_feat)

        # Transform each feature independently then weighted-sum
        # x_i: (batch, seq_len, 1) -> (batch, seq_len, hidden_dim)
        transformed = []
        for i in range(n_feat):
            feat_i = x[:, :, i:i+1]  # (batch, seq_len, 1)
            transformed.append(self.feature_transforms[i](feat_i))
        transformed = torch.stack(transformed, dim=-1)  # (batch, seq_len, hidden_dim, n_feat)

        # Weighted combination
        w = var_weights.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_feat)
        selected = (transformed * w).sum(dim=-1)  # (batch, seq_len, hidden_dim)

        # 2. LSTM Encoder
        lstm_out, _ = self.lstm(selected)  # (batch, seq_len, hidden_dim)

        # Gated residual
        gate = torch.sigmoid(self.gate_linear(lstm_out))
        lstm_out = self.gate_norm(gate * lstm_out + (1 - gate) * selected)

        # 3. Multi-Head Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # 4. Feed Forward + residual
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(ffn_out + attn_out)

        # Return last time step
        return ffn_out[:, -1, :]  # (batch, hidden_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim
