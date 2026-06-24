"""PatchTST — 시계열을 패치(조각)로 잘라서 Transformer로 학습.

2026년 벤치마크 3위 (LSTM과 조합 시 꼬리 위험 관리 최고).
이미지의 ViT처럼, 시계열을 일정 길이의 패치로 분할하여 각 패치를
하나의 토큰으로 취급한다. 채널 독립(channel-independent) 설계.

구조:
  [20일 시계열] → [5일씩 4개 패치로 분할] → [패치 임베딩] → [Transformer] → 출력
"""

import math

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("patchtst")
class PatchTSTNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 patch_len: int = 5, stride: int = 5,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, seq_len: int = 20, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        self.features_per_step = input_dim // seq_len

        # Number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding: flatten each patch then project
        patch_input_dim = patch_len * self.features_per_step
        self.patch_embed = nn.Linear(patch_input_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Transformer Encoder
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

        batch, seq_len, n_feat = x.shape

        # 1. Split into patches: extract sliding windows
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]  # (batch, patch_len, n_feat)
            patch = patch.reshape(batch, -1)  # (batch, patch_len * n_feat)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_input_dim)

        # 2. Patch embedding
        x = self.patch_embed(patches)  # (batch, num_patches, hidden_dim)

        # 3. Add positional encoding
        x = x + self.pos_encoding

        # 4. Transformer Encoder
        x = self.transformer(x)  # (batch, num_patches, hidden_dim)

        # 5. Mean pooling over patches
        x = self.norm(x.mean(dim=1))  # (batch, hidden_dim)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
