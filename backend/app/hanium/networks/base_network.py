"""네트워크 추상 클래스.

모든 네트워크(DNN, LSTM, CNN 등)는 이 클래스를 상속하고
forward와 output_dim을 구현해야 한다.

에이전트는 네트워크의 구체 구조를 몰라도 된다 —
output_dim만 알면 행동 헤드를 붙일 수 있다.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseNetwork(ABC, nn.Module):
    """네트워크 공통 인터페이스.

    Parameters
    ----------
    input_dim : int
        입력 특성 차원 (상태 벡터 크기).
    """

    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 텐서를 받아 특성 벡터를 출력한다.

        Parameters
        ----------
        x : torch.Tensor
            shape: (batch, ..., input_dim) — 네트워크 구조에 따라 다름.

        Returns
        -------
        torch.Tensor
            shape: (batch, output_dim) — 에이전트가 행동 헤드를 붙일 특성.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """forward 출력의 마지막 차원 크기.

        에이전트가 이 값을 참조하여 행동/가치 헤드의 입력 크기를 결정한다.
        """
        ...
