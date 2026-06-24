"""강화학습 에이전트 추상 클래스.

모든 알고리즘(DQN, A2C, PPO 등)은 이 클래스를 상속하고
아래 추상 메서드를 구현해야 한다.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn


class BaseAgent(ABC):
    """에이전트 공통 인터페이스.

    Parameters
    ----------
    network : nn.Module
        행동 결정에 사용할 네트워크. 알고리즘과 독립적으로 교체 가능.
    lr : float
        학습률.
    gamma : float
        할인율.
    device : str
        "cpu" 또는 "cuda".
    """

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu", **kwargs):
        self.network = network.to(device)
        self.lr = lr
        self.gamma = gamma
        self.device = device

    @abstractmethod
    def select_action(self, state, explore: bool = True) -> int:
        """현재 상태에서 행동(0=매수, 1=매도, 2=관망)을 선택한다."""
        ...

    @abstractmethod
    def train_step(self, batch: dict = None) -> dict:
        """한 스텝(또는 배치) 학습을 수행한다."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """모델 체크포인트를 저장한다."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """모델 체크포인트를 불러온다."""
        ...

    def store_transition(self, state, action, reward, next_state, done):
        """경험을 버퍼에 저장한다. 서브클래스에서 오버라이드."""
        pass

    def on_episode_end(self, episode: int):
        """에피소드 종료 후 호출. epsilon 감소, 타깃 갱신 등."""
        pass
