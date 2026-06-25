"""DQN (Deep Q-Network) 에이전트.

epsilon-greedy 탐험 + 리플레이 버퍼 + 타깃 네트워크.
"""

import copy
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent
from .registry import AgentRegistry


class ReplayBuffer:
    """고정 크기 경험 리플레이 버퍼."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def __len__(self):
        return len(self.buffer)


@AgentRegistry.register("dqn")
class DQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu",
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update: int = 10,
                 replay_size: int = 10000, batch_size: int = 64,
                 num_actions: int = 3, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.num_actions = num_actions

        # Q-value head: network 출력 → 행동 수
        self.q_head = nn.Linear(network.output_dim, num_actions).to(device)

        # 타깃 네트워크 (network + q_head 를 묶어서 복사)
        self.target_network = copy.deepcopy(network).to(device)
        self.target_q_head = copy.deepcopy(self.q_head).to(device)
        self.target_network.eval()
        self.target_q_head.eval()

        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(replay_size)

        # 옵티마이저 (network + q_head 파라미터)
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.q_head.parameters()),
            lr=lr,
        )

        self.train_count = 0  # 타깃 네트워크 업데이트 카운터

    def _get_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """상태 텐서로부터 Q-값 계산."""
        features = self.network(state_tensor)
        return self.q_head(features)

    def _get_target_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """타깃 네트워크로 Q-값 계산."""
        with torch.no_grad():
            features = self.target_network(state_tensor)
            return self.target_q_head(features)

    def select_action(self, state, explore: bool = True) -> int:
        """epsilon-greedy 행동 선택."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._get_q_values(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch: dict = None) -> dict:
        """리플레이 버퍼에서 샘플링하여 Q-learning 업데이트.

        batch가 None이면 내부 리플레이 버퍼에서 샘플링.
        버퍼가 부족하면 학습을 건너뛴다.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "skipped": True}

        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # 현재 Q-값: Q(s, a)
        q_values = self._get_q_values(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 타깃 Q-값: r + gamma * max_a' Q_target(s', a')
        next_q_values = self._get_target_q_values(next_states)
        max_next_q = next_q_values.max(dim=1).values
        target = rewards + self.gamma * max_next_q * (1 - dones)

        # 손실 계산 및 역전파
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "skipped": False}

    def update_target(self):
        """타깃 네트워크를 현재 네트워크로 갱신."""
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())

    def decay_epsilon(self):
        """epsilon 감소."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def on_episode_end(self, episode: int):
        """에피소드 종료: epsilon 감소 + 타깃 네트워크 갱신."""
        self.decay_epsilon()
        if episode % self.target_update == 0:
            self.update_target()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "q_head": self.q_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_q_head": self.target_q_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.q_head.load_state_dict(checkpoint["q_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_q_head.load_state_dict(checkpoint["target_q_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
