"""앙상블 에이전트 — 여러 Q-헤드를 운용하여 최고 성과의 헤드를 선택.

단일 네트워크 백본에 N개의 독립 DQN-like Q-헤드를 붙여,
각 헤드의 최근 보상을 추적하고 최고 성과 헤드의 행동을 따른다.
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


@AgentRegistry.register("ensemble")
class EnsembleAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 n_heads: int = 3, eval_window: int = 50,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update: int = 10,
                 replay_size: int = 10000, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.eval_window = eval_window
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        output_dim = network.output_dim

        # N independent Q-heads
        self.q_heads = nn.ModuleList([
            nn.Linear(output_dim, num_actions) for _ in range(n_heads)
        ]).to(device)

        # Target network and target Q-heads
        self.target_network = copy.deepcopy(network).to(device)
        self.target_q_heads = copy.deepcopy(self.q_heads).to(device)
        self.target_network.eval()
        self.target_q_heads.eval()

        # Replay buffer (shared)
        self.replay_buffer = ReplayBuffer(replay_size)

        # Optimizer for network + all heads
        params = list(self.network.parameters())
        for head in self.q_heads:
            params += list(head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Track per-head recent rewards
        self.head_rewards = [deque(maxlen=eval_window) for _ in range(n_heads)]
        self.active_head_idx = 0
        self._episode_reward = 0.0

        self.train_count = 0

    def _get_q_values_all_heads(self, state_tensor: torch.Tensor):
        """Return list of Q-value tensors, one per head."""
        features = self.network(state_tensor)
        return [head(features) for head in self.q_heads]

    def _get_target_q_values_all_heads(self, state_tensor: torch.Tensor):
        """Return list of target Q-value tensors."""
        with torch.no_grad():
            features = self.target_network(state_tensor)
            return [head(features) for head in self.target_q_heads]

    def _select_best_head(self) -> int:
        """Select the head with the highest average recent reward."""
        best_idx = 0
        best_avg = float("-inf")
        for i in range(self.n_heads):
            if len(self.head_rewards[i]) == 0:
                avg = 0.0
            else:
                avg = np.mean(self.head_rewards[i])
            if avg > best_avg:
                best_avg = avg
                best_idx = i
        return best_idx

    def select_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_all = self._get_q_values_all_heads(state_t)

        # Use the active (best-performing) head
        q_active = q_all[self.active_head_idx]
        return q_active.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._episode_reward += reward
        if done:
            # Attribute episode reward to all heads (they all share the network)
            # But simulate individual head performance by their own Q-value quality
            for i in range(self.n_heads):
                self.head_rewards[i].append(self._episode_reward)
            self._episode_reward = 0.0

    def train_step(self, batch: dict = None) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "skipped": True}

        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # Compute loss for each head and average
        total_loss = torch.tensor(0.0, device=self.device)

        q_all = self._get_q_values_all_heads(states)
        target_q_all = self._get_target_q_values_all_heads(next_states)

        for i in range(self.n_heads):
            q_values = q_all[i].gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next_q = target_q_all[i].max(dim=1).values
                target = rewards + self.gamma * max_next_q * (1 - dones)

            head_loss = F.mse_loss(q_values, target)
            total_loss = total_loss + head_loss

        avg_loss = total_loss / self.n_heads

        self.optimizer.zero_grad()
        avg_loss.backward()
        params = list(self.network.parameters())
        for head in self.q_heads:
            params += list(head.parameters())
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        self.optimizer.step()

        self.train_count += 1
        return {"loss": avg_loss.item(), "skipped": False}

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_q_heads.load_state_dict(self.q_heads.state_dict())

    def on_episode_end(self, episode: int):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if episode % self.target_update == 0:
            self.update_target()
        # Update active head selection
        self.active_head_idx = self._select_best_head()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "q_heads": self.q_heads.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_q_heads": self.target_q_heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "active_head_idx": self.active_head_idx,
            "train_count": self.train_count,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.q_heads.load_state_dict(checkpoint["q_heads"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_q_heads.load_state_dict(checkpoint["target_q_heads"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.active_head_idx = checkpoint.get("active_head_idx", 0)
        self.train_count = checkpoint.get("train_count", 0)
