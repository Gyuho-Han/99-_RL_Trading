"""IQN (Implicit Quantile Networks) — 수익 분포를 학습하는 분포형 RL.

리스크 관리 최고 — CVaR(조건부 VaR)를 직접 최적화 가능.
위험 성향을 조절할 수 있어 보수적/공격적 전략 모두 지원.

구조:
  상태 → [네트워크] → 특성 벡터
  τ ~ U(0,1) → cos(πiτ) → 분위수 임베딩
  특성 × 분위수 임베딩 → 분위수별 Q값
"""

import copy
import math
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


@AgentRegistry.register("iqn")
class IQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 5e-5,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 n_quantiles: int = 32, embedding_dim: int = 64,
                 kappa: float = 1.0, cvar_alpha: float = 1.0,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update: int = 10,
                 replay_size: int = 10000, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.n_quantiles = n_quantiles
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.cvar_alpha = cvar_alpha
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        output_dim = network.output_dim

        # Quantile embedding: cos(i * pi * tau) for i in [0, embedding_dim)
        # then project to output_dim
        self.quantile_embed = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        ).to(device)

        # Q-head: from combined features to per-action Q-values
        self.q_head = nn.Linear(output_dim, num_actions).to(device)

        # Target network
        self.target_network = copy.deepcopy(network).to(device)
        self.target_quantile_embed = copy.deepcopy(self.quantile_embed).to(device)
        self.target_q_head = copy.deepcopy(self.q_head).to(device)
        self.target_network.eval()
        self.target_quantile_embed.eval()
        self.target_q_head.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        # Optimizer
        params = (list(self.network.parameters())
                  + list(self.quantile_embed.parameters())
                  + list(self.q_head.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.train_count = 0

    def _compute_quantile_embedding(self, taus: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Compute cosine quantile embedding.

        Args:
            taus: (batch, n_quantiles) quantile fractions in [0, 1]
            target: whether to use target embedding network

        Returns:
            (batch, n_quantiles, output_dim) quantile embeddings
        """
        batch_size, n_q = taus.shape
        # i = 0, 1, ..., embedding_dim - 1
        i_pi = (torch.arange(0, self.embedding_dim, device=self.device).float()
                * math.pi)  # (embedding_dim,)
        # cos(i * pi * tau): (batch, n_q, embedding_dim)
        cos_features = torch.cos(taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))

        embed_net = self.target_quantile_embed if target else self.quantile_embed
        # (batch * n_q, embedding_dim) -> (batch * n_q, output_dim)
        cos_flat = cos_features.reshape(-1, self.embedding_dim)
        embedded = embed_net(cos_flat)
        return embedded.reshape(batch_size, n_q, -1)

    def _get_quantile_values(self, states: torch.Tensor, taus: torch.Tensor,
                             target: bool = False) -> torch.Tensor:
        """Compute Q-values for each (action, quantile) pair.

        Args:
            states: (batch, state_dim)
            taus: (batch, n_quantiles)
            target: use target networks

        Returns:
            (batch, n_quantiles, num_actions) Q-values per quantile
        """
        net = self.target_network if target else self.network
        head = self.target_q_head if target else self.q_head

        features = net(states)  # (batch, output_dim)
        quantile_emb = self._compute_quantile_embedding(taus, target=target)  # (batch, n_q, output_dim)

        # Element-wise multiply: broadcast features to (batch, n_q, output_dim)
        combined = features.unsqueeze(1) * quantile_emb  # (batch, n_q, output_dim)

        # Flatten, pass through Q-head, reshape
        batch_size, n_q, dim = combined.shape
        q_flat = head(combined.reshape(-1, dim))  # (batch * n_q, num_actions)
        return q_flat.reshape(batch_size, n_q, self.num_actions)

    def select_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Sample taus in [0, cvar_alpha] for risk-averse selection
            taus = torch.rand(1, self.n_quantiles, device=self.device) * self.cvar_alpha
            q_quantiles = self._get_quantile_values(state_t, taus)  # (1, n_q, num_actions)
            # Mean across quantiles to get expected Q per action
            q_mean = q_quantiles.mean(dim=1)  # (1, num_actions)
        return q_mean.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

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

        bs = states.shape[0]
        n_q = self.n_quantiles

        # Sample taus for current and next
        taus = torch.rand(bs, n_q, device=self.device)
        taus_prime = torch.rand(bs, n_q, device=self.device)

        # Current quantile values for chosen actions
        q_quantiles = self._get_quantile_values(states, taus)  # (bs, n_q, num_actions)
        # Gather for selected actions: (bs, n_q)
        q_sa = q_quantiles.gather(2, actions.unsqueeze(1).unsqueeze(2).expand(-1, n_q, -1)).squeeze(2)

        # Target quantile values
        with torch.no_grad():
            # Use online net to select best action (Double-DQN style)
            online_next_q = self._get_quantile_values(next_states, taus_prime)  # (bs, n_q, num_actions)
            online_mean = online_next_q.mean(dim=1)  # (bs, num_actions)
            best_actions = online_mean.argmax(dim=1)  # (bs,)

            target_next_q = self._get_quantile_values(next_states, taus_prime, target=True)
            # (bs, n_q)
            next_q_sa = target_next_q.gather(
                2, best_actions.unsqueeze(1).unsqueeze(2).expand(-1, n_q, -1)
            ).squeeze(2)

            # Target: r + gamma * Z(s', a*) per quantile
            target = rewards.unsqueeze(1) + self.gamma * next_q_sa * (1 - dones.unsqueeze(1))

        # Quantile Huber loss
        # td_error: (bs, n_q, n_q) = q_sa[:, :, None] - target[:, None, :]
        td_error = q_sa.unsqueeze(2) - target.unsqueeze(1)  # (bs, n_q_current, n_q_target)

        # Huber loss element-wise
        huber = torch.where(
            td_error.abs() <= self.kappa,
            0.5 * td_error ** 2,
            self.kappa * (td_error.abs() - 0.5 * self.kappa),
        )

        # Quantile weights: |tau - I(td_error < 0)|
        tau_expanded = taus.unsqueeze(2)  # (bs, n_q, 1)
        quantile_weight = torch.abs(tau_expanded - (td_error < 0).float())

        loss = (quantile_weight * huber).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters())
            + list(self.quantile_embed.parameters())
            + list(self.q_head.parameters()),
            10.0,
        )
        self.optimizer.step()

        self.train_count += 1
        return {"loss": loss.item(), "skipped": False}

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_quantile_embed.load_state_dict(self.quantile_embed.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())

    def on_episode_end(self, episode: int):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if episode % self.target_update == 0:
            self.update_target()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "quantile_embed": self.quantile_embed.state_dict(),
            "q_head": self.q_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_quantile_embed": self.target_quantile_embed.state_dict(),
            "target_q_head": self.target_q_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_count": self.train_count,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.quantile_embed.load_state_dict(checkpoint["quantile_embed"])
        self.q_head.load_state_dict(checkpoint["q_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_quantile_embed.load_state_dict(checkpoint["target_quantile_embed"])
        self.target_q_head.load_state_dict(checkpoint["target_q_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_count = checkpoint.get("train_count", 0)
