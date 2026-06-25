"""DDPG-Discrete (Deep Deterministic Policy Gradient for Discrete Actions).

DDPG를 이산 행동 공간에 적용. Actor는 softmax 정책을 출력하고
탐험은 epsilon-greedy로 수행한다.
TD3, SAC의 기반이 되는 알고리즘으로 베이스라인 비교에 사용.
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


@AgentRegistry.register("ddpg")
class DDPGAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 tau: float = 0.005,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 replay_size: int = 10000, **kwargs):
        """
        Parameters
        ----------
        tau : float
            타깃 네트워크 소프트 업데이트 비율.
        epsilon_start : float
            초기 탐험 확률.
        epsilon_end : float
            최소 탐험 확률.
        epsilon_decay : float
            에피소드마다 epsilon에 곱해지는 감소율.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        output_dim = network.output_dim

        # Actor: outputs action probabilities (softmax)
        self.actor_head = nn.Linear(output_dim, num_actions).to(device)

        # Critic: outputs Q-values for all actions
        self.critic_head = nn.Linear(output_dim, num_actions).to(device)

        # Target networks (separate backbone + heads)
        self.target_network = copy.deepcopy(network).to(device)
        self.target_actor_head = copy.deepcopy(self.actor_head).to(device)
        self.target_critic_head = copy.deepcopy(self.critic_head).to(device)
        self.target_network.eval()
        self.target_actor_head.eval()
        self.target_critic_head.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.actor_head.parameters()),
            lr=lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_head.parameters()),
            lr=lr,
        )

    def _get_action_probs(self, state_tensor: torch.Tensor,
                          network: nn.Module = None,
                          head: nn.Module = None):
        """상태에서 행동 확률을 반환한다."""
        net = network if network is not None else self.network
        hd = head if head is not None else self.actor_head
        features = net(state_tensor)
        logits = hd(features)
        probs = F.softmax(logits, dim=-1)
        return probs

    def _get_q_values(self, state_tensor: torch.Tensor,
                      network: nn.Module = None,
                      head: nn.Module = None):
        """상태에서 Q-values를 반환한다."""
        net = network if network is not None else self.network
        hd = head if head is not None else self.critic_head
        features = net(state_tensor)
        return hd(features)

    def select_action(self, state, explore: bool = True) -> int:
        """epsilon-greedy 행동 선택."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self._get_action_probs(state_t)
        return probs.argmax(dim=-1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch: dict = None) -> dict:
        """DDPG-Discrete 학습 스텝.

        1. Critic 업데이트 (MSE loss)
        2. Actor 업데이트 (Critic을 통한 정책 기울기)
        3. 타깃 네트워크 소프트 업데이트
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

        # --- Critic update ---
        with torch.no_grad():
            # Target actor probs for next states
            target_probs = self._get_action_probs(
                next_states, self.target_network, self.target_actor_head
            )
            # Target Q-values for next states
            target_q = self._get_q_values(
                next_states, self.target_network, self.target_critic_head
            )
            # Expected next Q: sum_a pi(a|s') * Q_target(s', a)
            next_q = (target_probs * target_q).sum(dim=-1)
            target = rewards + self.gamma * next_q * (1 - dones)

        # Current Q-values for taken actions
        q_values = self._get_q_values(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q_a, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        probs = self._get_action_probs(states)
        # Use current critic Q-values (detached from critic grad)
        with torch.no_grad():
            q_det = self._get_q_values(states)
        # Actor loss: maximize expected Q = sum_a pi(a|s) * Q(s, a)
        actor_loss = -(probs * q_det).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft target update ---
        self._soft_update(self.target_network, self.network)
        self._soft_update(self.target_actor_head, self.actor_head)
        self._soft_update(self.target_critic_head, self.critic_head)

        total_loss = critic_loss.item() + actor_loss.item()
        return {"loss": total_loss, "skipped": False}

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """타깃 네트워크 소프트 업데이트: target = tau*source + (1-tau)*target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def on_episode_end(self, episode: int):
        """에피소드 종료: epsilon 감소."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic_head": self.critic_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_actor_head": self.target_actor_head.state_dict(),
            "target_critic_head": self.target_critic_head.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_actor_head.load_state_dict(checkpoint["target_actor_head"])
        self.target_critic_head.load_state_dict(checkpoint["target_critic_head"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.epsilon = checkpoint["epsilon"]
