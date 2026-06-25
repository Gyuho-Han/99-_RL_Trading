"""TD3-Discrete (Twin Delayed DDPG for Discrete Actions).

쌍둥이 비평가 + 지연 정책 업데이트를 이산 행동 공간에 적용.
DDPG의 과대평가 문제를 쌍둥이 Critic과 지연 업데이트로 해결한다.
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


@AgentRegistry.register("td3")
class TD3Agent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_std: float = 0.2, noise_clip: float = 0.5,
                 replay_size: int = 10000, **kwargs):
        """
        Parameters
        ----------
        policy_delay : int
            Critic N번 업데이트마다 Actor 1번 업데이트.
        noise_std : float
            타깃 정책 스무딩 노이즈 표준편차.
        noise_clip : float
            노이즈 클리핑 범위.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.num_actions = num_actions

        output_dim = network.output_dim

        # Actor: outputs action probabilities (softmax policy)
        self.actor_head = nn.Linear(output_dim, num_actions).to(device)

        # Twin Critics: Q1, Q2
        self.q1_head = nn.Linear(output_dim, num_actions).to(device)
        self.q2_head = nn.Linear(output_dim, num_actions).to(device)

        # Target networks (full copies: backbone + heads)
        self.target_network = copy.deepcopy(network).to(device)
        self.target_actor_head = copy.deepcopy(self.actor_head).to(device)
        self.target_q1_head = copy.deepcopy(self.q1_head).to(device)
        self.target_q2_head = copy.deepcopy(self.q2_head).to(device)
        self.target_network.eval()
        self.target_actor_head.eval()
        self.target_q1_head.eval()
        self.target_q2_head.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.actor_head.parameters()),
            lr=lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1_head.parameters()) + list(self.q2_head.parameters()),
            lr=lr,
        )

        # Update counter for delayed policy update
        self.update_count = 0

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

    def _get_q_values(self, state_tensor: torch.Tensor):
        """Twin Q-values 계산."""
        features = self.network(state_tensor)
        q1 = self.q1_head(features)
        q2 = self.q2_head(features)
        return q1, q2

    def _get_target_q_values(self, state_tensor: torch.Tensor):
        """타깃 네트워크로 Twin Q-values 계산."""
        with torch.no_grad():
            features = self.target_network(state_tensor)
            q1 = self.target_q1_head(features)
            q2 = self.target_q2_head(features)
        return q1, q2

    def select_action(self, state, explore: bool = True) -> int:
        """행동 선택: explore=True이면 확률적 샘플링, False이면 greedy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self._get_action_probs(state_t)
        if explore:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch: dict = None) -> dict:
        """TD3-Discrete 학습 스텝.

        1. 타깃 행동 확률에 클리핑된 노이즈 추가 (스무딩)
        2. 쌍둥이 Critic 업데이트 (min Q 타깃)
        3. policy_delay마다 Actor 업데이트
        4. 타깃 네트워크 소프트 업데이트
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

        self.update_count += 1

        # --- Critic update ---
        with torch.no_grad():
            # Target action probs with smoothing noise
            target_probs = self._get_action_probs(
                next_states, self.target_network, self.target_actor_head
            )
            # Add clipped noise to target action probs for smoothing
            noise = torch.randn_like(target_probs) * self.noise_std
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            smoothed_probs = target_probs + noise
            # Re-normalize: clamp to positive then normalize
            smoothed_probs = smoothed_probs.clamp(min=1e-8)
            smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True)

            # Target Q-values
            target_q1, target_q2 = self._get_target_q_values(next_states)
            min_target_q = torch.min(target_q1, target_q2)

            # Expected target Q: sum_a pi(a|s') * min_Q(s', a)
            next_q = (smoothed_probs * min_target_q).sum(dim=-1)
            target = rewards + self.gamma * next_q * (1 - dones)

        # Current Q-values for taken actions
        q1, q2 = self._get_q_values(states)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        total_loss = critic_loss.item()

        # --- Delayed Actor update ---
        if self.update_count % self.policy_delay == 0:
            probs = self._get_action_probs(states)
            # Actor loss: maximize expected Q — use Q1
            with torch.no_grad():
                q1_det, _ = self._get_q_values(states)
            actor_loss = -(probs * q1_det).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            total_loss += actor_loss.item()

            # Soft target update (only when actor is updated)
            self._soft_update(self.target_network, self.network)
            self._soft_update(self.target_actor_head, self.actor_head)
            self._soft_update(self.target_q1_head, self.q1_head)
            self._soft_update(self.target_q2_head, self.q2_head)

        return {"loss": total_loss, "skipped": False}

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """타깃 네트워크 소프트 업데이트: target = tau*source + (1-tau)*target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def on_episode_end(self, episode: int):
        """에피소드 종료 후 처리 (TD3는 특별한 에피소드 단위 작업 없음)."""
        pass

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "q1_head": self.q1_head.state_dict(),
            "q2_head": self.q2_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_actor_head": self.target_actor_head.state_dict(),
            "target_q1_head": self.target_q1_head.state_dict(),
            "target_q2_head": self.target_q2_head.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.q1_head.load_state_dict(checkpoint["q1_head"])
        self.q2_head.load_state_dict(checkpoint["q2_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_actor_head.load_state_dict(checkpoint["target_actor_head"])
        self.target_q1_head.load_state_dict(checkpoint["target_q1_head"])
        self.target_q2_head.load_state_dict(checkpoint["target_q2_head"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = checkpoint["update_count"]
