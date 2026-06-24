"""SAC-Discrete (Soft Actor-Critic for Discrete Actions).

엔트로피 최대화 기반 Actor-Critic을 이산 행동 공간에 적용.
보상 최대화와 동시에 탐험(엔트로피)도 최대화하여
하이퍼파라미터에 강건하고 안정적인 학습을 제공한다.
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


@AgentRegistry.register("sac")
class SACAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 tau: float = 0.005, alpha: float = 0.2,
                 auto_alpha: bool = True, replay_size: int = 10000,
                 **kwargs):
        """
        Parameters
        ----------
        tau : float
            타깃 네트워크 소프트 업데이트 비율.
        alpha : float
            엔트로피 계수 (auto_alpha=True이면 자동 조절).
        auto_alpha : bool
            True이면 엔트로피 계수를 자동으로 학습.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.batch_size = batch_size
        self.num_actions = num_actions

        output_dim = network.output_dim

        # Actor: categorical policy over discrete actions
        self.actor_head = nn.Linear(output_dim, num_actions).to(device)

        # Twin Critics: Q1, Q2 — each outputs Q-values for all actions
        self.q1_head = nn.Linear(output_dim, num_actions).to(device)
        self.q2_head = nn.Linear(output_dim, num_actions).to(device)

        # Target critics (separate network backbone + heads)
        self.target_network = copy.deepcopy(network).to(device)
        self.target_q1_head = copy.deepcopy(self.q1_head).to(device)
        self.target_q2_head = copy.deepcopy(self.q2_head).to(device)
        self.target_network.eval()
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

        # Auto-tuned alpha
        if self.auto_alpha:
            self.target_entropy = -np.log(1.0 / num_actions) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def _get_action_probs(self, state_tensor: torch.Tensor):
        """상태에서 행동 확률과 로그 확률을 반환한다."""
        features = self.network(state_tensor)
        logits = self.actor_head(features)
        probs = F.softmax(logits, dim=-1)
        # 수치 안정성을 위해 작은 값 클리핑
        probs = probs.clamp(min=1e-8)
        log_probs = torch.log(probs)
        return probs, log_probs

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
        """행동 선택: explore=True이면 확률적, False이면 greedy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self._get_action_probs(state_t)
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
        """SAC-Discrete 학습 스텝.

        1. Twin Critic 업데이트 (min target Q - alpha * log_pi)
        2. Actor 업데이트 (min(Q1,Q2) - alpha * log_pi 최대화)
        3. auto_alpha이면 alpha 업데이트
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

        # --- Critic update ---
        with torch.no_grad():
            next_probs, next_log_probs = self._get_action_probs(next_states)
            target_q1, target_q2 = self._get_target_q_values(next_states)
            min_target_q = torch.min(target_q1, target_q2)
            # V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log pi(a|s'))
            next_v = (next_probs * (min_target_q - self.alpha * next_log_probs)).sum(dim=-1)
            target = rewards + self.gamma * next_v * (1 - dones)

        # Current Q-values for taken actions
        q1, q2 = self._get_q_values(states)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        probs, log_probs = self._get_action_probs(states)
        # Detach Q-values (don't backprop through critic heads for actor update)
        with torch.no_grad():
            q1_det, q2_det = self._get_q_values(states)
            min_q = torch.min(q1_det, q2_det)

        # Actor loss: minimize alpha * log_pi - Q  (maximize Q - alpha * log_pi)
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update ---
        if self.auto_alpha:
            # entropy = -sum(pi * log_pi)
            with torch.no_grad():
                probs_det, log_probs_det = self._get_action_probs(states)
                entropy = -(probs_det * log_probs_det).sum(dim=-1).mean()
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft target update ---
        self._soft_update(self.target_network, self.network)
        self._soft_update(self.target_q1_head, self.q1_head)
        self._soft_update(self.target_q2_head, self.q2_head)

        total_loss = critic_loss.item() + actor_loss.item()
        return {"loss": total_loss, "skipped": False}

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """타깃 네트워크 소프트 업데이트: target = tau*source + (1-tau)*target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def on_episode_end(self, episode: int):
        """에피소드 종료 후 처리 (SAC는 특별한 에피소드 단위 작업 없음)."""
        pass

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "network": self.network.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "q1_head": self.q1_head.state_dict(),
            "q2_head": self.q2_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_q1_head": self.target_q1_head.state_dict(),
            "target_q2_head": self.target_q2_head.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.q1_head.load_state_dict(checkpoint["q1_head"])
        self.q2_head.load_state_dict(checkpoint["q2_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_q1_head.load_state_dict(checkpoint["target_q1_head"])
        self.target_q2_head.load_state_dict(checkpoint["target_q2_head"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha = checkpoint["alpha"]
        if self.auto_alpha and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
