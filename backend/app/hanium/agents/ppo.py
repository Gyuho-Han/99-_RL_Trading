"""PPO (Proximal Policy Optimization) 에이전트.

클리핑 기반 정책 업데이트 + GAE (Generalized Advantage Estimation).
rollout_steps 만큼 경험을 모은 뒤 ppo_epochs 번 미니배치 업데이트한다.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("ppo")
class PPOAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 ppo_epochs: int = 4, rollout_steps: int = 1024,
                 **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.rollout_steps = rollout_steps

        # Actor head: policy logits
        self.actor_head = nn.Linear(network.output_dim, num_actions).to(device)
        # Critic head: state value
        self.critic_head = nn.Linear(network.output_dim, 1).to(device)

        # Optimizer over all parameters
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters()),
            lr=lr,
        )

        # Rollout buffer
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []
        self._log_probs = []

        # Cache the last log_prob from select_action for store_transition
        self._last_log_prob = 0.0

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, state_tensor: torch.Tensor):
        """Return (action_probs, state_value) given a batched state tensor."""
        features = self.network(state_tensor)
        logits = self.actor_head(features)
        probs = F.softmax(logits, dim=-1)
        value = self.critic_head(features).squeeze(-1)
        return probs, value

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def select_action(self, state, explore: bool = True) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self._forward(state_t)

        if explore:
            dist = Categorical(probs)
            action = dist.sample()
            self._last_log_prob = dist.log_prob(action).item()
            action = action.item()
        else:
            action = probs.argmax(dim=-1).item()
            self._last_log_prob = 0.0
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """롤아웃 버퍼에 경험과 log_prob를 저장한다."""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)
        self._log_probs.append(self._last_log_prob)

    def train_step(self, batch: dict = None) -> dict:
        """롤아웃 버퍼가 rollout_steps에 도달하면 PPO 업데이트를 수행한다."""
        if len(self._states) < self.rollout_steps:
            return {"loss": 0.0, "skipped": True}

        loss_val = self._ppo_update()
        return {"loss": loss_val, "skipped": False}

    def on_episode_end(self, episode: int):
        """에피소드 종료 시 버퍼가 rollout_steps 이상이면 업데이트 후 클리어.
        그렇지 않으면 다음 에피소드로 롤아웃 데이터를 이월한다."""
        if len(self._states) >= self.rollout_steps:
            self._ppo_update()

    # ------------------------------------------------------------------
    # PPO core
    # ------------------------------------------------------------------

    def _compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation을 계산한다."""
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _ppo_update(self) -> float:
        """PPO clipped surrogate 업데이트를 ppo_epochs 반복 수행한다."""
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self._states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(self._actions).to(self.device)
        rewards_t = torch.FloatTensor(self._rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self._next_states, dtype=np.float32)).to(self.device)
        dones_t = torch.FloatTensor(self._dones).to(self.device)
        old_log_probs = torch.FloatTensor(self._log_probs).to(self.device)

        # Compute values and GAE (no grad for advantage targets)
        with torch.no_grad():
            _, values = self._forward(states)
            _, next_values = self._forward(next_states)
            advantages, returns = self._compute_gae(rewards_t, values, next_values, dones_t)
            # Normalize advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(self._states)
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Forward pass
                probs, values_pred = self._forward(mb_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred, mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        # Clear buffer after update
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_states.clear()
        self._dones.clear()
        self._log_probs.clear()

        return total_loss / max(num_updates, 1)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic_head": self.critic_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
