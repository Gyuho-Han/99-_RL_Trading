"""A2C (Advantage Actor-Critic) 에이전트.

Actor-Critic 구조로 정책(actor)과 가치(critic)를 동시 학습.
에피소드 종료 시 전체 롤아웃으로 업데이트한다.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("a2c")
class A2CAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 7e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

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

        # Episode rollout buffer
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []

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
            action = dist.sample().item()
        else:
            action = probs.argmax(dim=-1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """에피소드 버퍼에 경험을 저장한다."""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def train_step(self, batch: dict = None) -> dict:
        """A2C는 on-policy이므로, 외부 호출 시에는 학습을 건너뛴다.
        실제 업데이트는 on_episode_end()에서 수행한다."""
        return {"loss": 0.0, "skipped": True}

    def on_episode_end(self, episode: int):
        """에피소드 종료 시 롤아웃 데이터로 actor+critic 업데이트."""
        if len(self._states) == 0:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self._states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(self._actions).to(self.device)
        rewards = torch.FloatTensor(self._rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self._next_states, dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(self._dones).to(self.device)

        # Compute returns (discounted cumulative reward from each step)
        probs, values = self._forward(states)
        with torch.no_grad():
            _, next_values = self._forward(next_states)

        # Advantage: A(s,a) = r + gamma * V(s') * (1-done) - V(s)
        targets = rewards + self.gamma * next_values * (1.0 - dones)
        advantages = targets - values.detach()

        # Policy loss: -log_prob(a) * advantage
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss: MSE between V(s) and target return
        value_loss = F.mse_loss(values, targets.detach())

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode buffer
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_states.clear()
        self._dones.clear()

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
