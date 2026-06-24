"""Rainbow DQN — 핵심 개선 기법을 결합한 강화 DQN.

포함된 기법:
  1. Double Q-learning (과대평가 방지)
  2. Prioritized Experience Replay (중요 경험 우선 학습)
  3. Dueling Network (가치/이점 분리)
  4. Multi-step Returns (n-step 부트스트래핑)
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


class PrioritizedReplayBuffer:
    """Proportional priority replay buffer using a parallel priority list."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self._max_priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # importance-sampling weights
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "indices": indices,
            "weights": np.array(weights, dtype=np.float32),
        }

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + 1e-6
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    """Accumulates n-step transitions before pushing to the main buffer."""

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def append(self, transition):
        self.buffer.append(transition)

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def get(self):
        """Compute n-step return and return (state, action, n_step_reward, nth_next_state, done)."""
        state, action = self.buffer[0][0], self.buffer[0][1]
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                # Episode ended within n steps
                return state, action, n_step_reward, self.buffer[i][3], True
        last = self.buffer[-1]
        return state, action, n_step_reward, last[3], last[4]

    def flush(self):
        """Flush remaining transitions at episode end."""
        results = []
        while len(self.buffer) > 0:
            state, action = self.buffer[0][0], self.buffer[0][1]
            n_step_reward = 0.0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    results.append((state, action, n_step_reward, self.buffer[i][3], True))
                    break
            else:
                last = self.buffer[-1]
                results.append((state, action, n_step_reward, last[3], last[4]))
            self.buffer.popleft()
        return results


@AgentRegistry.register("rainbow")
class RainbowDQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 6.25e-5,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 n_step: int = 3,
                 priority_alpha: float = 0.6, priority_beta: float = 0.4,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update: int = 10,
                 replay_size: int = 10000, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.n_step = n_step
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        output_dim = network.output_dim

        # Dueling heads
        self.value_head = nn.Linear(output_dim, 1).to(device)
        self.advantage_head = nn.Linear(output_dim, num_actions).to(device)

        # Target network
        self.target_network = copy.deepcopy(network).to(device)
        self.target_value_head = copy.deepcopy(self.value_head).to(device)
        self.target_advantage_head = copy.deepcopy(self.advantage_head).to(device)
        self.target_network.eval()
        self.target_value_head.eval()
        self.target_advantage_head.eval()

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(replay_size, alpha=priority_alpha)

        # N-step buffer
        self.n_step_buffer = NStepBuffer(n_step, gamma)

        # Optimizer
        params = (list(self.network.parameters())
                  + list(self.value_head.parameters())
                  + list(self.advantage_head.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.train_count = 0

    def _get_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Dueling Q-values: Q(s,a) = V(s) + A(s,a) - mean(A)."""
        features = self.network(state_tensor)
        value = self.value_head(features)            # (batch, 1)
        advantage = self.advantage_head(features)    # (batch, num_actions)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def _get_target_q_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Target dueling Q-values."""
        with torch.no_grad():
            features = self.target_network(state_tensor)
            value = self.target_value_head(features)
            advantage = self.target_advantage_head(features)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def select_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._get_q_values(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if self.n_step_buffer.is_ready():
            s, a, r, ns, d = self.n_step_buffer.get()
            self.replay_buffer.push(s, a, r, ns, d)
        if done:
            for transition in self.n_step_buffer.flush():
                self.replay_buffer.push(*transition)

    def train_step(self, batch: dict = None) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "skipped": True}

        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size, beta=self.priority_beta)

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        weights = torch.FloatTensor(batch["weights"]).to(self.device)
        indices = batch["indices"]

        # Current Q-values
        q_values = self._get_q_values(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects action, target evaluates
        with torch.no_grad():
            online_next_q = self._get_q_values(next_states)
            best_actions = online_next_q.argmax(dim=1, keepdim=True)
            target_next_q = self._get_target_q_values(next_states)
            next_q = target_next_q.gather(1, best_actions).squeeze(1)
            # n-step return: gamma^n
            target = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        td_errors = (q_values - target).detach().cpu().numpy()

        # Weighted loss
        loss = (weights * F.smooth_l1_loss(q_values, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters())
            + list(self.value_head.parameters())
            + list(self.advantage_head.parameters()),
            10.0,
        )
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        self.train_count += 1
        return {"loss": loss.item(), "skipped": False}

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())
        self.target_advantage_head.load_state_dict(self.advantage_head.state_dict())

    def on_episode_end(self, episode: int):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if episode % self.target_update == 0:
            self.update_target()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "value_head": self.value_head.state_dict(),
            "advantage_head": self.advantage_head.state_dict(),
            "target_network": self.target_network.state_dict(),
            "target_value_head": self.target_value_head.state_dict(),
            "target_advantage_head": self.target_advantage_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_count": self.train_count,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.advantage_head.load_state_dict(checkpoint["advantage_head"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_value_head.load_state_dict(checkpoint["target_value_head"])
        self.target_advantage_head.load_state_dict(checkpoint["target_advantage_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_count = checkpoint.get("train_count", 0)
