"""Decision Transformer — RL을 시퀀스 예측 문제로 변환.

과거 거래 이력에서 바로 학습 (Offline RL).
실시간 시뮬레이션 없이 과거 데이터만으로 학습 가능.

원리:
  GPT처럼 (목표수익, 상태, 행동) 시퀀스를 입력받아
  다음 행동을 예측. 목표수익을 높게 설정하면 고수익 전략,
  낮게 설정하면 보수적 전략을 생성.

구조:
  [R_1, s_1, a_1, R_2, s_2, a_2, ...] -> GPT-style Transformer -> a_next
"""

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent
from .registry import AgentRegistry


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@AgentRegistry.register("decision_transformer")
class DecisionTransformerAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 num_actions: int = 3, batch_size: int = 64,
                 context_len: int = 20, target_return: float = 0.5,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, max_episodes: int = 1000,
                 **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.context_len = context_len
        self.target_return = target_return
        self.max_episodes = max_episodes

        output_dim = network.output_dim
        # The transformer embed_dim = output_dim (feature dimension from network)
        embed_dim = output_dim

        # 3 token types per timestep: (return-to-go, state, action)
        max_seq_len = 3 * context_len

        # Embeddings
        self.return_embed = nn.Linear(1, embed_dim).to(device)
        # State embedding uses the network, then a projection if needed
        self.state_embed = nn.Linear(output_dim, embed_dim).to(device)
        self.action_embed = nn.Embedding(num_actions, embed_dim).to(device)
        # Timestep embedding (for position within episode)
        self.timestep_embed = nn.Embedding(4096, embed_dim).to(device)
        # Layer norm after embedding
        self.embed_ln = nn.LayerNorm(embed_dim).to(device)
        self.embed_drop = nn.Dropout(dropout).to(device)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ]).to(device)

        # Action prediction head (predict from state token position)
        self.action_head = nn.Linear(embed_dim, num_actions).to(device)

        # Optimizer: network + all transformer params
        all_params = (
            list(self.network.parameters())
            + list(self.return_embed.parameters())
            + list(self.state_embed.parameters())
            + list(self.action_embed.parameters())
            + list(self.timestep_embed.parameters())
            + list(self.embed_ln.parameters())
            + list(self.blocks.parameters())
            + list(self.action_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

        # Episode storage for online collection
        self._current_episode = []  # list of (state, action, reward)
        self._episodes = []  # list of completed episodes
        self._max_stored_episodes = max_episodes

        # Running context for inference
        self._context_states = []
        self._context_actions = []
        self._context_returns = []
        self._context_timesteps = []
        self._current_return_to_go = target_return
        self._current_timestep = 0

    def _build_sequence(self, returns_to_go, states, actions, timesteps):
        """Build interleaved token sequence: (R, s, a, R, s, a, ...).

        Args:
            returns_to_go: (batch, seq_len, 1)
            states: (batch, seq_len, state_dim) raw states
            actions: (batch, seq_len) action indices
            timesteps: (batch, seq_len) timestep indices

        Returns:
            token_embeddings: (batch, 3 * seq_len, embed_dim)
        """
        batch_size, seq_len = actions.shape

        # Get state features through the network
        # states may be (batch, seq_len, *state_shape) — flatten per timestep
        flat_states = states.reshape(batch_size * seq_len, -1)
        state_features = self.network(flat_states)
        state_features = state_features.reshape(batch_size, seq_len, -1)

        # Embed each modality
        r_emb = self.return_embed(returns_to_go)  # (batch, seq, embed_dim)
        s_emb = self.state_embed(state_features)   # (batch, seq, embed_dim)
        a_emb = self.action_embed(actions)          # (batch, seq, embed_dim)

        # Add timestep embeddings
        t_emb = self.timestep_embed(timesteps)      # (batch, seq, embed_dim)
        r_emb = r_emb + t_emb
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Interleave: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (batch, 3 * seq_len, embed_dim)
        tokens = torch.zeros(batch_size, 3 * seq_len, r_emb.shape[-1],
                             device=self.device)
        tokens[:, 0::3] = r_emb
        tokens[:, 1::3] = s_emb
        tokens[:, 2::3] = a_emb

        tokens = self.embed_ln(tokens)
        tokens = self.embed_drop(tokens)
        return tokens

    def select_action(self, state, explore: bool = True) -> int:
        # Add current state to context
        self._context_states.append(np.array(state, dtype=np.float32))
        self._context_returns.append(self._current_return_to_go)
        self._context_timesteps.append(self._current_timestep)

        # Trim context to context_len
        ctx_len = self.context_len
        ctx_states = self._context_states[-ctx_len:]
        ctx_returns = self._context_returns[-ctx_len:]
        ctx_timesteps = self._context_timesteps[-ctx_len:]
        # Actions: we have one fewer action than states (no action for current state yet)
        ctx_actions = self._context_actions[-(ctx_len - 1):] if self._context_actions else []

        seq_len = len(ctx_states)

        if seq_len < 2 or len(ctx_actions) == 0:
            # Not enough context, pick randomly or greedily
            if explore:
                action = random.randrange(self.num_actions)
            else:
                action = 0
            self._context_actions.append(action)
            self._current_timestep += 1
            return action

        # Build tensors for inference
        # We need equal length for R, s, a; pad actions with 0 for the last position
        padded_actions = list(ctx_actions) + [0]  # placeholder for current action
        assert len(padded_actions) == seq_len

        states_t = torch.FloatTensor(np.array(ctx_states)).unsqueeze(0).to(self.device)
        returns_t = torch.FloatTensor(ctx_returns).unsqueeze(0).unsqueeze(-1).to(self.device)
        actions_t = torch.LongTensor(padded_actions).unsqueeze(0).to(self.device)
        timesteps_t = torch.LongTensor(ctx_timesteps).unsqueeze(0).to(self.device)
        timesteps_t = timesteps_t.clamp(max=4095)

        with torch.no_grad():
            tokens = self._build_sequence(returns_t, states_t, actions_t, timesteps_t)
            for block in self.blocks:
                tokens = block(tokens)

            # Predict action from the last state token position
            # State tokens are at positions 1, 4, 7, ... i.e., 3*i + 1
            last_state_idx = 3 * (seq_len - 1) + 1
            state_token = tokens[:, last_state_idx]
            logits = self.action_head(state_token)  # (1, num_actions)

        if explore:
            # Sample from softmax distribution
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        else:
            action = logits.argmax(dim=-1).item()

        self._context_actions.append(action)
        self._current_timestep += 1
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self._current_episode.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
        ))
        if done:
            if len(self._current_episode) >= 2:
                self._episodes.append(list(self._current_episode))
                if len(self._episodes) > self._max_stored_episodes:
                    self._episodes.pop(0)
            self._current_episode = []

    def _compute_returns_to_go(self, rewards):
        """Compute return-to-go for each timestep."""
        rtg = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            rtg[t] = running
        return rtg

    def _sample_training_batch(self):
        """Sample a batch of sequences from stored episodes."""
        if not self._episodes:
            return None

        batch_states = []
        batch_actions = []
        batch_returns = []
        batch_timesteps = []
        batch_targets = []

        for _ in range(self.batch_size):
            ep = random.choice(self._episodes)
            ep_len = len(ep)

            states = [t[0] for t in ep]
            actions = [t[1] for t in ep]
            rewards = [t[2] for t in ep]

            rtg = self._compute_returns_to_go(rewards)

            # Random starting point
            if ep_len <= self.context_len:
                start = 0
                end = ep_len
            else:
                start = random.randint(0, ep_len - self.context_len)
                end = start + self.context_len

            seq_len = end - start
            # Pad to context_len if needed
            pad_len = self.context_len - seq_len

            s = np.array(states[start:end], dtype=np.float32)
            a = np.array(actions[start:end], dtype=np.int64)
            r = rtg[start:end]
            t = np.arange(start, end, dtype=np.int64)

            if pad_len > 0:
                pad_shape = (pad_len,) + s.shape[1:]
                s = np.concatenate([np.zeros(pad_shape, dtype=np.float32), s])
                a = np.concatenate([np.zeros(pad_len, dtype=np.int64), a])
                r = np.concatenate([np.zeros(pad_len, dtype=np.float32), r])
                t = np.concatenate([np.zeros(pad_len, dtype=np.int64), t])

            batch_states.append(s)
            batch_actions.append(a)
            batch_returns.append(r)
            batch_timesteps.append(t)
            batch_targets.append(a.copy())  # target = actual actions taken

        return {
            "states": np.array(batch_states),
            "actions": np.array(batch_actions),
            "returns_to_go": np.array(batch_returns),
            "timesteps": np.array(batch_timesteps),
            "targets": np.array(batch_targets),
        }

    def train_step(self, batch: dict = None) -> dict:
        if not self._episodes:
            return {"loss": 0.0, "skipped": True}

        if batch is None:
            batch = self._sample_training_batch()

        if batch is None:
            return {"loss": 0.0, "skipped": True}

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        returns_to_go = torch.FloatTensor(batch["returns_to_go"]).unsqueeze(-1).to(self.device)
        timesteps = torch.LongTensor(batch["timesteps"]).clamp(max=4095).to(self.device)
        targets = torch.LongTensor(batch["targets"]).to(self.device)

        # Forward pass
        tokens = self._build_sequence(returns_to_go, states, actions, timesteps)
        for block in self.blocks:
            tokens = block(tokens)

        # Extract state token positions (1, 4, 7, ...)
        seq_len = actions.shape[1]
        state_indices = torch.arange(seq_len, device=self.device) * 3 + 1
        state_tokens = tokens[:, state_indices]  # (batch, seq_len, embed_dim)

        # Predict actions
        logits = self.action_head(state_tokens)  # (batch, seq_len, num_actions)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_actions),
            targets.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters())
            + list(self.return_embed.parameters())
            + list(self.state_embed.parameters())
            + list(self.action_embed.parameters())
            + list(self.timestep_embed.parameters())
            + list(self.blocks.parameters())
            + list(self.action_head.parameters()),
            1.0,
        )
        self.optimizer.step()

        return {"loss": loss.item(), "skipped": False}

    def on_episode_end(self, episode: int):
        """Reset context for next episode."""
        self._context_states = []
        self._context_actions = []
        self._context_returns = []
        self._context_timesteps = []
        self._current_return_to_go = self.target_return
        self._current_timestep = 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "return_embed": self.return_embed.state_dict(),
            "state_embed": self.state_embed.state_dict(),
            "action_embed": self.action_embed.state_dict(),
            "timestep_embed": self.timestep_embed.state_dict(),
            "embed_ln": self.embed_ln.state_dict(),
            "blocks": self.blocks.state_dict(),
            "action_head": self.action_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "target_return": self.target_return,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.return_embed.load_state_dict(checkpoint["return_embed"])
        self.state_embed.load_state_dict(checkpoint["state_embed"])
        self.action_embed.load_state_dict(checkpoint["action_embed"])
        self.timestep_embed.load_state_dict(checkpoint["timestep_embed"])
        self.embed_ln.load_state_dict(checkpoint["embed_ln"])
        self.blocks.load_state_dict(checkpoint["blocks"])
        self.action_head.load_state_dict(checkpoint["action_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.target_return = checkpoint.get("target_return", self.target_return)
