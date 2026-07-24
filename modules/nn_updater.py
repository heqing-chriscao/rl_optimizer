"""
Block 3 (NN version): Transformer policy for smart candidate generation.

Implements the same interface as SPSAUpdater:
    compute_update(state, candidates, candidate_scores, current_score) -> str

Online PPO training is driven by the optimizer calling:
    record_outcome(reward, done)  after each env.step()

Architecture
------------
One token per history step; each token is the full candidate batch
flattened to shape [full_batch_size * (4L+1)].  The last-step
representation decodes to per-position nucleotide logits (policy head)
and a scalar value estimate (critic head).

TorchRL components used
-----------------------
- TensorDict          : rollout buffer storage and minibatch construction
- ProbabilisticActor  : actor that samples actions and stores log_prob
- ClipPPOLoss         : clipped surrogate + value + entropy losses

Note: TorchRL's GAE requires storing next-state observations for each
transition, which is expensive here.  Advantages are computed with a
manual GAE loop instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss

from .env import AptamerState

NUCLEOTIDES = ['A', 'C', 'G', 'T']


# ── Factored categorical distribution ─────────────────────────────────────────

class _FactoredCategorical:
    """
    Product of L independent Categoricals over 4 nucleotides.
    log_prob(action) sums the per-position log-probs.
    Accepts logits: [..., L, 4] and samples [..., L].
    """
    def __new__(cls, logits: torch.Tensor) -> Independent:
        return Independent(Categorical(logits=logits), 1)


# ── Model ─────────────────────────────────────────────────────────────────────

class AptamerTransformer(nn.Module):
    """
    Actor-critic Transformer for aptamer sequence generation.

    Parameters
    ----------
    history_length  : int   H — number of history steps (tokens)
    full_batch_size : int   N+2 — slots per step (dumb + smart + current)
    seq_length      : int   L — aptamer length in nucleotides
    d_model         : int   transformer hidden dimension
    nhead           : int   attention heads (must divide d_model)
    num_layers      : int   TransformerEncoder layers
    dropout         : float
    """

    def __init__(
        self,
        history_length: int,
        full_batch_size: int,
        seq_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_length = seq_length
        token_dim = full_batch_size * (4 * seq_length + 1)

        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embed  = nn.Embedding(history_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, seq_length * 4)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : [B, H, N+2, 4L+1]

        Returns
        -------
        logits : [B, L, 4]
        value  : [B]
        """
        B, H, N, F = obs.shape
        x = obs.reshape(B, H, N * F)
        x = self.input_proj(x)
        x = x + self.pos_embed(torch.arange(H, device=obs.device))
        x = self.transformer(x)
        x = x[:, -1, :]
        logits = self.policy_head(x).reshape(B, self.seq_length, 4)
        value  = self.value_head(x).squeeze(-1)
        return logits, value


# ── TorchRL thin wrappers ─────────────────────────────────────────────────────

class _PolicyModule(nn.Module):
    def __init__(self, transformer: AptamerTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.transformer(obs)
        return logits   # [B, L, 4]


class _CriticModule(nn.Module):
    def __init__(self, transformer: AptamerTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.transformer(obs)
        return value    # [B]


# ── Updater ───────────────────────────────────────────────────────────────────

class NNUpdater:
    """
    Online PPO-trained updater wrapping AptamerTransformer.

    Satisfies the same interface as SPSAUpdater.  PPO updates are
    triggered automatically every `rollout_length` steps when the
    optimizer calls record_outcome() after each env.step().

    The reward passed to record_outcome() should be the rank improvement
    of the smart candidate specifically:
        reward = rank(current_seq) - rank(smart_candidate)

    Parameters
    ----------
    model           : AptamerTransformer
    seq_length      : int
    device          : str
    deterministic   : bool   argmax if True, sample if False (training mode)
    rollout_length  : int    steps to collect before each PPO update
    ppo_epochs      : int    gradient passes per rollout
    minibatch_size  : int
    clip_eps        : float  PPO clip epsilon
    gamma           : float  discount factor
    gae_lambda      : float  GAE lambda
    entropy_coef    : float  entropy bonus weight
    value_coef      : float  value loss weight
    lr              : float  Adam learning rate
    max_grad_norm   : float  gradient clip norm
    """

    def __init__(
        self,
        model: AptamerTransformer,
        seq_length: int,
        device: str = 'cpu',
        deterministic: bool = False,
        rollout_length: int = 20,
        ppo_epochs: int = 4,
        minibatch_size: int = 8,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
    ):
        self.model          = model.to(device)
        self.seq_length     = seq_length
        self.device         = device
        self.deterministic  = deterministic
        self.rollout_length = rollout_length
        self.ppo_epochs     = ppo_epochs
        self.minibatch_size = minibatch_size
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.max_grad_norm  = max_grad_norm

        # ── TorchRL actor (used by ClipPPOLoss to recompute log_prob) ─────────
        self._actor = ProbabilisticActor(
            module=TensorDictModule(
                _PolicyModule(model),
                in_keys=["obs"],
                out_keys=["logits"],
            ),
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=_FactoredCategorical,
            return_log_prob=True,
            log_prob_key="sample_log_prob",
        )

        # ── TorchRL critic (used by ClipPPOLoss to recompute state_value) ─────
        self._critic = TensorDictModule(
            _CriticModule(model),
            in_keys=["obs"],
            out_keys=["state_value"],
        )

        # ── TorchRL PPO loss ───────────────────────────────────────────────────
        self._loss_fn = ClipPPOLoss(
            actor_network     = self._actor,
            critic_network    = self._critic,
            clip_epsilon      = clip_eps,
            entropy_bonus     = True,
            entropy_coeff     = entropy_coef,
            critic_coeff      = value_coef,
            normalize_advantage = False,
        )

        self._optim = torch.optim.Adam(model.parameters(), lr=lr)

        # Rollout buffer: list of scalar-batch TensorDicts
        self._buf: List[TensorDict] = []
        self._pending: Optional[TensorDict] = None

        self.total_steps   = 0
        self.total_updates = 0
        self.last_stats: Dict[str, float] = {}

    # ── Interface ──────────────────────────────────────────────────────────────

    def compute_update(
        self,
        state: AptamerState,
        candidates,        # ignored — already encoded in state.obs
        candidate_scores,  # ignored
        current_score,     # ignored
    ) -> str:
        """Sample (or argmax) a smart candidate from the policy."""
        obs_t = torch.tensor(state.obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_t)   # [1, L, 4], [1]

        logits_sq = logits.squeeze(0)           # [L, 4]

        if self.deterministic:
            action = logits_sq.argmax(dim=-1)   # [L]
        else:
            dist     = _FactoredCategorical(logits_sq)
            action   = dist.sample()             # [L]
            log_prob = dist.log_prob(action)     # scalar

            self._pending = TensorDict({
                "obs":             obs_t.squeeze(0),     # [H, N, F]
                "action":          action,               # [L]
                "sample_log_prob": log_prob,             # scalar
                "state_value":     value.squeeze(0),    # scalar
            }, batch_size=[])

        return ''.join(NUCLEOTIDES[i] for i in action.cpu().tolist())

    def record_outcome(self, reward: float, done: bool = False) -> None:
        """
        Complete the pending transition with its reward and trigger PPO if full.

        reward : rank(current_seq) - rank(smart_candidate)  (positive = improvement)
        done   : True on the last step of an optimization episode
        """
        if self._pending is None:
            return

        td = self._pending.clone()
        td["reward"] = torch.tensor(reward, dtype=torch.float32)
        td["done"]   = torch.tensor(float(done), dtype=torch.float32)
        self._buf.append(td)
        self._pending = None
        self.total_steps += 1

        if len(self._buf) >= self.rollout_length:
            self._ppo_update()

    # ── PPO ───────────────────────────────────────────────────────────────────

    def _ppo_update(self) -> None:
        T   = len(self._buf)
        dev = self.device

        obs     = torch.stack([td["obs"]             for td in self._buf]).to(dev)  # [T, H, N, F]
        actions = torch.stack([td["action"]          for td in self._buf]).to(dev)  # [T, L]
        old_lp  = torch.stack([td["sample_log_prob"] for td in self._buf]).to(dev)  # [T]
        values  = torch.stack([td["state_value"]     for td in self._buf]).to(dev)  # [T]
        rewards = torch.stack([td["reward"]          for td in self._buf]).to(dev)  # [T]
        dones   = torch.stack([td["done"]            for td in self._buf]).to(dev)  # [T]

        advantages = self._gae(rewards, values, dones)
        returns    = (advantages + values).detach()
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()

        pg_losses, v_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            for idx in torch.randperm(T).split(self.minibatch_size):
                mini = TensorDict({
                    "obs":             obs[idx],
                    "action":          actions[idx],
                    "sample_log_prob": old_lp[idx],
                    "advantage":       advantages[idx],
                    "value_target":    returns[idx],
                    "state_value":     values[idx],
                }, batch_size=[len(idx)]).to(dev)

                loss_td = self._loss_fn(mini)
                total   = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )

                self._optim.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optim.step()

                pg_losses.append(loss_td["loss_objective"].item())
                v_losses.append(loss_td["loss_critic"].item())
                entropies.append(loss_td["entropy"].item())

        self.total_updates += 1
        self.last_stats = {
            "pg_loss":  float(np.mean(pg_losses)),
            "v_loss":   float(np.mean(v_losses)),
            "entropy":  float(np.mean(entropies)),
            "mean_ret": float(returns.mean().item()),
        }
        self._buf.clear()

    def _gae(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
        dones:   torch.Tensor,
    ) -> torch.Tensor:
        T   = len(rewards)
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1].item() if t + 1 < T else 0.0
            delta    = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae      = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t]   = gae
        return adv

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "model":   self.model.state_dict(),
            "optim":   self._optim.state_dict(),
            "steps":   self.total_steps,
            "updates": self.total_updates,
        }, path)
        print(f"Saved checkpoint -> {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            self._optim.load_state_dict(ckpt["optim"])
        self.total_steps   = ckpt.get("steps",   0)
        self.total_updates = ckpt.get("updates",  0)
        print(f"Loaded checkpoint <- {path}")
