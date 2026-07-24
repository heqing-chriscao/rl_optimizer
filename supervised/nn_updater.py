"""
Block 3 (NN version) — Idea 2: flat sequence with multi-level positional encoding.

Architecture
------------
obs [B, H, N+2, 4L+1]
  split → onehot [B, H, N+2, L, 4]  +  score [B, H, N+2, 1]
  broadcast score → [B, H, N+2, L, 1]
  concat → [B, H, N+2, L, 5]
  Linear 5 → d_model
  + PE_H  : sinusoidal over H          (when in history)
  + E_type: learned embedding          (dumb / smart / current)
  + PE_L  : sinusoidal over L          (which nucleotide position)
  flatten → [B, H·(N+2)·L, d_model]
  TransformerEncoder
  policy head : tokens of current slot, last step → [B, L, d_model] → Linear → [B, L, 4]
  value  head : mean pool all tokens → [B, d_model] → Linear → [B]

H, N, L are inferred from obs shape at runtime — no fixed-size parameters.

NNUpdater interface is identical to modules/nn_updater.py and is drop-in
compatible with SPSAOptimizer.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent
# from tensordict import TensorDict
# from tensordict.nn import TensorDictModule
# from torchrl.modules import ProbabilisticActor
# from torchrl.objectives import ClipPPOLoss

from modules.env import AptamerState

NUCLEOTIDES = ['A', 'C', 'G', 'T']

# Slot type indices
_DUMB    = 0
_SMART   = 1
_CURRENT = 2


# ── Factored categorical distribution ─────────────────────────────────────────

class _FactoredCategorical:
    """Product of L independent Categoricals; log_prob sums over positions."""
    def __new__(cls, logits: torch.Tensor) -> Independent:
        return Independent(Categorical(logits=logits), 1)


# ── Model ─────────────────────────────────────────────────────────────────────

class AptamerTransformer(nn.Module):
    """
    Flat-sequence Transformer for aptamer sequence generation.

    All three dimensions (H, N, L) are read from obs at runtime, so the
    same model works for any sequence length, any number of candidates,
    and any history window without retraining.

    Parameters
    ----------
    d_model    : transformer hidden dimension
    nhead      : attention heads (must divide d_model)
    num_layers : TransformerEncoder layers
    dropout    : float
    """

    def __init__(
        self,
        d_model:    int   = 128,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj      = nn.Linear(5, d_model)   # onehot(4) + score(1)
        self.slot_type_embed = nn.Embedding(3, d_model) # dumb / smart / current

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, 4)
        self.value_head  = nn.Linear(d_model, 1)

    @staticmethod
    def _sinusoidal(n: int, d: int, device) -> torch.Tensor:
        """Sinusoidal positional encoding [n, d], computed on the fly."""
        pos    = torch.arange(n, device=device).float().unsqueeze(1)   # [n, 1]
        dim    = torch.arange(0, d, 2, device=device).float()          # [d/2]
        angles = pos / (10000 ** (dim / d))                            # [n, d/2]
        enc    = torch.zeros(n, d, device=device)
        enc[:, 0::2] = torch.sin(angles)
        enc[:, 1::2] = torch.cos(angles)
        return enc

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
        B, H, N2, feat = obs.shape
        L = (feat - 1) // 4
        N = N2 - 2          # number of dumb candidates
        dev = obs.device

        # ── Split one-hot and score ───────────────────────────────────────────
        onehot = obs[..., :4 * L].reshape(B, H, N2, L, 4)            # [B,H,N2,L,4]
        score  = obs[..., 4 * L:].unsqueeze(-2).expand(-1,-1,-1,L,-1) # [B,H,N2,L,1]
        x = self.input_proj(torch.cat([onehot, score], dim=-1))       # [B,H,N2,L,d]

        # ── Multi-level positional encoding ───────────────────────────────────
        pe_h   = self._sinusoidal(H,  self.d_model, dev)  # [H,  d]
        pe_l   = self._sinusoidal(L,  self.d_model, dev)  # [L,  d]

        slot_ids = torch.zeros(N2, dtype=torch.long, device=dev)
        slot_ids[N]   = _SMART
        slot_ids[N+1] = _CURRENT
        e_type = self.slot_type_embed(slot_ids)            # [N2, d]

        x = x + pe_h.view(1, H,  1,  1, self.d_model)
        x = x + e_type.view(1,  1, N2, 1, self.d_model)
        x = x + pe_l.view(1,  1,  1,  L, self.d_model)

        # ── Transformer ───────────────────────────────────────────────────────
        x = x.reshape(B, H * N2 * L, self.d_model)        # [B, H·N2·L, d]
        x = self.transformer(x)
        x = x.reshape(B, H, N2, L, self.d_model)

        # ── Policy head: current slot of last history step ────────────────────
        logits = self.policy_head(x[:, -1, N+1, :, :])    # [B, L, 4]

        # ── Value head: mean pool all tokens ─────────────────────────────────
        value  = self.value_head(x.mean(dim=(1, 2, 3))).squeeze(-1)  # [B]

        return logits, value


# ── TorchRL thin wrappers (unchanged from modules/nn_updater.py) ──────────────

class _PolicyModule(nn.Module):
    def __init__(self, transformer: AptamerTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.transformer(obs)
        return logits


class _CriticModule(nn.Module):
    def __init__(self, transformer: AptamerTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.transformer(obs)
        return value


# ── Updater (interface identical to modules/nn_updater.py) ───────────────────

class NNUpdater:
    """
    Online PPO-trained updater wrapping AptamerTransformer.

    Drop-in replacement for modules/nn_updater.NNUpdater — same constructor
    signature and same public methods (compute_update, record_outcome,
    save, load).
    """

    def __init__(
        self,
        model:          AptamerTransformer,
        seq_length:     int,
        device:         str   = 'cpu',
        deterministic:  bool  = False,
        rollout_length: int   = 20,
        ppo_epochs:     int   = 4,
        minibatch_size: int   = 8,
        clip_eps:       float = 0.2,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        entropy_coef:   float = 0.01,
        value_coef:     float = 0.5,
        lr:             float = 3e-4,
        max_grad_norm:  float = 0.5,
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

        self._critic = TensorDictModule(
            _CriticModule(model),
            in_keys=["obs"],
            out_keys=["state_value"],
        )

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
        self._buf:     List[TensorDict]           = []
        self._pending: Optional[TensorDict]       = None
        self.total_steps   = 0
        self.total_updates = 0
        self.last_stats: Dict[str, float]         = {}

    # ── Interface ──────────────────────────────────────────────────────────────

    def compute_update(
        self,
        state:            AptamerState,
        candidates,       # ignored — encoded in state.obs
        candidate_scores, # ignored
        current_score,    # ignored
    ) -> str:
        obs_t = torch.tensor(state.obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_t)   # [1, L, 4], [1]

        logits_sq = logits.squeeze(0)           # [L, 4]

        if self.deterministic:
            action = logits_sq.argmax(dim=-1)
        else:
            dist     = _FactoredCategorical(logits_sq)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

            self._pending = TensorDict({
                "obs":             obs_t.squeeze(0),
                "action":          action,
                "sample_log_prob": log_prob,
                "state_value":     value.squeeze(0),
            }, batch_size=[])

        return ''.join(NUCLEOTIDES[i] for i in action.cpu().tolist())

    def record_outcome(self, reward: float, done: bool = False) -> None:
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

    # ── PPO (identical to modules/nn_updater.py) ──────────────────────────────

    def _ppo_update(self) -> None:
        T, dev = len(self._buf), self.device
        obs     = torch.stack([td["obs"]             for td in self._buf]).to(dev)
        actions = torch.stack([td["action"]          for td in self._buf]).to(dev)
        old_lp  = torch.stack([td["sample_log_prob"] for td in self._buf]).to(dev)
        values  = torch.stack([td["state_value"]     for td in self._buf]).to(dev)
        rewards = torch.stack([td["reward"]          for td in self._buf]).to(dev)
        dones   = torch.stack([td["done"]            for td in self._buf]).to(dev)

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
                total   = (loss_td["loss_objective"]
                           + loss_td["loss_critic"]
                           + loss_td["loss_entropy"])
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

    def _gae(self, rewards, values, dones) -> torch.Tensor:
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
