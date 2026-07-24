"""
Collect (obs, action) pairs by running the SPSA optimizer across
randomly sampled protein weight configurations.

Each episode uses a freshly sampled weight vector, rebuilding the
ranking table for that objective.  The scores embedded in obs already
reflect the active weights, so the NN can learn weight-conditioned
SPSA behavior from obs alone.

Usage:
    python collect_data.py
    python collect_data.py --episodes 200 --out data.npz
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from config import get_6mer_config, TARGET_PROTEINS, COUNTER_PROTEINS
from modules.updater import SPSAUpdater
from modules.env import AptamerState
from optimizer import SPSAOptimizer, OptimizerConfig

NT_TO_INT    = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ALL_PROTEINS = TARGET_PROTEINS + COUNTER_PROTEINS


class _RecordingSPSAUpdater(SPSAUpdater):
    """SPSAUpdater that also saves (obs, action_indices) at each call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[tuple[np.ndarray, list[int]]] = []

    def compute_update(
        self,
        state: AptamerState,
        candidates,
        candidate_scores,
        current_score,
    ) -> str:
        seq = super().compute_update(state, candidates, candidate_scores, current_score)
        self.records.append((
            state.obs.copy(),
            [NT_TO_INT[c] for c in seq],
        ))
        return seq


def _sample_episode(rng: np.random.Generator):
    """
    Sample random protein roles and weights for one episode.

    Roles: randomly splits the 6 proteins into targets and counter-targets;
    n_targets is drawn from 0..6 (0 = all counters, 6 = all targets).
    Weights: all 6 drawn independently from [0, 5].
    Resampled only if every weight is zero.

    Returns
    -------
    targets  : list[str]   — proteins acting as targets this episode
    counters : list[str]   — proteins acting as counter-targets
    weights  : np.ndarray  — weights in (targets + counters) order
    """
    while True:
        perm     = rng.permutation(ALL_PROTEINS).tolist()
        n_target = int(rng.integers(0, len(ALL_PROTEINS) + 1))  # 0..6
        targets  = perm[:n_target]
        counters = perm[n_target:]
        weights  = rng.integers(0, 6, size=len(ALL_PROTEINS)).astype(float)
        if weights.sum() > 0:
            break

    return targets, counters, weights


NUCLEOTIDES = ['A', 'C', 'G', 'T']


def _random_seq(rng: np.random.Generator, length: int = 6) -> str:
    return ''.join(NUCLEOTIDES[i] for i in rng.integers(0, 4, size=length))


def _make_config(
    targets: list[str],
    counters: list[str],
    weights: np.ndarray,
    initial_seq: str,
    max_iterations: int,
) -> OptimizerConfig:
    base = get_6mer_config(max_iterations=max_iterations, initial_seq=initial_seq)
    base.target_protein_names  = targets
    base.counter_protein_names = counters
    base.protein_weights       = weights
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max_iter', type=int, default=60)
    p.add_argument('--out',      type=str, default='data.npz')
    p.add_argument('--seed',     type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    all_obs:     list[np.ndarray] = []
    all_actions: list[list[int]]  = []

    print(f"Collecting {args.episodes} episodes × {args.max_iter} steps "
          f"with randomised protein weights …\n")

    import io, contextlib

    for ep in range(1, args.episodes + 1):
        targets, counters, weights = _sample_episode(rng)
        initial_seq = _random_seq(rng)
        cfg = _make_config(targets, counters, weights, initial_seq, args.max_iter)

        updater = _RecordingSPSAUpdater(
            max_changes_per_round=cfg.max_changes_per_round,
            fixed_positions=cfg.fixed_positions,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            opt = SPSAOptimizer(cfg, updater=updater)
            opt.run(seed=int(rng.integers(1, 10_000)))

        all_obs.extend(r[0] for r in updater.records)
        all_actions.extend(r[1] for r in updater.records)

        if ep % 10 == 0 or ep == 1:
            print(f"  episode {ep:4d}/{args.episodes}"
                  f"  init={initial_seq}"
                  f"  targets={targets}  counters={counters}"
                  f"  weights={weights.astype(int).tolist()}"
                  f"  samples so far: {len(all_obs)}")

    obs_arr    = np.stack(all_obs).astype(np.float32)
    action_arr = np.array(all_actions, dtype=np.int64)

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    np.savez(out_path, obs=obs_arr, action=action_arr)
    print(f"\nSaved {len(all_obs)} samples -> {out_path}")
    print(f"obs shape: {obs_arr.shape}   action shape: {action_arr.shape}")


if __name__ == '__main__':
    main()
