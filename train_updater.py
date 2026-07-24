"""
Online PPO training for the NN updater.

The optimizer runs for `max_iterations` steps per episode.  The NNUpdater
collects transitions and triggers PPO updates internally every
`rollout_length` steps.

Usage:
    python train_updater.py
    python train_updater.py --episodes 200 --device cuda
    python train_updater.py --checkpoint nn_updater_ep0050.pt --episodes 100
"""

from __future__ import annotations

import argparse

import torch

from config import get_6mer_config
from modules.nn_updater import AptamerTransformer, NNUpdater
from optimizer import SPSAOptimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--episodes',    type=int,   default=100,
                   help='Number of optimisation episodes to train over')
    p.add_argument('--device',      type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint',  type=str,   default=None,
                   help='Path to a saved checkpoint to resume from')
    p.add_argument('--save_every',  type=int,   default=10,
                   help='Save a checkpoint every N episodes')
    p.add_argument('--d_model',     type=int,   default=128)
    p.add_argument('--num_layers',  type=int,   default=2)
    p.add_argument('--rollout',     type=int,   default=20,
                   help='Steps collected before each PPO update')
    p.add_argument('--ppo_epochs',  type=int,   default=4)
    p.add_argument('--lr',          type=float, default=3e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = get_6mer_config(max_iterations=60)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AptamerTransformer(
        history_length  = cfg.history_length,
        full_batch_size = cfg.num_internal_reps + 2,
        seq_length      = len(cfg.initial_sequence),
        d_model         = args.d_model,
        nhead           = 4,
        num_layers      = args.num_layers,
    )

    nn_updater = NNUpdater(
        model          = model,
        seq_length     = len(cfg.initial_sequence),
        device         = args.device,
        deterministic  = False,
        rollout_length = args.rollout,
        ppo_epochs     = args.ppo_epochs,
        lr             = args.lr,
    )

    if args.checkpoint:
        nn_updater.load(args.checkpoint)

    # ── Optimizer (built once — cache load and ranking table are expensive) ───
    opt = SPSAOptimizer(cfg, updater=nn_updater)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"Training on {args.device} for {args.episodes} episodes.\n")

    for ep in range(1, args.episodes + 1):
        df = opt.run(seed=ep)   # different seed per episode for diverse trajectories

        final_rank   = int(df['rank'].iloc[-1])
        total_reward = float(df['reward'].dropna().sum())
        stats        = nn_updater.last_stats

        stats_str = (
            f"  pg={stats['pg_loss']:.4f}  vl={stats['v_loss']:.4f}"
            f"  ent={stats['entropy']:.4f}  ret={stats['mean_ret']:.3f}"
            if stats else ""
        )
        print(
            f"Episode {ep:4d}/{args.episodes}"
            f"  final_rank={final_rank:5d}"
            f"  reward_sum={total_reward:+.0f}"
            f"  steps={nn_updater.total_steps}"
            f"  updates={nn_updater.total_updates}"
            + stats_str
        )

        if ep % args.save_every == 0:
            nn_updater.save(f'nn_updater_ep{ep:04d}.pt')

    print("\nTraining complete.")
    nn_updater.save('nn_updater_final.pt')


if __name__ == '__main__':
    main()
