"""
Supervised pretraining: teach AptamerTransformer to mimic SPSAUpdater.

Loads (obs, action) pairs collected by collect_data.py and trains the
policy head with per-position cross-entropy loss.  The value head is
left untouched (it will be fine-tuned during PPO).

Usage:
    python train_sl.py
    python train_sl.py --data data.npz --epochs 50 --lr 1e-3
    python train_sl.py --data data.npz --out ../nn_updater_sl.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from nn_updater import AptamerTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str,   default='data.npz')
    p.add_argument('--out',        type=str,   default='../nn_updater_sl.pt')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--val_frac',   type=float, default=0.1,
                   help='Fraction of data held out for validation')
    p.add_argument('--d_model',    type=int,   default=128)
    p.add_argument('--num_layers', type=int,   default=2)
    p.add_argument('--device',     type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), args.data)
    data = np.load(data_path)
    obs     = torch.tensor(data['obs'],    dtype=torch.float32)   # [N, H, N+2, 4L+1]
    actions = torch.tensor(data['action'], dtype=torch.long)      # [N, L]
    L = (obs.shape[-1] - 1) // 4
    print(f"Loaded {len(obs)} samples from {data_path}  (seq_length={L})")

    dataset  = TensorDataset(obs, actions)
    n_val    = max(1, int(len(dataset) * args.val_frac))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AptamerTransformer(
        d_model    = args.d_model,
        nhead      = 4,
        num_layers = args.num_layers,
    ).to(args.device)

    optim   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        train_losses, train_accs = [], []

        for step, (obs_b, act_b) in enumerate(train_dl, 1):
            obs_b = obs_b.to(args.device)   # [B, H, N, F]
            act_b = act_b.to(args.device)   # [B, L]

            logits, _ = model(obs_b)         # [B, L, 4]

            # Cross-entropy over all positions jointly
            loss = loss_fn(
                logits.reshape(-1, 4),       # [B*L, 4]
                act_b.reshape(-1),           # [B*L]
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            preds = logits.argmax(dim=-1)    # [B, L]
            acc   = (preds == act_b).float().mean().item()
            train_losses.append(loss.item())
            train_accs.append(acc)

            if step % 100 == 0:
                print(
                    f"  Epoch {epoch}/{args.epochs}  step {step}/{len(train_dl)}"
                    f"  loss={loss.item():.4f}  acc={acc:.3f}"
                )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for obs_b, act_b in val_dl:
                obs_b = obs_b.to(args.device)
                act_b = act_b.to(args.device)
                logits, _ = model(obs_b)
                loss = loss_fn(logits.reshape(-1, 4), act_b.reshape(-1))
                preds = logits.argmax(dim=-1)
                val_losses.append(loss.item())
                val_accs.append((preds == act_b).float().mean().item())

        val_loss = float(np.mean(val_losses))
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={np.mean(train_losses):.4f}"
            f"  train_acc={np.mean(train_accs):.3f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={np.mean(val_accs):.3f}"
            f"  time={time.time()-t_epoch:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                os.path.dirname(__file__), args.out + '.best'
            ))

    # ── Save final checkpoint ─────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    torch.save({"model": model.state_dict(), "steps": 0, "updates": 0}, out_path)
    print(f"Saved checkpoint -> {out_path}")
    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Final checkpoint saved -> {out_path}")
    print(f"Best checkpoint saved  -> {out_path}.best")


if __name__ == '__main__':
    main()
