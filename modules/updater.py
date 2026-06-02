"""
Block 3: Smart Update (SPSA Gradient Step)
Produces a gradient-informed candidate from the current batch of sequences+scores.

NN REPLACEMENT POINT
--------------------
Replace `SPSAUpdater.compute_update` with a learned policy.
The interface is:
    compute_update(
        state           : AptamerState,  # sliding-window history of (seq, rank) pairs
        candidates      : List[str],     # dumb perturbations only
        candidate_scores: np.ndarray,    # effective score for each candidate
        current_score   : float,
    ) -> str   # proposed new sequence

A RL policy would receive the AptamerState (with full history) and output a
sequence directly, without the manual gradient arithmetic below.
"""

import random
import numpy as np
from typing import List, Optional

from .env import AptamerState

NUCLEOTIDES = ['A', 'C', 'G', 'T']
NT_TO_INT   = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _seq_to_onehot(seq: str) -> np.ndarray:
    """Convert nucleotide string -> 4×L one-hot matrix (A=row0, C=row1, G=row2, T=row3)."""
    L   = len(seq)
    mat = np.zeros((4, L), dtype=float)
    for i, c in enumerate(seq):
        mat[NT_TO_INT[c], i] = 1.0
    return mat


def _onehot_to_seq(mat: np.ndarray) -> str:
    """Convert 4×L one-hot matrix -> nucleotide string."""
    return ''.join(NUCLEOTIDES[np.argmax(mat[:, i])] for i in range(mat.shape[1]))


class SPSAUpdater:
    """
    Produces one 'smart' candidate via SPSA-style gradient estimation.

    Mirrors MATLAB's UpdateCurrentAptamer exactly:
      1. For each non-fixed position i, accumulate:
             grad[:,i] += (score_j - current_score) * onehot(candidate_j[i])
         across all dumb candidates j.
      2. Average over candidate count.
      3. Zero out positive gradient entries (only accept score-decreasing moves).
      4. For each position, keep only the single nucleotide with the most-negative
         gradient (break ties randomly).
      5. Retain at most `max_changes_per_round` changing positions
         (the ones with the most-negative values).
      6. Apply changes to current_seq.

    Parameters
    ----------
    max_changes_per_round : int
        Maximum number of positions changed in one smart step.
    fixed_positions : list of int, optional
        0-based positions that are never mutated.
    """

    def __init__(
        self,
        max_changes_per_round: int,
        fixed_positions: Optional[List[int]] = None,
    ):
        self.max_changes_per_round = max_changes_per_round
        self.fixed_positions = set(fixed_positions or [])

    def compute_update(
        self,
        state: AptamerState,
        candidates: List[str],
        candidate_scores: np.ndarray,
        current_score: float,
    ) -> str:
        """
        Estimate SPSA gradient and return an improved candidate sequence.

        Parameters
        ----------
        state            : AptamerState with sliding-window history; state.current_seq
                           is the sequence to update from
        candidates       : list of dumb-perturbation sequences (length NumInternalReps)
        candidate_scores : effective score per candidate (same length as candidates)
        current_score    : effective score of state.current_seq

        Returns
        -------
        str : new candidate sequence (the 'smart update')
        """
        current_seq = state.current_seq
        L         = len(current_seq)
        non_fixed = [i for i in range(L) if i not in self.fixed_positions]
        grad      = np.zeros((4, L), dtype=float)

        # ── Accumulate SPSA gradient ──────────────────────────────────────────
        for cand, score in zip(candidates, candidate_scores):
            if not cand:
                continue
            score_diff = score - current_score
            for i in non_fixed:
                delta = np.zeros(4)
                delta[NT_TO_INT[cand[i]]] = 1.0
                grad[:, i] += score_diff * delta

        n = max(len(candidates), 1)
        grad /= n

        # ── Keep only score-decreasing moves ─────────────────────────────────
        grad[grad > 0] = 0.0

        # ── Per-position: keep only the single best (most-negative) nucleotide ─
        for i in non_fixed:
            col     = grad[:, i]
            min_val = col.min()
            if min_val >= 0:            # no beneficial move at this position
                grad[:, i] = 0.0
                continue
            tied = np.where(col == min_val)[0]
            keep = int(np.random.choice(tied))
            mask = np.zeros(4, dtype=bool)
            mask[keep] = True
            grad[~mask, i] = 0.0

        # ── Restrict to max_changes_per_round positions ───────────────────────
        nonzero_vals = grad[grad < 0]
        if len(nonzero_vals) > 0:
            k = min(self.max_changes_per_round, len(nonzero_vals))
            # threshold: k-th most-negative value
            threshold = np.sort(nonzero_vals)[k - 1]   # e.g. 4th smallest negative
            grad[grad > threshold] = 0.0                # remove less-important changes

        # ── Apply changes ─────────────────────────────────────────────────────
        new_seq = list(current_seq)
        for i in range(L):
            col = grad[:, i]
            if col.any():
                new_seq[i] = NUCLEOTIDES[int(np.argmax(col != 0))]

        return ''.join(new_seq)
