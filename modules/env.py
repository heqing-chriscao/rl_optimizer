"""
RL Environment for Aptamer Optimisation.

State  : AptamerState — fixed-shape tensor obs [history_length, full_batch_size, 4L+1]
         plus current_seq (str) and current_rank (int) for infrastructure use.
Action : next sequence (str)
Reward : rank improvement = rank(s_t) - rank(s_{t+1})  (positive = better)

obs layout per step:
  obs[t, 0..N-1, :]  = dumb candidates (one-hot + score)
  obs[t, N,      :]  = smart candidate (zeros until env.step() is called)
  obs[t, N+1,    :]  = current sequence (one-hot + score)

Rankings are computed from the full k-mer energy cache using the same weighted
FakeML pairwise scoring as PairwiseScorer, but applied to ALL sequences at once
via an O(n log n) sort-based method (avoids the O(n²) pairwise matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .docking import DockingFeatureExtractor, generate_all_kmers


NT_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _seq_to_onehot_flat(seq: str) -> np.ndarray:
    """Convert nucleotide string -> flat one-hot array [4L]."""
    arr = np.zeros(4 * len(seq), dtype=float)
    for i, c in enumerate(seq):
        arr[4 * i + NT_TO_INT[c]] = 1.0
    return arr


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class AptamerState:
    """
    obs          : [history_length, full_batch_size, 4L+1] tensor.
                   Zero-padded for steps not yet taken.
                   Smart slot (index N) is zeros until env.step() fills it in.
    current_seq  : most recently chosen sequence string (used by mutator/updater).
    current_rank : ground-truth rank of current_seq (used for reward and logging).
    """
    obs: np.ndarray
    current_seq: str
    current_rank: int


# ── Efficient all-sequence scoring ────────────────────────────────────────────

def _compute_all_effective_scores(
    energy_matrix: np.ndarray,
    protein_weights: np.ndarray,
    n_targets: int,
) -> np.ndarray:
    """
    Compute per-sequence effective score for every row of energy_matrix using
    a sort-based equivalent of FakeML pairwise comparison — O(n log n), no
    O(n²) matrix required.

    Mathematically equivalent to running PairwiseScorer.compute_scores() with
    all sequences in one batch.

    For each protein p and sequence i:
        per_protein[i,p] = (2 * #{j≠i: energy[j,p] <= energy[i,p]} - (n-1)) / (n-1)

    This equals the FakeML row-sum / (n-1) because:
        pairwise[i,p,j,p] = +1 if energy[i,p] >= energy[j,p], -1 otherwise
        row_sum = #{j≠i: +1} - #{j≠i: -1} = 2*#{j≠i: energy[j]<=energy[i]} - (n-1)

    NaN energies are treated as +inf (worst binding).
    """
    n_seqs, n_proteins = energy_matrix.shape
    per_protein = np.zeros((n_seqs, n_proteins))

    for p in range(n_proteins):
        col = energy_matrix[:, p].copy()
        col[np.isnan(col)] = np.inf
        sorted_col = np.sort(col)
        count_le = np.searchsorted(sorted_col, col, side='right') - 1
        per_protein[:, p] = (2 * count_le - (n_seqs - 1)) / (n_seqs - 1)

    weight_sum = np.sum(np.abs(protein_weights))
    weighted = per_protein * protein_weights[np.newaxis, :]
    if weight_sum > 0:
        weighted /= weight_sum

    effective = weighted[:, :n_targets].sum(axis=1) - weighted[:, n_targets:].sum(axis=1)
    return effective


# ── Environment ───────────────────────────────────────────────────────────────

class AptamerEnv:
    """
    RL environment wrapping the aptamer sequence space.

    On construction, computes a complete ranking table for all 4^k sequences
    using the energy cache and protein weights — no CSV file needed.

    Parameters
    ----------
    feature_extractor : DockingFeatureExtractor
        Must already have the full k-mer cache loaded.
    protein_weights : np.ndarray  shape [n_proteins]
        Same weights as OptimizerConfig.protein_weights.
    n_targets : int
        Number of target proteins (rest are counter-targets).
    seq_length : int
        k (6 or 7).
    full_batch_size : int
        num_internal_reps + 2  (dumb candidates + smart + current).
    history_length : int
        Number of steps to keep in AptamerState.obs (older steps are dropped).
    """

    def __init__(
        self,
        feature_extractor: DockingFeatureExtractor,
        protein_weights: np.ndarray,
        n_targets: int,
        seq_length: int,
        full_batch_size: int,
        history_length: int = 5,
    ):
        self.history_length = history_length
        self.seq_length = seq_length
        self.full_batch_size = full_batch_size
        self.feature_dim = 4 * seq_length + 1   # one-hot (4L) + score (1)

        print(f"Building ranking table for all {4**seq_length} {seq_length}-mers …")
        all_seqs = generate_all_kmers(seq_length)
        energy_matrix = feature_extractor.get_energy_matrix(all_seqs)
        effective = _compute_all_effective_scores(energy_matrix, protein_weights, n_targets)

        order = np.argsort(effective)
        self.rankings: Dict[str, int] = {
            all_seqs[idx]: int(rank + 1) for rank, idx in enumerate(order)
        }
        self.total_sequences = len(all_seqs)
        print(f"Ranking table ready. "
              f"Best: {all_seqs[order[0]]} (rank 1), "
              f"Worst: {all_seqs[order[-1]]} (rank {self.total_sequences})")

    def get_rank(self, seq: str) -> int:
        """Return the ground-truth rank of a sequence (1 = best)."""
        return self.rankings.get(seq, self.total_sequences)

    def reset(self, initial_seq: str) -> AptamerState:
        """Return the initial state for a new episode (obs is all zeros)."""
        obs = np.zeros(
            (self.history_length, self.full_batch_size, self.feature_dim),
            dtype=float,
        )
        return AptamerState(
            obs=obs,
            current_seq=initial_seq,
            current_rank=self.get_rank(initial_seq),
        )

    def observe(
        self,
        state: AptamerState,
        dumb_candidates: List[str],
        dumb_scores: np.ndarray,
        current_score: float,
    ) -> AptamerState:
        """
        Partially fill in the current step and slide the obs window.

        Encodes dumb candidates (indices 0..N-1) and the current sequence
        (index N+1) with their scores. Smart slot (index N) stays zeros.
        Called before the updater so the NN sees the current batch context.
        """
        N = len(dumb_candidates)
        L = self.seq_length
        step_obs = np.zeros((self.full_batch_size, self.feature_dim), dtype=float)

        for i, (seq, score) in enumerate(zip(dumb_candidates, dumb_scores)):
            step_obs[i, :4*L] = _seq_to_onehot_flat(seq)
            step_obs[i, -1]   = float(score)

        # smart slot (index N): zeros — placeholder for the updater's output

        step_obs[N+1, :4*L] = _seq_to_onehot_flat(state.current_seq)
        step_obs[N+1, -1]   = float(current_score)

        new_obs = np.roll(state.obs, -1, axis=0)
        new_obs[-1] = step_obs

        return AptamerState(
            obs=new_obs,
            current_seq=state.current_seq,
            current_rank=state.current_rank,
        )

    def step(
        self,
        state: AptamerState,
        best_seq: str,
        full_batch: List[str],
        all_scores: np.ndarray,
    ) -> Tuple[AptamerState, float]:
        """
        Finalise the current step: fill in the smart candidate slot, then advance.

        full_batch : [dumb_candidates..., smart_candidate, current_seq]
        all_scores : effective score for each entry in full_batch
        reward     : rank(current) - rank(best)  — positive means improvement.
        """
        N = self.full_batch_size - 2   # number of dumb candidates
        L = self.seq_length

        new_obs = state.obs.copy()
        new_obs[-1, N, :4*L] = _seq_to_onehot_flat(full_batch[N])   # fill smart one-hot
        new_obs[-1, :, -1]   = all_scores.astype(float)              # overwrite all scores with full-batch context

        next_rank = self.get_rank(best_seq)
        reward    = float(state.current_rank - next_rank)

        return AptamerState(
            obs=new_obs,
            current_seq=best_seq,
            current_rank=next_rank,
        ), reward
