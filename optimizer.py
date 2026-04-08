"""
SPSA Optimizer — Main Loop

Three NN-replaceable blocks are cleanly separated as injectable objects:
  - mutator  (Block 1): generates dumb candidates from current sequence
  - scorer   (Block 2): maps energy features -> per-aptamer scores
  - updater  (Block 3): produces the smart candidate from gradient estimate

To swap any block for a neural network, pass a subclass or duck-typed object
that satisfies the same interface documented in each module.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from modules.mutator  import RandomMutator
from modules.scorer   import PairwiseScorer
from modules.updater  import SPSAUpdater
from modules.docking  import DockingFeatureExtractor, generate_all_kmers


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    # Sequence
    initial_sequence: str
    fixed_positions: List[int]

    # Loop
    max_iterations: int
    num_internal_reps: int      # number of dumb random candidates per round
    max_mutation_rate: int
    less_than_or_equal: bool    # True  -> mutate 1..max_mutation_rate positions
                                # False -> always mutate exactly max_mutation_rate
    max_changes_per_round: int  # smart-update aggressiveness

    # Proteins (targets first, then counter-targets)
    target_protein_names: List[str]
    counter_protein_names: List[str]

    # Per-protein weights in the same order as proteins above
    # Matches MATLAB Weights(12:17) (1-indexed) = weights[11:17] (0-indexed)
    protein_weights: np.ndarray  # shape [n_targets + n_counter_targets]

    # Docking
    docker_dest_folder: str
    docker_name: str            # 'hdock' | 'autodock' | 'autodockgpu'

    # Output
    output_csv: str = 'evolution_data.csv'

    # Cache: path to a pickle file for pre-loaded energies.
    # If the file exists it is loaded; if not, all k-mers are preloaded and saved there.
    # Set to '' to disable (use lazy per-sequence loading instead).
    cache_path: str = ''


# ── Optimizer ─────────────────────────────────────────────────────────────────

class SPSAOptimizer:
    """
    Runs the SPSA aptamer optimisation loop.

    The three injectable blocks mirror MATLAB's structure:
      Block 1 — self.mutator  : generates NumInternalReps dumb candidates
      Block 2 — self.scorer   : maps docking energies -> weighted scores
      Block 3 — self.updater  : SPSA gradient -> one smart candidate

    To replace any block with a neural network, pass the NN object in the
    constructor.  It only needs to implement the same method signature.

    Parameters
    ----------
    config       : OptimizerConfig
    mutator      : optional override for Block 1
    scorer       : optional override for Block 2
    updater      : optional override for Block 3
    """

    def __init__(
        self,
        config: OptimizerConfig,
        mutator=None,
        scorer=None,
        updater=None,
    ):
        cfg = config
        self.cfg = cfg
        all_proteins = cfg.target_protein_names + cfg.counter_protein_names

        # ── Block 1 ───────────────────────────────────────────────────────────
        self.mutator = mutator or RandomMutator(
            max_mutation_rate=cfg.max_mutation_rate,
            fixed_positions=cfg.fixed_positions,
            less_than_or_equal=cfg.less_than_or_equal,
        )

        # ── Block 2 ───────────────────────────────────────────────────────────
        self.scorer = scorer or PairwiseScorer(
            protein_weights=cfg.protein_weights,
            n_targets=len(cfg.target_protein_names),
        )

        # ── Block 3 ───────────────────────────────────────────────────────────
        self.updater = updater or SPSAUpdater(
            max_changes_per_round=cfg.max_changes_per_round,
            fixed_positions=cfg.fixed_positions,
        )

        # ── Infrastructure (not replaceable) ──────────────────────────────────
        self.feature_extractor = DockingFeatureExtractor(
            dest_folder=cfg.docker_dest_folder,
            protein_names=all_proteins,
            docker_name=cfg.docker_name,
        )

        self.history: List[Dict[str, Any]] = []

        # ── Cache warm-up ──────────────────────────────────────────────────────
        if cfg.cache_path:
            k = len(cfg.initial_sequence)
            if os.path.exists(cfg.cache_path):
                self.feature_extractor.load_cache(cfg.cache_path)
            else:
                self.feature_extractor.preload_all(k)
                self.feature_extractor.save_cache(cfg.cache_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run the full optimisation loop and return the evolution table."""
        np.random.seed(0)
        cfg = self.cfg

        current_seq = cfg.initial_sequence
        self.history.append({
            'iteration':  0,
            'aptamer_seq': current_seq,
            'score':       np.nan,
            'update_type': 'initial',
        })

        for t in range(1, cfg.max_iterations + 1):
            print(f"\n── Iteration {t}/{cfg.max_iterations}  current: {current_seq} ──")

            # ── Block 1: generate dumb candidates ─────────────────────────────
            dumb_candidates = self.mutator.generate_candidates(
                current_seq, cfg.num_internal_reps
            )

            # ── Score batch-1 [current + dumb] to obtain current_score ────────
            batch1_seqs   = [current_seq] + dumb_candidates
            batch1_scores = self._effective_scores(batch1_seqs)
            current_score = batch1_scores[0]
            dumb_scores   = batch1_scores[1:]


            # ── Block 3: produce smart candidate ──────────────────────────────
            smart_candidate = self.updater.compute_update(
                current_seq=current_seq,
                candidates=dumb_candidates,
                candidate_scores=dumb_scores,
                current_score=current_score,
            )

            # ── Score full batch [dumb + smart + current] ─────────────────────
            # Indices: 0..N-1 = dumb, N = smart, N+1 = current
            full_batch    = dumb_candidates + [smart_candidate, current_seq]
            all_scores    = self._effective_scores(full_batch)  # [N+2]

            best_idx  = int(np.argmin(all_scores))
            best_seq  = full_batch[best_idx]
            best_score = float(all_scores[best_idx])

            N = cfg.num_internal_reps
            if best_idx < N:
                update_type = 'dumb'
            elif best_idx == N:
                update_type = 'smart'
            else:
                update_type = 'no_update'


            print(f"   {update_type}: {current_seq} -> {best_seq}  "
                  f"(score {best_score:.4f})")

            current_seq = best_seq
            self.history.append({
                'iteration':  t,
                'aptamer_seq': current_seq,
                'score':       best_score,
                'update_type': update_type,
            })

            pd.DataFrame(self.history).to_csv(cfg.output_csv, index=False)

        print("\nOptimisation complete.")
        return pd.DataFrame(self.history)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _effective_scores(self, sequences: List[str]) -> np.ndarray:
        """
        Compute effective score = target_score - counter_score for each sequence.

        Deduplicates before calling the feature extractor and scorer so that
        identical sequences in the batch are docked only once (cache hit on
        subsequent rounds).

        Returns np.ndarray of shape [len(sequences)].
        """
        unique_seqs, reverse_map = _deduplicate(sequences)

        # Feature extraction (infrastructure; cached internally)
        energy_matrix = self.feature_extractor.get_energy_matrix(unique_seqs)
        # [n_unique, n_proteins]

        # Block 2: score unique sequences
        target_scores, counter_scores = self.scorer.compute_scores(energy_matrix)
        unique_effective = target_scores - counter_scores    # [n_unique]

        # Broadcast back to the full (possibly duplicated) batch
        return unique_effective[reverse_map]


# ── Utility ───────────────────────────────────────────────────────────────────

def _deduplicate(sequences: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Return (unique_seqs, reverse_map) where reverse_map[i] is the index into
    unique_seqs that corresponds to sequences[i].
    """
    seen: Dict[str, int] = {}
    unique_seqs: List[str] = []
    reverse_map = np.empty(len(sequences), dtype=int)

    for i, seq in enumerate(sequences):
        if seq not in seen:
            seen[seq] = len(unique_seqs)
            unique_seqs.append(seq)
        reverse_map[i] = seen[seq]

    return unique_seqs, reverse_map
