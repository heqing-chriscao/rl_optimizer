"""
Block 2: Score Computation from Pairwise Matrix
Converts the 4-D pairwise tensor into per-aptamer scalar scores.

NN REPLACEMENT POINT
--------------------
Replace `PairwiseScorer.compute_scores` with a learned scoring function.
The interface is:
    compute_scores(top_energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        top_energies : [n_apts, n_proteins]  (targets first, then counter-targets)
        returns      : (target_scores [n_apts], counter_scores [n_apts])

A NN could map the raw energy matrix directly to scores without going through
the pairwise matrix, or it could learn a better aggregation of the pairwise tensor.

Score convention: lower (more negative) = better aptamer.
"""

import numpy as np
from typing import Tuple

from .pairwise import compute_fakeml_pairwise_matrix


def _aggregate_pairwise_to_scores(pairwise: np.ndarray) -> np.ndarray:
    """
    Row-sum the pairwise tensor (after zeroing the diagonal) and normalise.

    Maps MATLAB's GetScoresFromPairwiseRelationshipsMatrix exactly:
      - Reshape [n_apts, n_prot, n_apts, n_prot] -> [n_apts*n_prot, n_apts*n_prot]
      - Zero the diagonal (self-comparison)
      - Sum each row, divide by (n_apts - 1)
      - Reshape back to [n_apts, n_prot]

    Returns
    -------
    scores : [n_apts, n_proteins]
        More-negative score on a given protein means the aptamer outperforms others.
    """
    n_apts, n_proteins = pairwise.shape[0], pairwise.shape[1]
    # NumPy C-order reshape: row r = i*n_proteins + p  <->  (apt_i, prot_p)
    flat = pairwise.reshape(n_apts * n_proteins, n_apts * n_proteins).copy()
    np.fill_diagonal(flat, 0.0)
    row_sums = flat.sum(axis=1)
    if n_apts > 1:
        row_sums /= (n_apts - 1)
    return row_sums.reshape(n_apts, n_proteins)


class PairwiseScorer:
    """
    Computes weighted target and counter-target scores via the FakeML pairwise rule.

    Parameters
    ----------
    protein_weights : np.ndarray, shape [n_proteins]
        Per-protein weights in the order [targets..., counter_targets...].
        Matches MATLAB Weights(12:17) (1-indexed) = Python weights[11:17] (0-indexed).
    n_targets : int
        Number of target proteins (the remaining columns are counter-targets).
    """

    def __init__(self, protein_weights: np.ndarray, n_targets: int):
        self.protein_weights = np.asarray(protein_weights, dtype=float)
        self.n_targets = n_targets

    def compute_scores(
        self,
        top_energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-aptamer target and counter-target scores.

        Parameters
        ----------
        top_energies : [n_apts, n_proteins]
            Top-1 docking energy per (aptamer, protein).  NaN is tolerated.

        Returns
        -------
        target_scores   : [n_apts]  (lower = better binding to targets)
        counter_scores  : [n_apts]  (higher = worse binding to counter-targets, i.e. desirable)
        """
        pairwise = compute_fakeml_pairwise_matrix(top_energies)
        per_protein = _aggregate_pairwise_to_scores(pairwise)   # [n_apts, n_proteins]

        # Apply per-protein weights and normalise by total absolute weight
        weight_sum = np.sum(np.abs(self.protein_weights))
        weighted = per_protein * self.protein_weights[np.newaxis, :]
        if weight_sum > 0:
            weighted /= weight_sum

        target_scores  = weighted[:, :self.n_targets].sum(axis=1)
        counter_scores = weighted[:, self.n_targets:].sum(axis=1)
        return target_scores, counter_scores
