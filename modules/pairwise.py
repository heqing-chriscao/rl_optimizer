"""
FakeML Pairwise Relationship Matrix
Translates raw docking energies into a 4-D comparison tensor.

Convention (matches MATLAB GetPairwiseRelationshipsMatrixFromFeatureWithFakeML):
    entry[i, p, j, p] = -1  if  energy[i,p] < energy[j,p]   (apt-i binds better)
                       = +1  if  energy[i,p] >= energy[j,p]  (apt-j binds better)
    Cross-protein entries are always 0.

This module is intentionally pure-NumPy and stateless so it can be hot-swapped
with an ML model that produces the same shaped output tensor.
"""

import numpy as np


def compute_fakeml_pairwise_matrix(top_energies: np.ndarray) -> np.ndarray:
    """
    Build the 4-D pairwise comparison tensor from top-1 docking energies.

    Parameters
    ----------
    top_energies : np.ndarray, shape [n_apts, n_proteins]
        Best (most negative) docking energy for each (aptamer, protein) pair.

    Returns
    -------
    pairwise : np.ndarray, shape [n_apts, n_proteins, n_apts, n_proteins]
        pairwise[i, p, j, p] in {-1, +1}; all [i, p, j, q] with p != q are 0.
    """
    n_apts, n_proteins = top_energies.shape

    # Broadcast: ei[i,p,1,1] vs ej[1,1,j,q]
    ei = top_energies[:, :, np.newaxis, np.newaxis]   # [n_apts, n_prot, 1,      1     ]
    ej = top_energies[np.newaxis, np.newaxis, :, :]   # [1,      1,      n_apts, n_prot]

    # +1 where energy_i >= energy_j, -1 where energy_i < energy_j
    pairwise = np.where(ei >= ej, 1.0, -1.0)          # [n_apts, n_prot, n_apts, n_prot]

    # Zero out cross-protein comparisons: only keep entries where p == q
    p_idx = np.arange(n_proteins)
    same_protein = (p_idx[:, np.newaxis] == p_idx[np.newaxis, :])   # [n_prot, n_prot]
    same_protein = same_protein[np.newaxis, :, np.newaxis, :]        # [1, n_prot, 1, n_prot]
    pairwise *= same_protein

    return pairwise
