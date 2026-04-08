"""
Docking Feature Extraction
Reads pre-computed docking output files and returns top-1 binding energies.

This module is infrastructure (not one of the three NN-replaceable blocks).
It assumes all pdbqt/out files already exist on disk (no actual docking is run).

Naming conventions (matching MATLAB call_docker_once):
  hdock    : hdock_on{protein}__{apt_seq_stem}*RT_*.out
  autodock : autodock_on{protein}__{apt_seq_stem}*RT_*.pdbqt
  autodockgpu: autodockgpu_on{protein}__{apt_seq_stem}*RT_*.pdbqt

The aptamer file stem is:  APT_SEQ_{sequence}_RAND_TERM_{rand6chars}
so the search pattern strips the last 10 chars (.pdb suffix + 6-char random id)
from the full filename, which collapses to just APT_SEQ_{sequence}_RAND_TERM_*.
"""

import glob
import itertools
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── K-mer utilities ───────────────────────────────────────────────────────────

def generate_all_kmers(k: int) -> List[str]:
    """Return all 4^k nucleotide sequences of length k."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]


# ── File discovery ────────────────────────────────────────────────────────────

def _find_docking_file(
    dest_folder: str,
    protein_name: str,      # stem only, no extension
    aptamer_seq: str,
    docker_name: str,
) -> Optional[str]:
    """Return the first matching pre-computed docking output file, or None."""
    ext = '.out' if docker_name == 'hdock' else '.pdbqt'
    apt_stem = f"APT_SEQ_{aptamer_seq}_RAND_TERM_*"
    pattern = os.path.join(
        dest_folder,
        f"{docker_name}_on{protein_name}__{apt_stem}RT_*{ext}",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ── Energy parsers ────────────────────────────────────────────────────────────

def _parse_autodock_pdbqt(filepath: str, top_n: int) -> np.ndarray:
    """
    Extract binding energies from an AutoDock (CPU/GPU) pdbqt output.

    Handles two formats:
      1. CSV rows produced by generate_csv_with_geometrical_data_from_vina_pdbqt_file.py:
             -396.32, 0.000, 0.000, x, y, z, q1, q2, q3
         First field is the energy score.
      2. Vina-style REMARK lines:
             REMARK VINA RESULT:  <energy>  ...
      3. AutoDock GPU dlg-style:
             ... Estimated Free Energy of Binding    =   <energy> ...

    Returns an array of length top_n (most-negative first, NaN-padded).
    """
    energies: List[float] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Format 1: CSV — first token is a float energy value
            first = line.split(',')[0].strip()
            try:
                energies.append(float(first))
                continue
            except ValueError:
                pass
            # Format 2: Vina REMARK
            if line.startswith('REMARK VINA RESULT'):
                try:
                    energies.append(float(line.split()[3]))
                except (IndexError, ValueError):
                    pass
            # Format 3: AutoDock GPU dlg
            elif 'Estimated Free Energy of Binding' in line:
                try:
                    energies.append(float(line.split('=')[1].split()[0]))
                except (IndexError, ValueError):
                    pass
    energies.sort()                     # most-negative (best) first
    out = np.full(top_n, np.nan)
    n   = min(len(energies), top_n)
    out[:n] = energies[:n]
    return out


def _parse_hdock_out(filepath: str, top_n: int) -> np.ndarray:
    """
    Extract binding scores from an HDOCK .out file.
    HDOCK output format: one model per line, second field is the docking score.
    Returns an array of length top_n (most-negative first, NaN-padded).
    """
    energies: List[float] = []
    with open(filepath) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[1]))
                except ValueError:
                    pass
    energies.sort()
    out = np.full(top_n, np.nan)
    n   = min(len(energies), top_n)
    out[:n] = energies[:n]
    return out


# ── Main extractor class ──────────────────────────────────────────────────────

class DockingFeatureExtractor:
    """
    Retrieves the top-1 docking energy for each (aptamer, protein) pair from
    pre-computed docking output files.

    Results are cached in memory so repeated queries for the same pair are free.

    Parameters
    ----------
    dest_folder : str
        Directory containing the docking output files.
    protein_names : List[str]
        Protein name stems in the desired column order
        (targets first, then counter-targets).
    docker_name : str
        One of 'hdock', 'autodock', 'autodockgpu'.
    top_n : int
        How many models to read from each file (only top_n=1 energy is used
        by FakeML, but reading more enables future methods).
    """

    _TOP_N = {'hdock': 4392, 'autodock': 100, 'autodockgpu': 100}

    def __init__(
        self,
        dest_folder: str,
        protein_names: List[str],
        docker_name: str = 'autodock',
        top_n: Optional[int] = None,
    ):
        self.dest_folder   = dest_folder
        self.protein_names = protein_names
        self.docker_name   = docker_name
        self.top_n         = top_n or self._TOP_N.get(docker_name, 100)
        self._cache: Dict[Tuple[str, str], float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_top_energy(self, aptamer_seq: str, protein_name: str) -> float:
        """Return best (most-negative) docking energy; NaN if file not found."""
        key = (aptamer_seq, protein_name)
        if key not in self._cache:
            self._cache[key] = self._load_top_energy(aptamer_seq, protein_name)
        return self._cache[key]

    def get_energy_matrix(self, aptamer_seqs: List[str]) -> np.ndarray:
        """
        Return top-1 docking energies for a list of sequences.

        Parameters
        ----------
        aptamer_seqs : List[str]  (should contain only unique sequences)

        Returns
        -------
        energy_matrix : np.ndarray, shape [n_apts, n_proteins]
        """
        n_apts    = len(aptamer_seqs)
        n_proteins = len(self.protein_names)
        matrix = np.full((n_apts, n_proteins), np.nan)
        for i, seq in enumerate(aptamer_seqs):
            for j, prot in enumerate(self.protein_names):
                matrix[i, j] = self.get_top_energy(seq, prot)
        return matrix

    def preload_all(self, k: int) -> None:
        """
        Pre-warm the cache with all 4^k sequences × all proteins.

        On a cold cache this reads every docking file once and stores the
        result in memory.  Subsequent calls are no-ops for already-cached pairs.

        Parameters
        ----------
        k : int
            Sequence length (6 or 7).
        """
        seqs = generate_all_kmers(k)
        total = len(seqs) * len(self.protein_names)
        print(f"Preloading {len(seqs)} {k}-mers × {len(self.protein_names)} proteins "
              f"= {total} entries …")
        for i, seq in enumerate(seqs):
            for prot in self.protein_names:
                self.get_top_energy(seq, prot)
            if (i + 1) % 512 == 0:
                print(f"  {i + 1}/{len(seqs)} sequences loaded")
        print(f"Preload complete. Cache size: {len(self._cache)} entries.")

    def save_cache(self, path: str) -> None:
        """Persist the in-memory cache to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        print(f"Cache saved to {path}  ({len(self._cache)} entries)")

    def load_cache(self, path: str) -> None:
        """Load a previously saved cache from a pickle file."""
        with open(path, 'rb') as f:
            self._cache = pickle.load(f)
        print(f"Cache loaded from {path}  ({len(self._cache)} entries)")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_top_energy(self, aptamer_seq: str, protein_name: str) -> float:
        filepath = _find_docking_file(
            self.dest_folder, protein_name, aptamer_seq, self.docker_name
        )
        if filepath is None:
            return np.nan

        if self.docker_name == 'hdock':
            energies = _parse_hdock_out(filepath, self.top_n)
        else:
            energies = _parse_autodock_pdbqt(filepath, self.top_n)

        valid = energies[~np.isnan(energies)]
        return float(valid[0]) if len(valid) > 0 else np.nan
