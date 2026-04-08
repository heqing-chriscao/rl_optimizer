"""
Block 1: Sequence Mutation
Generates candidate aptamer sequences by randomly mutating the current sequence.

NN REPLACEMENT POINT
--------------------
Replace `RandomMutator.generate_candidates` with a learned generator.
The interface is:
    generate_candidates(sequence: str, n_variants: int) -> List[str]
The NN receives the current sequence (and optionally history/scores) and proposes
n_variants candidate sequences to explore in the next round.
"""

import random
from typing import List, Optional

NUCLEOTIDES = ['A', 'C', 'G', 'T']
NT_TO_INT   = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _single_mutant(seq: str, positions: List[int], n_mutations: int) -> str:
    """Mutate exactly n_mutations positions. Each mutation shifts the nucleotide
    by a random offset of 1, 2, or 3 (mod 4), guaranteeing a change."""
    seq_ints = [NT_TO_INT[c] for c in seq]
    sites = random.sample(positions, n_mutations)
    for site in sites:
        delta = random.randint(1, 3)
        seq_ints[site] = (seq_ints[site] + delta) % 4
    return ''.join(NUCLEOTIDES[x] for x in seq_ints)


class RandomMutator:
    """
    Generates random mutations of the input sequence.

    Parameters
    ----------
    max_mutation_rate : int
        Maximum (or exact) number of nucleotide substitutions per variant.
    fixed_positions : list of int, optional
        0-based positions that must not be mutated.
    less_than_or_equal : bool
        If True, draw n_mutations uniformly from [1, max_mutation_rate].
        If False, always mutate exactly max_mutation_rate positions.
        Matches MATLAB's LessThanOrEqualNumberOfMutationsFlag.
    """

    def __init__(
        self,
        max_mutation_rate: int,
        fixed_positions: Optional[List[int]] = None,
        less_than_or_equal: bool = True,
    ):
        self.max_mutation_rate = max_mutation_rate
        self.fixed_positions   = set(fixed_positions or [])
        self.less_than_or_equal = less_than_or_equal

    def generate_candidates(self, sequence: str, n_variants: int) -> List[str]:
        """
        Generate n_variants mutated sequences from `sequence`.

        Parameters
        ----------
        sequence : str
            Current aptamer (e.g. 'CACCCT').
        n_variants : int
            Number of candidates to produce.

        Returns
        -------
        List[str] of length n_variants.
        """
        non_fixed = [i for i in range(len(sequence)) if i not in self.fixed_positions]
        max_mut = min(self.max_mutation_rate, len(non_fixed))

        candidates = []
        for _ in range(n_variants):
            n_mut = random.randint(1, max_mut) if self.less_than_or_equal else max_mut
            candidates.append(_single_mutant(sequence, non_fixed, n_mut))
        return candidates
