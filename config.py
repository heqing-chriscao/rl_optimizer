"""
Default configurations mirroring params_temp.m.

Protein order in the 6-protein setup (targets first, then counter-targets):
  Targets       : Human-AC2 (w=2),  MERS-Spike (w=3),  RV-VP1 (w=2)
  Counter-targets: SARS-Spike (w=0), H1N1-HA (w=0),    RSV-Fusion (w=1)

Weights array (0-indexed, length 6) matches MATLAB Weights(12:17):
  index 0 -> Weights(12) = 2  (Human-AC2)
  index 1 -> Weights(13) = 3  (MERS-Spike)
  index 2 -> Weights(14) = 2  (RV-VP1)
  index 3 -> Weights(15) = 0  (SARS-Spike  — counter)
  index 4 -> Weights(16) = 0  (H1N1-HA     — counter)
  index 5 -> Weights(17) = 1  (RSV-Fusion  — counter)
"""

import numpy as np
from optimizer import OptimizerConfig

# ── Proteins ──────────────────────────────────────────────────────────────────
TARGET_PROTEINS  = ['Human-AC2', 'MERS-Spike', 'RV-VP1']
COUNTER_PROTEINS = ['SARS-Spike', 'H1N1-HA', 'RSV-Fusion']
PROTEIN_WEIGHTS  = np.array([2.0, 3.0, 2.0, 0.0, 0.0, 1.0])
NUM_INTERNAL_REPS = 25
MAX_MUTATION_RATE = 2
MAX_CHANGES_PER_ROUND = 4
LESS_THAN_OR_EQUAL = False

def get_6mer_config(
    initial_seq: str = 'CACCCT',
    docker_dest: str = '/data/alibd/Code/digitalSELEX-2/dockers/autodockgpu/autodockgpu_output_files/',
    max_iterations: int = 60,
    output_csv: str = 'evolution_data_6mer.csv',
    cache_path: str = '/data/alibd/Code/digitalSELEX-2/python_spsa_optimizer/cache_6mer_gpu.pkl',
    history_length: int = 5,
) -> OptimizerConfig:
    return OptimizerConfig(
        initial_sequence=initial_seq,
        fixed_positions=[],
        max_iterations=max_iterations,
        num_internal_reps=NUM_INTERNAL_REPS,
        max_mutation_rate=MAX_MUTATION_RATE,
        less_than_or_equal=LESS_THAN_OR_EQUAL,
        max_changes_per_round=MAX_CHANGES_PER_ROUND,
        target_protein_names=TARGET_PROTEINS,
        counter_protein_names=COUNTER_PROTEINS,
        protein_weights=PROTEIN_WEIGHTS,
        docker_dest_folder=docker_dest,
        docker_name='autodockgpu',
        output_csv=output_csv,
        cache_path=cache_path,
        history_length=history_length,
    )


def get_7mer_config(
    initial_seq: str = 'CACCCTA',
    docker_dest: str = '/data/alibd/Code/digitalSELEX-2/dockers/autodockgpu/autodockgpu_output_files/',
    max_iterations: int = 60,
    output_csv: str = 'evolution_data_7mer.csv',
    cache_path: str = '/data/alibd/Code/digitalSELEX-2/python_spsa_optimizer/cache_7mer_gpu.pkl',
    history_length: int = 5,
) -> OptimizerConfig:
    return OptimizerConfig(
        initial_sequence=initial_seq,
        fixed_positions=[],
        max_iterations=max_iterations,
        num_internal_reps=NUM_INTERNAL_REPS,
        max_mutation_rate=MAX_MUTATION_RATE,
        less_than_or_equal=LESS_THAN_OR_EQUAL,
        max_changes_per_round=MAX_CHANGES_PER_ROUND,
        target_protein_names=TARGET_PROTEINS,
        counter_protein_names=COUNTER_PROTEINS,
        protein_weights=PROTEIN_WEIGHTS,
        docker_dest_folder=docker_dest,
        docker_name='autodockgpu',
        output_csv=output_csv,
        cache_path=cache_path,
        history_length=history_length,
    )
