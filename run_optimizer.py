"""
Entry point for the SPSA aptamer optimizer.

Usage
-----
# 6-mer with defaults from params_temp.m:
    python run_optimizer.py --seq_len 6

# 7-mer, custom start sequence and output:
    python run_optimizer.py --seq_len 7 --initial_seq ACGTACG \
        --docker_dest /path/to/pdbqt/ --output evolution_7mer.csv

# Override max iterations:
    python run_optimizer.py --seq_len 6 --max_iter 30
"""

import argparse
from config import get_6mer_config, get_7mer_config
from optimizer import SPSAOptimizer


def parse_args():
    p = argparse.ArgumentParser(description='SPSA Aptamer Optimizer')
    p.add_argument('--seq_len',      type=int, choices=[6, 7], default=6)
    p.add_argument('--initial_seq',  type=str, default=None,
                   help='Starting sequence (default from config)')
    p.add_argument('--docker_dest',  type=str, default=None,
                   help='Folder containing pre-computed docking output files')
    p.add_argument('--max_iter',     type=int, default=None)
    p.add_argument('--output',       type=str, default=None,
                   help='Output CSV path')
    return p.parse_args()


def main():
    args = parse_args()

    kwargs = {}
    if args.initial_seq:
        kwargs['initial_seq'] = args.initial_seq
    if args.docker_dest:
        kwargs['docker_dest'] = args.docker_dest
    if args.max_iter:
        kwargs['max_iterations'] = args.max_iter
    if args.output:
        kwargs['output_csv'] = args.output

    config = get_6mer_config(**kwargs) if args.seq_len == 6 else get_7mer_config(**kwargs)

    opt = SPSAOptimizer(config)
    df  = opt.run()

    print('\n── Evolution table ──')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
