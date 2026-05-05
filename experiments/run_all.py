"""
run_all.py — Main experiment runner

Usage:
    python run_all.py                       # Run all experiments
    python run_all.py --exp spectral        # Experiment 1: Spectral profiles
    python run_all.py --exp comparison      # Experiment 2: PEFT comparison
    python run_all.py --exp selection       # Experiment 3: Selection algorithm
    python run_all.py --exp assumption_a    # Experiment 4: Assumption A (from earlier)
    python run_all.py --fast                # Quick mode (fewer epochs, fewer tasks)
"""

import argparse
import time
import json
from config import ExperimentConfig, ensure_dirs


def main():
    parser = argparse.ArgumentParser(description='Run paper experiments')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'spectral', 'comparison', 'selection',
                                 'lr_sweep', 'task_structure', 'assumption_a'],
                        help='Which experiment to run')
    parser.add_argument('--fast', action='store_true',
                        help='Quick mode with reduced epochs and tasks')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    # Configuration
    if args.fast:
        config = ExperimentConfig(
            epochs=10,
            pilot_epochs=3,
            n_train=500,
            n_val=100,
            output_dir=args.output_dir,
        )
        print("Running in FAST mode (reduced epochs and data)")
    else:
        config = ExperimentConfig(output_dir=args.output_dir)

    ensure_dirs(config)
    all_results = {}
    start_time = time.time()

    # ---- Experiment 1: Spectral Analysis ----
    if args.exp in ['all', 'spectral']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: SPECTRAL PROFILE ANALYSIS")
        print("=" * 70)
        from exp1_spectral import run_spectral_analysis
        results = run_spectral_analysis(config)
        all_results['spectral'] = results

    # ---- Experiment 2: PEFT Comparison ----
    if args.exp in ['all', 'comparison']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: PEFT METHOD COMPARISON")
        print("=" * 70)
        from exp2_comparison import run_comparison
        results = run_comparison(config)
        all_results['comparison'] = results

    # ---- Experiment 3: Selection Algorithm ----
    if args.exp in ['all', 'selection']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: SELECTION ALGORITHM BENCHMARK")
        print("=" * 70)
        from exp3_selection import run_selection_benchmark
        results = run_selection_benchmark(config)
        all_results['selection'] = results

    # ---- Experiment 4: LR Sweep ----
    if args.exp in ['all', 'lr_sweep']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: LEARNING RATE SWEEP")
        print("=" * 70)
        from exp4_lr_sweep import run_lr_sweep
        results = run_lr_sweep(config)
        all_results['lr_sweep'] = results

    # ---- Experiment 5: Task Structure (Training-Free) ----
    if args.exp in ['all', 'task_structure']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: TASK STRUCTURE ANALYSIS")
        print("=" * 70)
        from exp5_task_structure import run_task_structure_analysis
        results = run_task_structure_analysis(config)
        all_results['task_structure'] = results

    # ---- Experiment 4: Assumption A ----
    if args.exp in ['all', 'assumption_a']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: ASSUMPTION A VERIFICATION")
        print("=" * 70)
        print("Run separately: python validate_assumption_a.py")
        print("(Requires pretrained model download)")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results saved to {config.output_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
