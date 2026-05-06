"""
Multi-seed runs for error bars.

Runs 5 seeds on the most important tasks with key methods.
Reports mean ± std for the paper's tables.

Usage:
    python run_seeds.py                        # all key tasks
    python run_seeds.py --task mnist           # single task
    python run_seeds.py --task mnist --seeds 3 # fewer seeds for quick test
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
import time
from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Subset

from config import ExperimentConfig, setup_device, ensure_dirs
from exp1_spectral import load_real_dataset
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


def run_single_seed(task_name, n_classes, base_model, config, device, seed):
    """Run key methods on one task with one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = load_real_dataset(task_name, n_classes, config, max_samples=1000)
    if dataset is None:
        return None

    n_total = len(dataset)
    n_val = min(200, n_total // 5)
    n_train = n_total - n_val

    # Seeded split
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=2,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=2)

    results = {}

    # Key methods (not all 15 — just the ones that matter for comparisons)
    methods = [
        ('LP', lambda m: apply_linear_probe(m, config)),
        ('LoRA_r1', lambda m: apply_lora(m, 1, config)),
        ('LoRA_r4', lambda m: apply_lora(m, 4, config)),
        ('LoRA_r8', lambda m: apply_lora(m, 8, config)),
        ('LoRA_r16', lambda m: apply_lora(m, 16, config)),
        ('VPT_p1', lambda m: apply_vpt(m, 1, config)),
        ('VPT_p5', lambda m: apply_vpt(m, 5, config)),
        ('VPT_p10', lambda m: apply_vpt(m, 10, config)),
        ('VPT_p20', lambda m: apply_vpt(m, 20, config)),
        ('Adapter_r8', lambda m: apply_adapter(m, 8, config)),
    ]

    for method_name, apply_fn in methods:
        torch.manual_seed(seed + hash(method_name) % 10000)
        model = deepcopy(base_model).to(device)
        model.head = nn.Linear(config.embed_dim, n_classes).to(device)
        model = apply_fn(model)
        acc = train_and_evaluate(model, train_loader, val_loader, config,
                                 device, f'{method_name}')
        results[method_name] = acc
        del model

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None,
                        help='Run single task')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds (default: 5)')
    args = parser.parse_args()

    config = ExperimentConfig()
    device = setup_device()
    ensure_dirs(config)

    print("Loading pretrained model...")
    base_model = timm.create_model(config.model_name, pretrained=True,
                                    img_size=config.img_size)

    # Key tasks covering all regions of the prediction space
    all_tasks = {
        # LP-sufficient (gap < 0.10)
        'cifar10':        (10,  'natural'),
        'stl10':          (10,  'natural'),
        'eurosat':        (10,  'specialized'),
        # LoRA wins (moderate-high gap, low attn_var)
        'cifar100':       (100, 'natural'),
        'dtd':            (47,  'natural'),
        'svhn':           (10,  'structured'),
        'gtsrb':          (43,  'structured'),
        'fgvc_aircraft':  (100, 'natural'),
        'pcam':           (2,   'specialized'),
        # VPT competitive (high attn_var)
        'mnist':          (10,  'structured'),
        'emnist_letters': (26,  'structured'),
        'fashionmnist':   (10,  'structured'),
        # Borderline
        'emnist_digits':  (10,  'structured'),
        'food101':        (101, 'natural'),
    }

    if args.task:
        if args.task in all_tasks:
            all_tasks = {args.task: all_tasks[args.task]}
        else:
            print(f"Unknown task: {args.task}")
            print(f"Available: {', '.join(all_tasks.keys())}")
            return

    seeds = list(range(42, 42 + args.seeds))
    all_results = {}

    # Load existing results
    results_path = os.path.join(config.output_dir, 'exp2_seeds.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    for task_name, (n_classes, category) in all_tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category}) — {args.seeds} seeds")
        print(f"{'='*60}")

        if task_name in all_results and len(all_results[task_name].get('seed_results', [])) >= args.seeds:
            print(f"  Already have {args.seeds} seeds, skipping")
            continue

        seed_results = []
        task_start = time.time()

        for i, seed in enumerate(seeds):
            print(f"\n  Seed {i+1}/{args.seeds} (seed={seed})")
            try:
                result = run_single_seed(task_name, n_classes, base_model,
                                        config, device, seed)
                if result:
                    seed_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                torch.cuda.empty_cache()

        if not seed_results:
            continue

        elapsed = time.time() - task_start

        # Compute mean ± std for each method
        methods = seed_results[0].keys()
        summary = {}
        for method in methods:
            accs = [r[method] for r in seed_results if method in r]
            summary[method] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'values': accs,
            }

        # Best LoRA vs best VPT
        lora_methods = [m for m in methods if 'LoRA' in m]
        vpt_methods = [m for m in methods if 'VPT' in m]
        best_lora = max(lora_methods, key=lambda m: summary[m]['mean'])
        best_vpt = max(vpt_methods, key=lambda m: summary[m]['mean'])

        lora_mean = summary[best_lora]['mean']
        lora_std = summary[best_lora]['std']
        vpt_mean = summary[best_vpt]['mean']
        vpt_std = summary[best_vpt]['std']

        # Statistical significance (paired t-test)
        from scipy import stats
        lora_vals = [r[best_lora] for r in seed_results]
        vpt_vals = [r[best_vpt] for r in seed_results]
        if len(lora_vals) >= 3:
            t_stat, p_val = stats.ttest_rel(lora_vals, vpt_vals)
        else:
            t_stat, p_val = 0, 1.0

        winner = "LoRA" if lora_mean > vpt_mean + 0.005 else \
                 "VPT" if vpt_mean > lora_mean + 0.005 else "TIE"
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

        print(f"\n  Results ({elapsed:.0f}s):")
        print(f"    LP:           {summary['LP']['mean']:.4f} ± {summary['LP']['std']:.4f}")
        print(f"    Best LoRA:    {best_lora} = {lora_mean:.4f} ± {lora_std:.4f}")
        print(f"    Best VPT:     {best_vpt} = {vpt_mean:.4f} ± {vpt_std:.4f}")
        print(f"    Winner:       {winner} (p={p_val:.3f}) {sig}")

        all_results[task_name] = {
            'category': category,
            'summary': summary,
            'best_lora': best_lora,
            'best_vpt': best_vpt,
            'winner': winner,
            'p_value': p_val,
            'n_seeds': len(seed_results),
            'seed_results': seed_results,
        }

        # Save after each task
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary table
    print("\n" + "=" * 70)
    print(f"MULTI-SEED RESULTS ({args.seeds} seeds)")
    print("=" * 70)
    print(f"\n  {'Task':<18s} {'LP':>12s} {'Best LoRA':>12s} {'Best VPT':>12s} "
          f"{'Winner':>8s} {'p-val':>7s}")
    print(f"  {'-'*75}")

    for task_name, r in all_results.items():
        s = r['summary']
        lp = f"{s['LP']['mean']:.3f}±{s['LP']['std']:.3f}"
        bl = r['best_lora']
        bv = r['best_vpt']
        lora_str = f"{s[bl]['mean']:.3f}±{s[bl]['std']:.3f}"
        vpt_str = f"{s[bv]['mean']:.3f}±{s[bv]['std']:.3f}"
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else \
              "*" if r['p_value'] < 0.10 else ""
        print(f"  {task_name:<18s} {lp:>12s} {lora_str:>12s} {vpt_str:>12s} "
              f"{r['winner']:>8s} {r['p_value']:>6.3f}{sig}")


if __name__ == '__main__':
    main()
