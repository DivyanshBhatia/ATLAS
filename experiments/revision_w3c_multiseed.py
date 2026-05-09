"""
REVISION W3c: Multi-Seed Cross-Backbone Experiments

Reviewer concern: "All quantitative comparisons should include confidence
intervals or p-values. Many claims of 'LoRA wins' are based on single-run
differences of 1-3%."

This experiment runs 3 seeds on key task-backbone pairs to add confidence
intervals to the cross-backbone table.

Priority pairs (where the result matters most):
  - Tasks where LoRA vs VPT is close or VPT wins
  - DeiT-III (most VPT-friendly)
  - Supervised (VPT wins on some tasks)

Usage:
    python revision_w3c_multiseed.py
    python revision_w3c_multiseed.py --backbone deit3 --tasks svhn dtd food101
    python revision_w3c_multiseed.py --seeds 5
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
from copy import deepcopy
from scipy import stats
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig, setup_device
from exp2_comparison import (apply_lora, apply_vpt, apply_linear_probe,
                              train_and_evaluate)
from run_all_backbones import BACKBONES, TASKS, get_transforms, load_dataset


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def run_one_seed(base_model, task_name, bb_key, device, config, seed):
    """Run best LoRA and best VPT for one seed."""
    set_seed(seed)
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]

    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}
    embed_dim = config.embed_dim

    # LoRA at ranks 1, 4, 8
    for r in [1, 4, 8]:
        set_seed(seed)
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        model = apply_lora(model, r, config)
        acc = train_and_evaluate(model, train_loader, val_loader, config, device)
        results[f'LoRA_r{r}'] = acc
        del model; torch.cuda.empty_cache()

    # VPT at prompts 1, 5, 10
    for p in [1, 5, 10]:
        set_seed(seed)
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        model = apply_vpt(model, p, config)
        acc = train_and_evaluate(model, train_loader, val_loader, config, device)
        results[f'VPT_p{p}'] = acc
        del model; torch.cuda.empty_cache()

    best_lora = max(v for k, v in results.items() if k.startswith('LoRA'))
    best_vpt = max(v for k, v in results.items() if k.startswith('VPT'))
    best_lora_k = max((k for k in results if k.startswith('LoRA')), key=lambda k: results[k])
    best_vpt_k = max((k for k in results if k.startswith('VPT')), key=lambda k: results[k])

    return {
        'best_lora': best_lora,
        'best_vpt': best_vpt,
        'best_lora_method': best_lora_k,
        'best_vpt_method': best_vpt_k,
        'all': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='deit3')
    parser.add_argument('--tasks', nargs='+',
                        default=['svhn', 'dtd', 'food101', 'cifar100', 'eurosat'])
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 75)
    print(f"REVISION W3c: Multi-Seed on {bb['name']}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 75)

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    # Setup config for this backbone
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    all_results = {}

    for task in args.tasks:
        if task not in TASKS:
            continue

        print(f"\n{'='*55}")
        print(f"  {bb['name']} × {task}")
        print(f"{'='*55}")

        seed_results = []
        for seed in range(args.seeds):
            print(f"\n  Seed {seed+1}/{args.seeds}...")
            res = run_one_seed(base_model, task, bb_key, device, config, seed=seed+42)
            seed_results.append(res)
            print(f"    LoRA={res['best_lora']:.3f} ({res['best_lora_method']})  "
                  f"VPT={res['best_vpt']:.3f} ({res['best_vpt_method']})")

        # Statistical analysis
        lora_accs = [r['best_lora'] for r in seed_results]
        vpt_accs = [r['best_vpt'] for r in seed_results]

        lora_mean = np.mean(lora_accs)
        lora_std = np.std(lora_accs, ddof=1) if len(lora_accs) > 1 else 0
        vpt_mean = np.mean(vpt_accs)
        vpt_std = np.std(vpt_accs, ddof=1) if len(vpt_accs) > 1 else 0

        # Paired t-test
        if len(lora_accs) > 1:
            t_stat, p_value = stats.ttest_rel(lora_accs, vpt_accs)
        else:
            t_stat, p_value = 0, 1.0

        # Determine winner
        delta = lora_mean - vpt_mean
        if p_value < 0.05:
            winner = "LoRA **" if delta > 0 else "VPT **"
        elif abs(delta) < 0.01:
            winner = "TIE"
        else:
            winner = "LoRA (n.s.)" if delta > 0 else "VPT (n.s.)"

        print(f"\n  Summary: LoRA {lora_mean:.3f}±{lora_std:.3f}  "
              f"VPT {vpt_mean:.3f}±{vpt_std:.3f}  "
              f"Δ={delta:+.3f}  p={p_value:.3f}  → {winner}")

        all_results[task] = {
            'lora_mean': lora_mean, 'lora_std': lora_std,
            'vpt_mean': vpt_mean, 'vpt_std': vpt_std,
            'delta': delta, 'p_value': p_value,
            'winner': winner,
            'seeds': seed_results,
        }

    # Summary table
    print(f"\n{'='*75}")
    print(f"SUMMARY: {bb['name']} multi-seed ({args.seeds} seeds)")
    print(f"{'='*75}")
    print(f"  {'Task':<14s} {'LoRA':>15s} {'VPT':>15s} {'Δ':>7s} {'p':>6s} {'Winner':>12s}")
    print(f"  {'-'*72}")

    for task, res in all_results.items():
        print(f"  {task:<14s} {res['lora_mean']:.3f}±{res['lora_std']:.3f}"
              f"     {res['vpt_mean']:.3f}±{res['vpt_std']:.3f}"
              f"  {res['delta']:>+6.3f} {res['p_value']:>6.3f} {res['winner']:>12s}")

    del base_model; torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/revision_w3c_multiseed_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {fname}")


if __name__ == '__main__':
    main()
