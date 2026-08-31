"""
Fair LR Comparison at Original Protocol (img_size=224)

Uses the EXACT same data loading and protocol as run_all_backbones.py,
but with the best LRs discovered from the sweep.

From the sweep results:
  - VPT_p5 best LR: 1e-3
  - VPT_p1 best LR: 5e-3  
  - LoRA_r8 best LR: 2e-3
  - Default: VPT LR=1e-2, LoRA LR=1e-3

Runs 3 seeds for statistical confidence.

Usage:
    python revision2_fair_comparison.py
    python revision2_fair_comparison.py --tasks cifar100 svhn
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

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora, apply_vpt, train_and_evaluate
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]

# Best LRs from sweep (update these after full sweep completes)
BEST_LRS = {
    'dinov2': {
        'vpt_p5': 1e-3,
        'vpt_p1': 5e-3,
        'lora_r8': 2e-3,
    },
    'dinov2reg': {
        'vpt_p5': None,   # fill after sweep
        'vpt_p1': None,
        'lora_r8': None,
    },
    'deit3': {
        'vpt_p5': 1e-2,   # default (control — should work)
        'vpt_p1': 1e-2,
        'lora_r8': 1e-3,
    },
    'supervised': {
        'vpt_p5': 1e-2,
        'vpt_p1': 1e-2,
        'lora_r8': 1e-3,
    },
}

# Default LRs (original protocol)
DEFAULT_LRS = {
    'vpt_p5': 1e-2,
    'vpt_p1': 1e-2,
    'lora_r8': 1e-3,
}


def run_method(base_model, config, method, capacity, lr, train_loader, val_loader, device):
    """Run one method with given LR."""
    model = deepcopy(base_model)
    model.head = nn.Linear(config.embed_dim, config.num_classes).to(device)

    if method == 'vpt':
        model = apply_vpt(model, capacity, config)
    elif method == 'lora':
        model = apply_lora(model, capacity, config)

    model = model.to(device)
    config.lr = lr
    acc = train_and_evaluate(model, train_loader, val_loader, config, device)

    del model
    torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar100', 'svhn', 'gtsrb'])
    parser.add_argument('--backbones', nargs='+',
                        default=['dinov2', 'dinov2reg', 'deit3'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("FAIR COMPARISON: Best LR vs Default LR (img_size=224)")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    all_results = {}

    for bb_key in args.backbones:
        if bb_key not in BACKBONES:
            continue
        bb = BACKBONES[bb_key]
        img_size = bb['img_size']  # 224 — matches original

        print(f"\n  Loading {bb['name']} (img_size={img_size})...")
        base_model = timm.create_model(bb['model'], pretrained=True,
                                        img_size=img_size).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        best_lrs = BEST_LRS.get(bb_key, DEFAULT_LRS)

        for task in args.tasks:
            if task not in TASKS:
                continue
            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            print(f"\n{'='*55}")
            print(f"  {bb['name']} × {task} (img_size={img_size})")
            print(f"{'='*55}")

            methods = [
                ('LoRA_r8_default', 'lora', 8, DEFAULT_LRS['lora_r8']),
                ('LoRA_r8_best',    'lora', 8, best_lrs.get('lora_r8', DEFAULT_LRS['lora_r8'])),
                ('VPT_p5_default',  'vpt',  5, DEFAULT_LRS['vpt_p5']),
                ('VPT_p5_best',     'vpt',  5, best_lrs.get('vpt_p5', DEFAULT_LRS['vpt_p5'])),
                ('VPT_p1_default',  'vpt',  1, DEFAULT_LRS['vpt_p1']),
                ('VPT_p1_best',     'vpt',  1, best_lrs.get('vpt_p1', DEFAULT_LRS['vpt_p1'])),
            ]

            # Skip methods with None LR (not yet determined)
            methods = [(n, m, c, lr) for n, m, c, lr in methods if lr is not None]

            task_results = {}

            for method_name, method, capacity, lr in methods:
                seed_accs = []
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    ds = load_dataset(task, img_size, max_samples=1000)
                    n_val = min(200, len(ds) // 5)
                    train_ds, val_ds = random_split(
                        ds, [len(ds) - n_val, n_val],
                        generator=torch.Generator().manual_seed(seed))
                    train_loader = DataLoader(train_ds, batch_size=64,
                                              shuffle=True, num_workers=2)
                    val_loader = DataLoader(val_ds, batch_size=64,
                                            shuffle=False, num_workers=2)

                    acc = run_method(base_model, config, method, capacity, lr,
                                     train_loader, val_loader, device)
                    seed_accs.append(acc)

                mean_acc = np.mean(seed_accs)
                std_acc = np.std(seed_accs)
                task_results[method_name] = {
                    'lr': lr, 'seeds': seed_accs,
                    'mean': float(mean_acc), 'std': float(std_acc)
                }
                print(f"    {method_name:<20s} (lr={lr:.0e}): "
                      f"{mean_acc:.3f} ± {std_acc:.3f}  "
                      f"({', '.join(f'{a:.3f}' for a in seed_accs)})")

            # Summary
            key = f"{bb_key}_{task}"
            all_results[key] = task_results

            lora_def = task_results.get('LoRA_r8_default', {}).get('mean', 0)
            lora_best = task_results.get('LoRA_r8_best', {}).get('mean', 0)
            vpt5_def = task_results.get('VPT_p5_default', {}).get('mean', 0)
            vpt5_best = task_results.get('VPT_p5_best', {}).get('mean', 0)

            print(f"\n    Default gap:  LoRA {lora_def:.3f} vs VPT {vpt5_def:.3f} = "
                  f"{lora_def - vpt5_def:+.3f}")
            print(f"    Fair gap:     LoRA {lora_best:.3f} vs VPT {vpt5_best:.3f} = "
                  f"{lora_best - vpt5_best:+.3f}")

            if abs(lora_best - vpt5_best) < 0.02:
                print(f"    → WITHIN NOISE after fair tuning")
            elif lora_best > vpt5_best:
                print(f"    → LoRA still wins after fair tuning (+{lora_best-vpt5_best:.3f})")
            else:
                print(f"    → VPT wins after fair tuning ({lora_best-vpt5_best:+.3f})")

        del base_model
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: Default vs Fair Comparison")
    print(f"{'='*70}")
    for key, results in all_results.items():
        lora_def = results.get('LoRA_r8_default', {}).get('mean', 0)
        lora_best = results.get('LoRA_r8_best', {}).get('mean', 0)
        vpt_def = results.get('VPT_p5_default', {}).get('mean', 0)
        vpt_best = results.get('VPT_p5_best', {}).get('mean', 0)
        print(f"  {key:<25s}: default gap={lora_def-vpt_def:+.3f}, "
              f"fair gap={lora_best-vpt_best:+.3f}")

    os.makedirs('results', exist_ok=True)
    with open('results/revision2_fair_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_fair_comparison.json")


if __name__ == '__main__':
    main()
