"""
Larger-n Validation: Does the theory hold beyond n=800?

Tests LoRA vs VPT at n = 800, 2000, 5000, 10000 on multiple tasks.
Theory predictions:
  - At small n: LoRA dominates (VPT exceeds capacity budget)
  - At large n: Methods converge (both have sufficient capacity)
  - The crossover point depends on σ²_P and d/(2d_h)

Uses CIFAR-100 (50K available) and SVHN (73K available) which have 
enough data for large-n experiments.

Usage:
    python revision_larger_n.py --backbone dinov2
    python revision_larger_n.py --backbone deit3
    python revision_larger_n.py --backbone dinov2 --tasks cifar100 svhn
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
from torch.utils.data import DataLoader, Subset


N_VALUES = [800, 2000, 5000, 10000]


def run_at_n(base_model, task_name, n_train, bb_key, device, config):
    """Run LoRA and VPT at a specific training set size."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    
    # Load full dataset
    max_samples = n_train + 500  # extra for validation
    ds = load_dataset(task_name, bb['img_size'], max_samples=max_samples)
    
    if len(ds) < n_train + 200:
        print(f"    WARNING: Only {len(ds)} samples available, need {n_train}+200")
        return None
    
    n_val = min(500, len(ds) - n_train)
    train_ds = Subset(ds, list(range(n_train)))
    val_ds = Subset(ds, list(range(n_train, n_train + n_val)))
    
    batch_size = min(64, n_train // 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    results = {}
    
    methods = [
        ('LP', None, 1e-2),
        ('LoRA_r4', lambda m: apply_lora(m, 4, config), 1e-3),
        ('LoRA_r8', lambda m: apply_lora(m, 8, config), 1e-3),
        ('VPT_p1', lambda m: apply_vpt(m, 1, config), 1e-2),
        ('VPT_p5', lambda m: apply_vpt(m, 5, config), 1e-2),
        ('VPT_p10', lambda m: apply_vpt(m, 10, config), 1e-2),
    ]
    
    for name, apply_fn, lr in methods:
        model = deepcopy(base_model)
        model.head = nn.Linear(config.embed_dim, num_classes).to(device)
        
        if apply_fn is not None:
            model = apply_fn(model)
        else:
            # LP: freeze backbone
            for param in model.parameters():
                param.requires_grad_(False)
            for param in model.head.parameters():
                param.requires_grad_(True)
        
        model = model.to(device)
        config.lr = lr
        acc = train_and_evaluate(model, train_loader, val_loader, config, device)
        results[name] = acc
        
        del model; torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+', default=['cifar100', 'svhn', 'eurosat'])
    parser.add_argument('--n', nargs='+', type=int, default=[800, 2000, 5000, 10000],
                        help='Training set sizes to test')
    args = parser.parse_args()
    
    n_values = args.n
    
    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100
    
    bb_key = args.backbone
    bb = BACKBONES[bb_key]
    
    print("=" * 70)
    print(f"Larger-n Validation: {bb['name']}")
    print(f"n values: {n_values}")
    print("=" * 70)
    
    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)
    
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
        
        task_results = {}
        
        for n in n_values:
            print(f"\n  n = {n}:")
            results = run_at_n(base_model, task, n, bb_key, device, config)
            if results:
                task_results[str(n)] = results
                
                best_lora = max(results.get(f'LoRA_r{r}', 0) for r in [4, 8])
                best_vpt = max(results.get(f'VPT_p{p}', 0) for p in [1, 5, 10])
                gap = best_lora - best_vpt
                winner = "LoRA" if gap > 0.005 else "VPT" if gap < -0.005 else "TIE"
                
                print(f"    LP={results.get('LP',0):.3f}  "
                      f"bestLoRA={best_lora:.3f}  bestVPT={best_vpt:.3f}  "
                      f"gap={gap:+.3f}  → {winner}")
        
        all_results[task] = task_results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {bb['name']} — LoRA-VPT gap vs n")
    print(f"{'='*70}")
    
    for task, task_results in all_results.items():
        print(f"\n  {task}:")
        print(f"    {'n':>8s} {'LP':>6s} {'L_r4':>6s} {'L_r8':>6s} "
              f"{'V_p1':>6s} {'V_p5':>6s} {'V_p10':>6s} {'Gap':>6s} {'Winner':>7s}")
        
        for n_str, results in sorted(task_results.items(), key=lambda x: int(x[0])):
            best_lora = max(results.get(f'LoRA_r{r}', 0) for r in [4, 8])
            best_vpt = max(results.get(f'VPT_p{p}', 0) for p in [1, 5, 10])
            gap = best_lora - best_vpt
            winner = "LoRA" if gap > 0.005 else "VPT" if gap < -0.005 else "TIE"
            
            print(f"    {n_str:>8s} {results.get('LP',0):>6.3f} "
                  f"{results.get('LoRA_r4',0):>6.3f} {results.get('LoRA_r8',0):>6.3f} "
                  f"{results.get('VPT_p1',0):>6.3f} {results.get('VPT_p5',0):>6.3f} "
                  f"{results.get('VPT_p10',0):>6.3f} "
                  f"{gap:>+6.3f} {winner:>7s}")
    
    del base_model; torch.cuda.empty_cache()
    
    os.makedirs('results', exist_ok=True)
    fname = f'results/larger_n_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
