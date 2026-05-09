"""
REVISION W3a: Multi-Scale Validation on Multiple Tasks and Backbones

Reviewer concern: "Multi-scale results only for CIFAR-100 on DINOv2. It is
essential to verify [n-dependent predictions] on at least 3-4 tasks across
multiple backbones."

This experiment runs n ∈ {400, 800, 2000, 5000} on:
  - 3 tasks: cifar100, svhn, dtd
  - 3 backbones: dinov2, deit3, clip

Validates: LoRA-VPT gap narrows with n, VPT degradation reverses at large n.

Usage:
    python revision_w3a_multiscale.py
    python revision_w3a_multiscale.py --tasks cifar100 svhn --backbones dinov2 deit3
    python revision_w3a_multiscale.py --scales 400 800 2000
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
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig, setup_device
from exp2_comparison import (apply_lora, apply_vpt, apply_linear_probe,
                              train_and_evaluate)
from run_all_backbones import BACKBONES, TASKS, get_transforms, load_dataset


def run_single_scale(base_model, task_name, n_train, bb_key, device, config):
    """Run LoRA and VPT at a single scale."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    embed_dim = config.embed_dim

    ds = load_dataset(task_name, bb['img_size'], max_samples=max(n_train + 200, 6000))
    n_val = min(200, len(ds) // 5)
    actual_train = min(n_train, len(ds) - n_val)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    if actual_train < len(train_ds):
        indices = torch.randperm(len(train_ds))[:actual_train].tolist()
        train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    # LP
    model_lp = deepcopy(base_model)
    model_lp.head = nn.Linear(embed_dim, num_classes).to(device)
    model_lp = apply_linear_probe(model_lp, config)
    results['LP'] = train_and_evaluate(model_lp, train_loader, val_loader, config, device)
    del model_lp; torch.cuda.empty_cache()

    # LoRA at multiple ranks
    for r in [1, 4, 8, 16]:
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        model = apply_lora(model, r, config)
        results[f'LoRA_r{r}'] = train_and_evaluate(model, train_loader, val_loader, config, device)
        del model; torch.cuda.empty_cache()

    # VPT at multiple prompt counts
    for p in [1, 5, 10, 20]:
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        model = apply_vpt(model, p, config)
        results[f'VPT_p{p}'] = train_and_evaluate(model, train_loader, val_loader, config, device)
        del model; torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['cifar100', 'svhn', 'dtd'])
    parser.add_argument('--backbones', nargs='+', default=['dinov2', 'deit3', 'clip'])
    parser.add_argument('--scales', nargs='+', type=int, default=[400, 800, 2000, 5000])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 75)
    print("REVISION W3a: Multi-Scale Validation")
    print(f"  Tasks: {args.tasks}")
    print(f"  Backbones: {args.backbones}")
    print(f"  Scales: {args.scales}")
    print("=" * 75)

    all_results = {}

    for bb_key in args.backbones:
        bb = BACKBONES[bb_key]
        print(f"\n{'='*60}")
        print(f"  Loading {bb['name']}...")
        base_model = timm.create_model(bb['model'], pretrained=True,
                                        img_size=bb['img_size']).to(device)

        # Setup config for this backbone
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        # σ_P²
        total_norm, cnt = 0.0, 0
        for name, param in base_model.named_parameters():
            if any(t in name for t in ['qkv.weight', 'proj.weight']):
                total_norm += param.float().norm().item() ** 2
                cnt += 1
        d_h = base_model.embed_dim // base_model.blocks[0].attn.num_heads
        sigma_sq = total_norm / (cnt * d_h) if cnt > 0 else 1.0

        for task in args.tasks:
            if task not in TASKS:
                continue

            print(f"\n  --- {bb['name']} × {task} ---")
            print(f"  σ²_P={sigma_sq:.1f}")

            task_results = {}
            for n in args.scales:
                # Compute theory predictions
                L, d = 12, 768
                r_star = int(2 * n * sigma_sq / (L * d_h))
                p_star = int(4 * n * sigma_sq / (L * d))

                print(f"\n    n={n}, r*={r_star}, p*={p_star}")

                try:
                    results = run_single_scale(base_model, task, n, bb_key, device, config)
                    task_results[str(n)] = results

                    best_lora = max(v for k, v in results.items() if k.startswith('LoRA'))
                    best_vpt = max(v for k, v in results.items() if k.startswith('VPT'))
                    lp = results['LP']
                    best_vpt_key = max((k for k in results if k.startswith('VPT')),
                                       key=lambda k: results[k])
                    best_lora_key = max((k for k in results if k.startswith('LoRA')),
                                        key=lambda k: results[k])

                    gap = best_lora - best_vpt
                    print(f"    LP={lp:.3f}  {best_lora_key}={best_lora:.3f}  "
                          f"{best_vpt_key}={best_vpt:.3f}  Δ(L-V)={gap:+.3f}")

                    # VPT degradation check
                    vpt1 = results.get('VPT_p1', 0)
                    vpt20 = results.get('VPT_p20', 0)
                    if vpt20 > vpt1:
                        print(f"    VPT healthy: p1={vpt1:.3f} < p20={vpt20:.3f} ✓")
                    else:
                        print(f"    VPT degradation: p1={vpt1:.3f} > p20={vpt20:.3f}")

                except Exception as e:
                    print(f"    ERROR: {str(e)[:60]}")
                    task_results[str(n)] = {'error': str(e)}

            all_results[f'{bb_key}_{task}'] = {
                'sigma_sq': sigma_sq,
                'scales': task_results
            }

        del base_model; torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*75}")
    print("SUMMARY: LoRA-VPT gap vs n")
    print(f"{'='*75}")
    print(f"  {'Backbone × Task':<25s}", end='')
    for n in args.scales:
        print(f"  n={n:>5d}", end='')
    print()

    for key, data in all_results.items():
        print(f"  {key:<25s}", end='')
        for n in args.scales:
            ns = str(n)
            if ns in data['scales'] and 'error' not in data['scales'][ns]:
                r = data['scales'][ns]
                best_l = max(v for k, v in r.items() if k.startswith('LoRA'))
                best_v = max(v for k, v in r.items() if k.startswith('VPT'))
                gap = best_l - best_v
                print(f"  {gap:>+6.3f}", end='')
            else:
                print(f"  {'ERR':>6s}", end='')
        print()

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/revision_w3a_multiscale.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/revision_w3a_multiscale.json")


if __name__ == '__main__':
    main()
