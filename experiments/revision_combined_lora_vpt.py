"""
Combined LoRA+VPT Experiment

Key insight from NOAH (NeurIPS 2022): combining PEFT methods can outperform
individual methods. Instead of binary LoRA-or-VPT selection, use BOTH with
shared capacity budget.

Theory: KL_total = KL_LoRA(r) + KL_VPT(p) < 2n
So LoRA_r4 + VPT_p1 uses KL = (4×64 + 1×768)η²/(2σ²_P) which is easily affordable.

This tests whether the combination beats both LoRA_r8 and the oracle of 
individual methods.

Usage:
    python revision_combined_lora_vpt.py
    python revision_combined_lora_vpt.py --backbone deit3 --tasks svhn dtd food101
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


def apply_combined(model, lora_rank, vpt_prompts, config):
    """Apply BOTH LoRA and VPT to the same model.
    
    Key: apply_vpt may freeze LoRA params, so we manually unfreeze both.
    """
    # First apply LoRA (modifies attention weights)
    model = apply_lora(model, lora_rank, config)
    # Then apply VPT (adds prompt tokens)
    model = apply_vpt(model, vpt_prompts, config)
    
    # Ensure BOTH LoRA and VPT params are trainable
    for name, param in model.named_parameters():
        if any(k in name for k in ['lora_', 'prompt', 'head']):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    return model


def run_task(base_model, task_name, bb_key, device, config):
    """Run individual and combined methods on one task."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    embed_dim = config.embed_dim

    # Load data
    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    # Individual baselines
    configs_individual = [
        ('LoRA_r4', lambda m: apply_lora(m, 4, config)),
        ('LoRA_r8', lambda m: apply_lora(m, 8, config)),
        ('VPT_p1', lambda m: apply_vpt(m, 1, config)),
        ('VPT_p5', lambda m: apply_vpt(m, 5, config)),
    ]

    for name, apply_fn in configs_individual:
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        model = apply_fn(model)
        model = model.to(device)
        acc = train_and_evaluate(model, train_loader, val_loader, config, device)
        results[name] = acc
        del model; torch.cuda.empty_cache()
        print(f"    {name}: {acc:.3f}")

    # Combined LoRA+VPT configurations
    combos = [
        ('L2+V1', 2, 1),
        ('L4+V1', 4, 1),
        ('L4+V3', 4, 3),
        ('L4+V5', 4, 5),
        ('L8+V1', 8, 1),
    ]

    for name, lr, vp in combos:
        try:
            model = deepcopy(base_model)
            model.head = nn.Linear(embed_dim, num_classes).to(device)
            model = apply_combined(model, lr, vp, config)
            model = model.to(device)

            # Count params
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            results[name] = acc
            print(f"    {name}: {acc:.3f} ({trainable:,} params)")
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {name}: ERROR - {str(e)[:60]}")
            results[name] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat',
                                 'fashionmnist', 'gtsrb', 'food101', 'mnist'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 75)
    print(f"Combined LoRA+VPT Experiment on {bb['name']}")
    print("=" * 75)

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    # Setup config
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

        results = run_task(base_model, task, bb_key, device, config)
        all_results[task] = results

    # Summary
    print(f"\n{'='*75}")
    print(f"SUMMARY: {bb['name']}")
    print(f"{'='*75}")

    methods = ['LoRA_r4', 'LoRA_r8', 'VPT_p1', 'VPT_p5',
               'L2+V1', 'L4+V1', 'L4+V3', 'L4+V5', 'L8+V1']

    print(f"  {'Task':<14s}", end='')
    for m in methods:
        print(f" {m:>7s}", end='')
    print(f" {'Best':>8s}")
    print(f"  {'-'*100}")

    for task, results in all_results.items():
        print(f"  {task:<14s}", end='')
        best_method = ''
        best_acc = 0
        for m in methods:
            acc = results.get(m, 0)
            print(f" {acc:>7.3f}", end='')
            if acc > best_acc:
                best_acc = acc
                best_method = m
        print(f" {best_method:>8s}")

    # Average comparison
    print(f"\n  {'Average':<14s}", end='')
    for m in methods:
        avg = np.mean([r.get(m, 0) for r in all_results.values()])
        print(f" {avg:>7.3f}", end='')
    print()

    # How often does combined beat individual?
    combined_methods = ['L2+V1', 'L4+V1', 'L4+V3', 'L4+V5', 'L8+V1']
    individual_methods = ['LoRA_r4', 'LoRA_r8', 'VPT_p1', 'VPT_p5']

    combined_wins = 0
    total = 0
    for task, results in all_results.items():
        best_individual = max(results.get(m, 0) for m in individual_methods)
        best_combined = max(results.get(m, 0) for m in combined_methods)
        if best_combined > best_individual:
            combined_wins += 1
        total += 1

    print(f"\n  Combined beats best individual on {combined_wins}/{total} tasks")

    del base_model; torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/revision_combined_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
