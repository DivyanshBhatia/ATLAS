"""
Layer-Adaptive LoRA: Training-Free Rank Allocation

Key idea: Instead of uniform rank across layers, allocate rank proportional
to per-layer σ²_l (weight norm). Same TOTAL parameter budget as uniform LoRA,
but smarter allocation.

This is "training-free AdaLoRA" — AdaLoRA requires training to discover
layer importance; we compute it from pretrained weights in <1 second.

Theory extension: Per-layer PAC-Bayes bound
  KL_l = r_l × d_h × η²_l / (2σ²_l)
  Optimal: r_l ∝ σ²_l (layers with higher prior variance afford more rank)

Allocation strategies:
1. Uniform (baseline): all layers get rank r
2. σ²_l-proportional: r_l ∝ σ²_l (more rank where weights are larger)
3. Increasing: r grows linearly with layer depth (later layers get more)
4. Top-heavy: top 4 layers get 2× rank, bottom 4 get 0.5× rank

Usage:
    python revision_layer_adaptive.py
    python revision_layer_adaptive.py --backbone deit3 --tasks svhn dtd cifar100
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
import math
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import train_and_evaluate
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader


class LayerAdaptiveLoRA(nn.Module):
    """LoRA with per-layer rank allocation."""
    def __init__(self, original_linear, rank, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        if rank > 0:
            d_out, d_in = original_linear.weight.shape
            self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
            self.scaling = alpha / rank
        
    def forward(self, x):
        out = self.original(x)
        if self.rank > 0:
            out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out


def compute_per_layer_sigma(model):
    """Compute σ²_l for each layer from attention weight norms."""
    d_h = model.embed_dim // model.blocks[0].attn.num_heads
    layer_sigmas = []
    
    for block in model.blocks:
        total_norm = 0.0
        count = 0
        for name, param in block.attn.named_parameters():
            if 'weight' in name and ('qkv' in name or 'proj' in name):
                total_norm += param.float().norm().item() ** 2
                count += 1
        sigma_l = total_norm / (count * d_h) if count > 0 else 1.0
        layer_sigmas.append(sigma_l)
    
    return layer_sigmas


def allocate_ranks(layer_sigmas, total_budget, strategy='proportional', min_rank=1):
    """Allocate per-layer ranks given a total parameter budget.
    
    total_budget: total rank budget (sum of all layer ranks = L × r_uniform)
    """
    L = len(layer_sigmas)
    
    if strategy == 'uniform':
        r_per_layer = total_budget // L
        return [r_per_layer] * L
    
    elif strategy == 'proportional':
        # r_l ∝ σ²_l
        sigmas = np.array(layer_sigmas)
        weights = sigmas / sigmas.sum()
        ranks = np.maximum(min_rank, np.round(weights * total_budget)).astype(int)
        # Adjust to match budget exactly
        while ranks.sum() > total_budget:
            ranks[ranks.argmax()] -= 1
        while ranks.sum() < total_budget:
            ranks[ranks.argmin()] += 1
        return ranks.tolist()
    
    elif strategy == 'increasing':
        # Linear increase: later layers get more rank
        weights = np.linspace(0.5, 1.5, L)
        weights = weights / weights.sum()
        ranks = np.maximum(min_rank, np.round(weights * total_budget)).astype(int)
        while ranks.sum() > total_budget:
            ranks[ranks.argmax()] -= 1
        while ranks.sum() < total_budget:
            ranks[ranks.argmin()] += 1
        return ranks.tolist()
    
    elif strategy == 'top_heavy':
        # Top 4 layers get 2× rank, bottom 4 get 0.5×
        weights = np.array([0.5]*4 + [1.0]*4 + [2.0]*4)
        weights = weights / weights.sum()
        ranks = np.maximum(min_rank, np.round(weights * total_budget)).astype(int)
        while ranks.sum() > total_budget:
            ranks[ranks.argmax()] -= 1
        while ranks.sum() < total_budget:
            ranks[ranks.argmin()] += 1
        return ranks.tolist()
    
    elif strategy == 'sqrt_proportional':
        # r_l ∝ √σ²_l (less aggressive than proportional)
        sigmas = np.sqrt(np.array(layer_sigmas))
        weights = sigmas / sigmas.sum()
        ranks = np.maximum(min_rank, np.round(weights * total_budget)).astype(int)
        while ranks.sum() > total_budget:
            ranks[ranks.argmax()] -= 1
        while ranks.sum() < total_budget:
            ranks[ranks.argmin()] += 1
        return ranks.tolist()


def apply_adaptive_lora(model, per_layer_ranks, num_classes):
    """Apply LoRA with different ranks per layer."""
    embed_dim = model.embed_dim
    
    # Replace head
    model.head = nn.Linear(embed_dim, num_classes)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Apply LoRA to each layer with its specific rank
    for i, block in enumerate(model.blocks):
        rank = per_layer_ranks[i]
        if rank > 0:
            # Replace qkv projection
            old_qkv = block.attn.qkv
            block.attn.qkv = LayerAdaptiveLoRA(old_qkv, rank)
            
            # Replace output projection
            old_proj = block.attn.proj
            block.attn.proj = LayerAdaptiveLoRA(old_proj, rank)
    
    # Unfreeze LoRA params and head
    for name, param in model.named_parameters():
        if 'lora_' in name or 'head' in name:
            param.requires_grad_(True)
    
    return model


def run_task(base_model, task_name, bb_key, device, config, layer_sigmas):
    """Run uniform and adaptive LoRA on one task."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    L = len(base_model.blocks)

    # Load data
    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    # Test at two budget levels: r_avg=4 (budget=48) and r_avg=8 (budget=96)
    for r_avg in [4, 8]:
        total_budget = L * r_avg  # Same total params as uniform

        strategies = ['uniform', 'proportional', 'sqrt_proportional', 'increasing', 'top_heavy']

        for strategy in strategies:
            ranks = allocate_ranks(layer_sigmas, total_budget, strategy)

            model = deepcopy(base_model).to(device)
            model = apply_adaptive_lora(model, ranks, num_classes)
            model = model.to(device)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            acc = train_and_evaluate(model, train_loader, val_loader, config, device)

            key = f"r{r_avg}_{strategy}"
            results[key] = {'acc': acc, 'ranks': ranks, 'params': trainable}
            
            rank_str = '-'.join(str(r) for r in ranks)
            print(f"    {key:<25s}: {acc:.3f}  ranks=[{rank_str}]  ({trainable:,}p)")
            
            del model; torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat',
                                 'gtsrb', 'fashionmnist', 'food101', 'mnist'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 75)
    print(f"Layer-Adaptive LoRA on {bb['name']}")
    print("=" * 75)

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    # Setup config
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    # Compute per-layer σ²_l
    layer_sigmas = compute_per_layer_sigma(base_model)
    print(f"\n  Per-layer σ²_l:")
    for i, s in enumerate(layer_sigmas):
        bar = '█' * int(s * 3)
        print(f"    Layer {i:>2d}: σ²={s:>7.3f} {bar}")
    print(f"    Mean σ²_P = {np.mean(layer_sigmas):.3f}")

    # Show rank allocations for r_avg=8
    total_budget = 12 * 8
    print(f"\n  Rank allocations for budget={total_budget} (equiv to uniform r=8):")
    for strategy in ['uniform', 'proportional', 'sqrt_proportional', 'increasing', 'top_heavy']:
        ranks = allocate_ranks(layer_sigmas, total_budget, strategy)
        print(f"    {strategy:<20s}: {ranks}")

    all_results = {}

    for task in args.tasks:
        if task not in TASKS:
            continue

        print(f"\n{'='*55}")
        print(f"  {bb['name']} × {task}")
        print(f"{'='*55}")

        results = run_task(base_model, task, bb_key, device, config, layer_sigmas)
        all_results[task] = results

    # Summary
    print(f"\n{'='*75}")
    print(f"SUMMARY: {bb['name']}")
    print(f"{'='*75}")

    strategies = ['uniform', 'proportional', 'sqrt_proportional', 'increasing', 'top_heavy']
    
    for r_avg in [4, 8]:
        print(f"\n  Budget = r_avg={r_avg} (total rank budget = {12*r_avg}):")
        print(f"  {'Task':<14s}", end='')
        for s in strategies:
            print(f" {s[:8]:>9s}", end='')
        print(f" {'Best':>10s}")

        avgs = {s: [] for s in strategies}
        for task, results in all_results.items():
            print(f"  {task:<14s}", end='')
            best_s, best_a = '', 0
            for s in strategies:
                key = f"r{r_avg}_{s}"
                acc = results.get(key, {}).get('acc', 0)
                avgs[s].append(acc)
                print(f" {acc:>9.3f}", end='')
                if acc > best_a:
                    best_a = acc
                    best_s = s
            print(f" {best_s[:10]:>10s}")

        print(f"  {'AVERAGE':<14s}", end='')
        for s in strategies:
            print(f" {np.mean(avgs[s]):>9.3f}", end='')
        print()

        # How often does adaptive beat uniform?
        for s in strategies:
            if s == 'uniform':
                continue
            wins = sum(1 for t in all_results 
                      if all_results[t].get(f'r{r_avg}_{s}',{}).get('acc',0) > 
                         all_results[t].get(f'r{r_avg}_uniform',{}).get('acc',0))
            print(f"    {s} beats uniform on {wins}/{len(all_results)} tasks")

    del base_model; torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/revision_adaptive_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
