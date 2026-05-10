"""
Measure spectral decay exponent α of trained LoRA weight updates.

Validates Assumption 2: σ_i ~ i^{-α} with α > 1/2.

For each trained LoRA model, extracts ΔW = BA, computes SVD,
and fits power-law decay to singular values.

Usage:
    python validate_spectral_decay.py
    python validate_spectral_decay.py --backbone deit3
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
from scipy import stats
import json
import os
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora, train_and_evaluate
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader


def extract_lora_delta_W(model):
    """Extract ΔW = BA from all LoRA layers."""
    delta_Ws = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            A = module.lora_A.data.float()  # (r, d_in)
            B = module.lora_B.data.float()  # (d_out, r)
            scaling = getattr(module, 'scaling', 1.0)
            delta_W = (B @ A) * scaling  # (d_out, d_in)
            delta_Ws.append({
                'name': name,
                'delta_W': delta_W.cpu().numpy(),
                'rank': A.shape[0],
            })
    return delta_Ws


def fit_spectral_decay(singular_values, min_sv_ratio=1e-4):
    """Fit power-law decay σ_i ~ i^{-α} to singular values."""
    sv = np.array(singular_values)
    sv = sv[sv > sv[0] * min_sv_ratio]  # Filter near-zero SVs
    
    if len(sv) < 3:
        return 0.0, 0.0  # Not enough data
    
    log_i = np.log(np.arange(1, len(sv) + 1))
    log_sv = np.log(sv)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_i, log_sv)
    
    alpha = -slope  # σ_i ~ i^{-α} means log(σ_i) ~ -α·log(i)
    r_squared = r_value ** 2
    
    return alpha, r_squared


def run_task(base_model, task_name, bb_key, device, config, rank=8):
    """Train LoRA and measure spectral decay."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    embed_dim = config.embed_dim

    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Train LoRA
    model = deepcopy(base_model)
    model.head = nn.Linear(embed_dim, num_classes).to(device)
    model = apply_lora(model, rank, config)
    model = model.to(device)
    acc = train_and_evaluate(model, train_loader, val_loader, config, device)

    # Extract ΔW and measure spectral decay
    delta_Ws = extract_lora_delta_W(model)
    
    results = []
    for dw in delta_Ws:
        U, S, Vt = np.linalg.svd(dw['delta_W'], full_matrices=False)
        alpha, r2 = fit_spectral_decay(S)
        results.append({
            'layer': dw['name'],
            'alpha': alpha,
            'r_squared': r2,
            'top_sv': S[:5].tolist(),
            'sv_ratio': float(S[0] / S[-1]) if S[-1] > 0 else float('inf'),
        })
    
    mean_alpha = np.mean([r['alpha'] for r in results])
    mean_r2 = np.mean([r['r_squared'] for r in results])
    
    del model; torch.cuda.empty_cache()
    
    return {
        'acc': acc,
        'mean_alpha': mean_alpha,
        'mean_r2': mean_r2,
        'per_layer': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat'])
    parser.add_argument('--rank', type=int, default=8)
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 70)
    print(f"Spectral Decay Validation on {bb['name']} (LoRA rank={args.rank})")
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
        print(f"\n  {bb['name']} × {task}")
        result = run_task(base_model, task, bb_key, device, config, args.rank)
        all_results[task] = result
        print(f"    Acc: {result['acc']:.3f}")
        print(f"    Mean α: {result['mean_alpha']:.3f} (R²={result['mean_r2']:.3f})")
        
        if result['mean_alpha'] > 0.5:
            print(f"    → Assumption 2 VALIDATED (α={result['mean_alpha']:.2f} > 0.5)")
        else:
            print(f"    → Assumption 2 NOT MET (α={result['mean_alpha']:.2f} ≤ 0.5)")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {bb['name']}")
    print(f"{'='*70}")
    print(f"  {'Task':<14s} {'Acc':>6s} {'α':>6s} {'R²':>6s} {'Valid?':>8s}")
    for task, r in all_results.items():
        valid = "✓" if r['mean_alpha'] > 0.5 else "✗"
        print(f"  {task:<14s} {r['acc']:>6.3f} {r['mean_alpha']:>6.3f} {r['mean_r2']:>6.3f} {valid:>8s}")

    overall_alpha = np.mean([r['mean_alpha'] for r in all_results.values()])
    print(f"\n  Overall mean α = {overall_alpha:.3f}")
    print(f"  Assumption 2 (α > 0.5): {'VALIDATED' if overall_alpha > 0.5 else 'NOT MET'}")

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/spectral_decay_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
