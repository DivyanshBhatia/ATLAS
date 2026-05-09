"""
REVISION W1: Measure actual learned parameter magnitudes ||Δθ||² for LoRA vs VPT.

Reviewer concern: "η_eff is undefined. If η_eff differs between LoRA and VPT,
the 12× and 6× ratios do not follow."

This experiment:
1. Trains LoRA and VPT on 5 tasks
2. Measures ||Δθ||² for each method after training
3. Computes the per-parameter perturbation magnitude η²_eff = ||Δθ||² / D_eff
4. Reports ratio η²_eff(VPT) / η²_eff(LoRA)
   - If ratio ≈ 1: η_eff is matched → 6× claim holds from dimensionality alone
   - If ratio ≠ 1: 6× is modulated by training dynamics → reframe claim

Usage:
    python revision_w1_eta_eff.py
    python revision_w1_eta_eff.py --tasks cifar10 svhn dtd
    python revision_w1_eta_eff.py --backbones dinov2 deit3
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import json
from copy import deepcopy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig, setup_device
from exp2_comparison import (apply_lora, apply_vpt, apply_linear_probe,
                              train_and_evaluate)
from run_all_backbones import BACKBONES, TASKS, get_transforms, load_dataset


def measure_delta_theta_sq(model_before_state, model_after, method_name):
    """Compute ||Δθ||² = sum of squared differences in learned parameters."""
    total_delta_sq = 0.0
    total_params = 0
    param_details = {}

    for name, param in model_after.named_parameters():
        if not param.requires_grad:
            continue

        if name in model_before_state:
            delta = param.data - model_before_state[name].to(param.device)
        else:
            # New parameter (e.g., LoRA A/B, VPT prompts) — delta from zero
            delta = param.data

        delta_sq = (delta.float() ** 2).sum().item()
        n_params = param.numel()
        total_delta_sq += delta_sq
        total_params += n_params
        param_details[name] = {
            'delta_sq': delta_sq,
            'n_params': n_params,
            'mean_delta_sq': delta_sq / n_params if n_params > 0 else 0
        }

    return {
        'total_delta_sq': total_delta_sq,
        'total_params': total_params,
        'eta_eff_sq': total_delta_sq / total_params if total_params > 0 else 0,
        'details': param_details
    }


def run_single(backbone_key, task_name, device, config):
    """Train LoRA and VPT on one backbone-task pair, measure ||Δθ||²."""
    bb = BACKBONES[backbone_key]
    num_classes = TASKS[task_name][0]

    # Load model
    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    # Setup config for this backbone
    embed_dim = base_model.embed_dim
    num_layers = len(base_model.blocks)
    num_heads = base_model.blocks[0].attn.num_heads
    d_h = embed_dim // num_heads
    config.embed_dim = embed_dim
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.head_dim = d_h

    # Compute σ_P²
    total_norm, count = 0.0, 0
    for name, param in base_model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total_norm += param.float().norm().item() ** 2
            count += 1
    sigma_sq = total_norm / (count * d_h) if count > 0 else 1.0

    # Load data
    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    # --- LoRA at multiple ranks ---
    for r in [1, 4, 8]:
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        # Save state before training
        state_before = {n: p.data.clone() for n, p in model.named_parameters()}
        model = apply_lora(model, r, config)
        model = model.to(device)

        acc = train_and_evaluate(model, train_loader, val_loader, config, device)

        measurements = measure_delta_theta_sq(state_before, model, f'LoRA_r{r}')
        D_eff = 12 * r * d_h  # L * r * d_h
        results[f'LoRA_r{r}'] = {
            'accuracy': acc,
            'total_delta_sq': measurements['total_delta_sq'],
            'total_params': measurements['total_params'],
            'eta_eff_sq': measurements['eta_eff_sq'],
            'D_eff_theory': D_eff,
            'eta_eff_sq_theory': measurements['total_delta_sq'] / D_eff if D_eff > 0 else 0,
        }

        del model
        torch.cuda.empty_cache()

    # --- VPT at multiple prompt counts ---
    for p in [1, 5, 10]:
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, num_classes).to(device)
        state_before = {n: p_param.data.clone() for n, p_param in model.named_parameters()}
        model = apply_vpt(model, p, config)
        model = model.to(device)

        acc = train_and_evaluate(model, train_loader, val_loader, config, device)

        measurements = measure_delta_theta_sq(state_before, model, f'VPT_p{p}')
        d = embed_dim
        D_eff = 12 * p * d  # L * p * d
        results[f'VPT_p{p}'] = {
            'accuracy': acc,
            'total_delta_sq': measurements['total_delta_sq'],
            'total_params': measurements['total_params'],
            'eta_eff_sq': measurements['eta_eff_sq'],
            'D_eff_theory': D_eff,
            'eta_eff_sq_theory': measurements['total_delta_sq'] / D_eff if D_eff > 0 else 0,
        }

        del model
        torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()

    return results, sigma_sq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat'])
    parser.add_argument('--backbones', nargs='+', default=['dinov2', 'deit3'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("REVISION W1: Measuring η²_eff for LoRA vs VPT")
    print("=" * 70)

    all_results = {}

    for bb_key in args.backbones:
        for task in args.tasks:
            if task not in TASKS:
                continue

            print(f"\n{'=' * 50}")
            print(f"  {BACKBONES[bb_key]['name']} × {task}")
            print(f"{'=' * 50}")

            results, sigma_sq = run_single(bb_key, task, device, config)
            all_results[f'{bb_key}_{task}'] = {
                'sigma_sq': sigma_sq,
                'methods': results
            }

            # Print summary
            print(f"\n  σ²_P = {sigma_sq:.2f}")
            print(f"  {'Method':<12s} {'Acc':>6s} {'||Δθ||²':>12s} {'D_eff':>8s} "
                  f"{'η²_eff':>12s} {'η²_theory':>12s}")
            print(f"  {'-'*64}")
            for method, res in results.items():
                print(f"  {method:<12s} {res['accuracy']:>6.3f} "
                      f"{res['total_delta_sq']:>12.4f} "
                      f"{res['D_eff_theory']:>8d} "
                      f"{res['eta_eff_sq']:>12.6f} "
                      f"{res['eta_eff_sq_theory']:>12.6f}")

            # Compute ratio
            lora_eta = results.get('LoRA_r4', results.get('LoRA_r1', {}))
            vpt_eta = results.get('VPT_p1', {})
            if lora_eta and vpt_eta:
                ratio = vpt_eta.get('eta_eff_sq_theory', 0) / (lora_eta.get('eta_eff_sq_theory', 1e-10))
                print(f"\n  η²_eff ratio (VPT/LoRA): {ratio:.3f}")
                print(f"  If ≈1.0: 6× claim holds from dimensionality alone")
                print(f"  If ≠1.0: effective 6× ratio is {6.0 / ratio:.1f}×")

    # Save
    with open('results/revision_w1_eta_eff.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results/revision_w1_eta_eff.json")


if __name__ == '__main__':
    main()
