"""
Experiment 4: Learning Rate Sweep (Proposition 2)

Purpose: Validate that the optimal LR for each PEFT method is predicted by
the spectral structure of the task (σ₁, α, d_eff).

Predictions:
- η_opt ∝ 1/√d_eff across methods (FFT needs lowest, LP needs highest)
- η_opt ∝ 1/σ₁(ΔW*) across tasks (larger weight shift → smaller LR)
- Tasks with higher α tolerate higher LR at matched d_eff
"""

import torch
import torch.nn as nn
import timm
import numpy as np
import json
from copy import deepcopy
from torch.utils.data import DataLoader, random_split

from config import (ExperimentConfig, setup_device, ensure_dirs, save_results,
                    compute_svd_profile, fit_spectral_decay)
from exp1_spectral import load_real_dataset, SyntheticVTABDataset
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


def run_lr_sweep(config: ExperimentConfig):
    """Run LR sweep for each method × task combination."""
    print("=" * 70)
    print("EXPERIMENT 4: LEARNING RATE SWEEP (Proposition 2)")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    # Load model
    print("\nLoading pretrained model...")
    try:
        base_model = timm.create_model(config.model_name, pretrained=True,
                                        img_size=config.img_size)
    except Exception:
        base_model = timm.create_model(config.model_name, pretrained=False,
                                        img_size=config.img_size)

    pretrained_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    # LR grid
    lr_grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

    # Methods to sweep
    methods = {
        'LP': lambda m, c: apply_linear_probe(m, c),
        'LoRA_r4': lambda m, c: apply_lora(m, 4, c),
        'LoRA_r16': lambda m, c: apply_lora(m, 16, c),
        'VPT_p10': lambda m, c: apply_vpt(m, 10, c),
        'VPT_p50': lambda m, c: apply_vpt(m, 50, c),
        'Adapter_r16': lambda m, c: apply_adapter(m, 16, c),
        'FFT': None,  # special handling
    }

    # Tasks (representative from each category)
    tasks = {
        'cifar100': (100, 'natural'),
        'gtsrb': (43, 'structured'),
    }

    all_results = {}

    for task_name, (n_classes, category) in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*60}")

        # Load data — use 2000 samples for speed (LR sweep needs many runs)
        dataset = load_real_dataset(task_name, n_classes, config,
                                    max_samples=2000)
        if dataset is None:
            task_type = 'structured' if category == 'structured' else 'natural'
            dataset = SyntheticVTABDataset(2000, n_classes,
                                           config.img_size, task_type)

        n_total = len(dataset)
        n_val = max(200, n_total // 5)
        train_ds, val_ds = random_split(dataset, [n_total - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, num_workers=2)

        task_results = {}

        for method_name, apply_fn in methods.items():
            print(f"\n  Method: {method_name}")
            lr_results = {}

            for lr in lr_grid:
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)

                if method_name == 'FFT':
                    # All params trainable
                    for param in model.parameters():
                        param.requires_grad_(True)
                else:
                    model = apply_fn(model, config)

                d_eff = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

                # Short training (30 epochs for speed)
                acc = train_and_evaluate(model, train_loader, val_loader,
                                         config, device,
                                         f'{method_name} lr={lr:.0e}',
                                         max_epochs=30, lr=lr)

                lr_results[f'{lr:.0e}'] = {
                    'accuracy': acc,
                    'lr': lr,
                    'd_eff': d_eff,
                }
                del model

            # Find optimal LR
            best_lr_key = max(lr_results, key=lambda k: lr_results[k]['accuracy'])
            best_lr = lr_results[best_lr_key]['lr']
            best_acc = lr_results[best_lr_key]['accuracy']
            d_eff = lr_results[best_lr_key]['d_eff']

            task_results[method_name] = {
                'lr_sweep': lr_results,
                'optimal_lr': best_lr,
                'optimal_accuracy': best_acc,
                'd_eff': d_eff,
            }

            print(f"    Optimal LR: {best_lr:.0e} (acc={best_acc:.4f}, d_eff={d_eff:,})")

        all_results[task_name] = task_results

    # ========================================================================
    # Analysis: Validate Proposition 2
    # ========================================================================
    print("\n" + "=" * 70)
    print("PROPOSITION 2 VALIDATION")
    print("=" * 70)

    print("\n  η_opt vs √d_eff (expect: η_opt ∝ 1/√d_eff)")
    print(f"  {'Method':<15s} {'d_eff':>10s} {'√d_eff':>10s} {'η_opt':>10s} {'η·√d_eff':>10s}")
    print(f"  {'-'*55}")

    for task_name, task_res in all_results.items():
        print(f"\n  {task_name}:")
        products = []
        for method_name, res in sorted(task_res.items(),
                                        key=lambda x: x[1]['d_eff']):
            d_eff = res['d_eff']
            eta = res['optimal_lr']
            sqrt_d = np.sqrt(d_eff)
            product = eta * sqrt_d
            products.append(product)
            print(f"  {method_name:<15s} {d_eff:>10,d} {sqrt_d:>10.1f} "
                  f"{eta:>10.0e} {product:>10.4f}")

        # If η·√d_eff is roughly constant, Proposition 2 is validated
        if len(products) > 2:
            cv = np.std(products) / np.mean(products)
            print(f"  Coefficient of variation of η·√d_eff: {cv:.3f} "
                  f"({'GOOD (<0.5)' if cv < 0.5 else 'WEAK (>0.5)'})")

    save_results(all_results, 'experiment4_lr_sweep.json', config)
    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    run_lr_sweep(config)
