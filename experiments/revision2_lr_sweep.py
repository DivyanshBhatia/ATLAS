"""
Per-Backbone Learning Rate Sweep — Addresses R1.4

R1's concern: "One learning rate across backbones spanning 70× in weight norm...
DINOv2 VPT failures might be optimization failures, not DINO exceptionalism."

This experiment tests whether VPT failure on DINOv2/DINOv2-reg persists
across a wide LR sweep. If VPT still fails at ALL learning rates,
DINO exceptionalism is real. If VPT recovers at a different LR,
the original claim was a tuning artifact.

Usage:
    python revision2_lr_sweep.py
    python revision2_lr_sweep.py --cases dinov2_cifar100 dinov2_svhn
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
from torch.utils.data import DataLoader

# Wide LR range: 20× below to 2× above standard
VPT_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
LORA_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

CASES = {
    'dinov2_cifar100': ('dinov2', 'cifar100'),
    'dinov2_svhn': ('dinov2', 'svhn'),
    'dinov2reg_cifar100': ('dinov2reg', 'cifar100'),
    'dinov2reg_svhn': ('dinov2reg', 'svhn'),
    # Controls: backbones where VPT should work
    'deit3_svhn': ('deit3', 'svhn'),
    'supervised_svhn': ('supervised', 'svhn'),
}

BACKBONE_MODELS = {
    'dinov2': ('vit_base_patch14_dinov2.lvd142m', 518),
    'dinov2reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 518),
    'deit3': ('deit3_base_patch16_224.fb_in1k', 224),
    'supervised': ('vit_base_patch16_224.augreg_in1k', 224),
}

SEEDS = [42, 123, 456]


def run_single(base_model, config, method, capacity, lr, train_loader, val_loader, device):
    """Run a single training with given method, capacity, and LR."""
    model = deepcopy(base_model)
    num_classes = config.num_classes
    model.head = nn.Linear(config.embed_dim, num_classes).to(device)

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
    parser.add_argument('--cases', nargs='+', default=list(CASES.keys()))
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds per LR (1 for sweep, 3 for key results)')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}

    for case_name in args.cases:
        if case_name not in CASES:
            print(f"Unknown case: {case_name}")
            continue

        bb_key, task = CASES[case_name]
        model_name, img_size = BACKBONE_MODELS[bb_key]
        num_classes = TASKS[task][0]
        config.num_classes = num_classes

        print(f"\n{'='*70}")
        print(f"  LR SWEEP: {bb_key} × {task}")
        print(f"{'='*70}")

        base_model = timm.create_model(model_name, pretrained=True,
                                        img_size=img_size).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        case_results = {'vpt': {}, 'lora': {}}

        for seed_idx in range(args.seeds):
            seed = SEEDS[seed_idx]
            torch.manual_seed(seed)
            np.random.seed(seed)

            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = torch.utils.data.random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(seed))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            # VPT sweep
            print(f"\n  VPT_p5 LR sweep (seed {seed}):")
            for lr in VPT_LRS:
                acc = run_single(base_model, config, 'vpt', 5, lr,
                                 train_loader, val_loader, device)
                key = f"lr={lr}"
                if key not in case_results['vpt']:
                    case_results['vpt'][key] = []
                case_results['vpt'][key].append(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")

            # Also test VPT_p1
            print(f"\n  VPT_p1 LR sweep (seed {seed}):")
            for lr in VPT_LRS:
                acc = run_single(base_model, config, 'vpt', 1, lr,
                                 train_loader, val_loader, device)
                key = f"p1_lr={lr}"
                if key not in case_results['vpt']:
                    case_results['vpt'][key] = []
                case_results['vpt'][key].append(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")

            # LoRA sweep
            print(f"\n  LoRA_r8 LR sweep (seed {seed}):")
            for lr in LORA_LRS:
                acc = run_single(base_model, config, 'lora', 8, lr,
                                 train_loader, val_loader, device)
                key = f"lr={lr}"
                if key not in case_results['lora']:
                    case_results['lora'][key] = []
                case_results['lora'][key].append(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")

        del base_model
        torch.cuda.empty_cache()

        # Summary
        print(f"\n{'='*70}")
        print(f"  SUMMARY: {bb_key} × {task}")
        print(f"{'='*70}")

        best_vpt_lr = max(case_results['vpt'].keys(),
                          key=lambda k: np.mean(case_results['vpt'][k])
                          if not k.startswith('p1_') else -1)
        best_vpt_p1_lr = max([k for k in case_results['vpt'].keys() if k.startswith('p1_')],
                              key=lambda k: np.mean(case_results['vpt'][k]))
        best_lora_lr = max(case_results['lora'].keys(),
                           key=lambda k: np.mean(case_results['lora'][k]))

        best_vpt = np.mean(case_results['vpt'][best_vpt_lr])
        best_vpt_p1 = np.mean(case_results['vpt'][best_vpt_p1_lr])
        best_lora = np.mean(case_results['lora'][best_lora_lr])
        default_vpt = np.mean(case_results['vpt'].get('lr=0.01', [0]))
        default_lora = np.mean(case_results['lora'].get('lr=0.001', [0]))

        print(f"  Best VPT_p5:  {best_vpt:.3f} ({best_vpt_lr})")
        print(f"  Best VPT_p1:  {best_vpt_p1:.3f} ({best_vpt_p1_lr})")
        print(f"  Best LoRA_r8: {best_lora:.3f} ({best_lora_lr})")
        print(f"  Default VPT:  {default_vpt:.3f} (lr=1e-2)")
        print(f"  Default LoRA: {default_lora:.3f} (lr=1e-3)")
        print(f"  Gap (default): {default_lora - default_vpt:+.3f}")
        print(f"  Gap (best):    {best_lora - best_vpt:+.3f}")

        if best_vpt > default_vpt + 0.03:
            print(f"  ⚠ VPT IMPROVES with different LR! (+{best_vpt-default_vpt:.3f})")
            print(f"  → Original claim may be a tuning artifact")
        else:
            print(f"  ✓ VPT failure persists across all LRs")
            print(f"  → DINO exceptionalism is NOT a tuning artifact")

        if best_lora - best_vpt > 0.02:
            print(f"  ✓ LoRA still wins after fairness sweep ({best_lora-best_vpt:+.3f})")
        elif abs(best_lora - best_vpt) <= 0.02:
            print(f"  ~ Methods are within noise after sweep ({best_lora-best_vpt:+.3f})")
        else:
            print(f"  ⚠ VPT WINS after sweep ({best_lora-best_vpt:+.3f})")

        all_results[case_name] = case_results

    # Save all results
    os.makedirs('results', exist_ok=True)
    with open('results/revision2_lr_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_lr_sweep.json")


if __name__ == '__main__':
    main()
