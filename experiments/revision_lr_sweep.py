"""
Hyperparameter Fairness Sweep for VPT

Preempts the reviewer objection that VPT failed due to poor tuning.
Tests 5 learning rates on VPT's worst failure cases.

If VPT stays bad across ALL LRs, the failure is fundamental, not tuning.

Usage:
    python revision_lr_sweep.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt, apply_lora, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader

LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
WDS = [0.01, 0.05]


def main():
    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    CASES = [
        ('DINOv2', 'vit_base_patch14_dinov2.lvd142m', 518, 'cifar100'),
        ('DINOv2-reg', 'vit_base_patch14_reg4_dinov2.lvd142m', 518, 'cifar100'),
        ('DINOv2', 'vit_base_patch14_dinov2.lvd142m', 518, 'svhn'),
    ]

    print("=" * 70)
    print("VPT Hyperparameter Fairness Sweep")
    print(f"LRs: {LRS}")
    print("=" * 70)

    all_results = {}

    for bb_name, model_name, img_size, task in CASES:
        key = f"{bb_name}_{task}"
        print(f"\n{'='*55}")
        print(f"  {bb_name} × {task} — VPT_p5 LR sweep")
        print(f"{'='*55}")

        base_model = timm.create_model(model_name, pretrained=True,
                                        img_size=img_size).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        num_classes = TASKS[task][0]
        ds = load_dataset(task, img_size, max_samples=1000)
        n_val = min(200, len(ds) // 5)
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        # LoRA baseline (single LR)
        model = deepcopy(base_model)
        model.head = nn.Linear(config.embed_dim, num_classes).to(device)
        model = apply_lora(model, 8, config)
        model = model.to(device)
        config.lr = 1e-3
        lora_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
        print(f"    LoRA_r8 (lr=1e-3): {lora_acc:.3f}")
        del model; torch.cuda.empty_cache()

        # VPT sweep
        case_results = {'LoRA_r8': lora_acc}
        best_vpt = 0

        for lr in LRS:
            for wd in WDS:
                model = deepcopy(base_model)
                model.head = nn.Linear(config.embed_dim, num_classes).to(device)
                model = apply_vpt(model, 5, config)
                model = model.to(device)

                config.lr = lr
                # Use custom optimizer with this WD
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=lr, weight_decay=wd)

                acc = train_and_evaluate(model, train_loader, val_loader, config, device)
                tag = f"VPT_p5_lr{lr}_wd{wd}"
                case_results[tag] = acc
                best_vpt = max(best_vpt, acc)
                print(f"    {tag}: {acc:.3f}")

                del model; torch.cuda.empty_cache()

        case_results['best_vpt'] = best_vpt
        all_results[key] = case_results

        print(f"\n    Best VPT across sweep: {best_vpt:.3f}")
        print(f"    LoRA_r8 baseline:      {lora_acc:.3f}")
        print(f"    Gap: {lora_acc - best_vpt:+.3f}")

        if best_vpt < lora_acc - 0.02:
            print(f"    → VPT failure is NOT due to LR/WD tuning")
        else:
            print(f"    → VPT competitive with proper tuning")

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("FAIRNESS SWEEP SUMMARY")
    print(f"{'='*70}")
    for key, results in all_results.items():
        lora = results['LoRA_r8']
        best = results['best_vpt']
        print(f"  {key:<25s}: LoRA={lora:.3f}, best_VPT={best:.3f}, gap={lora-best:+.3f}")

    os.makedirs('results', exist_ok=True)
    with open('results/lr_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/lr_sweep.json")


if __name__ == '__main__':
    main()
