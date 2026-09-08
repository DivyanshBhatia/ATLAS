"""
DINOv2-reg 7-point VPT LR sweep.

σ²_P = 0.62 — the only backbone near the 0.7 threshold.
This is the critical test: does it crash like DINOv2 (0.22) or stay robust like DeiT-III (1.04)?

Usage:
    cd /content/ATLAS
    python experiments/dinov2_reg_sweep.py
    python experiments/dinov2_reg_sweep.py --tasks cifar100 svhn
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
from exp2_comparison import apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

VPT_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
TASKS_DEFAULT = ['cifar100', 'svhn']
SAVE_PATH = 'results/dinov2_reg_sweep.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=TASKS_DEFAULT)
    parser.add_argument('--prompt_counts', nargs='+', type=int, default=[1, 5])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    print(f"\n{'='*60}")
    print(f"  DINOv2-reg (σ²_P = 0.62) — VPT LR Sweep")
    print(f"  Tests whether σ²_P = 0.62 crashes like DINOv2 (0.22)")
    print(f"  or stays robust like DeiT-III (1.04)")
    print(f"{'='*60}")

    # Load DINOv2-reg (with 4 register tokens)
    base_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                                    pretrained=True, img_size=224).to(device)
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = config.embed_dim // config.num_heads

    print(f"  Model: vit_base_patch14_reg4_dinov2.lvd142m")
    print(f"  embed_dim={config.embed_dim}, layers={config.num_layers}")
    print(f"  Register tokens: 4")

    for task in args.tasks:
        if task not in TASKS:
            print(f"  Unknown task: {task}"); continue

        num_classes = TASKS[task][0]
        config.num_classes = num_classes

        for p in args.prompt_counts:
            key = f"p{p}_{task}"
            if key in all_results:
                print(f"\n  p={p} x {task}: already done"); continue

            print(f"\n  --- DINOv2-reg x {task} (VPT p={p}) ---")
            lr_results = {}

            for lr in VPT_LRS:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, p, config)
                m = m.to(device)
                config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                lr_results[f'{lr}'] = float(acc)
                print(f"    VPT p={p} LR={lr:.0e}: {acc:.4f}")

            best_lr = max(VPT_LRS, key=lambda x: lr_results[f'{x}'])
            best_acc = lr_results[f'{best_lr}']
            acc_at_1e2 = lr_results.get('0.01', None)

            # Compute drop relative to best LR ≤ 2e-3
            low_lrs = [f'{x}' for x in VPT_LRS if x <= 2e-3]
            best_low = max(low_lrs, key=lambda x: lr_results.get(x, 0))
            best_low_acc = lr_results.get(best_low, 0)
            drop = (acc_at_1e2 - best_low_acc) * 100 if acc_at_1e2 else None

            all_results[key] = {
                'prompt_count': p,
                'task': task,
                'lr_sweep': lr_results,
                'best_lr': float(best_lr),
                'best_acc': float(best_acc),
                'best_low_lr': best_low,
                'best_low_acc': float(best_low_acc),
                'acc_at_1e2': float(acc_at_1e2) if acc_at_1e2 else None,
                'drop_at_1e2': float(drop) if drop else None,
            }

            print(f"    Best: LR={best_lr:.0e} acc={best_acc:.3f}")
            print(f"    Best (≤2e-3): LR={best_low} acc={best_low_acc:.3f}")
            if drop is not None:
                crash_label = "CRASH" if drop < -10 else "moderate" if drop < -5 else "mild" if drop < -2 else "robust"
                print(f"    Drop at 1e-2: {drop:+.1f} pts ({crash_label})")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

    del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("DINOv2-reg (σ²_P = 0.62) SUMMARY")
    print(f"{'='*60}")
    print(f"  Threshold test: does σ²_P = 0.62 crash like < 0.7 or stay robust?")
    for key, data in sorted(all_results.items()):
        drop = data.get('drop_at_1e2', None)
        drop_str = f"{drop:+.1f} pts" if drop is not None else "N/A"
        crash = "CRASH" if drop and drop < -10 else "NO CRASH" if drop and drop >= -10 else "?"
        print(f"  p={data['prompt_count']} x {data['task']}: "
              f"best={data['best_acc']:.3f} @ {data['best_lr']:.0e}, "
              f"drop={drop_str} → {crash}")

    # Comparison with DINOv2 (no reg)
    print(f"\n  For comparison:")
    print(f"  DINOv2     (σ²_P=0.22): C-100 drop = -60.5 pts (CRASH)")
    print(f"  DINOv2-reg (σ²_P=0.62): C-100 drop = {all_results.get('p5_cifar100', {}).get('drop_at_1e2', '?')} pts")
    print(f"  DeiT-III   (σ²_P=1.04): C-100 drop = -5.5 pts (robust)")


if __name__ == '__main__':
    main()
