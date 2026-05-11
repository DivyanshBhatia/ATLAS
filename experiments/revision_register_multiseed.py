"""
Multi-Seed Register-Prompt Interference Validation

The register-token interference claim (VPT_p1=0.865 → VPT_p10=0.220 on
DINOv2-reg CIFAR-100) is dramatic and needs multi-seed validation.

Tests 3 seeds × 3 tasks × {VPT_p1, VPT_p5, VPT_p10, LoRA_r8}
on DINOv2-reg to confirm the degradation is systematic.

Usage:
    python revision_register_multiseed.py
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
from exp2_comparison import apply_lora, apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]
TEST_TASKS = ['cifar100', 'gtsrb', 'svhn']

# DINOv2 with registers
BB = {
    'model': 'vit_base_patch14_reg4_dinov2.lvd142m',
    'img_size': 518,
    'name': 'DINOv2-reg ViT-B/14',
}

METHODS = [
    ('LoRA_r8', lambda m, c: apply_lora(m, 8, c), 1e-3),
    ('VPT_p1', lambda m, c: apply_vpt(m, 1, c), 1e-2),
    ('VPT_p5', lambda m, c: apply_vpt(m, 5, c), 1e-2),
    ('VPT_p10', lambda m, c: apply_vpt(m, 10, c), 1e-2),
]


def main():
    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("Multi-Seed Register-Prompt Interference Validation")
    print(f"Seeds: {SEEDS}, Tasks: {TEST_TASKS}")
    print("=" * 70)

    base_model = timm.create_model(BB['model'], pretrained=True,
                                    img_size=BB['img_size']).to(device)
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    all_results = {}

    for task in TEST_TASKS:
        num_classes = TASKS[task][0]
        print(f"\n{'='*55}")
        print(f"  DINOv2-reg × {task}")
        print(f"{'='*55}")

        task_results = {}

        for method_name, apply_fn, lr in METHODS:
            seed_accs = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                ds = load_dataset(task, BB['img_size'], max_samples=1000)
                n_val = min(200, len(ds) // 5)
                train_ds, val_ds = torch.utils.data.random_split(
                    ds, [len(ds) - n_val, n_val],
                    generator=torch.Generator().manual_seed(seed))
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

                model = deepcopy(base_model)
                model.head = nn.Linear(config.embed_dim, num_classes).to(device)
                model = apply_fn(model, config)
                model = model.to(device)

                config.lr = lr
                acc = train_and_evaluate(model, train_loader, val_loader, config, device)
                seed_accs.append(acc)

                del model; torch.cuda.empty_cache()

            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            task_results[method_name] = {
                'seeds': seed_accs, 'mean': mean_acc, 'std': std_acc
            }
            print(f"    {method_name:<10s}: {mean_acc:.3f} ± {std_acc:.3f}  "
                  f"({', '.join(f'{a:.3f}' for a in seed_accs)})")

        all_results[task] = task_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Register-Prompt Interference (mean ± std)")
    print(f"{'='*70}")
    print(f"  {'Task':<12s} {'LoRA_r8':>14s} {'VPT_p1':>14s} {'VPT_p5':>14s} {'VPT_p10':>14s}")
    for task, results in all_results.items():
        parts = []
        for m in ['LoRA_r8', 'VPT_p1', 'VPT_p5', 'VPT_p10']:
            r = results[m]
            parts.append(f"{r['mean']:.3f}±{r['std']:.3f}")
        print(f"  {task:<12s} {'  '.join(parts)}")

    print(f"\n  Key question: Does VPT degrade as p increases on DINOv2-reg?")
    for task, results in all_results.items():
        v1 = results['VPT_p1']['mean']
        v10 = results['VPT_p10']['mean']
        degradation = v1 - v10
        print(f"    {task}: VPT_p1→VPT_p10 degradation = {degradation:+.3f} "
              f"({'CONFIRMED' if degradation > 0.05 else 'mild' if degradation > 0 else 'NOT CONFIRMED'})")

    del base_model; torch.cuda.empty_cache()

    os.makedirs('results', exist_ok=True)
    with open('results/register_multiseed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/register_multiseed.json")


if __name__ == '__main__':
    main()
