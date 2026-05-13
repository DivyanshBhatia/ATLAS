"""
Multi-Seed Validation of Key PEFT Comparison Cells

Validates the most striking claims in Table XXII with 3 seeds:
- VeRA collapse on SVHN (both backbones)
- SSF collapse on DINOv2 CIFAR-100
- DoRA ≈ LoRA (both backbones)

Usage:
    python revision_peft_multiseed.py --backbone dinov2
    python revision_peft_multiseed.py --backbone deit3
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
from revision_recent_peft import apply_dora, apply_vera, apply_ssf
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]

# Key cells to validate (most striking claims)
KEY_CELLS = {
    'dinov2': {
        'cifar100': ['LoRA_r8', 'DoRA_r8', 'SSF'],  # SSF collapse
        'svhn': ['LoRA_r8', 'VeRA_r8', 'VPT_p5'],   # VeRA collapse
    },
    'deit3': {
        'cifar100': ['LoRA_r8', 'DoRA_r8', 'SSF'],   # SSF wins
        'svhn': ['LoRA_r8', 'VeRA_r8', 'VPT_p5'],    # VeRA collapse
    },
}

METHOD_CONFIG = {
    'LoRA_r8': (lambda m, c: apply_lora(m, 8, c), 1e-3),
    'DoRA_r8': (lambda m, c: apply_dora(m, 8), 1e-3),
    'VeRA_r8': (lambda m, c: apply_vera(m, 8, c), 1e-3),
    'VPT_p5':  (lambda m, c: apply_vpt(m, 5, c), 1e-2),
    'SSF':     (lambda m, c: apply_ssf(m, c), 1e-2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 70)
    print(f"Multi-Seed PEFT Validation: {bb['name']}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    cells = KEY_CELLS.get(bb_key, {})
    all_results = {}

    for task, methods in cells.items():
        num_classes = TASKS[task][0]
        print(f"\n{'='*55}")
        print(f"  {bb['name']} × {task}")
        print(f"{'='*55}")

        task_results = {}

        for method_name in methods:
            apply_fn, lr = METHOD_CONFIG[method_name]
            seed_accs = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                ds = load_dataset(task, bb['img_size'], max_samples=1000)
                n_val = min(200, len(ds) // 5)
                train_ds, val_ds = torch.utils.data.random_split(
                    ds, [len(ds) - n_val, n_val],
                    generator=torch.Generator().manual_seed(seed))
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

                try:
                    model = deepcopy(base_model)
                    model.head = nn.Linear(config.embed_dim, num_classes).to(device)
                    model = apply_fn(model, config)
                    model = model.to(device)

                    config.lr = lr
                    acc = train_and_evaluate(model, train_loader, val_loader, config, device)
                    seed_accs.append(acc)
                except Exception as e:
                    print(f"      Seed {seed} ERROR: {str(e)[:60]}")
                    seed_accs.append(0.0)

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
    print(f"SUMMARY: {bb['name']} Multi-Seed PEFT (mean ± std)")
    print(f"{'='*70}")
    for task, results in all_results.items():
        print(f"\n  {task}:")
        for method, r in results.items():
            print(f"    {method:<10s}: {r['mean']:.3f} ± {r['std']:.3f}")

    del base_model; torch.cuda.empty_cache()

    os.makedirs('results', exist_ok=True)
    fname = f'results/peft_multiseed_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
