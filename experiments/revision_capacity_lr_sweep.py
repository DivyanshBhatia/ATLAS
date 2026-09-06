"""
VPT Capacity LR Sweep: Does the p=50 crash survive LR tuning?

Reviewer blocking issue: "you declared 'more capacity hurts' without sweeping LR
at the new capacity. You need at least a 3-5 point LR sweep at p=20 and p=50."

If p=50 still crashes after LR tuning → finding is real (structural).
If p=50 recovers with different LR → finding was our own confound.

Usage:
    cd /content/ATLAS
    python experiments/revision_capacity_lr_sweep.py --backbones DINOv2 CLIP DeiT-III
    python experiments/revision_capacity_lr_sweep.py --backbones DINOv2 --tasks cifar100
    python experiments/revision_capacity_lr_sweep.py --backbones MoCo-v3 --tasks cifar100
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

VPT_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
PROMPT_COUNTS = [5, 20, 50]
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/capacity_lr_sweep.json'

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 1.60),
    'MAE': ('vit_base_patch16_224.mae', 1.76),
    'DINOv1': ('vit_base_patch16_224.dino', 0.19),
    'MoCo-v3': (None, 2.31),
}

# Best LoRA accuracy per backbone-task (from main table, for comparison)
LORA_BEST = {
    'DINOv2': {'cifar100': 0.835, 'svhn': 0.877, 'gtsrb': 0.917, 'eurosat': 0.967, 'dtd': 0.793},
    'CLIP': {'cifar100': 0.762, 'svhn': 0.902, 'gtsrb': 0.967, 'eurosat': 0.982, 'dtd': 0.738},
    'DeiT-III': {'cifar100': 0.715, 'svhn': 0.875, 'gtsrb': 0.965, 'eurosat': 0.965, 'dtd': 0.673},
    'MoCo-v3': {'cifar100': 0.695, 'svhn': 0.887, 'gtsrb': 0.963, 'eurosat': 0.972, 'dtd': 0.648},
}


def load_model(name, device):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name][0], pretrained=True, img_size=224).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'CLIP', 'DeiT-III'])
    parser.add_argument('--tasks', nargs='+', default=['cifar100'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for bb_name in args.backbones:
        if bb_name not in BACKBONES:
            print(f"  Unknown: {bb_name}"); continue

        print(f"\n{'='*60}")
        print(f"  Capacity LR Sweep: {bb_name}")
        print(f"{'='*60}")

        base_model = load_model(bb_name, device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task in args.tasks:
            if task not in TASKS: continue
            task_key = f"{task}"
            if task_key in all_results.get(bb_name, {}):
                print(f"\n  {bb_name} x {task}: already done"); continue

            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            print(f"\n  --- {bb_name} x {task} ---")

            task_results = {}

            for p in PROMPT_COUNTS:
                print(f"\n    p={p} ({p * config.num_layers * config.embed_dim // 1000}K params):")
                lr_results = {}

                for lr in VPT_LRS:
                    # Use seed 42 for single-seed LR sweep
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
                    print(f"      LR={lr:.0e}: {acc:.3f}")

                best_lr = max(VPT_LRS, key=lambda x: lr_results[f'{x}'])
                best_acc = lr_results[f'{best_lr}']
                task_results[f'p{p}'] = {
                    'lr_sweep': lr_results,
                    'best_lr': float(best_lr),
                    'best_acc': float(best_acc),
                    'params': int(p * config.num_layers * config.embed_dim),
                }
                print(f"      Best: LR={best_lr:.0e} acc={best_acc:.3f}")

            all_results[bb_name][task_key] = task_results

            # Save incrementally
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Summary for this backbone-task
            lora_acc = LORA_BEST.get(bb_name, {}).get(task, None)
            print(f"\n    === {bb_name} x {task} SUMMARY ===")
            for p in PROMPT_COUNTS:
                key = f'p{p}'
                if key in task_results:
                    d = task_results[key]
                    lora_str = f" (LoRA={lora_acc:.3f})" if lora_acc else ""
                    print(f"    p={p:>3d}: best={d['best_acc']:.3f} @ LR={d['best_lr']:.0e}{lora_str}")

            if lora_acc:
                p5_best = task_results['p5']['best_acc']
                p50_best = task_results['p50']['best_acc']
                print(f"\n    p=5 vs LoRA:  {p5_best:.3f} vs {lora_acc:.3f} (gap: {p5_best-lora_acc:+.3f})")
                print(f"    p=50 vs LoRA: {p50_best:.3f} vs {lora_acc:.3f} (gap: {p50_best-lora_acc:+.3f})")
                if p50_best > p5_best + 0.02:
                    print(f"    --> CAPACITY HELPS even after LR tuning (+{p50_best-p5_best:.3f})")
                elif p50_best < p5_best - 0.02:
                    print(f"    --> CAPACITY HURTS even after LR tuning ({p50_best-p5_best:+.3f})")
                else:
                    print(f"    --> FLAT after LR tuning ({p50_best-p5_best:+.3f})")

        del base_model; torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL: Does capacity crash survive LR tuning?")
    print(f"{'='*60}")
    for bb, tasks in all_results.items():
        print(f"\n  {bb}:")
        for task, data in tasks.items():
            parts = []
            for p in PROMPT_COUNTS:
                key = f'p{p}'
                if key in data:
                    parts.append(f"p={p}: {data[key]['best_acc']:.3f} @ {data[key]['best_lr']:.0e}")
            print(f"    {task}: {', '.join(parts)}")


if __name__ == '__main__':
    main()
