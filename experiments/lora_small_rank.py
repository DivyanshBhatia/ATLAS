"""
LoRA r=1 and r=2 baseline: Does LoRA win at equal parameter count?

VPT p=5 uses ~46K params. LoRA r=1 uses ~37K, r=2 uses ~74K.
If LoRA still wins at matched params, the capacity explanation is dead.

Usage:
    cd /content/ATLAS
    python experiments/lora_small_rank.py
    python experiments/lora_small_rank.py --backbones DINOv2 MoCo-v3 --tasks cifar100 dtd
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
from exp2_comparison import apply_lora, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

LORA_RANKS = [1, 2]
LORA_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
TASKS_DEFAULT = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/lora_small_rank.json'

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'MoCo-v3': (None, 2.31),
}

# VPT p=5 best accuracy from Table 2 (for comparison)
VPT_P5_BEST = {
    'DINOv2': {'cifar100': 0.815, 'svhn': 0.848, 'gtsrb': 0.920, 'eurosat': 0.958, 'dtd': 0.788},
    'DeiT-III': {'cifar100': 0.652, 'svhn': 0.830, 'gtsrb': 0.925, 'eurosat': 0.958, 'dtd': 0.670},
    'MoCo-v3': {'cifar100': 0.738, 'svhn': 0.825, 'gtsrb': 0.948, 'eurosat': 0.968, 'dtd': 0.708},
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
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'MoCo-v3'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_DEFAULT)
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
        if bb_name not in BACKBONES: continue

        print(f"\n{'='*60}")
        print(f"  LoRA Small Rank: {bb_name}")
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

            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            for rank in LORA_RANKS:
                key = f"r{rank}_{task}"
                if key in all_results.get(bb_name, {}):
                    print(f"  {bb_name} r={rank} x {task}: already done"); continue

                n_params = 2 * config.num_layers * 2 * config.embed_dim * rank  # Q/V
                print(f"\n  --- {bb_name} x {task} (LoRA r={rank}, ~{n_params//1000}K params) ---")

                lr_results = {}
                for lr in LORA_LRS:
                    torch.manual_seed(42); np.random.seed(42)
                    ds = load_dataset(task, 224, max_samples=1000)
                    nv = min(200, len(ds) // 5)
                    tds, vds = random_split(ds, [len(ds)-nv, nv],
                        generator=torch.Generator().manual_seed(42))
                    tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                    vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                    m = deepcopy(base_model)
                    m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                    m = apply_lora(m, rank, config)
                    m = m.to(device)
                    config.lr = lr
                    acc = train_and_evaluate(m, tl, vl, config, device)
                    del m; torch.cuda.empty_cache()

                    lr_results[f'{lr}'] = float(acc)
                    print(f"    LR={lr:.0e}: {acc:.3f}")

                best_lr = max(LORA_LRS, key=lambda x: lr_results[f'{x}'])
                best_acc = lr_results[f'{best_lr}']
                vpt_acc = VPT_P5_BEST.get(bb_name, {}).get(task, None)

                all_results[bb_name][key] = {
                    'rank': rank,
                    'task': task,
                    'params': n_params,
                    'lr_sweep': lr_results,
                    'best_lr': float(best_lr),
                    'best_acc': float(best_acc),
                    'vpt_p5': vpt_acc,
                }

                vpt_str = f" (VPT p=5: {vpt_acc:.3f})" if vpt_acc else ""
                winner = ""
                if vpt_acc:
                    gap = best_acc - vpt_acc
                    winner = f" → {'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'} ({gap:+.1%})"
                print(f"    Best: LR={best_lr:.0e} acc={best_acc:.3f}{vpt_str}{winner}")

                os.makedirs('results', exist_ok=True)
                with open(SAVE_PATH, 'w') as f:
                    json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("EQUAL-PARAMETER COMPARISON")
    print(f"{'='*60}")
    print(f"  VPT p=5: ~46K params | LoRA r=1: ~37K | LoRA r=2: ~74K")
    for bb in sorted(all_results.keys()):
        print(f"\n  {bb}:")
        for task in TASKS_DEFAULT:
            parts = []
            for rank in LORA_RANKS:
                key = f"r{rank}_{task}"
                if key in all_results[bb]:
                    d = all_results[bb][key]
                    parts.append(f"r={rank}: {d['best_acc']:.3f}")
            vpt = VPT_P5_BEST.get(bb, {}).get(task, None)
            vpt_str = f"VPT={vpt:.3f}" if vpt else ""
            print(f"    {task:>8s}: {', '.join(parts)}  ({vpt_str})")


if __name__ == '__main__':
    main()
