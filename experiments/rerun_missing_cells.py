"""
Rerun 11 missing Table 2 cells + compute PAIRED gap std for ALL 40 cells.

Missing cells: CLIP C-100/SVHN, DeiT-III SVHN/GTSRB/Euro/DTD, Supervised x5
Also reruns cells we have to get per-seed gaps (paired std).

Outputs: results/table2_paired_std.json with per-seed L, V, gap for all cells.

Usage:
    cd /content/ATLAS
    python experiments/rerun_missing_cells.py
    python experiments/rerun_missing_cells.py --backbones CLIP --tasks cifar100 svhn
    python experiments/rerun_missing_cells.py --resume
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
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]
SAVE_PATH = 'results/table2_paired_std.json'
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']

# Best LRs from our sweeps (already determined)
BEST_LRS = {
    'CLIP': {
        'cifar100': {'lora': 2e-3, 'vpt': 2e-3},
        'svhn':     {'lora': 2e-3, 'vpt': 2e-3},
        'gtsrb':    {'lora': 2e-3, 'vpt': 1e-3},
        'eurosat':  {'lora': 2e-3, 'vpt': 1e-3},
        'dtd':      {'lora': 2e-3, 'vpt': 2e-3},
    },
    'DeiT-III': {
        'cifar100': {'lora': 5e-3, 'vpt': 1e-3},
        'svhn':     {'lora': 5e-3, 'vpt': 1e-3},
        'gtsrb':    {'lora': 2e-3, 'vpt': 5e-3},
        'eurosat':  {'lora': 2e-3, 'vpt': 5e-3},
        'dtd':      {'lora': 2e-3, 'vpt': 5e-3},
    },
    'Supervised': {
        'cifar100': {'lora': 2e-3, 'vpt': 5e-3},
        'svhn':     {'lora': 5e-3, 'vpt': 5e-3},
        'gtsrb':    {'lora': 2e-3, 'vpt': 5e-3},
        'eurosat':  {'lora': 2e-3, 'vpt': 5e-3},
        'dtd':      {'lora': 2e-3, 'vpt': 2e-3},
    },
}

BACKBONES = {
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 1.60),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['CLIP', 'DeiT-III', 'Supervised'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
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

        model_name, sigma_p = BACKBONES[bb_name]
        print(f"\n{'='*60}")
        print(f"  {bb_name} ({model_name})")
        print(f"{'='*60}")

        base_model = timm.create_model(model_name, pretrained=True, img_size=224).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task in args.tasks:
            if task not in TASKS: continue
            if task in all_results.get(bb_name, {}):
                print(f"  {bb_name} x {task}: already done"); continue

            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            lrs = BEST_LRS[bb_name][task]

            print(f"\n  --- {bb_name} x {task} (LoRA lr={lrs['lora']:.0e}, VPT lr={lrs['vpt']:.0e}) ---")

            lora_seeds, vpt_seeds, gap_seeds = [], [], []

            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(seed))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                # LoRA
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_lora(m, 8, config); m = m.to(device)
                config.lr = lrs['lora']
                la = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                # VPT
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config); m = m.to(device)
                config.lr = lrs['vpt']
                va = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                gap = la - va
                lora_seeds.append(float(la))
                vpt_seeds.append(float(va))
                gap_seeds.append(float(gap))
                print(f"    Seed {seed}: LoRA={la:.3f} VPT={va:.3f} gap={gap:+.3f}")

            lm, ls = np.mean(lora_seeds), np.std(lora_seeds)
            vm, vs = np.mean(vpt_seeds), np.std(vpt_seeds)
            gm, gs = np.mean(gap_seeds), np.std(gap_seeds)
            winner = 'LoRA' if gm > 0.02 else 'VPT' if gm < -0.02 else 'TIE'

            all_results[bb_name][task] = {
                'lora_seeds': lora_seeds,
                'vpt_seeds': vpt_seeds,
                'gap_seeds': gap_seeds,
                'lora_mean': float(lm), 'lora_std': float(ls),
                'vpt_mean': float(vm), 'vpt_std': float(vs),
                'gap_mean': float(gm), 'gap_std_paired': float(gs),
                'gap_std_unpaired': float(np.sqrt(ls**2 + vs**2)),
                'winner': winner,
            }

            print(f"  RESULT: LoRA={lm:.3f}±{ls:.3f} VPT={vm:.3f}±{vs:.3f}")
            print(f"          gap={gm:+.3f} paired_std={gs:.3f} unpaired_std={np.sqrt(ls**2+vs**2):.3f} -> {winner}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("TABLE 2 DATA (with paired gap std)")
    print(f"{'='*60}")
    for bb in sorted(all_results.keys()):
        print(f"\n  {bb}:")
        for task in TASKS_5:
            if task in all_results[bb]:
                d = all_results[bb][task]
                print(f"    {task:>8s}: gap={d['gap_mean']:+.1%} ± {d['gap_std_paired']:.1%} (paired) "
                      f"[{d['gap_std_unpaired']:.1%} unpaired] -> {d['winner']}")


if __name__ == '__main__':
    main()
