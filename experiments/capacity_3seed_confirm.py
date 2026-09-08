"""
3-seed capacity confirmation: DINOv2 CIFAR-100 p=50 crash.

Single seed shows -37.5 pts. Run 3 seeds at the best LR (1e-3) to confirm.

Usage:
    cd /content/ATLAS
    python experiments/capacity_3seed_confirm.py
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
from exp2_comparison import apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]
SAVE_PATH = 'results/capacity_3seed_confirm.json'


def main():
    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    base_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                                    pretrained=True, img_size=224).to(device)
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = config.embed_dim // config.num_heads
    config.num_classes = 100

    results = {}
    
    for p, lr in [(5, 2e-3), (50, 1e-3)]:
        key = f"p{p}"
        print(f"\n  --- DINOv2 x CIFAR-100, VPT p={p}, LR={lr:.0e} ---")
        accs = []
        
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            ds = load_dataset('cifar100', 224, max_samples=1000)
            nv = min(200, len(ds) // 5)
            tds, vds = random_split(ds, [len(ds)-nv, nv],
                generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
            vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

            m = deepcopy(base_model)
            m.head = nn.Linear(config.embed_dim, 100).to(device)
            m = apply_vpt(m, p, config)
            m = m.to(device)
            config.lr = lr
            acc = train_and_evaluate(m, tl, vl, config, device)
            accs.append(float(acc))
            print(f"    Seed {seed}: {acc:.3f}")
            del m; torch.cuda.empty_cache()
        
        results[key] = {
            'seeds': accs,
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'lr': float(lr),
        }
        print(f"    Mean: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    gap = results['p50']['mean'] - results['p5']['mean']
    print(f"\n  p=5: {results['p5']['mean']:.3f} ± {results['p5']['std']:.3f}")
    print(f"  p=50: {results['p50']['mean']:.3f} ± {results['p50']['std']:.3f}")
    print(f"  Δ: {gap*100:+.1f} pts")
    print(f"  {'CRASH CONFIRMED' if gap < -0.1 else 'Mild' if gap < -0.02 else 'Flat'}")

    os.makedirs('results', exist_ok=True)
    with open(SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
