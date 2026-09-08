"""
Weight Decay Sweep for VPT: Are the 38/40 results robust to WD tuning?

Reviewer: "VPT is known to be sensitive to weight decay and prompt init scale.
A paper titled 'Not a Fair Fight' is exposed to its own critique if VPT's
hyperparameter space is under-explored."

Tests VPT at best LR with multiple weight decay values.
If VPT improves significantly with WD tuning, the 38/40 count may shift.

Usage:
    cd /content/ATLAS
    python experiments/weight_decay_sweep.py
    python experiments/weight_decay_sweep.py --backbones DINOv2 MoCo-v3 DeiT-III
    python experiments/weight_decay_sweep.py --backbones MoCo-v3 --tasks cifar100 dtd
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

WD_VALUES = [0, 1e-5, 1e-4, 1e-3, 1e-2]
TASKS_DEFAULT = ['cifar100', 'svhn', 'gtsrb']
SAVE_PATH = 'results/weight_decay_sweep.json'

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'MoCo-v3': (None, 2.31),
}

# Best VPT LR per backbone (from σ²_P-guided search)
BEST_VPT_LR = {
    'DINOv2': {'cifar100': 2e-3, 'svhn': 2e-3, 'gtsrb': 1e-3},
    'DeiT-III': {'cifar100': 1e-3, 'svhn': 1e-3, 'gtsrb': 5e-3},
    'MoCo-v3': {'cifar100': 1e-2, 'svhn': 5e-3, 'gtsrb': 5e-3},
}

# LoRA best accuracy for comparison
LORA_BEST = {
    'DINOv2': {'cifar100': 0.835, 'svhn': 0.877, 'gtsrb': 0.917},
    'DeiT-III': {'cifar100': 0.715, 'svhn': 0.898, 'gtsrb': 0.920},
    'MoCo-v3': {'cifar100': 0.695, 'svhn': 0.887, 'gtsrb': 0.963},
}

# VPT at default WD (1e-4) for comparison
VPT_DEFAULT = {
    'DINOv2': {'cifar100': 0.815, 'svhn': 0.848, 'gtsrb': 0.920},
    'DeiT-III': {'cifar100': 0.652, 'svhn': 0.830, 'gtsrb': 0.925},
    'MoCo-v3': {'cifar100': 0.738, 'svhn': 0.825, 'gtsrb': 0.948},
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


def train_with_wd(model, train_loader, val_loader, config, device, weight_decay):
    """Train with specific weight decay."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / total
        best_acc = max(best_acc, acc)

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'MoCo-v3', 'DeiT-III'])
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
        print(f"  Weight Decay Sweep: {bb_name}")
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
            if task in all_results.get(bb_name, {}):
                print(f"  {bb_name} x {task}: already done"); continue

            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            config.lr = BEST_VPT_LR[bb_name][task]

            print(f"\n  --- {bb_name} x {task} (VPT p=5, LR={config.lr:.0e}) ---")

            wd_results = {}
            for wd in WD_VALUES:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config)
                m = m.to(device)

                acc = train_with_wd(m, tl, vl, config, device, wd)
                del m; torch.cuda.empty_cache()

                wd_results[f'{wd}'] = float(acc)
                print(f"    WD={wd:.0e}: {acc:.3f}")

            best_wd = max(WD_VALUES, key=lambda x: wd_results[f'{x}'])
            best_acc = wd_results[f'{best_wd}']
            default_acc = VPT_DEFAULT.get(bb_name, {}).get(task, None)
            lora_acc = LORA_BEST.get(bb_name, {}).get(task, None)

            all_results[bb_name][task] = {
                'wd_sweep': wd_results,
                'best_wd': float(best_wd),
                'best_acc': float(best_acc),
                'default_wd_acc': default_acc,
                'lora_acc': lora_acc,
                'improvement': float(best_acc - default_acc) if default_acc else None,
            }

            print(f"    Best: WD={best_wd:.0e} acc={best_acc:.3f}")
            if default_acc:
                print(f"    vs default WD=1e-4: {default_acc:.3f} (gain: {best_acc-default_acc:+.3f})")
            if lora_acc:
                print(f"    vs LoRA: {lora_acc:.3f} (gap: {best_acc-lora_acc:+.3f})")
                still_loses = best_acc < lora_acc - 0.02
                print(f"    {'LoRA still wins' if still_loses else 'VPT catches up!' if best_acc > lora_acc - 0.02 else 'Tie'}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("WEIGHT DECAY SWEEP SUMMARY")
    print(f"{'='*60}")
    flips = 0
    for bb, tasks in all_results.items():
        print(f"\n  {bb}:")
        for task, data in tasks.items():
            imp = data.get('improvement', 0) or 0
            lora = data.get('lora_acc', 0) or 0
            best = data['best_acc']
            still_loses = best < lora - 0.02
            status = "LoRA still wins" if still_loses else "VPT CATCHES UP"
            if not still_loses and imp > 0.02:
                flips += 1
            print(f"    {task}: default={data.get('default_wd_acc', '?'):.3f} → best={best:.3f} "
                  f"(WD={data['best_wd']:.0e}, +{imp:.3f}) | {status}")

    print(f"\n  Cells where WD tuning flips the winner: {flips}")
    if flips == 0:
        print("  Results are robust to weight decay — 38/40 holds.")


if __name__ == '__main__':
    main()
