"""
VPT Full LR Sweep (7 values, unconstrained)
Uses the same code pattern as revision3_priority.py which worked correctly.

Validates that σ²_P rule recovers the best LR without restricting the search.

Usage:
    python revision_vpt_full_sweep.py --backbones DINOv2 CLIP DeiT-III
    python revision_vpt_full_sweep.py --backbones DINOv2 --tasks cifar100
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

VPT_LRS_FULL = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/vpt_full_lr_sweep.json'

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 1.60),
    'MoCo-v3': (None, 2.31),
    'MAE': ('vit_base_patch16_224.mae', 1.76),
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
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for bb_name in args.backbones:
        if bb_name not in BACKBONES:
            print(f"  Unknown backbone: {bb_name}"); continue

        done_tasks = list(all_results.get(bb_name, {}).keys())
        pending = [t for t in args.tasks if t in TASKS and t not in done_tasks]
        if not pending:
            print(f"\n  {bb_name}: all done"); continue

        print(f"\n{'='*55}")
        print(f"  VPT Full LR Sweep: {bb_name}")
        print(f"  Pending: {pending}")
        print(f"{'='*55}")

        base_model = load_model(bb_name, device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task in pending:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            print(f"\n  --- {bb_name} x {task} ---")

            task_results = {}
            for lr in VPT_LRS_FULL:
                # Fresh data load each time (same pattern as working scripts)
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                # Fresh model copy each time
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config)
                m = m.to(device)
                config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                task_results[f'{lr}'] = float(acc)
                print(f"    VPT p=5 LR={lr:.0e}: {acc:.3f}")

            best_lr = max(VPT_LRS_FULL, key=lambda x: task_results[f'{x}'])
            task_results['best_lr'] = float(best_lr)
            task_results['best_acc'] = float(task_results[f'{best_lr}'])
            all_results[bb_name][task] = task_results

            # Save incrementally
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

            print(f"    Best: LR={best_lr:.0e} acc={task_results['best_acc']:.3f}")

        del base_model; torch.cuda.empty_cache()

    # Validate σ²_P rule
    print(f"\n{'='*55}")
    print("σ²_P RULE VALIDATION")
    print(f"{'='*55}")
    sigma_vals = {k: v[1] for k, v in BACKBONES.items()}
    correct = 0; total = 0
    for bb, tasks in all_results.items():
        sigma = sigma_vals.get(bb, None)
        if sigma is None: continue
        rule_lr = 1e-3 if sigma < 0.7 else 1e-2
        for task, data in tasks.items():
            best = data['best_lr']
            # Rule matches if best is in the right neighborhood
            match = (best <= 2e-3 and rule_lr == 1e-3) or (best >= 5e-3 and rule_lr == 1e-2)
            total += 1
            if match: correct += 1
            m = 'MATCH' if match else 'MISS'
            print(f"  {bb:<12s} {task:<10s}: σ²_P={sigma:.2f} rule={rule_lr:.0e} "
                  f"actual={best:.0e} {m}")
    if total > 0:
        print(f"\n  Rule accuracy: {correct}/{total} = {correct/total*100:.0f}%")


if __name__ == '__main__':
    main()
