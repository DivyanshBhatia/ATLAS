"""
VPT Capacity Sweep: p = {5, 20, 50}

Addresses reviewer concern: "13x parameter gap, VPT never gets to vary capacity"
Uses same code pattern as revision3_priority.py (which works).

Usage:
    python revision_capacity_sweep.py --backbones DINOv2 CLIP DeiT-III
    python revision_capacity_sweep.py --backbones DINOv2 --tasks cifar100
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
from exp2_comparison import apply_vpt, apply_lora, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]
PROMPT_COUNTS = [5, 20, 50]
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/vpt_capacity_sweep.json'

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 1.60),
    'MAE': ('vit_base_patch16_224.mae', 1.76),
    'DINOv1': ('vit_base_patch16_224.dino', 0.19),
    'MoCo-v3': (None, 2.31),
    'iBOT': (None, 0.15),
}

# Best VPT LR per backbone (from our sweeps)
BEST_VPT_LR = {
    'DINOv2': 1e-3, 'CLIP': 2e-3, 'DeiT-III': 5e-3,
    'Supervised': 5e-3, 'MAE': 1e-2, 'DINOv1': 1e-3,
    'MoCo-v3': 1e-2, 'iBOT': 1e-3,
}


def load_model(name, device, ibot_checkpoint=None):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        msg = model.load_state_dict(sd, strict=False)
        print(f"  MoCo-v3 loaded (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
        return model.to(device)
    elif name == 'iBOT':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        # Find checkpoint
        candidates = [ibot_checkpoint] if ibot_checkpoint else []
        candidates.extend([
            '/content/ibot/checkpoint_teacher.pth',
            '/content/checkpoint_teacher.pth',
            'checkpoint_teacher.pth',
        ])
        ckpt_path = next((x for x in candidates if x and os.path.isfile(x)), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"iBOT checkpoint not found. Tried: {candidates}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'teacher' in checkpoint:
            sd = checkpoint['teacher']
        else:
            sd = checkpoint
        cleaned = {}
        for key, value in sd.items():
            new_key = key
            for prefix in ('module.', 'teacher.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if new_key.startswith(('head.', 'last_layer.', 'student_head.', 'teacher_head.')):
                continue
            cleaned[new_key] = value
        msg = model.load_state_dict(cleaned, strict=False)
        print(f"  iBOT loaded from {ckpt_path} (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name][0], pretrained=True, img_size=224).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'CLIP', 'DeiT-III'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
    parser.add_argument('--ibot_checkpoint', default=None,
                        help='Path to iBOT checkpoint_teacher.pth')
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

        done_tasks = list(all_results.get(bb_name, {}).keys())
        pending = [t for t in args.tasks if t in TASKS and t not in done_tasks]
        if not pending:
            print(f"\n  {bb_name}: all done"); continue

        print(f"\n{'='*55}")
        print(f"  VPT Capacity Sweep: {bb_name}")
        print(f"  Pending: {pending}")
        print(f"{'='*55}")

        base_model = load_model(bb_name, device, ibot_checkpoint=args.ibot_checkpoint)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        vpt_lr = BEST_VPT_LR.get(bb_name, 1e-3)

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task in pending:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            print(f"\n  --- {bb_name} x {task} (VPT LR={vpt_lr:.0e}) ---")

            task_results = {}

            for p in PROMPT_COUNTS:
                accs = []
                for seed in SEEDS:
                    torch.manual_seed(seed); np.random.seed(seed)
                    ds = load_dataset(task, 224, max_samples=1000)
                    nv = min(200, len(ds) // 5)
                    tds, vds = random_split(ds, [len(ds)-nv, nv],
                        generator=torch.Generator().manual_seed(seed))
                    tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                    vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                    m = deepcopy(base_model)
                    m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                    m = apply_vpt(m, p, config)
                    m = m.to(device)
                    config.lr = vpt_lr
                    acc = train_and_evaluate(m, tl, vl, config, device)
                    del m; torch.cuda.empty_cache()
                    accs.append(float(acc))

                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                params = p * config.num_layers * config.embed_dim
                task_results[f'p{p}'] = {
                    'mean': float(mean_acc),
                    'std': float(std_acc),
                    'seeds': accs,
                    'params': int(params),
                }
                print(f"    p={p:>3d} ({params/1000:.0f}K params): "
                      f"{mean_acc:.3f} ± {std_acc:.3f}")

            all_results[bb_name][task] = task_results

            # Save incrementally
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*55}")
    print("CAPACITY SWEEP SUMMARY")
    print(f"{'='*55}")
    print(f"  Does VPT at higher p close the gap with LoRA?")
    print(f"  LoRA r=8 uses ~590K params; VPT p=50 uses ~460K params (comparable)\n")

    for bb, tasks in all_results.items():
        print(f"  {bb}:")
        for task, data in tasks.items():
            parts = []
            for p in PROMPT_COUNTS:
                key = f'p{p}'
                if key in data:
                    parts.append(f"p={p}: {data[key]['mean']:.3f}")
            print(f"    {task:<10s}: {', '.join(parts)}")


if __name__ == '__main__':
    main()
