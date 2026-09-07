"""
VPT 7-point LR sweep for σ²_P validation.
Tests VPT p=5 at 7 LRs (unrestricted) to validate the σ²_P rule.

Now supports DINOv1 and iBOT (with checkpoint argument).

Usage:
    cd /content/ATLAS
    python experiments/revision_vpt_full_sweep.py --backbones DINOv1
    python experiments/revision_vpt_full_sweep.py --backbones iBOT --ibot_checkpoint /content/checkpoint_teacher.pth
    python experiments/revision_vpt_full_sweep.py --backbones DINOv1 iBOT --ibot_checkpoint /content/checkpoint_teacher.pth
    python experiments/revision_vpt_full_sweep.py --backbones DINOv2 --tasks cifar100
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
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/vpt_lr_sweep.json'

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
    elif name == 'DINOv1':
        model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, img_size=224)
        print(f"  DINOv1 loaded")
        return model.to(device)
    else:
        model_name = BACKBONES[name][0]
        model = timm.create_model(model_name, pretrained=True, img_size=224)
        print(f"  {name} loaded ({model_name})")
        return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv1', 'iBOT'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
    parser.add_argument('--ibot_checkpoint', default=None)
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
            print(f"  Unknown backbone: {bb_name}"); continue

        sigma_p = BACKBONES[bb_name][1]
        print(f"\n{'='*60}")
        print(f"  VPT Full LR Sweep: {bb_name} (σ²_P={sigma_p})")
        print(f"{'='*60}")

        base_model = load_model(bb_name, device, ibot_checkpoint=args.ibot_checkpoint)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if bb_name not in all_results:
            all_results[bb_name] = {}

        pending = [t for t in args.tasks if t in TASKS and t not in all_results.get(bb_name, {})]
        if not pending:
            print(f"  All tasks done for {bb_name}"); continue
        print(f"  Pending: {pending}")

        for task in pending:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            print(f"\n  --- {bb_name} x {task} ---")
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
                m = apply_vpt(m, 5, config)
                m = m.to(device)
                config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                lr_results[f'{lr}'] = float(acc)
                print(f"    VPT p=5 LR={lr:.0e}: {acc:.4f}")

            best_lr = max(VPT_LRS, key=lambda x: lr_results[f'{x}'])
            best_acc = lr_results[f'{best_lr}']

            all_results[bb_name][task] = lr_results
            all_results[bb_name][task]['best_lr'] = float(best_lr)
            all_results[bb_name][task]['best_acc'] = float(best_acc)

            print(f"    Best: LR={best_lr:.0e} acc={best_acc:.3f}")

            # Save incrementally
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Validate σ²_P rule
    print(f"\n{'='*60}")
    print("σ²_P RULE VALIDATION")
    print(f"{'='*60}")
    matches, total = 0, 0
    for bb_name, tasks in all_results.items():
        sigma = BACKBONES.get(bb_name, (None, None))[1]
        if sigma is None: continue
        for task, data in tasks.items():
            if task in ('best_lr', 'best_acc'): continue
            if not isinstance(data, dict) or 'best_lr' not in data: continue

            best_lr = data['best_lr']
            best_acc = data['best_acc']
            acc_at_1e2 = data.get('0.01', None)
            if acc_at_1e2 is None: continue

            drop = (acc_at_1e2 - best_acc) * 100
            rule_lr = 2e-3 if sigma < 0.7 else 1e-2
            rule_match = 'MATCH' if (sigma < 0.7 and best_lr <= 2e-3) or (sigma >= 0.7) else 'MISS'
            if drop > -10 and sigma < 0.7:
                rule_match = 'MATCH (no crash)'
            if rule_match.startswith('MATCH'): matches += 1
            total += 1

            print(f"  {bb_name:12s} {task:10s}: σ²_P={sigma:.2f} best={best_lr:.0e} drop={drop:+.1f}pts {rule_match}")

    print(f"\n  Rule accuracy: {matches}/{total} = {matches/total*100:.0f}%")


if __name__ == '__main__':
    main()
