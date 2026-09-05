"""
LoRA LR Sweep for DINOv1 and iBOT

Completes the full-sweep protocol for the two remaining single-seed backbones.
Uses the same 5-point LoRA grid and 3-point VPT grid as revision3_priority.py.

DINOv1: loads from timm (vit_base_patch16_224.dino)
iBOT: loads from official checkpoint via IBOT_ROOT

Usage:
    # DINOv1 only:
    python revision3_dinov1_ibot_sweep.py --backbones DINOv1

    # iBOT only:
    python revision3_dinov1_ibot_sweep.py --backbones iBOT --ibot_checkpoint /path/to/checkpoint_teacher.pth

    # Both:
    python revision3_dinov1_ibot_sweep.py --backbones DINOv1 iBOT --ibot_checkpoint /path/to/checkpoint_teacher.pth

    # Specific tasks:
    python revision3_dinov1_ibot_sweep.py --backbones DINOv1 --tasks cifar100 svhn

    # Resume:
    python revision3_dinov1_ibot_sweep.py --backbones DINOv1 --resume
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
LORA_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
VPT_LRS_LOW = [5e-4, 1e-3, 2e-3]
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']
SAVE_PATH = 'results/revision3_dinov1_ibot_sweep.json'


def load_dinov1(device):
    """Load DINOv1 from timm."""
    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, img_size=224)
    return model.to(device), 'vit_base_patch16_224.dino'


def load_ibot(device, checkpoint_path, ibot_root='/content/ibot'):
    """Load iBOT from official checkpoint."""
    if ibot_root not in sys.path:
        sys.path.insert(0, ibot_root)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint: {type(checkpoint)}")

    # Clean keys
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ('module.', 'teacher.', 'backbone.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if new_key.startswith(('head.', 'last_layer.', 'student_head.', 'teacher_head.')):
            continue
        cleaned[new_key] = value

    # Build model
    try:
        import models as ibot_models
        model = ibot_models.__dict__['vit_base'](patch_size=16, num_classes=0)
    except ImportError:
        # Fallback: use timm ViT-B and load weights
        print("  iBOT models not found, using timm ViT-B as architecture")
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)

    msg = model.load_state_dict(cleaned, strict=False)
    print(f"  iBOT loaded (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
    return model.to(device), 'ibot_vit_base_patch16'


def run_sweep(base_model, bb_name, device, config, tasks, all_results):
    """Run full LoRA + VPT sweep on one backbone."""
    sigma_p_val = compute_sigma(base_model)
    vpt_lrs = VPT_LRS_LOW  # both DINOv1 and iBOT have low sigma

    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = config.embed_dim // config.num_heads

    if bb_name not in all_results:
        all_results[bb_name] = {'sigma_p': float(sigma_p_val), 'tasks': {}}

    done_tasks = list(all_results[bb_name].get('tasks', {}).keys())
    pending = [t for t in tasks if t in TASKS and t not in done_tasks]

    if not pending:
        print(f"\n  {bb_name}: all done")
        return

    print(f"\n{'='*55}")
    print(f"  {bb_name} (sigma_P={sigma_p_val:.4f})")
    print(f"  Pending: {pending}")
    print(f"{'='*55}")

    for task in pending:
        num_classes = TASKS[task][0]
        config.num_classes = num_classes
        print(f"\n  --- {bb_name} x {task} ---")

        task_results = {}

        # LoRA sweep (seed 42)
        print(f"  LoRA LR sweep:")
        best_lora_lr, best_lora_acc = 1e-3, 0
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
            m = apply_lora(m, 8, config); m = m.to(device); config.lr = lr
            acc = train_and_evaluate(m, tl, vl, config, device)
            del m; torch.cuda.empty_cache()
            task_results[f'lora_lr{lr}'] = float(acc)
            print(f"    LR={lr:.0e}: {acc:.3f}")
            if acc > best_lora_acc:
                best_lora_acc = acc; best_lora_lr = lr

        # VPT sweep (seed 42)
        print(f"  VPT LR sweep:")
        best_vpt_lr, best_vpt_acc = vpt_lrs[1], 0
        for lr in vpt_lrs:
            torch.manual_seed(42); np.random.seed(42)
            ds = load_dataset(task, 224, max_samples=1000)
            nv = min(200, len(ds) // 5)
            tds, vds = random_split(ds, [len(ds)-nv, nv],
                generator=torch.Generator().manual_seed(42))
            tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
            vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)
            m = deepcopy(base_model)
            m.head = nn.Linear(config.embed_dim, num_classes).to(device)
            m = apply_vpt(m, 5, config); m = m.to(device); config.lr = lr
            acc = train_and_evaluate(m, tl, vl, config, device)
            del m; torch.cuda.empty_cache()
            task_results[f'vpt_lr{lr}'] = float(acc)
            print(f"    LR={lr:.0e}: {acc:.3f}")
            if acc > best_vpt_acc:
                best_vpt_acc = acc; best_vpt_lr = lr

        # 3-seed at best LRs
        print(f"\n  3-seed (LoRA lr={best_lora_lr:.0e}, VPT lr={best_vpt_lr:.0e}):")
        lora_seeds, vpt_seeds = [], []
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
            m = apply_lora(m, 8, config); m = m.to(device); config.lr = best_lora_lr
            la = train_and_evaluate(m, tl, vl, config, device)
            del m; torch.cuda.empty_cache()

            m = deepcopy(base_model)
            m.head = nn.Linear(config.embed_dim, num_classes).to(device)
            m = apply_vpt(m, 5, config); m = m.to(device); config.lr = best_vpt_lr
            va = train_and_evaluate(m, tl, vl, config, device)
            del m; torch.cuda.empty_cache()

            lora_seeds.append(la); vpt_seeds.append(va)
            print(f"    Seed {seed}: LoRA={la:.3f} VPT={va:.3f}")

        lm, ls = np.mean(lora_seeds), np.std(lora_seeds)
        vm, vs = np.mean(vpt_seeds), np.std(vpt_seeds)
        gap = lm - vm
        winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'

        task_results['best_lora_lr'] = float(best_lora_lr)
        task_results['best_vpt_lr'] = float(best_vpt_lr)
        task_results['lora_3seed'] = {'mean': float(lm), 'std': float(ls),
                                       'seeds': [float(x) for x in lora_seeds]}
        task_results['vpt_3seed'] = {'mean': float(vm), 'std': float(vs),
                                      'seeds': [float(x) for x in vpt_seeds]}
        task_results['gap'] = float(gap)

        print(f"  RESULT: LoRA={lm:.3f}+/-{ls:.3f} VPT={vm:.3f}+/-{vs:.3f} "
              f"gap={gap:+.3f} -> {winner}")

        all_results[bb_name]['tasks'][task] = task_results

        # Incremental save
        os.makedirs('results', exist_ok=True)
        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def compute_sigma(model):
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_qkv = block.attn.qkv.weight.float()
        W_q, W_k, W_v = W_qkv[:d], W_qkv[d:2*d], W_qkv[2*d:]
        W_o = block.attn.proj.weight.float()
        g_attn = W_q.norm().item() * W_k.norm().item()
        g_val = W_v.norm().item() * W_o.norm().item()
        per_layer.append((g_attn + g_val) / (2 * d))
    return np.mean(per_layer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv1'],
                        choices=['DINOv1', 'iBOT'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
    parser.add_argument('--ibot_checkpoint', default=None,
                        help='Path to iBOT checkpoint_teacher.pth')
    parser.add_argument('--ibot_root', default='/content/ibot')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 60)
    print(f"LoRA + VPT SWEEP: {args.backbones}")
    print("=" * 60)

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Resuming: {list(all_results.keys())}")

    for bb in args.backbones:
        if bb == 'DINOv1':
            model, source = load_dinov1(device)
            print(f"  Loaded DINOv1: {source}")
            run_sweep(model, 'DINOv1', device, config, args.tasks, all_results)
            del model; torch.cuda.empty_cache()

        elif bb == 'iBOT':
            # Find checkpoint
            candidates = []
            if args.ibot_checkpoint:
                candidates.append(args.ibot_checkpoint)
            candidates.extend([
                os.path.join(args.ibot_root, 'checkpoint_teacher.pth'),
                '/content/checkpoint_teacher.pth',
                'checkpoint_teacher.pth',
            ])
            ckpt = next((x for x in candidates if os.path.isfile(x)), None)
            if ckpt is None:
                print(f"  iBOT checkpoint not found. Tried: {candidates}")
                print(f"  Use --ibot_checkpoint /path/to/checkpoint_teacher.pth")
                continue

            model, source = load_ibot(device, ckpt, args.ibot_root)
            print(f"  Loaded iBOT: {source}")
            run_sweep(model, 'iBOT', device, config, args.tasks, all_results)
            del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for bb, data in all_results.items():
        print(f"\n  {bb} (sigma_P={data['sigma_p']:.4f}):")
        for task, tdata in data.get('tasks', {}).items():
            l3 = tdata.get('lora_3seed', {})
            v3 = tdata.get('vpt_3seed', {})
            if l3 and v3:
                gap = l3['mean'] - v3['mean']
                w = 'L' if gap > 0.02 else 'V' if gap < -0.02 else 'T'
                print(f"    {task}: LoRA={l3['mean']:.3f} VPT={v3['mean']:.3f} "
                      f"gap={gap:+.3f} {w} "
                      f"(LoRA lr={tdata['best_lora_lr']:.0e}, VPT lr={tdata['best_vpt_lr']:.0e})")


if __name__ == '__main__':
    main()
