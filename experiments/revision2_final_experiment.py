"""
Final Required Experiment Before TMLR Submission

The reviewer says: "This is the one experiment I would not submit without."

Three components:
1. LoRA LR sweep on MoCo-v3 and MAE (currently fixed at 1e-3, may be suboptimal)
2. Multi-seed (3 seeds) for MoCo-v3 key cells (the +11.5% VPT win is single-seed)
3. Multi-seed for MAE key cells (the 0.350/0.130 CIFAR-100 needs confirmation)

If MoCo-v3 LoRA improves with different LR and closes the +11.5% gap,
the sole VPT hit disappears and the paper's claims need adjustment.

Usage:
    python revision2_final_experiment.py
    python revision2_final_experiment.py --resume
    python revision2_final_experiment.py --models MoCo-v3 --tasks cifar100 dtd
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

SAVE_PATH = 'results/revision2_final.json'
SEEDS = [42, 123, 456]
LORA_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
VPT_LRS_LOW_SIGMA = [5e-4, 1e-3, 2e-3]   # for low sigma_P
VPT_LRS_HIGH_SIGMA = [5e-3, 1e-2, 2e-2]  # for high sigma_P
TEST_TASKS = ['cifar100', 'svhn', 'dtd', 'eurosat', 'gtsrb']


def get_vit_base():
    return timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)


MODELS = {
    'MoCo-v3': {
        'sigma_p': 2.31,
        'vpt_lrs': VPT_LRS_HIGH_SIGMA,
        'load': lambda: load_moco_v3(),
    },
    'MAE': {
        'sigma_p': 1.76,
        'vpt_lrs': VPT_LRS_HIGH_SIGMA,
        'load': lambda: timm.create_model('vit_base_patch16_224.mae', pretrained=True, img_size=224),
    },
    # Controls
    'DINOv2': {
        'sigma_p': 0.22,
        'vpt_lrs': VPT_LRS_LOW_SIGMA,
        'load': lambda: timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, img_size=224),
    },
    'CLIP': {
        'sigma_p': 0.18,
        'vpt_lrs': VPT_LRS_LOW_SIGMA,
        'load': lambda: timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, img_size=224),
    },
}


def load_moco_v3():
    """Load MoCo-v3 from Facebook's checkpoint."""
    model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
    url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        state_dict = {k.replace('module.', '').replace('base_encoder.', ''): v
                     for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"    MoCo-v3 loaded (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
    return model


def run_single(base_model, config, method, capacity, lr, train_loader, val_loader, device):
    model = deepcopy(base_model)
    model.head = nn.Linear(config.embed_dim, config.num_classes).to(device)
    if method == 'lora':
        model = apply_lora(model, capacity, config)
    elif method == 'vpt':
        model = apply_vpt(model, capacity, config)
    model = model.to(device)
    config.lr = lr
    acc = train_and_evaluate(model, train_loader, val_loader, config, device)
    del model; torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['MoCo-v3', 'MAE'])
    parser.add_argument('--tasks', nargs='+', default=TEST_TASKS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("FINAL EXPERIMENT: LoRA LR Sweep + Multi-Seed Validation")
    print(f"Models: {args.models}, Tasks: {args.tasks}")
    print("=" * 70)

    # Load existing
    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Resuming: {list(all_results.keys())}")

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"  Unknown: {model_key}")
            continue

        model_info = MODELS[model_key]
        done_tasks = list(all_results.get(model_key, {}).get('tasks', {}).keys())
        pending = [t for t in args.tasks if t in TASKS and t not in done_tasks]

        if not pending:
            print(f"\n  {model_key}: all done, skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  {model_key} (sigma_P={model_info['sigma_p']:.2f})")
        print(f"  Pending: {pending}")
        print(f"{'='*55}")

        try:
            base_model = model_info['load']().to(device)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if model_key not in all_results:
            all_results[model_key] = {'sigma_p': model_info['sigma_p'], 'tasks': {}}

        for task in pending:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            print(f"\n  --- {model_key} x {task} ---")

            task_results = {}

            # Part 1: LoRA LR sweep (single seed first for speed)
            print(f"  LoRA LR sweep (seed 42):")
            best_lora_lr = 1e-3
            best_lora_acc = 0
            for lr in LORA_LRS:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                n_val = min(200, len(ds) // 5)
                train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                    generator=torch.Generator().manual_seed(42))
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
                acc = run_single(base_model, config, 'lora', 8, lr,
                               train_loader, val_loader, device)
                task_results[f'lora_lr{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")
                if acc > best_lora_acc:
                    best_lora_acc = acc
                    best_lora_lr = lr

            # Part 2: VPT LR sweep (single seed)
            print(f"  VPT LR sweep (seed 42):")
            best_vpt_lr = model_info['vpt_lrs'][1]  # middle default
            best_vpt_acc = 0
            for lr in model_info['vpt_lrs']:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                n_val = min(200, len(ds) // 5)
                train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                    generator=torch.Generator().manual_seed(42))
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
                acc = run_single(base_model, config, 'vpt', 5, lr,
                               train_loader, val_loader, device)
                task_results[f'vpt_lr{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")
                if acc > best_vpt_acc:
                    best_vpt_acc = acc
                    best_vpt_lr = lr

            # Part 3: Multi-seed at best LRs
            print(f"\n  Multi-seed at best LRs (LoRA lr={best_lora_lr:.0e}, VPT lr={best_vpt_lr:.0e}):")
            lora_seeds = []
            vpt_seeds = []
            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed)
                ds = load_dataset(task, 224, max_samples=1000)
                n_val = min(200, len(ds) // 5)
                train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                    generator=torch.Generator().manual_seed(seed))
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

                l_acc = run_single(base_model, config, 'lora', 8, best_lora_lr,
                                  train_loader, val_loader, device)
                v_acc = run_single(base_model, config, 'vpt', 5, best_vpt_lr,
                                  train_loader, val_loader, device)
                lora_seeds.append(l_acc)
                vpt_seeds.append(v_acc)
                print(f"    Seed {seed}: LoRA={l_acc:.3f} VPT={v_acc:.3f}")

            lora_mean = np.mean(lora_seeds)
            lora_std = np.std(lora_seeds)
            vpt_mean = np.mean(vpt_seeds)
            vpt_std = np.std(vpt_seeds)
            gap = lora_mean - vpt_mean

            task_results['best_lora_lr'] = float(best_lora_lr)
            task_results['best_vpt_lr'] = float(best_vpt_lr)
            task_results['lora_3seed'] = {'mean': float(lora_mean), 'std': float(lora_std),
                                          'seeds': [float(x) for x in lora_seeds]}
            task_results['vpt_3seed'] = {'mean': float(vpt_mean), 'std': float(vpt_std),
                                         'seeds': [float(x) for x in vpt_seeds]}
            task_results['gap'] = float(gap)

            winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'
            print(f"\n  RESULT: LoRA={lora_mean:.3f}+/-{lora_std:.3f} "
                  f"VPT={vpt_mean:.3f}+/-{vpt_std:.3f} gap={gap:+.3f} -> {winner}")

            all_results[model_key]['tasks'][task] = task_results

            # Incremental save
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        del base_model; torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model+Task':<25s} {'LoRA best':>12s} {'VPT best':>12s} {'Gap':>8s} {'Winner'}")

    for model_key, mdata in all_results.items():
        for task, tdata in mdata.get('tasks', {}).items():
            l3 = tdata.get('lora_3seed', {})
            v3 = tdata.get('vpt_3seed', {})
            if l3 and v3:
                lm = l3['mean']; ls = l3['std']
                vm = v3['mean']; vs = v3['std']
                gap = lm - vm
                winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'
                print(f"  {model_key+' '+task:<25s} {lm:.3f}+/-{ls:.3f} {vm:.3f}+/-{vs:.3f} {gap:+.3f}   {winner}")

    # Key question
    print(f"\n  KEY: Does MoCo-v3 CIFAR-100 VPT win survive LoRA LR tuning + 3 seeds?")
    moco_c100 = all_results.get('MoCo-v3', {}).get('tasks', {}).get('cifar100', {})
    if moco_c100:
        l3 = moco_c100.get('lora_3seed', {})
        v3 = moco_c100.get('vpt_3seed', {})
        if l3 and v3:
            gap = l3['mean'] - v3['mean']
            if gap < -0.02:
                print(f"  YES: VPT still wins by {-gap:.1%} after LoRA LR tuning and 3 seeds")
            elif gap > 0.02:
                print(f"  NO: LoRA wins by {gap:.1%} after tuning. The +11.5% was a tuning artifact.")
                print(f"  -> Paper needs adjustment: remove MoCo-v3 as VPT example")
            else:
                print(f"  TIE: gap is {gap:+.3f}, within noise. MoCo-v3 is not a clear VPT win.")

    with open(SAVE_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
