"""
Experiment #2: Extended Fair Comparison on More Tasks
Experiment #4: DeiT-III VPT LR Sweep

#2: We only have CIFAR-100 and SVHN. Need EuroSAT, DTD, GTSRB for broader coverage.
#4: DeiT-III VPT lost CIFAR-100 by 5.5% — might be wrong LR.

Usage:
    # Run both experiments:
    python revision2_extended.py

    # Just the extended fair comparison:
    python revision2_extended.py --mode fair

    # Just the DeiT-III LR sweep:
    python revision2_extended.py --mode sweep

    # Specific backbones/tasks:
    python revision2_extended.py --mode fair --backbones dinov2 deit3 --tasks eurosat dtd

    # Resume after crash:
    python revision2_extended.py --resume
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
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]

# Best LRs from sweep results
BEST_LRS = {
    'dinov2':     {'lora': 2e-3, 'vpt_p5': 1e-3, 'vpt_p1': 5e-3},
    'dinov2reg':  {'lora': 2e-3, 'vpt_p5': 1e-3, 'vpt_p1': 5e-3},
    'deit3':      {'lora': 1e-3, 'vpt_p5': 1e-2, 'vpt_p1': 1e-2},
    'supervised': {'lora': 1e-3, 'vpt_p5': 1e-2, 'vpt_p1': 1e-2},
    'clip':       {'lora': 1e-3, 'vpt_p5': 1e-2, 'vpt_p1': 1e-2},
}

# DeiT-III VPT LR sweep range
DEIT3_VPT_LRS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]


def run_method(base_model, config, method, capacity, lr, train_loader, val_loader, device):
    model = deepcopy(base_model)
    model.head = nn.Linear(config.embed_dim, config.num_classes).to(device)
    if method == 'vpt':
        model = apply_vpt(model, capacity, config)
    elif method == 'lora':
        model = apply_lora(model, capacity, config)
    model = model.to(device)
    config.lr = lr
    acc = train_and_evaluate(model, train_loader, val_loader, config, device)
    del model; torch.cuda.empty_cache()
    return acc


def run_fair_comparison(device, config, backbones, tasks, existing_results=None):
    """Experiment #2: Fair comparison on extended task set."""
    print("\n" + "=" * 70)
    print("EXPERIMENT #2: Extended Fair Comparison")
    print("=" * 70)

    SAVE_PATH = 'results/revision2_extended_fair.json'
    all_results = existing_results or {}

    for bb_key in backbones:
        if bb_key not in BACKBONES:
            continue
        bb = BACKBONES[bb_key]
        img_size = bb['img_size']
        lrs = BEST_LRS.get(bb_key, {'lora': 1e-3, 'vpt_p5': 1e-2, 'vpt_p1': 1e-2})

        # Check which tasks are done
        pending_tasks = []
        for task in tasks:
            key = f"{bb_key}_{task}"
            if key in all_results:
                print(f"  {key}: already done, skipping")
            elif task in TASKS:
                pending_tasks.append(task)
        
        if not pending_tasks:
            continue

        print(f"\n  Loading {bb['name']} (img_size={img_size})...")
        base_model = timm.create_model(bb['model'], pretrained=True,
                                        img_size=img_size).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        for task in pending_tasks:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            print(f"\n  {'='*50}")
            print(f"  {bb['name']} × {task}")
            print(f"  {'='*50}")

            methods = [
                ('LoRA_r8',  'lora', 8, lrs['lora']),
                ('VPT_p5',   'vpt',  5, lrs['vpt_p5']),
                ('VPT_p1',   'vpt',  1, lrs['vpt_p1']),
            ]

            task_results = {}
            for method_name, method, capacity, lr in methods:
                seed_accs = []
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    ds = load_dataset(task, img_size, max_samples=1000)
                    n_val = min(200, len(ds) // 5)
                    train_ds, val_ds = random_split(
                        ds, [len(ds) - n_val, n_val],
                        generator=torch.Generator().manual_seed(seed))
                    train_loader = DataLoader(train_ds, batch_size=64,
                                              shuffle=True, num_workers=2)
                    val_loader = DataLoader(val_ds, batch_size=64,
                                            shuffle=False, num_workers=2)
                    acc = run_method(base_model, config, method, capacity, lr,
                                     train_loader, val_loader, device)
                    seed_accs.append(acc)

                mean_acc = np.mean(seed_accs)
                std_acc = np.std(seed_accs)
                task_results[method_name] = {
                    'lr': lr, 'seeds': seed_accs,
                    'mean': float(mean_acc), 'std': float(std_acc)
                }
                print(f"    {method_name:<12s} (lr={lr:.0e}): "
                      f"{mean_acc:.3f} ± {std_acc:.3f}  "
                      f"({', '.join(f'{a:.3f}' for a in seed_accs)})")

            key = f"{bb_key}_{task}"
            all_results[key] = task_results

            lora = task_results['LoRA_r8']['mean']
            best_vpt = max(task_results['VPT_p5']['mean'],
                           task_results['VPT_p1']['mean'])
            best_vpt_name = 'VPT_p5' if task_results['VPT_p5']['mean'] >= task_results['VPT_p1']['mean'] else 'VPT_p1'
            gap = lora - best_vpt
            winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'
            print(f"\n    LoRA={lora:.3f} vs best VPT({best_vpt_name})={best_vpt:.3f} "
                  f"→ gap={gap:+.3f} → {winner}")

            # INCREMENTAL SAVE
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        del base_model; torch.cuda.empty_cache()

    return all_results


def run_deit3_vpt_sweep(device, config):
    """Experiment #4: DeiT-III VPT LR sweep on CIFAR-100."""
    print("\n" + "=" * 70)
    print("EXPERIMENT #4: DeiT-III VPT LR Sweep")
    print("=" * 70)

    bb = BACKBONES['deit3']
    img_size = bb['img_size']

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=img_size).to(device)
    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    sweep_results = {}

    for task in ['cifar100', 'svhn', 'dtd']:
        if task not in TASKS:
            continue
        num_classes = TASKS[task][0]
        config.num_classes = num_classes

        print(f"\n  {'='*50}")
        print(f"  DeiT-III × {task} — VPT LR sweep")
        print(f"  {'='*50}")

        task_results = {}

        # LoRA baseline (single LR, 3 seeds)
        seed_accs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(seed))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
            acc = run_method(base_model, config, 'lora', 8, 1e-3,
                             train_loader, val_loader, device)
            seed_accs.append(acc)
        task_results['LoRA_r8'] = {
            'lr': 1e-3, 'mean': float(np.mean(seed_accs)),
            'std': float(np.std(seed_accs)), 'seeds': seed_accs
        }
        print(f"    LoRA_r8 (lr=1e-3): {np.mean(seed_accs):.3f} ± {np.std(seed_accs):.3f}")

        # VPT_p5 sweep across LRs (single seed for speed)
        print(f"\n    VPT_p5 LR sweep:")
        for lr in DEIT3_VPT_LRS:
            torch.manual_seed(42); np.random.seed(42)
            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
            acc = run_method(base_model, config, 'vpt', 5, lr,
                             train_loader, val_loader, device)
            task_results[f'VPT_p5_lr{lr}'] = {'lr': lr, 'acc': float(acc)}
            print(f"      LR={lr:.0e}: {acc:.3f}")

        # VPT_p1 sweep
        print(f"\n    VPT_p1 LR sweep:")
        for lr in DEIT3_VPT_LRS:
            torch.manual_seed(42); np.random.seed(42)
            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
            acc = run_method(base_model, config, 'vpt', 1, lr,
                             train_loader, val_loader, device)
            task_results[f'VPT_p1_lr{lr}'] = {'lr': lr, 'acc': float(acc)}
            print(f"      LR={lr:.0e}: {acc:.3f}")

        # Summary
        best_vpt5 = max(
            [(v['acc'], v['lr']) for k, v in task_results.items() 
             if k.startswith('VPT_p5')],
            key=lambda x: x[0])
        best_vpt1 = max(
            [(v['acc'], v['lr']) for k, v in task_results.items() 
             if k.startswith('VPT_p1')],
            key=lambda x: x[0])
        lora_mean = task_results['LoRA_r8']['mean']

        print(f"\n    Summary:")
        print(f"      LoRA_r8:      {lora_mean:.3f}")
        print(f"      Best VPT_p5:  {best_vpt5[0]:.3f} (lr={best_vpt5[1]:.0e})")
        print(f"      Best VPT_p1:  {best_vpt1[0]:.3f} (lr={best_vpt1[1]:.0e})")
        best_vpt = max(best_vpt5[0], best_vpt1[0])
        print(f"      Gap: {lora_mean - best_vpt:+.3f}")

        sweep_results[task] = task_results

    del base_model; torch.cuda.empty_cache()
    return sweep_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='both', choices=['both', 'fair', 'sweep'])
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved results')
    parser.add_argument('--backbones', nargs='+', default=['dinov2', 'deit3'],
                        help='Backbones for fair comparison')
    parser.add_argument('--tasks', nargs='+', default=['eurosat', 'dtd', 'gtsrb'],
                        help='Tasks for fair comparison')
    parser.add_argument('--sweep_tasks', nargs='+', default=['cifar100', 'svhn', 'dtd'],
                        help='Tasks for DeiT-III sweep')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}

    if args.mode in ['both', 'fair']:
        # Load existing fair results if resuming
        existing_fair = {}
        if args.resume and os.path.exists('results/revision2_extended_fair.json'):
            with open('results/revision2_extended_fair.json') as f:
                existing_fair = json.load(f)
            print(f"  Resuming fair comparison: {len(existing_fair)} pairs done")

        fair_results = run_fair_comparison(
            device, config,
            backbones=args.backbones,
            tasks=args.tasks,
            existing_results=existing_fair
        )
        all_results['fair_comparison'] = fair_results

    if args.mode in ['both', 'sweep']:
        sweep_results = run_deit3_vpt_sweep(device, config)
        all_results['deit3_sweep'] = sweep_results

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    if 'fair_comparison' in all_results:
        print("\n  Extended Fair Comparison:")
        for key, results in all_results['fair_comparison'].items():
            lora = results['LoRA_r8']['mean']
            best_vpt = max(results['VPT_p5']['mean'], results['VPT_p1']['mean'])
            gap = lora - best_vpt
            winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'
            print(f"    {key:<25s}: LoRA={lora:.3f} VPT={best_vpt:.3f} "
                  f"gap={gap:+.3f} → {winner}")

    if 'deit3_sweep' in all_results:
        print("\n  DeiT-III VPT LR Sweep:")
        for task, results in all_results['deit3_sweep'].items():
            lora = results['LoRA_r8']['mean']
            vpt5_accs = [(v['acc'], v['lr']) for k, v in results.items() 
                         if k.startswith('VPT_p5')]
            if vpt5_accs:
                best = max(vpt5_accs, key=lambda x: x[0])
                print(f"    {task}: LoRA={lora:.3f}, best VPT_p5={best[0]:.3f} "
                      f"(lr={best[1]:.0e}), gap={lora-best[0]:+.3f}")

    os.makedirs('results', exist_ok=True)
    with open('results/revision2_extended.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_extended.json")


if __name__ == '__main__':
    main()
