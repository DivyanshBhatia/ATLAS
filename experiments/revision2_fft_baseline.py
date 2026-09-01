"""
FFT (Full Fine-Tuning) Baseline — R1 flagged missing

Compares LoRA and VPT against full fine-tuning to show:
1. How close PEFT methods get to FFT
2. Whether FFT overfits at n=800 (it should)

Usage:
    python revision2_fft_baseline.py
    python revision2_fft_baseline.py --backbones dinov2 deit3 --tasks cifar100 svhn eurosat
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

SAVE_PATH = 'results/revision2_fft_baseline.json'

# Best LRs from sweep
BEST_LRS = {
    'dinov2':     {'lora': 2e-3, 'vpt_p5': 1e-3, 'fft': 1e-5},
    'deit3':      {'lora': 1e-3, 'vpt_p5': 1e-2, 'fft': 1e-5},
    'clip':       {'lora': 1e-3, 'vpt_p5': 1e-3, 'fft': 1e-5},
}


def apply_fft(model):
    """Full fine-tuning: unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def run_method(base_model, config, method, lr, train_loader, val_loader, device):
    model = deepcopy(base_model)
    num_classes = config.num_classes
    model.head = nn.Linear(config.embed_dim, num_classes).to(device)

    if method == 'fft':
        model = apply_fft(model)
    elif method == 'lora':
        model = apply_lora(model, 8, config)
    elif method == 'vpt':
        model = apply_vpt(model, 5, config)
    elif method == 'lp':
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.head.parameters():
            param.requires_grad_(True)

    model = model.to(device)
    config.lr = lr
    acc = train_and_evaluate(model, train_loader, val_loader, config, device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    del model; torch.cuda.empty_cache()
    return acc, trainable, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['dinov2', 'deit3'])
    parser.add_argument('--tasks', nargs='+', default=['cifar100', 'svhn', 'eurosat'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("FFT BASELINE COMPARISON")
    print(f"Backbones: {args.backbones}, Tasks: {args.tasks}")
    print("=" * 70)

    # Load existing results
    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} pairs done")

    for bb_key in args.backbones:
        if bb_key not in BACKBONES:
            continue
        bb = BACKBONES[bb_key]
        img_size = bb['img_size']
        lrs = BEST_LRS.get(bb_key, {'lora': 1e-3, 'vpt_p5': 1e-2, 'fft': 1e-5})

        print(f"\n  Loading {bb['name']}...")
        base_model = timm.create_model(bb['model'], pretrained=True,
                                        img_size=img_size).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        for task in args.tasks:
            if task not in TASKS:
                continue

            key = f"{bb_key}_{task}"
            if key in all_results:
                print(f"\n  {key}: already done, skipping")
                continue

            num_classes = TASKS[task][0]
            config.num_classes = num_classes

            print(f"\n  {'='*50}")
            print(f"  {bb['name']} × {task}")
            print(f"  {'='*50}")

            methods = [
                ('LP',       'lp',   1e-2),
                ('LoRA_r8',  'lora', lrs['lora']),
                ('VPT_p5',   'vpt',  lrs['vpt_p5']),
                ('FFT',      'fft',  lrs['fft']),
            ]

            task_results = {}

            for method_name, method, lr in methods:
                seed_accs = []
                trainable_count = 0

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

                    acc, trainable, total = run_method(
                        base_model, config, method, lr,
                        train_loader, val_loader, device)
                    seed_accs.append(acc)
                    trainable_count = trainable

                mean_acc = np.mean(seed_accs)
                std_acc = np.std(seed_accs)
                pct = trainable_count / total * 100 if total > 0 else 100

                task_results[method_name] = {
                    'lr': float(lr),
                    'seeds': seed_accs,
                    'mean': float(mean_acc),
                    'std': float(std_acc),
                    'trainable': trainable_count,
                    'pct_params': float(pct),
                }
                print(f"    {method_name:<12s} (lr={lr:.0e}, {pct:>5.1f}%): "
                      f"{mean_acc:.3f} ± {std_acc:.3f}")

            all_results[key] = task_results

            # Incremental save
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

            # Summary
            fft = task_results['FFT']['mean']
            lora = task_results['LoRA_r8']['mean']
            vpt = task_results['VPT_p5']['mean']
            lp = task_results['LP']['mean']

            print(f"\n    LP={lp:.3f} → LoRA={lora:.3f} → VPT={vpt:.3f} → FFT={fft:.3f}")
            print(f"    LoRA recovers {(lora-lp)/(fft-lp+1e-6)*100:.0f}% of FFT-LP gap")
            print(f"    VPT recovers  {(vpt-lp)/(fft-lp+1e-6)*100:.0f}% of FFT-LP gap")

        del base_model; torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print("FFT BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Case':<20s} {'LP':>7s} {'LoRA':>7s} {'VPT':>7s} {'FFT':>7s} {'LoRA/FFT':>9s} {'Overfit?':>9s}")
    for key, results in all_results.items():
        lp = results['LP']['mean']
        lora = results['LoRA_r8']['mean']
        vpt = results['VPT_p5']['mean']
        fft = results['FFT']['mean']
        recovery = (lora - lp) / (fft - lp + 1e-6) * 100
        overfit = "YES" if fft < lora else "no"
        print(f"  {key:<20s} {lp:>7.3f} {lora:>7.3f} {vpt:>7.3f} {fft:>7.3f} {recovery:>8.0f}% {overfit:>9s}")

    with open(SAVE_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
