"""
Multi-Scale Architecture Validation: ViT-S, ViT-B, ViT-L

Tests whether the capacity ratio d/(2d_h) correctly predicts 
method behavior at different ViT scales:
  ViT-S: d=384, d_h=64, ratio = 384/128 = 3
  ViT-B: d=768, d_h=64, ratio = 768/128 = 6  (current paper)
  ViT-L: d=1024, d_h=64, ratio = 1024/128 = 8

Usage:
    python revision_vit_scale.py --scale large
    python revision_vit_scale.py --scale small
    python revision_vit_scale.py --scale all
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

from config import ExperimentConfig, setup_device, compute_sigma_p_sq
from exp2_comparison import train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader


VIT_SCALES = {
    'small': {
        'model': 'vit_small_patch16_224.augreg_in1k',
        'img_size': 224,
        'name': 'ViT-S/16 (d=384)',
        'd': 384, 'd_h': 64, 'heads': 6, 'layers': 12,
        'ratio': 384 // 128,  # = 3
    },
    'base': {
        'model': 'vit_base_patch16_224.augreg_in1k',
        'img_size': 224,
        'name': 'ViT-B/16 (d=768)',
        'd': 768, 'd_h': 64, 'heads': 12, 'layers': 12,
        'ratio': 768 // 128,  # = 6
    },
    'large': {
        'model': 'vit_large_patch16_224.augreg_in1k',
        'img_size': 224,
        'name': 'ViT-L/16 (d=1024)',
        'd': 1024, 'd_h': 64, 'heads': 16, 'layers': 24,
        'ratio': 1024 // 128,  # = 8
    },
}

TEST_TASKS = ['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat']


class LoRALayer(nn.Module):
    def __init__(self, original_linear, rank, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        d_out, d_in = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class VPTLayer(nn.Module):
    def __init__(self, block, num_prompts, embed_dim):
        super().__init__()
        self.block = block
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        x = self.block(x)
        x = torch.cat([x[:, :1], x[:, 1+self.prompts.shape[1]:]], dim=1)
        return x


def apply_lora_generic(model, rank):
    """Apply LoRA to any timm ViT model."""
    for param in model.parameters():
        param.requires_grad_(False)

    for block in model.blocks:
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRALayer(old_qkv, rank)
        old_proj = block.attn.proj
        block.attn.proj = LoRALayer(old_proj, rank)

    for name, param in model.named_parameters():
        if 'lora_' in name or 'head' in name:
            param.requires_grad_(True)

    return model


def apply_vpt_generic(model, num_prompts):
    """Apply VPT to any timm ViT model."""
    for param in model.parameters():
        param.requires_grad_(False)

    embed_dim = model.embed_dim
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(VPTLayer(block, num_prompts, embed_dim))
    model.blocks = new_blocks

    for name, param in model.named_parameters():
        if 'prompt' in name or 'head' in name:
            param.requires_grad_(True)

    return model


def run_scale(scale_key, tasks, device, n_train=800):
    """Run LoRA vs VPT comparison at a given ViT scale."""
    scale = VIT_SCALES[scale_key]
    print(f"\n{'='*70}")
    print(f"  {scale['name']}: d={scale['d']}, d_h={scale['d_h']}, "
          f"ratio d/(2d_h)={scale['ratio']}")
    print(f"{'='*70}")

    # Load model
    model = timm.create_model(scale['model'], pretrained=True,
                               img_size=scale['img_size']).to(device)

    # Compute σ²_P
    config = ExperimentConfig()
    config.embed_dim = model.embed_dim
    config.num_layers = len(model.blocks)
    config.num_heads = model.blocks[0].attn.num_heads
    config.head_dim = model.embed_dim // model.blocks[0].attn.num_heads
    config.epochs = 100

    sigma_p = compute_sigma_p_sq(model.state_dict(), config)
    r_star = int(2 * n_train * sigma_p / (config.num_layers * config.head_dim))
    p_star = int(4 * n_train * sigma_p / (config.num_layers * config.embed_dim))

    print(f"  σ²_P = {sigma_p:.2f}, r* = {r_star}, p* = {p_star}")
    print(f"  Predicted ratio: d/(2d_h) = {scale['d']}/{2*scale['d_h']} = {scale['ratio']}")

    all_results = {}

    for task_name in tasks:
        if task_name not in TASKS:
            continue

        num_classes = TASKS[task_name][0]
        print(f"\n  {scale['name']} × {task_name}")

        # Load data
        ds = load_dataset(task_name, scale['img_size'], max_samples=1000)
        n_val = min(200, len(ds) // 5)
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        task_results = {}

        # Methods to test
        methods = {
            'LoRA_r1': ('lora', 1),
            'LoRA_r4': ('lora', 4),
            'LoRA_r8': ('lora', 8),
            'VPT_p1':  ('vpt', 1),
            'VPT_p5':  ('vpt', 5),
            'VPT_p10': ('vpt', 10),
        }

        for method_name, (method_type, capacity) in methods.items():
            m = deepcopy(model)
            m.head = nn.Linear(config.embed_dim, num_classes).to(device)

            if method_type == 'lora':
                m = apply_lora_generic(m, capacity)
            else:
                m = apply_vpt_generic(m, capacity)

            m = m.to(device)
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

            acc = train_and_evaluate(m, train_loader, val_loader, config, device)
            task_results[method_name] = acc
            print(f"    {method_name}: {acc:.3f} ({trainable:,}p)")

            del m; torch.cuda.empty_cache()

        all_results[task_name] = task_results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {scale['name']}")
    print(f"σ²_P = {sigma_p:.2f}, r* = {r_star}, p* = {p_star}, "
          f"ratio = {scale['ratio']}")
    print(f"{'='*70}")

    methods_list = ['LoRA_r1', 'LoRA_r4', 'LoRA_r8', 'VPT_p1', 'VPT_p5', 'VPT_p10']
    print(f"  {'Task':<12s}", end='')
    for m in methods_list:
        print(f" {m:>8s}", end='')
    print(f" {'Winner':>8s}")

    lora_wins, vpt_wins, ties = 0, 0, 0
    for task, results in all_results.items():
        best_lora = max(results.get(f'LoRA_r{r}', 0) for r in [1,4,8])
        best_vpt = max(results.get(f'VPT_p{p}', 0) for p in [1,5,10])
        winner = "LoRA" if best_lora > best_vpt + 0.005 else "VPT" if best_vpt > best_lora + 0.005 else "TIE"
        if winner == "LoRA": lora_wins += 1
        elif winner == "VPT": vpt_wins += 1
        else: ties += 1

        print(f"  {task:<12s}", end='')
        for m in methods_list:
            print(f" {results.get(m, 0):>8.3f}", end='')
        print(f" {winner:>8s}")

    print(f"\n  L/T/V = {lora_wins}/{ties}/{vpt_wins}")

    del model; torch.cuda.empty_cache()

    return {
        'scale': scale['name'],
        'd': scale['d'], 'd_h': scale['d_h'],
        'ratio': scale['ratio'],
        'sigma_p': sigma_p, 'r_star': r_star, 'p_star': p_star,
        'results': all_results,
        'lora_wins': lora_wins, 'ties': ties, 'vpt_wins': vpt_wins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, default='large',
                        choices=['small', 'base', 'large', 'all'])
    parser.add_argument('--tasks', nargs='+', default=TEST_TASKS)
    args = parser.parse_args()

    device = setup_device()

    scales = ['small', 'base', 'large'] if args.scale == 'all' else [args.scale]

    all_scale_results = {}
    for scale in scales:
        result = run_scale(scale, args.tasks, device)
        all_scale_results[scale] = result

    # Cross-scale summary
    if len(all_scale_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-SCALE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Scale':<20s} {'d':>5s} {'d/(2d_h)':>8s} {'σ²_P':>6s} {'r*':>4s} {'p*':>4s} {'L/T/V':>8s}")
        for s, r in all_scale_results.items():
            print(f"  {r['scale']:<20s} {r['d']:>5d} {r['ratio']:>8d} "
                  f"{r['sigma_p']:>6.1f} {r['r_star']:>4d} {r['p_star']:>4d} "
                  f"{r['lora_wins']}/{r['ties']}/{r['vpt_wins']:>8s}")

    # Save
    os.makedirs('results', exist_ok=True)
    for s, r in all_scale_results.items():
        fname = f'results/vit_scale_{s}.json'
        with open(fname, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
