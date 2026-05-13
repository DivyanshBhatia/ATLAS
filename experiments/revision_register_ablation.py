"""
Register-Token Causal Ablation

Tests whether VPT collapse on DINOv2-reg is CAUSED by register tokens
or is just a property of the DINOv2-reg checkpoint.

Ablations:
1. DINOv2 (no reg) + VPT → baseline (should work)
2. DINOv2-reg + VPT → collapse (known)
3. DINOv2-reg + VPT with registers REMOVED from sequence → causal test
4. DINOv2-reg + LoRA → stable baseline

If ablation 3 recovers VPT performance, the collapse is causally due to
register-prompt competition, not the checkpoint itself.

Usage:
    python revision_register_ablation.py
    python revision_register_ablation.py --tasks cifar100 gtsrb
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
from torch.utils.data import DataLoader


class RegisterMaskedVPTBlock(nn.Module):
    """VPT block that masks attention between prompts and register tokens."""
    def __init__(self, block, num_prompts, num_registers, embed_dim, mode='remove'):
        super().__init__()
        self.block = block
        self.num_prompts = num_prompts
        self.num_registers = num_registers
        self.mode = mode  # 'remove' or 'mask'
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)

        if self.mode == 'remove':
            # Remove register tokens, add prompts, run block, re-add registers
            # Registers are tokens 1:1+num_registers (after CLS)
            cls = x[:, :1]
            registers = x[:, 1:1+self.num_registers]
            patches = x[:, 1+self.num_registers:]

            # Run block WITHOUT registers, WITH prompts
            x_no_reg = torch.cat([cls, prompts, patches], dim=1)
            x_no_reg = self.block(x_no_reg)

            # Remove prompts from output
            cls_out = x_no_reg[:, :1]
            patches_out = x_no_reg[:, 1+self.num_prompts:]

            # Re-insert registers (unchanged)
            x_out = torch.cat([cls_out, registers, patches_out], dim=1)
            return x_out

        else:  # standard VPT (prompts added normally alongside registers)
            x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
            x = self.block(x)
            x = torch.cat([x[:, :1], x[:, 1+self.num_prompts:]], dim=1)
            return x


def apply_vpt_register_ablation(model, num_prompts, num_registers, config, mode='remove'):
    """Apply VPT with register ablation."""
    for param in model.parameters():
        param.requires_grad_(False)

    d = config.embed_dim
    new_blocks = nn.Sequential()
    for i, block in enumerate(model.blocks):
        new_blocks.add_module(str(i),
            RegisterMaskedVPTBlock(block, num_prompts, num_registers, d, mode=mode))
    model.blocks = new_blocks

    for name, param in model.named_parameters():
        if 'prompt' in name or 'head' in name:
            param.requires_grad_(True)

    return model


def count_registers(model):
    """Detect number of register tokens in a DINOv2-reg model."""
    # DINOv2-reg uses 4 register tokens
    if hasattr(model, 'num_register_tokens'):
        return model.num_register_tokens
    # Default for DINOv2-reg
    return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['cifar100', 'gtsrb', 'svhn'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    MODELS = {
        'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 518, 0),
        'DINOv2-reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 518, 4),
    }

    print("=" * 70)
    print("Register-Token Causal Ablation")
    print("=" * 70)

    all_results = {}

    for task in args.tasks:
        if task not in TASKS:
            continue

        num_classes = TASKS[task][0]
        print(f"\n{'='*55}")
        print(f"  Task: {task}")
        print(f"{'='*55}")

        task_results = {}

        for model_name, (timm_name, img_size, n_reg) in MODELS.items():
            base_model = timm.create_model(timm_name, pretrained=True,
                                            img_size=img_size).to(device)
            config.embed_dim = base_model.embed_dim
            config.num_layers = len(base_model.blocks)
            config.num_heads = base_model.blocks[0].attn.num_heads
            config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = torch.utils.data.random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            conditions = [
                (f'{model_name}+LoRA_r8', lambda m: apply_lora(m, 8, config), 1e-3),
                (f'{model_name}+VPT_p5', lambda m: apply_vpt(m, 5, config), 1e-2),
            ]

            # Add register ablation only for DINOv2-reg
            if n_reg > 0:
                conditions.append(
                    (f'{model_name}+VPT_p5(no_reg)',
                     lambda m: apply_vpt_register_ablation(m, 5, n_reg, config, mode='remove'),
                     1e-2)
                )

            for cond_name, apply_fn, lr in conditions:
                model = deepcopy(base_model)
                model.head = nn.Linear(config.embed_dim, num_classes).to(device)
                model = apply_fn(model)
                model = model.to(device)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                config.lr = lr
                acc = train_and_evaluate(model, train_loader, val_loader, config, device)

                task_results[cond_name] = acc
                print(f"    {cond_name:<30s}: {acc:.3f} ({trainable:,}p)")

                del model; torch.cuda.empty_cache()

            del base_model; torch.cuda.empty_cache()

        all_results[task] = task_results

    # Summary
    print(f"\n{'='*70}")
    print("CAUSAL ABLATION SUMMARY")
    print(f"{'='*70}")

    for task, results in all_results.items():
        print(f"\n  {task}:")
        dinov2_vpt = results.get('DINOv2+VPT_p5', 0)
        reg_vpt = results.get('DINOv2-reg+VPT_p5', 0)
        reg_ablated = results.get('DINOv2-reg+VPT_p5(no_reg)', 0)
        reg_lora = results.get('DINOv2-reg+LoRA_r8', 0)

        print(f"    DINOv2 + VPT_p5:           {dinov2_vpt:.3f}  (baseline: VPT works)")
        print(f"    DINOv2-reg + VPT_p5:       {reg_vpt:.3f}  (collapse expected)")
        print(f"    DINOv2-reg + VPT_p5(no_reg): {reg_ablated:.3f}  (causal test)")
        print(f"    DINOv2-reg + LoRA_r8:      {reg_lora:.3f}  (stable reference)")

        if reg_ablated > reg_vpt + 0.05:
            print(f"    → CAUSAL: removing registers recovers VPT by {reg_ablated-reg_vpt:+.3f}")
        elif reg_ablated < reg_vpt + 0.02:
            print(f"    → NOT CAUSAL: removing registers does not help")
        else:
            print(f"    → INCONCLUSIVE: small difference {reg_ablated-reg_vpt:+.3f}")

    os.makedirs('results', exist_ok=True)
    with open('results/register_ablation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/register_ablation.json")


if __name__ == '__main__':
    main()
