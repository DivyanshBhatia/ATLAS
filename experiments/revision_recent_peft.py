"""
Recent PEFT Methods Comparison

Compares LoRA, VPT with recent methods to validate that the theoretical
framework's predictions extend to modern variants:

1. DoRA (Liu et al., ICML 2024): LoRA + learned magnitude vector
   Theory prediction: Should match or slightly beat LoRA (same subspace,
   better magnitude calibration). Capacity ratio unchanged.

2. VeRA (Kopiczko et al., ICLR 2024): Shared frozen random B,A + learned 
   scaling vectors. Extremely parameter-efficient.
   Theory prediction: Should underperform LoRA at equal rank (frozen random
   matrices can't adapt direction, only scale).

3. SSF (Lian et al., NeurIPS 2022): Scale and Shift Features.
   Minimal baseline — one scale and one shift per feature dimension per layer.
   Theory prediction: Should work for easy tasks (low γ) but fail on hard
   tasks requiring substantial feature transformation.

Usage:
    python revision_recent_peft.py --backbone dinov2
    python revision_recent_peft.py --backbone deit3
    python revision_recent_peft.py --backbone dinov2 --tasks cifar10 svhn dtd
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
from torch.utils.data import DataLoader


# ============================================================
# DoRA: Weight-Decomposed Low-Rank Adaptation
# ============================================================
class DoRALayer(nn.Module):
    """DoRA = LoRA + learned magnitude vector.
    
    Decomposes ΔW into magnitude (m) and direction (LoRA):
    W' = m * (W + BA) / ||W + BA||_col
    
    Simplified version: W' = W + BA * diag(m)
    where m is a learned per-output-dim magnitude vector.
    """
    def __init__(self, original_linear, rank, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        d_out, d_in = original_linear.weight.shape
        
        # Standard LoRA components
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank
        
        # DoRA magnitude vector (initialized to column norms of W)
        with torch.no_grad():
            col_norms = original_linear.weight.float().norm(dim=1)
        self.magnitude = nn.Parameter(col_norms.float())
    
    def forward(self, x):
        # Compute LoRA update
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # Original output
        orig_out = self.original(x)
        
        # Combined with magnitude scaling
        combined = orig_out + lora_out
        
        # Normalize and rescale by learned magnitude
        col_norms = (self.original.weight.float() + 
                     (self.lora_B @ self.lora_A * self.scaling).float()).norm(dim=1, keepdim=True).T
        combined = combined / (col_norms + 1e-8) * self.magnitude.unsqueeze(0)
        
        return combined


def apply_dora(model, rank, config):
    """Apply DoRA to all attention layers."""
    for param in model.parameters():
        param.requires_grad_(False)
    
    for block in model.blocks:
        old_qkv = block.attn.qkv
        block.attn.qkv = DoRALayer(old_qkv, rank)
        old_proj = block.attn.proj  
        block.attn.proj = DoRALayer(old_proj, rank)
    
    for name, param in model.named_parameters():
        if any(k in name for k in ['lora_', 'magnitude', 'head']):
            param.requires_grad_(True)
    
    return model


# ============================================================
# VeRA: Vector-based Random Matrix Adaptation
# ============================================================
class VeRALayer(nn.Module):
    """VeRA: Shared frozen random B,A with learned scaling vectors.
    
    W' = W + diag(d) @ B_frozen @ diag(b) @ A_frozen
    
    Only d (d_out-dim) and b (rank-dim) are learned.
    B_frozen and A_frozen are shared across layers.
    """
    def __init__(self, original_linear, rank, shared_A, shared_B):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        d_out, d_in = original_linear.weight.shape
        
        # Frozen random matrices (shared across layers)
        self.register_buffer('vera_A', shared_A[:rank, :d_in].clone())  # (rank, d_in)
        self.register_buffer('vera_B', shared_B[:d_out, :rank].clone())  # (d_out, rank)
        
        # Learned scaling vectors
        self.d_vec = nn.Parameter(torch.ones(d_out))   # per-output scaling
        self.b_vec = nn.Parameter(torch.ones(rank))    # per-rank scaling
    
    def forward(self, x):
        orig_out = self.original(x)
        
        # VeRA: x @ A^T @ diag(b) @ B^T @ diag(d)
        h = x @ self.vera_A.T          # (B, seq, rank)
        h = h * self.b_vec.unsqueeze(0)  # scale by b
        h = h @ self.vera_B.T          # (B, seq, d_out)
        h = h * self.d_vec.unsqueeze(0)  # scale by d
        
        return orig_out + h * 0.1  # small scaling


def apply_vera(model, rank, config):
    """Apply VeRA to all attention layers with shared random matrices."""
    for param in model.parameters():
        param.requires_grad_(False)
    
    d = config.embed_dim
    max_dim = d * 3  # QKV projection is 3d x d
    
    # Generate shared random matrices (same for all layers)
    torch.manual_seed(42)
    shared_A = torch.randn(rank, max_dim) * 0.02
    shared_B = torch.randn(max_dim, rank) * 0.02
    
    for block in model.blocks:
        old_qkv = block.attn.qkv
        block.attn.qkv = VeRALayer(old_qkv, rank, shared_A, shared_B)
        old_proj = block.attn.proj
        block.attn.proj = VeRALayer(old_proj, rank, shared_A, shared_B)
    
    for name, param in model.named_parameters():
        if any(k in name for k in ['d_vec', 'b_vec', 'head']):
            param.requires_grad_(True)
    
    return model


# ============================================================
# SSF: Scale and Shift Features
# ============================================================
class SSFLayer(nn.Module):
    """Scale and Shift Features applied after attention output."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return x * self.scale + self.shift


def apply_ssf(model, config):
    """Apply SSF after each attention block."""
    for param in model.parameters():
        param.requires_grad_(False)
    
    d = config.embed_dim
    
    for block in model.blocks:
        # Add SSF after attention
        original_forward = block.forward
        ssf = SSFLayer(d)
        block.ssf = ssf
        
        def make_forward(orig_fn, ssf_mod):
            def new_forward(x):
                out = orig_fn(x)
                return ssf_mod(out)
            return new_forward
        
        block.forward = make_forward(original_forward, ssf)
    
    for name, param in model.named_parameters():
        if any(k in name for k in ['ssf', 'scale', 'shift', 'head']):
            param.requires_grad_(True)
    
    return model


# ============================================================
# Main experiment
# ============================================================
def run_task(base_model, task_name, bb_key, device, config):
    """Run all methods on one task."""
    bb = BACKBONES[bb_key]
    num_classes = TASKS[task_name][0]
    embed_dim = config.embed_dim

    ds = load_dataset(task_name, bb['img_size'], max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}

    methods = [
        # Existing baselines
        ('LoRA_r4', lambda m: apply_lora(m, 4, config), 1e-3),
        ('LoRA_r8', lambda m: apply_lora(m, 8, config), 1e-3),
        ('VPT_p5', lambda m: apply_vpt(m, 5, config), 1e-2),
        ('VPT_p10', lambda m: apply_vpt(m, 10, config), 1e-2),
        # Recent methods
        ('DoRA_r4', lambda m: apply_dora(m, 4), 1e-3),
        ('DoRA_r8', lambda m: apply_dora(m, 8), 1e-3),
        ('VeRA_r4', lambda m: apply_vera(m, 4, config), 1e-3),
        ('VeRA_r8', lambda m: apply_vera(m, 8, config), 1e-3),
        ('SSF', lambda m: apply_ssf(m, config), 1e-2),
    ]

    for name, apply_fn, lr in methods:
        try:
            model = deepcopy(base_model)
            model.head = nn.Linear(embed_dim, num_classes).to(device)
            model = apply_fn(model)
            model = model.to(device)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Override LR in config temporarily
            old_lr = getattr(config, 'lr', 1e-3)
            config.lr = lr
            acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            config.lr = old_lr

            results[name] = {'acc': acc, 'params': trainable}
            print(f"    {name:<12s}: {acc:.3f}  ({trainable:,}p)")
            
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {name:<12s}: ERROR - {str(e)[:80]}")
            results[name] = {'acc': 0.0, 'params': 0, 'error': str(e)[:200]}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+',
                        default=['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 70)
    print(f"Recent PEFT Methods Comparison on {bb['name']}")
    print("=" * 70)

    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    config.embed_dim = base_model.embed_dim
    config.num_layers = len(base_model.blocks)
    config.num_heads = base_model.blocks[0].attn.num_heads
    config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

    all_results = {}

    for task in args.tasks:
        if task not in TASKS:
            continue
        print(f"\n{'='*55}")
        print(f"  {bb['name']} × {task}")
        print(f"{'='*55}")
        results = run_task(base_model, task, bb_key, device, config)
        all_results[task] = results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {bb['name']}")
    print(f"{'='*70}")

    methods = ['LoRA_r4', 'LoRA_r8', 'VPT_p5', 'VPT_p10',
               'DoRA_r4', 'DoRA_r8', 'VeRA_r4', 'VeRA_r8', 'SSF']

    print(f"  {'Task':<12s}", end='')
    for m in methods:
        print(f" {m:>8s}", end='')
    print(f" {'Best':>8s}")
    print(f"  {'-'*100}")

    for task, results in all_results.items():
        print(f"  {task:<12s}", end='')
        best_m, best_a = '', 0
        for m in methods:
            acc = results.get(m, {}).get('acc', 0)
            print(f" {acc:>8.3f}", end='')
            if acc > best_a:
                best_a = acc
                best_m = m
        print(f" {best_m:>8s}")

    # Category averages
    print(f"\n  Category averages:")
    categories = {
        'LoRA': ['LoRA_r4', 'LoRA_r8'],
        'VPT': ['VPT_p5', 'VPT_p10'],
        'DoRA': ['DoRA_r4', 'DoRA_r8'],
        'VeRA': ['VeRA_r4', 'VeRA_r8'],
        'SSF': ['SSF'],
    }
    for cat, cat_methods in categories.items():
        accs = []
        for task, results in all_results.items():
            best = max(results.get(m, {}).get('acc', 0) for m in cat_methods)
            accs.append(best)
        print(f"    {cat:<8s}: {np.mean(accs):.3f} (avg best across tasks)")

    # Theory predictions
    print(f"\n  Theory predictions:")
    print(f"    DoRA ≈ LoRA: Same subspace, better magnitude → expect DoRA ≥ LoRA")
    print(f"    VeRA < LoRA: Frozen random directions → expect VeRA < LoRA")
    print(f"    SSF < LoRA:  Minimal (scale/shift only) → expect SSF < LoRA on hard tasks")

    del base_model; torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/recent_peft_{bb_key}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
