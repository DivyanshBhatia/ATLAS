"""
Canonical σ²_P Definition — Addresses R1.3

R1's concern: "σ²_P is not invariant to rescalings that leave the network 
function unchanged (W_Q → cW_Q with W_K → W_K/c after LayerNorm)."

This script computes σ²_P under multiple canonical forms and proves
that the backbone ORDERING is invariant, even if absolute values differ.

Three canonical forms:
1. Raw (current): σ²_P = (1/|W|·d_h) Σ ||W||²_F
2. LayerNorm-absorbed: W_canonical = W · diag(γ_LN), absorbing LN scale
3. Spectral-normalized: σ²_P = (1/|W|) Σ (||W||_F / ||W||_spectral)²

Usage:
    python revision2_sigma_canonical.py
"""
import sys
sys.path.insert(0, '.')

import torch
import timm
import numpy as np
import json
import os
from scipy import stats

from config import setup_device


BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 518),
    'DINOv2-reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 518),
    'CLIP': ('vit_base_patch16_clip_224.openai', 224),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 224),
    'DeiT-III': ('deit3_base_patch16_224.fb_in1k', 224),
    'MAE': ('vit_base_patch16_mae', 224),
}


def compute_sigma_raw(model, d_h=64):
    """Original σ²_P: (1/|W|·d_h) Σ ||W||²_F over QKV+O matrices."""
    total = 0
    count = 0
    for block in model.blocks:
        # QKV combined
        W_qkv = block.attn.qkv.weight  # (3d, d)
        d = model.embed_dim
        W_q = W_qkv[:d]
        W_k = W_qkv[d:2*d]
        W_v = W_qkv[2*d:]
        W_o = block.attn.proj.weight  # (d, d)

        for W in [W_q, W_k, W_v, W_o]:
            total += W.float().norm()**2
            count += 1

    return (total / (count * d_h)).item()


def compute_sigma_ln_absorbed(model, d_h=64):
    """LayerNorm-absorbed σ²_P: W_canonical = W · diag(γ_LN).
    
    After LayerNorm(x) = γ * (x - μ) / σ + β, the effective weight 
    applied to the pre-LN activations is W · diag(γ). This makes σ²_P 
    invariant to γ ↔ W trading.
    """
    total = 0
    count = 0
    for block in model.blocks:
        # Get the LayerNorm scale before attention
        if hasattr(block, 'norm1'):
            gamma = block.norm1.weight.float()  # (d,)
        else:
            gamma = torch.ones(model.embed_dim)

        W_qkv = block.attn.qkv.weight.float()  # (3d, d)
        d = model.embed_dim
        W_q = W_qkv[:d]
        W_k = W_qkv[d:2*d]
        W_v = W_qkv[2*d:]

        # Absorb LayerNorm: W_canonical = W · diag(γ)
        for W in [W_q, W_k, W_v]:
            W_canon = W * gamma.unsqueeze(0)  # broadcast: (d, d) * (1, d)
            total += W_canon.norm()**2
            count += 1

        # Output projection has its own preceding context (attention output)
        W_o = block.attn.proj.weight.float()
        total += W_o.norm()**2
        count += 1

    return (total / (count * d_h)).item()


def compute_sigma_spectral_normalized(model):
    """Spectral-normalized σ²_P: (1/|W|) Σ (||W||_F / ||W||_spectral)².
    
    This is fully scale-invariant: multiplying W by c doesn't change
    ||cW||_F / ||cW||_spectral = ||W||_F / ||W||_spectral.
    
    Interpretation: measures the "effective rank" of each weight matrix.
    """
    total = 0
    count = 0
    for block in model.blocks:
        W_qkv = block.attn.qkv.weight.float()
        d = model.embed_dim
        W_q = W_qkv[:d]
        W_k = W_qkv[d:2*d]
        W_v = W_qkv[2*d:]
        W_o = block.attn.proj.weight.float()

        for W in [W_q, W_k, W_v, W_o]:
            fro = W.norm()
            spec = torch.linalg.svdvals(W)[0]
            ratio = (fro / spec) ** 2
            total += ratio.item()
            count += 1

    return total / count


def compute_sigma_per_layer(model, d_h=64):
    """Per-layer σ²_l for the supplementary table."""
    per_layer = []
    for block in model.blocks:
        W_qkv = block.attn.qkv.weight.float()
        d = model.embed_dim
        W_q = W_qkv[:d]
        W_k = W_qkv[d:2*d]
        W_v = W_qkv[2*d:]
        W_o = block.attn.proj.weight.float()

        layer_total = sum(W.norm()**2 for W in [W_q, W_k, W_v, W_o])
        per_layer.append((layer_total / (4 * d_h)).item())

    return per_layer


def main():
    device = setup_device()

    print("=" * 70)
    print("Canonical σ²_P Computation")
    print("Addressing R1.3: Scale Invariance")
    print("=" * 70)

    results = {}

    for bb_name, (model_name, img_size) in BACKBONES.items():
        print(f"\n  Loading {bb_name}...")
        try:
            model = timm.create_model(model_name, pretrained=True, img_size=img_size)
        except Exception:
            model = timm.create_model(model_name, pretrained=True)
        model.eval()

        d_h = model.embed_dim // model.blocks[0].attn.num_heads

        raw = compute_sigma_raw(model, d_h)
        ln_absorbed = compute_sigma_ln_absorbed(model, d_h)
        spectral = compute_sigma_spectral_normalized(model)
        per_layer = compute_sigma_per_layer(model, d_h)

        cv = np.std(per_layer) / np.mean(per_layer)
        max_min = max(per_layer) / min(per_layer)

        results[bb_name] = {
            'raw': raw,
            'ln_absorbed': ln_absorbed,
            'spectral_normalized': spectral,
            'per_layer_mean': np.mean(per_layer),
            'per_layer_cv': cv,
            'per_layer_max_min': max_min,
        }

        print(f"    Raw σ²_P:           {raw:.2f}")
        print(f"    LN-absorbed σ²_P:   {ln_absorbed:.2f}")
        print(f"    Spectral-norm σ²_P: {spectral:.2f}")
        print(f"    Per-layer mean:     {np.mean(per_layer):.2f} (CV={cv:.3f})")

        del model
        torch.cuda.empty_cache()

    # Ordering analysis
    print(f"\n{'='*70}")
    print("ORDERING ANALYSIS")
    print(f"{'='*70}")

    backbones = list(results.keys())
    raw_vals = [results[b]['raw'] for b in backbones]
    ln_vals = [results[b]['ln_absorbed'] for b in backbones]
    spec_vals = [results[b]['spectral_normalized'] for b in backbones]

    rho_raw_ln, _ = stats.spearmanr(raw_vals, ln_vals)
    rho_raw_spec, _ = stats.spearmanr(raw_vals, spec_vals)
    rho_ln_spec, _ = stats.spearmanr(ln_vals, spec_vals)

    print(f"  Spearman correlations:")
    print(f"    Raw vs LN-absorbed:  ρ = {rho_raw_ln:.3f}")
    print(f"    Raw vs Spectral:     ρ = {rho_raw_spec:.3f}")
    print(f"    LN vs Spectral:      ρ = {rho_ln_spec:.3f}")

    print(f"\n  Backbone ordering (Raw):        {sorted(backbones, key=lambda b: results[b]['raw'])}")
    print(f"  Backbone ordering (LN-absorbed): {sorted(backbones, key=lambda b: results[b]['ln_absorbed'])}")
    print(f"  Backbone ordering (Spectral):    {sorted(backbones, key=lambda b: results[b]['spectral_normalized'])}")

    if rho_raw_ln >= 0.9 and rho_raw_spec >= 0.8:
        print(f"\n  ✓ Ordering is INVARIANT across canonical forms")
        print(f"    → σ²_P captures a genuine backbone property")
    else:
        print(f"\n  ⚠ Ordering DIFFERS across canonical forms")
        print(f"    → Need to choose the most principled definition")

    # Table for paper: reconcile Table IV vs Supp Table I
    print(f"\n{'='*70}")
    print("TABLE RECONCILIATION (Table IV vs Supp Table I)")
    print(f"{'='*70}")
    print(f"  {'Backbone':<15s} {'Raw (QKV+O)':<15s} {'Per-layer mean':<15s} {'Ratio':<10s}")
    for bb in backbones:
        raw = results[bb]['raw']
        plm = results[bb]['per_layer_mean']
        ratio = plm / raw if raw > 0 else 0
        print(f"  {bb:<15s} {raw:<15.2f} {plm:<15.2f} {ratio:<10.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/revision2_sigma_canonical.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_sigma_canonical.json")


if __name__ == '__main__':
    main()
