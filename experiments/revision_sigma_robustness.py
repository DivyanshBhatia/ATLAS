"""
σ²_P Robustness Analysis

Tests whether σ²_P is a robust metric across:
1. Normalization variants (Frobenius, spectral, trace)
2. Weight subsets (QKV only, QKV+O, QKV+O+MLP)
3. Checkpoint variants (same arch, different pretraining)
4. Architecture families (ViT-S/B/L within same pretraining)

Key question: Does the ORDERING of backbones by σ²_P remain stable?
If yes, σ²_P is a robust backbone fingerprint regardless of exact formula.

Usage:
    python revision_sigma_robustness.py
"""
import torch
import timm
import numpy as np
from scipy import stats
import json
import os

MODELS = {
    # Same architecture, different pretraining
    'DINOv2-B': ('vit_base_patch14_dinov2.lvd142m', 518),
    'DeiT-III-B': ('deit3_base_patch16_224.fb_in1k', 224),
    'CLIP-B': ('vit_base_patch16_clip_224.openai', 224),
    'Sup-AugReg-B': ('vit_base_patch16_224.augreg_in1k', 224),
    'MAE-B': ('vit_base_patch16_224.mae', 224),
    
    # Same pretraining, different scale
    'DINOv2-S': ('vit_small_patch14_dinov2.lvd142m', 518),
    'DINOv2-L': ('vit_large_patch14_dinov2.lvd142m', 518),
    'DeiT-III-S': ('deit3_small_patch16_224.fb_in1k', 224),
    'DeiT-III-L': ('deit3_large_patch16_224.fb_in1k', 224),
    
    # Same architecture, different checkpoint
    'Sup-Orig-B': ('vit_base_patch16_224.orig_in21k', 224),
    'CLIP-B-LAION': ('vit_base_patch16_clip_224.laion2b', 224),
}


def compute_sigma_variants(model):
    """Compute σ²_P using multiple normalization variants."""
    d_h = model.embed_dim // model.blocks[0].attn.num_heads
    d = model.embed_dim
    L = len(model.blocks)
    
    # Collect weight norms by category
    attn_qkv_norms = []
    attn_proj_norms = []
    mlp_norms = []
    all_attn_norms = []
    spectral_norms = []
    
    for block in model.blocks:
        # Attention weights
        for name, param in block.attn.named_parameters():
            if 'weight' in name:
                norm_sq = param.float().norm().item() ** 2
                all_attn_norms.append(norm_sq)
                
                # Spectral norm (largest singular value)
                try:
                    s = torch.linalg.svdvals(param.float())
                    spectral_norms.append(s[0].item() ** 2)
                except:
                    spectral_norms.append(norm_sq)
                
                if 'qkv' in name:
                    attn_qkv_norms.append(norm_sq)
                elif 'proj' in name:
                    attn_proj_norms.append(norm_sq)
        
        # MLP weights
        for name, param in block.named_parameters():
            if 'mlp' in name and 'weight' in name:
                mlp_norms.append(param.float().norm().item() ** 2)
    
    results = {}
    
    # Variant 1: Original (QKV + proj, divide by count × d_h)
    qkv_proj = attn_qkv_norms + attn_proj_norms
    results['v1_original'] = sum(qkv_proj) / (len(qkv_proj) * d_h) if qkv_proj else 0
    
    # Variant 2: All attention weights, divide by count × d_h
    results['v2_all_attn'] = sum(all_attn_norms) / (len(all_attn_norms) * d_h) if all_attn_norms else 0
    
    # Variant 3: QKV + proj + MLP, divide by count × d_h  
    all_weights = all_attn_norms + mlp_norms
    results['v3_all_weights'] = sum(all_weights) / (len(all_weights) * d_h) if all_weights else 0
    
    # Variant 4: Per-parameter (divide by total param count, not d_h)
    total_params = sum(p.numel() for p in model.parameters())
    total_norm = sum(all_attn_norms)
    results['v4_per_param'] = total_norm / total_params if total_params > 0 else 0
    
    # Variant 5: Spectral (use max singular value instead of Frobenius)
    results['v5_spectral'] = sum(spectral_norms) / (len(spectral_norms) * d_h) if spectral_norms else 0
    
    # Variant 6: Trace-normalized (Frobenius² / d, giving mean eigenvalue)
    results['v6_trace_norm'] = sum(all_attn_norms) / (len(all_attn_norms) * d) if all_attn_norms else 0
    
    return results


def main():
    print("=" * 80)
    print("σ²_P ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    all_results = {}
    
    for name, (model_name, img_size) in MODELS.items():
        print(f"\n  Loading {name}: {model_name}...")
        try:
            model = timm.create_model(model_name, pretrained=True, img_size=img_size)
            model.eval()
            
            variants = compute_sigma_variants(model)
            all_results[name] = variants
            
            print(f"    d={model.embed_dim}, heads={model.blocks[0].attn.num_heads}, "
                  f"layers={len(model.blocks)}")
            for vname, val in variants.items():
                print(f"    {vname:<20s}: {val:>10.3f}")
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ERROR: {str(e)[:80]}")
            all_results[name] = {'error': str(e)[:200]}
    
    # Analyze ordering consistency
    print(f"\n{'='*80}")
    print("ORDERING CONSISTENCY ACROSS VARIANTS")
    print(f"{'='*80}")
    
    # Use only ViT-B models for fair comparison
    vit_b_models = ['DINOv2-B', 'DeiT-III-B', 'CLIP-B', 'Sup-AugReg-B', 'MAE-B']
    vit_b_results = {k: v for k, v in all_results.items() 
                     if k in vit_b_models and 'error' not in v}
    
    if len(vit_b_results) >= 3:
        variant_names = list(next(iter(vit_b_results.values())).keys())
        
        print(f"\n  Spearman rank correlation between σ²_P variants (ViT-B models only):")
        print(f"  {'':20s}", end='')
        for v in variant_names[:4]:
            print(f" {v[:12]:>12s}", end='')
        print()
        
        for v1 in variant_names[:4]:
            print(f"  {v1:<20s}", end='')
            vals1 = [vit_b_results[m].get(v1, 0) for m in vit_b_models if m in vit_b_results]
            for v2 in variant_names[:4]:
                vals2 = [vit_b_results[m].get(v2, 0) for m in vit_b_models if m in vit_b_results]
                if len(vals1) >= 3 and len(vals2) >= 3:
                    rho, p = stats.spearmanr(vals1, vals2)
                    print(f" {rho:>12.3f}", end='')
                else:
                    print(f" {'N/A':>12s}", end='')
            print()
        
        # Show orderings for each variant
        print(f"\n  Backbone ordering by each variant:")
        for vname in variant_names:
            models_sorted = sorted(vit_b_results.keys(), 
                                   key=lambda m: vit_b_results[m].get(vname, 0))
            order = ' < '.join(f"{m.split('-')[0]}({vit_b_results[m].get(vname,0):.1f})" 
                              for m in models_sorted)
            print(f"    {vname:<20s}: {order}")
    
    # Scale consistency within architecture family
    print(f"\n{'='*80}")
    print("SCALE CONSISTENCY (same pretraining, different model size)")
    print(f"{'='*80}")
    
    for family, models in [('DINOv2', ['DINOv2-S', 'DINOv2-B', 'DINOv2-L']),
                            ('DeiT-III', ['DeiT-III-S', 'DeiT-III-B', 'DeiT-III-L'])]:
        print(f"\n  {family} family:")
        for m in models:
            if m in all_results and 'error' not in all_results[m]:
                v1 = all_results[m].get('v1_original', 0)
                print(f"    {m:<15s}: σ²_P = {v1:.2f}")
    
    # Checkpoint dependence
    print(f"\n{'='*80}")
    print("CHECKPOINT DEPENDENCE (same architecture, different training)")  
    print(f"{'='*80}")
    
    checkpoint_pairs = [
        ('Sup-AugReg-B', 'Sup-Orig-B', 'Supervised ViT-B: AugReg vs Original'),
        ('CLIP-B', 'CLIP-B-LAION', 'CLIP ViT-B: OpenAI vs LAION'),
    ]
    
    for m1, m2, desc in checkpoint_pairs:
        if m1 in all_results and m2 in all_results:
            if 'error' not in all_results[m1] and 'error' not in all_results[m2]:
                v1_1 = all_results[m1].get('v1_original', 0)
                v1_2 = all_results[m2].get('v1_original', 0)
                ratio = v1_1 / v1_2 if v1_2 > 0 else float('inf')
                print(f"  {desc}")
                print(f"    {m1}: σ²_P = {v1_1:.2f}")
                print(f"    {m2}: σ²_P = {v1_2:.2f}")
                print(f"    Ratio: {ratio:.2f}×")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/sigma_robustness.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/sigma_robustness.json")


if __name__ == '__main__':
    main()
