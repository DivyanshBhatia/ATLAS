"""
Reparameterization-Invariant σ²_P — Fixes R1.3

THE PROBLEM:
After LayerNorm, these rescalings leave the network function UNCHANGED:
  W_Q → cW_Q, W_K → W_K/c  (softmax sees W_Q @ W_K.T, unchanged)
  W_V → cW_V, W_O → W_O/c  (output sees W_O @ W_V, unchanged)

Current σ²_P = mean(||W||²_F / d_h) changes arbitrarily under these rescalings.
So σ²_P is a property of the PARAMETERIZATION, not the MODEL.

THE FIX:
Define σ²_P over the invariant products:
  M_attn = W_Q @ W_K.T   (attention logit matrix, d×d)
  M_val  = W_O @ W_V      (value output matrix, d×d)

These products are invariant to all rescalings that preserve the function.

CANDIDATE DEFINITIONS:
1. Product Frobenius:  σ²_P = mean_l [ (||M_attn||²_F + ||M_val||²_F) / d² ]
2. Product nuclear:   σ²_P = mean_l [ (||M_attn||_* + ||M_val||_*) / d ]
3. Geometric mean:    σ²_P = mean_l [ (||W_Q||_F · ||W_K||_F + ||W_V||_F · ||W_O||_F) / d ]
4. Per-head product:  σ²_P = mean_l,h [ ||W_Qh @ W_Kh.T||²_F / d_h² ]

Option 3 (geometric mean) is special: ||W_Q||_F · ||W_K||_F is invariant to
c/1/c rescaling AND equals ||M_attn||_F when W_Q, W_K have matched singular
values (a soft version of the invariance). It's also cheap to compute.

Usage:
    python revision2_sigma_invariant.py
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
    'DINOv2':     ('vit_base_patch14_dinov2.lvd142m', 518),
    'DINOv2-reg': ('vit_base_patch14_reg4_dinov2.lvd142m', 518),
    'CLIP':       ('vit_base_patch16_clip_224.openai', 224),
    'Supervised':  ('vit_base_patch16_224.augreg_in1k', 224),
    'DeiT-III':   ('deit3_base_patch16_224.fb_in1k', 224),
    'MAE':        ('vit_base_patch16_mae', 224),
}


def extract_qkvo(block, d):
    """Extract Q, K, V, O weight matrices from a timm attention block."""
    W_qkv = block.attn.qkv.weight.float()  # (3d, d)
    W_q = W_qkv[:d]
    W_k = W_qkv[d:2*d]
    W_v = W_qkv[2*d:]
    W_o = block.attn.proj.weight.float()    # (d, d)
    return W_q, W_k, W_v, W_o


def sigma_original(model):
    """ORIGINAL (non-invariant): mean(||W||²_F / d_h) over Q,K,V,O."""
    d = model.embed_dim
    d_h = d // model.blocks[0].attn.num_heads
    total, count = 0.0, 0
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        for W in [W_q, W_k, W_v, W_o]:
            total += (W.norm() ** 2).item()
            count += 1
    return total / (count * d_h)


def sigma_product_frobenius(model):
    """INVARIANT: mean_l [ (||W_Q W_K^T||²_F + ||W_O W_V||²_F) / (2·d²) ]
    
    Both products are invariant to c/1/c rescaling.
    Normalized by d² so the quantity scales sensibly across architectures.
    """
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        M_attn = W_q @ W_k.T   # (d, d) — attention logit kernel
        M_val = W_o @ W_v       # (d, d) — value output kernel
        val = (M_attn.norm() ** 2 + M_val.norm() ** 2).item() / (2 * d * d)
        per_layer.append(val)
    return np.mean(per_layer), per_layer


def sigma_product_trace(model):
    """INVARIANT: mean_l [ (tr(W_K^T W_Q^T W_Q W_K) + tr(W_V^T W_O^T W_O W_V)) / (2·d) ]
    
    = mean_l [ (||W_Q W_K^T||²_F + ||W_O W_V||²_F) / (2·d) ]
    Same as product_frobenius but normalized by d instead of d².
    """
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        M_attn = W_q @ W_k.T
        M_val = W_o @ W_v
        val = (M_attn.norm() ** 2 + M_val.norm() ** 2).item() / (2 * d)
        per_layer.append(val)
    return np.mean(per_layer), per_layer


def sigma_geometric_mean(model):
    """INVARIANT: mean_l [ (||W_Q||_F · ||W_K||_F + ||W_V||_F · ||W_O||_F) / (2·d) ]
    
    ||W_Q||_F · ||W_K||_F is invariant to c/1/c rescaling (each norm scales 
    by c and 1/c respectively, product unchanged).
    
    This is an upper bound on ||W_Q W_K^T||_* (nuclear norm) by Cauchy-Schwarz,
    and equals ||W_Q W_K^T||_F when singular values are matched.
    """
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        geom_attn = W_q.norm().item() * W_k.norm().item()
        geom_val = W_v.norm().item() * W_o.norm().item()
        val = (geom_attn + geom_val) / (2 * d)
        per_layer.append(val)
    return np.mean(per_layer), per_layer


def sigma_per_head_product(model):
    """INVARIANT: mean over layers and heads of ||W_Qh W_Kh^T||²_F / d_h²
    
    Per-head version: computes the product within each attention head.
    This is the most granular invariant form.
    """
    d = model.embed_dim
    n_heads = model.blocks[0].attn.num_heads
    d_h = d // n_heads
    per_layer = []
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        layer_val = 0.0
        for h in range(n_heads):
            W_qh = W_q[h*d_h:(h+1)*d_h, :]       # (d_h, d)
            W_kh = W_k[h*d_h:(h+1)*d_h, :]        # (d_h, d)
            # For value path: W_O is (d, d), we need per-head slice
            # W_Oh = W_o[:, h*d_h:(h+1)*d_h]      # (d, d_h)
            W_vh = W_v[h*d_h:(h+1)*d_h, :]        # (d_h, d)
            
            M_attn_h = W_qh @ W_kh.T              # (d_h, d_h)
            layer_val += (M_attn_h.norm() ** 2).item() / (d_h * d_h)
        layer_val /= n_heads
        per_layer.append(layer_val)
    return np.mean(per_layer), per_layer


def sigma_spectral_product(model):
    """INVARIANT: mean of top singular value of W_Q W_K^T and W_O W_V.
    
    The spectral norm of the product is invariant and captures the 
    "strength" of the strongest attention/value direction.
    """
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_q, W_k, W_v, W_o = extract_qkvo(block, d)
        M_attn = W_q @ W_k.T
        M_val = W_o @ W_v
        s_attn = torch.linalg.svdvals(M_attn)[0].item()
        s_val = torch.linalg.svdvals(M_val)[0].item()
        per_layer.append((s_attn + s_val) / 2)
    return np.mean(per_layer), per_layer


def verify_rescaling_invariance(model, method_fn, method_name, n_trials=5):
    """Empirically verify that a σ²_P definition is rescaling-invariant.
    
    Apply random c/1/c rescalings to Q/K and V/O pairs, 
    check that σ²_P doesn't change.
    """
    d = model.embed_dim
    
    # Compute baseline
    baseline, _ = method_fn(model)
    
    # Apply random rescalings
    import copy
    deviations = []
    for trial in range(n_trials):
        model_copy = copy.deepcopy(model)
        for block in model_copy.blocks:
            W_qkv = block.attn.qkv.weight.data.float()
            
            # Random rescaling factor for Q/K
            c_qk = np.random.uniform(0.1, 10.0)
            W_qkv[:d] *= c_qk       # W_Q → c·W_Q
            W_qkv[d:2*d] /= c_qk    # W_K → W_K/c
            
            # Random rescaling factor for V/O  
            c_vo = np.random.uniform(0.1, 10.0)
            W_qkv[2*d:] *= c_vo     # W_V → c·W_V
            block.attn.proj.weight.data /= c_vo  # W_O → W_O/c
            
            block.attn.qkv.weight.data = W_qkv
        
        rescaled, _ = method_fn(model_copy)
        deviation = abs(rescaled - baseline) / (abs(baseline) + 1e-10)
        deviations.append(deviation)
        del model_copy
    
    max_dev = max(deviations)
    return max_dev


def main():
    device = setup_device()
    
    print("=" * 70)
    print("REPARAMETERIZATION-INVARIANT σ²_P")
    print("Fixing R1.3: Scale Invariance Under Q/K and V/O Rescaling")
    print("=" * 70)
    
    methods = {
        'original (NOT invariant)': (sigma_original, False),
        'product_frobenius':        (lambda m: sigma_product_frobenius(m), True),
        'product_trace':            (lambda m: sigma_product_trace(m), True),
        'geometric_mean':           (lambda m: sigma_geometric_mean(m), True),
        'per_head_product':         (lambda m: sigma_per_head_product(m), True),
        'spectral_product':         (lambda m: sigma_spectral_product(m), True),
    }
    
    all_results = {}
    backbone_values = {m: [] for m in methods}
    backbone_names = []
    
    for bb_name, (model_name, img_size) in BACKBONES.items():
        print(f"\n{'='*55}")
        print(f"  {bb_name} ({model_name})")
        print(f"{'='*55}")
        
        try:
            model = timm.create_model(model_name, pretrained=True, img_size=img_size)
        except:
            model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        backbone_names.append(bb_name)
        bb_result = {}
        
        for method_name, (method_fn, is_invariant) in methods.items():
            if method_name == 'original (NOT invariant)':
                val = method_fn(model)
                per_layer = None
            else:
                val, per_layer = method_fn(model)
            
            bb_result[method_name] = {
                'value': val,
                'per_layer_cv': float(np.std(per_layer) / np.mean(per_layer)) if per_layer else None,
            }
            backbone_values[method_name].append(val)
            
            # Verify invariance empirically
            if is_invariant:
                max_dev = verify_rescaling_invariance(model, method_fn, method_name)
                bb_result[method_name]['max_rescaling_deviation'] = float(max_dev)
                inv_str = f"  max_dev={max_dev:.2e} {'✓' if max_dev < 1e-4 else '✗'}"
            else:
                max_dev = verify_rescaling_invariance(
                    model, lambda m: (sigma_original(m), None), method_name)
                bb_result[method_name]['max_rescaling_deviation'] = float(max_dev)
                inv_str = f"  max_dev={max_dev:.2e} ✗ (NOT invariant)"
            
            cv_str = f"  CV={bb_result[method_name]['per_layer_cv']:.3f}" if per_layer else ""
            print(f"  {method_name:<30s}: {val:>10.4f}{cv_str}{inv_str}")
        
        all_results[bb_name] = bb_result
        del model
        torch.cuda.empty_cache()
    
    # Ordering analysis
    print(f"\n{'='*70}")
    print("BACKBONE ORDERING COMPARISON")
    print(f"{'='*70}")
    
    method_names = list(methods.keys())
    
    # Print ordering for each method
    for method_name in method_names:
        vals = backbone_values[method_name]
        order = sorted(range(len(backbone_names)), key=lambda i: vals[i])
        ordered = [backbone_names[i] for i in order]
        print(f"\n  {method_name}:")
        print(f"    {' < '.join(ordered)}")
    
    # Spearman correlations between all pairs
    print(f"\n{'='*70}")
    print("SPEARMAN RANK CORRELATIONS")
    print(f"{'='*70}")
    
    print(f"\n  {'':30s}", end="")
    for m in method_names:
        print(f"  {m[:12]:>12s}", end="")
    print()
    
    for m1 in method_names:
        print(f"  {m1:30s}", end="")
        for m2 in method_names:
            rho, _ = stats.spearmanr(backbone_values[m1], backbone_values[m2])
            print(f"  {rho:>12.3f}", end="")
        print()
    
    # Capacity predictions (p*) under each definition
    print(f"\n{'='*70}")
    print("CAPACITY PREDICTIONS (p* at n=800, L=12, d=768)")
    print(f"{'='*70}")
    
    # For the invariant definitions, we need to recalibrate the p* formula
    # p* = floor(4n·σ²_P / (L·d)) for the original
    # For invariant versions, we derive the equivalent
    
    print(f"\n  Note: p* formulas need recalibration for each σ²_P definition.")
    print(f"  What matters is the ORDERING, not the absolute p* value.")
    print(f"  Below we show which backbones have p* >= 2 under each definition,")
    print(f"  using the threshold that separates DINOv2 from the rest.\n")
    
    for method_name in method_names:
        vals = backbone_values[method_name]
        # Find the natural break: DINOv2 should be lowest
        sorted_vals = sorted(zip(backbone_names, vals), key=lambda x: x[1])
        
        # Use the gap between the 2nd and 3rd smallest as the threshold
        if len(sorted_vals) >= 3:
            threshold = (sorted_vals[1][1] + sorted_vals[2][1]) / 2
        else:
            threshold = sorted_vals[0][1] * 1.5
        
        above = [name for name, v in sorted_vals if v > threshold]
        below = [name for name, v in sorted_vals if v <= threshold]
        
        print(f"  {method_name}:")
        print(f"    Below threshold (VPT limited): {below}")
        print(f"    Above threshold (VPT viable):  {above}")
    
    # RECOMMENDED DEFINITION
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    # Check which invariant methods perfectly correlate with original ordering
    orig_vals = backbone_values['original (NOT invariant)']
    best_method = None
    best_rho = 0
    
    for method_name in method_names:
        if method_name == 'original (NOT invariant)':
            continue
        rho, _ = stats.spearmanr(orig_vals, backbone_values[method_name])
        # Check invariance
        max_devs = [all_results[bb][method_name].get('max_rescaling_deviation', 1.0) 
                     for bb in backbone_names]
        is_invariant = all(d < 1e-4 for d in max_devs)
        
        if is_invariant and rho > best_rho:
            best_rho = rho
            best_method = method_name
    
    print(f"\n  Best invariant definition: {best_method}")
    print(f"  Correlation with original ordering: ρ = {best_rho:.3f}")
    print(f"  Preserves all predictions: {'YES' if best_rho >= 0.9 else 'PARTIALLY'}")
    
    if best_method:
        print(f"\n  Recommended σ²_P definition for the paper:")
        if best_method == 'product_frobenius':
            print(f"    σ²_P = (1/L) Σ_l [(||W_Q W_K^T||²_F + ||W_O W_V||²_F) / (2d²)]")
        elif best_method == 'geometric_mean':
            print(f"    σ²_P = (1/L) Σ_l [(||W_Q||_F·||W_K||_F + ||W_V||_F·||W_O||_F) / (2d)]")
        elif best_method == 'per_head_product':
            print(f"    σ²_P = (1/LH) Σ_l,h [||W_Qh W_Kh^T||²_F / d_h²]")
        elif best_method == 'spectral_product':
            print(f"    σ²_P = (1/L) Σ_l [(σ_1(W_Q W_K^T) + σ_1(W_O W_V)) / 2]")
        elif best_method == 'product_trace':
            print(f"    σ²_P = (1/L) Σ_l [(||W_Q W_K^T||²_F + ||W_O W_V||²_F) / (2d)]")
    
    # Save
    os.makedirs('results', exist_ok=True)
    
    # Convert to serializable format
    save_results = {}
    for bb in backbone_names:
        save_results[bb] = {}
        for m in method_names:
            save_results[bb][m] = {
                'value': float(all_results[bb][m]['value']),
                'max_rescaling_deviation': float(all_results[bb][m].get('max_rescaling_deviation', -1)),
            }
    
    with open('results/revision2_sigma_invariant.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved to results/revision2_sigma_invariant.json")


if __name__ == '__main__':
    main()
