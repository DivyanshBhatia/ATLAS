"""
σ²_P Ablation: Is the full formula needed, or do simpler statistics work?

Reviewer: "ablate against simpler candidates (e.g., ‖W_Q‖‖W_K‖ alone, 
LN gain alone) to show the full formula is needed."

Computes multiple statistics per backbone and correlates each with
the max LR drop from the 7-point sweep data.

Usage:
    cd /content/ATLAS
    python experiments/sigma_ablation.py
"""
import sys
sys.path.insert(0, '.')

import torch
import timm
import numpy as np
import json
import os

BACKBONES = {
    'iBOT': (None, '/content/checkpoint_teacher.pth'),
    'CLIP': ('vit_base_patch16_clip_224.openai', None),
    'DINOv1': ('vit_base_patch16_224.dino', None),
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', None),
    'DINOv2-reg': ('vit_base_patch14_reg4_dinov2.lvd142m', None),
    'DeiT-III': ('deit3_base_patch16_224', None),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', None),
    'MAE': ('vit_base_patch16_224.mae', None),
    'MoCo-v3': (None, None),
}

# Max drop at LR=1e-2 from 7-point sweeps (across all tasks)
MAX_DROPS = {
    'iBOT': -10.0,
    'CLIP': -51.5,
    'DINOv1': -9.0,
    'DINOv2': -60.5,
    'DINOv2-reg': -7.5,
    'DeiT-III': -7.5,
    'Supervised': -4.0,
    'MAE': -1.5,
    'MoCo-v3': 2.5,
}


def load_model(name, ibot_checkpoint=None):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model
    elif name == 'iBOT':
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        candidates = [ibot_checkpoint, '/content/ibot/checkpoint_teacher.pth',
                      '/content/checkpoint_teacher.pth']
        ckpt_path = next((x for x in candidates if x and os.path.isfile(x)), None)
        if ckpt_path is None:
            print(f"  iBOT checkpoint not found, skipping")
            return None
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint.get('teacher', checkpoint.get('state_dict', checkpoint))
        cleaned = {}
        for key, value in sd.items():
            new_key = key
            for prefix in ('module.', 'teacher.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if not new_key.startswith(('head.', 'last_layer.')):
                cleaned[new_key] = value
        model.load_state_dict(cleaned, strict=False)
        return model
    else:
        return timm.create_model(BACKBONES[name][0], pretrained=True)


def compute_statistics(model, name):
    """Compute multiple weight statistics for a backbone."""
    stats = {}
    L = len(model.blocks)
    d = model.embed_dim

    qk_products = []
    vo_products = []
    q_norms = []
    k_norms = []
    v_norms = []
    o_norms = []
    ln_gains = []

    for l, block in enumerate(model.blocks):
        attn = block.attn
        
        # Extract Q, K, V, O weights
        if hasattr(attn, 'qkv'):
            qkv_w = attn.qkv.weight.data  # (3*d, d)
            Wq = qkv_w[:d]
            Wk = qkv_w[d:2*d]
            Wv = qkv_w[2*d:]
        else:
            Wq = attn.q_proj.weight.data if hasattr(attn, 'q_proj') else None
            Wk = attn.k_proj.weight.data if hasattr(attn, 'k_proj') else None
            Wv = attn.v_proj.weight.data if hasattr(attn, 'v_proj') else None

        Wo = attn.proj.weight.data  # (d, d)

        if Wq is not None:
            nq = torch.norm(Wq, 'fro').item()
            nk = torch.norm(Wk, 'fro').item()
            nv = torch.norm(Wv, 'fro').item()
            no = torch.norm(Wo, 'fro').item()

            q_norms.append(nq)
            k_norms.append(nk)
            v_norms.append(nv)
            o_norms.append(no)
            qk_products.append(nq * nk)
            vo_products.append(nv * no)

        # LayerNorm gains
        if hasattr(block, 'norm1'):
            ln_gain = block.norm1.weight.data.norm().item()
            ln_gains.append(ln_gain)

    # σ²_P (full formula)
    sigma_p = sum(qk + vo for qk, vo in zip(qk_products, vo_products)) / (2 * L * d)
    stats['sigma_p_full'] = sigma_p

    # Ablation 1: QK product only
    sigma_qk = sum(qk_products) / (L * d)
    stats['sigma_qk_only'] = sigma_qk

    # Ablation 2: VO product only
    sigma_vo = sum(vo_products) / (L * d)
    stats['sigma_vo_only'] = sigma_vo

    # Ablation 3: Mean Q norm only
    stats['mean_q_norm'] = np.mean(q_norms)

    # Ablation 4: Mean K norm only
    stats['mean_k_norm'] = np.mean(k_norms)

    # Ablation 5: QK product (no division by d)
    stats['qk_raw'] = np.mean(qk_products)

    # Ablation 6: LN gain
    stats['ln_gain_mean'] = np.mean(ln_gains) if ln_gains else 0

    # Ablation 7: LN gain × QK
    stats['ln_qk'] = np.mean(ln_gains) * np.mean(qk_products) if ln_gains else 0

    # Ablation 8: Total weight norm
    total_norm = sum(p.norm().item() for p in model.parameters())
    stats['total_weight_norm'] = total_norm

    # Ablation 9: Attention logit scale proxy (QK / sqrt(head_dim))
    head_dim = d // model.blocks[0].attn.num_heads
    stats['attn_logit_scale'] = np.mean(qk_products) / np.sqrt(head_dim)

    return stats


def main():
    print(f"{'='*70}")
    print("  σ²_P ABLATION: Which statistic best predicts LR sensitivity?")
    print(f"{'='*70}")

    all_stats = {}
    for name, (model_name, ckpt) in BACKBONES.items():
        print(f"\n  Loading {name}...")
        model = load_model(name, ckpt)
        if model is None:
            continue
        stats = compute_statistics(model, name)
        stats['max_drop'] = MAX_DROPS[name]
        all_stats[name] = stats
        del model

        print(f"    σ²_P (full):   {stats['sigma_p_full']:.4f}")
        print(f"    σ²_QK only:    {stats['sigma_qk_only']:.4f}")
        print(f"    σ²_VO only:    {stats['sigma_vo_only']:.4f}")
        print(f"    LN gain:       {stats['ln_gain_mean']:.4f}")
        print(f"    Attn logit:    {stats['attn_logit_scale']:.4f}")
        print(f"    Max drop:      {stats['max_drop']:.1f}")

    # Compute correlations
    print(f"\n{'='*70}")
    print("  CORRELATION WITH MAX DROP AT LR=1e-2")
    print(f"{'='*70}")

    drops = [all_stats[bb]['max_drop'] for bb in all_stats]
    
    stat_names = [
        ('sigma_p_full', 'σ²_P (full formula)'),
        ('sigma_qk_only', 'QK product only'),
        ('sigma_vo_only', 'VO product only'),
        ('mean_q_norm', 'Mean ‖W_Q‖ only'),
        ('mean_k_norm', 'Mean ‖W_K‖ only'),
        ('ln_gain_mean', 'LN gain only'),
        ('ln_qk', 'LN gain × QK'),
        ('total_weight_norm', 'Total weight norm'),
        ('attn_logit_scale', 'Attn logit scale'),
    ]

    print(f"\n  {'Statistic':<25s} {'Corr (Pearson)':>15s} {'Corr (Spearman)':>16s} {'Separates?':>12s}")
    print("  " + "-" * 70)

    for key, label in stat_names:
        values = [all_stats[bb][key] for bb in all_stats]
        
        # Pearson correlation
        pearson = np.corrcoef(values, drops)[0, 1]
        
        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(values, drops)
        
        # Does it separate crashing (drop < -10) from non-crashing?
        crash_vals = [v for v, d in zip(values, drops) if d < -10]
        safe_vals = [v for v, d in zip(values, drops) if d >= -10]
        separates = "Yes" if max(crash_vals) < min(safe_vals) else "No"
        
        print(f"  {label:<25s} {pearson:>15.3f} {spearman:>16.3f} {separates:>12s}")

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/sigma_ablation.json', 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n  Saved to results/sigma_ablation.json")
    print(f"\n  KEY QUESTION: Does any simpler statistic separate as cleanly as σ²_P?")


if __name__ == '__main__':
    main()
