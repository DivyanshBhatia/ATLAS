"""
Compute σ_P² (prior variance) across pretrained ViT backbones.

No dataset needed — purely model-dependent metrics.
Computes σ_P², r*, p* and predicts VPT competitiveness.

Usage:
    python compute_sigma.py
    python compute_sigma.py --model vit_base_patch16_clip_224
"""
import argparse
import torch
import timm
import numpy as np


# Curated list of ViT-B backbones (all ~86M params, comparable)
VIT_BASE_MODELS = {
    # Self-supervised
    'vit_base_patch14_dinov2.lvd142m':          'DINOv2 ViT-B/14',
    'vit_base_patch14_reg4_dinov2.lvd142m':     'DINOv2-reg ViT-B/14',

    # Supervised (ImageNet)
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'Supervised ViT-B/16 (AugReg)',
    'vit_base_patch16_224.orig_in21k_ft_in1k':    'Supervised ViT-B/16 (Original)',
    'deit_base_patch16_224':                       'DeiT-B/16',
    'deit3_base_patch16_224':                      'DeiT-III-B/16',

    # CLIP / Contrastive
    'vit_base_patch16_clip_224':                'CLIP ViT-B/16 (OpenAI)',
    'vit_base_patch32_clip_224':                'CLIP ViT-B/32 (OpenAI)',
    'eva02_base_patch16_clip_224':              'EVA02-CLIP ViT-B/16',

    # MAE / Masked
    'vit_base_patch16_224.mae':                 'MAE ViT-B/16',

    # SAM
    'samvit_base_patch16_224':                  'SAM ViT-B/16',

    # BEiT
    'beit_base_patch16_224':                    'BEiT ViT-B/16',
    'beitv2_base_patch16_224':                  'BEiTv2 ViT-B/16',
}

# Also test some larger/smaller models
VIT_OTHER_MODELS = {
    'vit_small_patch14_dinov2.lvd142m':         'DINOv2 ViT-S/14',
    'vit_large_patch14_dinov2.lvd142m':         'DINOv2 ViT-L/14',
    'vit_large_patch14_clip_224':               'CLIP ViT-L/14',
    'vit_base_patch16_clip_quickgelu_224':      'CLIP-QG ViT-B/16',
}


def compute_model_metrics(model_name, display_name=None):
    """Compute σ_P² and derived quantities for a pretrained model."""
    if display_name is None:
        display_name = model_name

    try:
        model = timm.create_model(model_name, pretrained=True)
    except Exception as e:
        return None, str(e)

    # Get architecture params
    if hasattr(model, 'blocks'):
        num_layers = len(model.blocks)
        embed_dim = model.embed_dim
        if hasattr(model.blocks[0].attn, 'num_heads'):
            num_heads = model.blocks[0].attn.num_heads
        else:
            num_heads = 12
        head_dim = embed_dim // num_heads
    else:
        return None, "Not a standard ViT"

    # Compute σ_P² from attention weights
    total_norm_sq = 0.0
    count = 0
    weight_norms = []

    for name, param in model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight',
                                     'k_proj.weight', 'v_proj.weight',
                                     'q_proj.weight', 'out_proj.weight']):
            norm_sq = param.float().norm().item() ** 2
            total_norm_sq += norm_sq
            weight_norms.append((name.split('.')[-2], norm_sq))
            count += 1

    if count == 0:
        # Try broader search
        for name, param in model.named_parameters():
            if 'attn' in name and 'weight' in name and param.dim() == 2:
                norm_sq = param.float().norm().item() ** 2
                total_norm_sq += norm_sq
                count += 1

    if count == 0:
        return None, "No attention weights found"

    sigma_p_sq = total_norm_sq / (count * head_dim)

    # Derived quantities at n=800
    n = 800
    L = num_layers
    d = embed_dim
    d_h = head_dim

    r_star = max(1, int(2 * n * sigma_p_sq / (L * d_h)))
    p_star = max(1, int(4 * n * sigma_p_sq / (L * d)))

    # Total params
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        'model_name': model_name,
        'display_name': display_name,
        'sigma_p_sq': sigma_p_sq,
        'r_star': r_star,
        'p_star': p_star,
        'num_layers': L,
        'embed_dim': d,
        'head_dim': d_h,
        'num_heads': num_heads,
        'n_weight_matrices': count,
        'total_params_M': total_params,
        'vpt_viable': p_star >= 2,
    }

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return result, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Compute for a single model')
    parser.add_argument('--all', action='store_true',
                        help='Include non-base models')
    args = parser.parse_args()

    if args.model:
        print(f"Computing σ_P² for {args.model}...")
        result, err = compute_model_metrics(args.model)
        if err:
            print(f"  Error: {err}")
        else:
            r = result
            print(f"\n  Model:      {r['display_name']}")
            print(f"  Architecture: {r['num_layers']}L × {r['embed_dim']}d × {r['num_heads']}H (d_h={r['head_dim']})")
            print(f"  Parameters:   {r['total_params_M']:.1f}M")
            print(f"  σ_P²:         {r['sigma_p_sq']:.4f}")
            print(f"  r* (n=800):   {r['r_star']}")
            print(f"  p* (n=800):   {r['p_star']}")
            print(f"  VPT viable:   {'Yes' if r['vpt_viable'] else 'No (p*=1)'}")
        return

    models = dict(VIT_BASE_MODELS)
    if args.all:
        models.update(VIT_OTHER_MODELS)

    print(f"Computing σ_P² across {len(models)} pretrained ViT models...")
    print(f"(No dataset needed — purely model weights)\n")

    results = []
    errors = []

    for model_name, display_name in models.items():
        print(f"  {display_name}...", end=" ", flush=True)
        result, err = compute_model_metrics(model_name, display_name)
        if err:
            print(f"SKIP ({err[:50]})")
            errors.append((display_name, err))
        else:
            print(f"σ_P² = {result['sigma_p_sq']:.2f}, p* = {result['p_star']}")
            results.append(result)

    # Sort by σ_P²
    results.sort(key=lambda r: r['sigma_p_sq'])

    print(f"\n{'='*80}")
    print(f"σ_P² ACROSS PRETRAINED ViT BACKBONES (sorted, n=800)")
    print(f"{'='*80}")
    print(f"\n  {'Model':<35s} {'σ_P²':>7s} {'r*':>4s} {'p*':>4s} {'VPT?':>6s} {'Pretraining':>15s}")
    print(f"  {'-'*75}")

    for r in results:
        # Classify pretraining type
        name = r['model_name'].lower()
        if 'dino' in name:
            pt = 'Self-sup (DINO)'
        elif 'clip' in name or 'eva' in name:
            pt = 'Contrastive'
        elif 'mae' in name:
            pt = 'Masked (MAE)'
        elif 'beit' in name:
            pt = 'Masked (BEiT)'
        elif 'sam' in name:
            pt = 'Segmentation'
        elif 'deit' in name:
            pt = 'Supervised+Dist'
        else:
            pt = 'Supervised'

        vpt = "✓ Yes" if r['vpt_viable'] else "✗ No"
        print(f"  {r['display_name']:<35s} {r['sigma_p_sq']:>7.2f} {r['r_star']:>4d} "
              f"{r['p_star']:>4d} {vpt:>6s} {pt:>15s}")

    print(f"\n  Key insight:")
    print(f"  - Low σ_P² (self-supervised): VPT constrained → LoRA dominates")
    print(f"  - High σ_P² (supervised/contrastive): VPT viable → method selection matters")

    if errors:
        print(f"\n  Skipped {len(errors)} models (not available or incompatible)")


if __name__ == '__main__':
    main()
