"""
Per-layer variance analysis to validate/invalidate isotropy assumption.

Measures σ²_l for each layer's attention weights across backbones.
If variance is relatively uniform across layers, isotropy holds.
If it varies by 10×+, the isotropic prior is substantially distortive.
"""
import torch
import timm
import numpy as np

BACKBONES = {
    'dinov2': ('vit_base_patch14_dinov2.lvd142m', 518),
    'deit3': ('deit3_base_patch16_224.fb_in1k', 224),
    'clip': ('vit_base_patch16_clip_224.openai', 224),
    'supervised': ('vit_base_patch16_224.augreg_in1k', 224),
    'mae': ('vit_base_patch16_224.mae', 224),
}

def analyze_per_layer_variance(model_name, img_size):
    model = timm.create_model(model_name, pretrained=True, img_size=img_size)
    model.eval()
    
    d_h = model.embed_dim // model.blocks[0].attn.num_heads
    
    layer_sigmas = []
    layer_details = []
    
    for i, block in enumerate(model.blocks):
        # Collect norms from EXACTLY the same weights as compute_sigma_p_sq
        norms = {}
        for name, param in block.attn.named_parameters():
            if 'qkv' in name and 'weight' in name:
                norms['qkv.weight'] = param.float().norm().item() ** 2
            elif 'proj' in name and 'weight' in name:
                norms['proj.weight'] = param.float().norm().item() ** 2
        
        # σ²_l = average squared norm / d_h  
        total_norm = sum(norms.values())
        count = len(norms)
        sigma_l = total_norm / (count * d_h)
        layer_sigmas.append(sigma_l)
        layer_details.append(norms)
    
    return layer_sigmas, layer_details


print("=" * 80)
print("Per-Layer Variance Analysis: Validating Isotropy Assumption")
print("=" * 80)

for bb_name, (model_name, img_size) in BACKBONES.items():
    print(f"\n{'='*60}")
    print(f"  {bb_name}: {model_name}")
    print(f"{'='*60}")
    
    try:
        sigmas, details = analyze_per_layer_variance(model_name, img_size)
        
        sigmas = np.array(sigmas)
        mean_sigma = np.mean(sigmas)
        std_sigma = np.std(sigmas)
        cv = std_sigma / mean_sigma  # coefficient of variation
        ratio = np.max(sigmas) / np.min(sigmas)
        
        print(f"  Mean σ²_l = {mean_sigma:.3f}")
        print(f"  Std  σ²_l = {std_sigma:.3f}")
        print(f"  CV (std/mean) = {cv:.3f}")
        print(f"  Max/Min ratio = {ratio:.2f}×")
        print(f"  Per-layer σ²_l:")
        for i, s in enumerate(sigmas):
            bar = '█' * int(s / mean_sigma * 20)
            print(f"    Layer {i:>2d}: {s:>7.3f}  ({s/mean_sigma:.2f}× mean)  {bar}")
        
        if ratio < 3:
            print(f"\n  → Isotropy REASONABLE: max/min = {ratio:.1f}× (< 3×)")
        elif ratio < 10:
            print(f"\n  → Isotropy APPROXIMATE: max/min = {ratio:.1f}× (moderate variation)")
        else:
            print(f"\n  → Isotropy QUESTIONABLE: max/min = {ratio:.1f}× (> 10×)")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If CV < 0.3 and max/min < 3× for most backbones: isotropy is a")
print("reasonable approximation. The capacity formulas use the MEAN σ²_P,")
print("which correctly predicts backbone-level behavior even if individual")
print("layers vary moderately.")
