"""
REVISION W6: Attention Perturbation Experiment

Reviewer concern: "Provide even preliminary evidence for the 'stability' 
hypothesis — e.g., measuring how much VPT prompts actually perturb attention 
distributions on DINO vs DeiT-III."

This experiment:
1. For each backbone, runs forward pass WITHOUT prompts → attention A_0
2. Runs forward pass WITH random VPT prompts → attention A_p
3. Computes KL(A_p || A_0), ||A_p - A_0||_F, and JS divergence
4. If DINO shows SMALLER perturbation → explains VPT resistance
   (DINO attention is "stable" / resistant to prompt perturbation)

This gives a training-free metric for Factor 3 (attention modifiability).

Usage:
    python revision_w6_attn_perturbation.py
    python revision_w6_attn_perturbation.py --tasks cifar10 svhn
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import setup_device
from run_all_backbones import BACKBONES, get_transforms, load_dataset


def measure_perturbation_simple(model, loader, device, num_prompts=5, max_batches=10):
    """
    Model-agnostic attention perturbation measurement.
    Instead of rebuilding forward_features, we hook into each block's input
    and prepend random tokens. Works for ALL models (incl. register models, DeiT).
    """
    model.eval()
    embed_dim = model.embed_dim
    num_layers = len(model.blocks)

    # ---- Pass 1: Normal forward, extract attention ----
    attn_base_all = [[] for _ in range(num_layers)]

    def make_attn_hook(layer_idx, storage):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            attn = F.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (d ** -0.5), dim=-1)
            # CLS→all attention (row 0)
            storage[layer_idx].append(attn[:, :, 0, :].detach().cpu())
        return hook_fn

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i, attn_base_all)))

    with torch.no_grad():
        for bi, (bx, _) in enumerate(loader):
            if bi >= max_batches:
                break
            model.forward_features(bx.to(device))

    for h in hooks:
        h.remove()

    # ---- Pass 2: Inject random prompts at each block input via hooks ----
    attn_prompted_all = [[] for _ in range(num_layers)]
    random_prompts = [torch.randn(1, num_prompts, embed_dim, device=device) * 0.02
                      for _ in range(num_layers)]

    def make_inject_hook(layer_idx):
        """Hook that prepends random tokens to block input."""
        def hook_fn(module, args):
            x = args[0]  # [B, N, d]
            prompt = random_prompts[layer_idx].expand(x.shape[0], -1, -1)
            x_new = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            return (x_new,) + args[1:]  # Replace input
        return hook_fn

    def make_remove_hook(layer_idx, np_):
        """Hook on block output to remove prompt tokens."""
        def hook_fn(module, input, output):
            # Remove the injected prompts from output
            return torch.cat([output[:, :1], output[:, np_+1:]], dim=1)
        return hook_fn

    inject_hooks = []
    remove_hooks = []
    attn_hooks = []
    for i, block in enumerate(model.blocks):
        inject_hooks.append(block.register_forward_pre_hook(make_inject_hook(i)))
        remove_hooks.append(block.register_forward_hook(make_remove_hook(i, num_prompts)))
        attn_hooks.append(block.attn.register_forward_hook(
            make_attn_hook(i, attn_prompted_all)))

    with torch.no_grad():
        for bi, (bx, _) in enumerate(loader):
            if bi >= max_batches:
                break
            model.forward_features(bx.to(device))

    for h in inject_hooks + remove_hooks + attn_hooks:
        h.remove()

    # ---- Compute divergences ----
    results_per_layer = []
    eps = 1e-10

    for l in range(num_layers):
        if not attn_base_all[l] or not attn_prompted_all[l]:
            continue

        a0 = torch.cat(attn_base_all[l], dim=0)   # [total_B, H, N_base]
        ap = torch.cat(attn_prompted_all[l], dim=0) # [total_B, H, N_prompted]

        # Extract CLS→patch attention (skip CLS self-attention, skip prompts)
        # Base: CLS attn to patches = columns 1: (skip col 0 = CLS self)
        p0 = a0[:, :, 1:]
        # Prompted: CLS attn to patches = columns after CLS and prompts
        pp_raw = ap[:, :, num_prompts+1:]

        # Truncate to same length
        min_len = min(p0.shape[-1], pp_raw.shape[-1])
        p0 = p0[:, :, :min_len].float() + eps
        pp = pp_raw[:, :, :min_len].float() + eps

        # Renormalize
        p0 = p0 / p0.sum(-1, keepdim=True)
        pp = pp / pp.sum(-1, keepdim=True)

        # KL(prompted || base)
        kl = (pp * (pp.log() - p0.log())).sum(-1).mean().item()

        # Jensen-Shannon
        m = 0.5 * (p0 + pp)
        js = 0.5 * (p0 * (p0.log() - m.log())).sum(-1).mean().item() + \
             0.5 * (pp * (pp.log() - m.log())).sum(-1).mean().item()

        # L2
        l2 = ((p0 - pp) ** 2).sum(-1).mean().item()

        # Total variation
        tv = 0.5 * (p0 - pp).abs().sum(-1).mean().item()

        results_per_layer.append({
            'layer': l, 'kl_divergence': kl, 'js_divergence': js,
            'l2_distance': l2, 'total_variation': tv,
        })

    mean_kl = np.mean([r['kl_divergence'] for r in results_per_layer])
    mean_js = np.mean([r['js_divergence'] for r in results_per_layer])
    mean_l2 = np.mean([r['l2_distance'] for r in results_per_layer])
    mean_tv = np.mean([r['total_variation'] for r in results_per_layer])

    return {
        'mean_kl': mean_kl, 'mean_js': mean_js,
        'mean_l2': mean_l2, 'mean_tv': mean_tv,
        'per_layer': results_per_layer,
    }


def run_backbone(backbone_key, task_name, device, num_prompts=5, max_batches=10):
    """Measure attention perturbation for one backbone on one task."""
    bb = BACKBONES[backbone_key]

    # Load model
    model = timm.create_model(bb['model'], pretrained=True,
                               img_size=bb['img_size']).to(device)
    model.eval()

    # Load data
    ds = load_dataset(task_name, bb['img_size'], max_samples=500)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Measure perturbation using hook-based approach (works for all models)
    divergences = measure_perturbation_simple(model, loader, device,
                                              num_prompts, max_batches)

    # Compute σ_P²
    total_norm, cnt = 0.0, 0
    embed_dim = model.embed_dim
    for name, param in model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total_norm += param.float().norm().item() ** 2
            cnt += 1
    d_h = embed_dim // model.blocks[0].attn.num_heads
    sigma_sq = total_norm / (cnt * d_h) if cnt > 0 else 1.0

    del model
    torch.cuda.empty_cache()

    return divergences, sigma_sq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['cifar10', 'svhn', 'gtsrb'])
    parser.add_argument('--prompts', type=int, default=5)
    args = parser.parse_args()
    
    device = setup_device()
    
    backbone_order = ['dinov2', 'dinov2reg', 'clip', 'supervised', 'mae', 'deit3']
    vpt_wins = {'dinov2': '0/9', 'dinov2reg': '0/9', 'clip': '1/9',
                'supervised': '2/9', 'mae': '1/9', 'deit3': '3/9'}
    
    print("=" * 75)
    print(f"REVISION W6: Attention Perturbation with {args.prompts} random VPT prompts")
    print("=" * 75)
    
    all_results = {}
    
    for task in args.tasks:
        print(f"\n{'=' * 65}")
        print(f"  Task: {task}")
        print(f"{'=' * 65}")
        print(f"\n  {'Backbone':<22s} {'KL div':>8s} {'JS div':>8s} {'L2':>8s} "
              f"{'TV':>8s} {'σ²_P':>6s} {'VPT wins':>10s}")
        print(f"  {'-'*72}")
        
        task_results = {}
        for bb_key in backbone_order:
            try:
                divs, sigma_sq = run_backbone(bb_key, task, device, args.prompts)
                
                print(f"  {BACKBONES[bb_key]['name']:<22s} "
                      f"{divs['mean_kl']:>8.5f} {divs['mean_js']:>8.5f} "
                      f"{divs['mean_l2']:>8.6f} {divs['mean_tv']:>8.5f} "
                      f"{sigma_sq:>6.1f} {vpt_wins.get(bb_key, '?'):>10s}")
                
                task_results[bb_key] = {
                    'sigma_sq': sigma_sq,
                    'divergences': divs,
                }
            except Exception as e:
                print(f"  {BACKBONES[bb_key]['name']:<22s} ERROR: {str(e)[:40]}")
        
        all_results[task] = task_results
        
        # Analyze: is DINO perturbation smaller?
        dino_kl = [task_results[k]['divergences']['mean_kl'] 
                   for k in ['dinov2', 'dinov2reg'] if k in task_results]
        non_dino_kl = [task_results[k]['divergences']['mean_kl']
                       for k in ['clip', 'supervised', 'mae', 'deit3'] if k in task_results]
        
        if dino_kl and non_dino_kl:
            print(f"\n  DINO mean KL:     {np.mean(dino_kl):.6f}")
            print(f"  Non-DINO mean KL: {np.mean(non_dino_kl):.6f}")
            ratio = np.mean(dino_kl) / np.mean(non_dino_kl)
            if ratio < 0.8:
                print(f"  → DINO attention is {1/ratio:.1f}× MORE STABLE (less perturbed)")
                print(f"  → SUPPORTS stability hypothesis for Factor 3")
            elif ratio > 1.2:
                print(f"  → DINO attention is {ratio:.1f}× LESS stable")
                print(f"  → CONTRADICTS stability hypothesis")
            else:
                print(f"  → Perturbation ratio ≈ {ratio:.2f} — no clear difference")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/revision_w6_attn_perturbation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"\nResults saved to results/revision_w6_attn_perturbation.json")


import os
if __name__ == '__main__':
    main()
