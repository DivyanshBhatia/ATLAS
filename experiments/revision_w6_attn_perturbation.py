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


class AttentionExtractor:
    """Hook-based attention extraction."""
    def __init__(self, model):
        self.attentions = []
        self.hooks = []
        for block in model.blocks:
            hook = block.attn.register_forward_hook(self._make_hook())
            self.hooks.append(hook)

    def _make_hook(self):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            attn = F.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (d ** -0.5), dim=-1)
            self.attentions.append(attn.detach())
        return hook_fn

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self.hooks:
            h.remove()


class VPTWrapper(nn.Module):
    """Minimal VPT wrapper that prepends learnable prompts."""
    def __init__(self, base_model, num_prompts, embed_dim, num_layers=12, random_init=True):
        super().__init__()
        self.base_model = base_model
        self.num_prompts = num_prompts
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_prompts, embed_dim) * 0.02)
            for _ in range(num_layers)
        ])
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def forward_features(self, x):
        """Modified forward that injects prompts at each layer."""
        x = self.base_model.patch_embed(x)
        cls_token = self.base_model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.base_model.pos_embed

        for i, block in enumerate(self.base_model.blocks):
            # Prepend prompts
            prompt = self.prompts[i].expand(x.shape[0], -1, -1)
            x_with_prompts = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            x_out = block(x_with_prompts)
            # Remove prompts for next layer
            x = torch.cat([x_out[:, :1], x_out[:, self.num_prompts+1:]], dim=1)

        return x


def compute_attention_divergence(attn_base, attn_prompted, eps=1e-10):
    """
    Compute multiple divergence measures between base and prompted attention.
    Input: attn_base, attn_prompted: lists of [B, H, N, N] tensors per layer
    """
    results_per_layer = []
    
    for l, (a0, ap) in enumerate(zip(attn_base, attn_prompted)):
        # CLS token attention (row 0) — most relevant for classification
        # a0: [B, H, N_base, N_base], ap: [B, H, N_prompted, N_prompted]
        # We need to extract CLS→patch attention (excluding prompt tokens)
        
        # For base model: CLS is token 0, patches are tokens 1:
        cls_attn_base = a0[:, :, 0, 1:]  # [B, H, N_patches]
        
        # For prompted model: CLS is token 0, prompts are 1:p+1, patches are p+1:
        n_prompts = ap.shape[2] - a0.shape[2]
        if n_prompts > 0:
            cls_attn_prompted = ap[:, :, 0, n_prompts+1:]  # [B, H, N_patches]
            # Renormalize to sum to 1 over patches only
            cls_attn_prompted = cls_attn_prompted / (cls_attn_prompted.sum(-1, keepdim=True) + eps)
        else:
            cls_attn_prompted = ap[:, :, 0, 1:]

        # Normalize base attention
        cls_attn_base = cls_attn_base / (cls_attn_base.sum(-1, keepdim=True) + eps)

        # Truncate to same length
        min_len = min(cls_attn_base.shape[-1], cls_attn_prompted.shape[-1])
        p0 = cls_attn_base[:, :, :min_len] + eps
        pp = cls_attn_prompted[:, :, :min_len] + eps
        
        # Renormalize
        p0 = p0 / p0.sum(-1, keepdim=True)
        pp = pp / pp.sum(-1, keepdim=True)

        # KL(prompted || base) — how much prompts change the attention
        kl = (pp * (pp.log() - p0.log())).sum(-1).mean().item()
        
        # Symmetric KL (Jensen-Shannon)
        m = 0.5 * (p0 + pp)
        js = 0.5 * (p0 * (p0.log() - m.log())).sum(-1).mean().item() + \
             0.5 * (pp * (pp.log() - m.log())).sum(-1).mean().item()
        
        # L2 distance
        l2 = ((p0 - pp) ** 2).sum(-1).mean().item()
        
        # Total variation
        tv = 0.5 * (p0 - pp).abs().sum(-1).mean().item()

        results_per_layer.append({
            'layer': l,
            'kl_divergence': kl,
            'js_divergence': js,
            'l2_distance': l2,
            'total_variation': tv,
        })

    # Aggregate across layers
    mean_kl = np.mean([r['kl_divergence'] for r in results_per_layer])
    mean_js = np.mean([r['js_divergence'] for r in results_per_layer])
    mean_l2 = np.mean([r['l2_distance'] for r in results_per_layer])
    mean_tv = np.mean([r['total_variation'] for r in results_per_layer])
    
    return {
        'mean_kl': mean_kl,
        'mean_js': mean_js,
        'mean_l2': mean_l2,
        'mean_tv': mean_tv,
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
    tfm = get_transforms(bb['img_size'])
    if task_name in ['mnist', 'fashionmnist']:
        tfm = transforms.Compose([
            transforms.Resize((bb['img_size'], bb['img_size'])),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    train_ds, _ = load_dataset(task_name, tfm, n_train=500, n_val=100)
    loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # === Pass 1: Base model (no prompts) ===
    extractor_base = AttentionExtractor(model)
    all_attn_base = [[] for _ in range(len(model.blocks))]
    
    with torch.no_grad():
        for bi, (bx, _) in enumerate(loader):
            if bi >= max_batches:
                break
            extractor_base.clear()
            model.forward_features(bx.to(device))
            for l, attn in enumerate(extractor_base.attentions):
                all_attn_base[l].append(attn.cpu())
    
    extractor_base.remove()
    
    # Concatenate across batches
    attn_base = [torch.cat(layer_attns, dim=0) for layer_attns in all_attn_base]
    
    # === Pass 2: Model with random VPT prompts ===
    embed_dim = model.embed_dim
    num_layers = len(model.blocks)
    vpt_model = VPTWrapper(model, num_prompts, embed_dim, num_layers).to(device)
    vpt_model.eval()
    
    extractor_vpt = AttentionExtractor(vpt_model.base_model)
    all_attn_vpt = [[] for _ in range(num_layers)]
    
    with torch.no_grad():
        for bi, (bx, _) in enumerate(loader):
            if bi >= max_batches:
                break
            extractor_vpt.clear()
            vpt_model.forward_features(bx.to(device))
            for l, attn in enumerate(extractor_vpt.attentions):
                all_attn_vpt[l].append(attn.cpu())
    
    extractor_vpt.remove()
    
    attn_prompted = [torch.cat(layer_attns, dim=0) for layer_attns in all_attn_vpt]
    
    # === Compute divergences ===
    divergences = compute_attention_divergence(attn_base, attn_prompted)
    
    # Compute σ_P²
    total_norm, cnt = 0.0, 0
    for name, param in model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total_norm += param.float().norm().item() ** 2
            cnt += 1
    d_h = embed_dim // model.blocks[0].attn.num_heads
    sigma_sq = total_norm / (cnt * d_h) if cnt > 0 else 1.0
    
    del model, vpt_model
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
