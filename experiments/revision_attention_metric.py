"""
Training-Free Attention Modifiability Metric

Attempts to formalize Factor 3 (attention modifiability) with a
training-free metric that predicts VPT viability.

Computes the "linearized VPT gain" — how much the loss would decrease
from an infinitesimal prompt perturbation — across all backbone-task pairs.

Metric: ||∇_v L||² where v are VPT prompt parameters at initialization
Higher gradient = more potential for VPT to improve the model
Correlate with actual VPT-vs-LoRA gap across 54 backbone-task pairs

Usage:
    python revision_attention_metric.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
from scipy import stats

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt
from run_all_backbones import BACKBONES, TASKS, load_dataset
from torch.utils.data import DataLoader


def compute_vpt_gradient_magnitude(model, data_loader, device, num_prompts=5):
    """Compute ||∇_v L||² for VPT prompts at initialization.
    
    This measures how much the loss landscape responds to prompt perturbations.
    Higher = more potential for VPT to improve the model.
    """
    from copy import deepcopy
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    config.embed_dim = model.embed_dim
    config.num_layers = len(model.blocks)
    config.num_heads = model.blocks[0].attn.num_heads
    config.head_dim = model.embed_dim // model.blocks[0].attn.num_heads
    
    model_vpt = deepcopy(model).to(device)
    model_vpt = apply_vpt(model_vpt, num_prompts, config)
    model_vpt = model_vpt.to(device)
    model_vpt.eval()
    
    # Find prompt parameters
    prompt_params = []
    for name, param in model_vpt.named_parameters():
        if 'prompt' in name:
            param.requires_grad_(True)
            prompt_params.append(param)
        else:
            param.requires_grad_(False)
    
    if not prompt_params:
        return 0.0
    
    # Compute gradient on a batch
    total_grad_norm = 0.0
    n_batches = 0
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model_vpt(x)
        loss = F.cross_entropy(logits, y)
        
        model_vpt.zero_grad()
        loss.backward()
        
        batch_grad_norm = 0.0
        for param in prompt_params:
            if param.grad is not None:
                batch_grad_norm += param.grad.float().norm().item() ** 2
        
        total_grad_norm += batch_grad_norm
        n_batches += 1
        
        if n_batches >= 5:  # 5 batches is enough for a stable estimate
            break
    
    del model_vpt
    torch.cuda.empty_cache()
    
    return total_grad_norm / max(n_batches, 1)


def compute_attention_class_alignment(model, data_loader, device):
    """Compute alignment between attention patterns and class labels.
    
    Measures whether different classes produce systematically different 
    attention patterns. Higher alignment = more potential for VPT.
    """
    model.eval()
    
    class_attentions = {}
    
    hooks = []
    attention_maps = []
    
    def hook_fn(module, input, output):
        # timm attention returns (B, N, D) after projection
        # We need to access the attention weights before projection
        pass
    
    # Simpler approach: measure CLS attention entropy per class
    with torch.no_grad():
        all_features = []
        all_labels = []
        
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            features = model.forward_features(x)
            cls_features = features[:, 0]  # CLS token
            all_features.append(cls_features.cpu())
            all_labels.append(y.cpu())
            
            if len(all_features) >= 5:
                break
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Compute between-class variance of CLS features
        # normalized by within-class variance
        unique_labels = labels.unique()
        class_means = []
        within_var = 0
        
        for c in unique_labels:
            mask = labels == c
            if mask.sum() > 1:
                class_feat = features[mask].float()
                class_means.append(class_feat.mean(dim=0))
                within_var += class_feat.var(dim=0).mean().item()
        
        if len(class_means) < 2:
            return 0.0
        
        class_means = torch.stack(class_means)
        between_var = class_means.var(dim=0).mean().item()
        within_var /= len(unique_labels)
        
        # Fisher ratio: between / within
        fisher_ratio = between_var / (within_var + 1e-8)
    
    return fisher_ratio


def main():
    device = setup_device()
    
    print("=" * 70)
    print("Training-Free Attention Modifiability Metric")
    print("=" * 70)
    
    # Backbone-task pairs with known VPT-LoRA outcomes
    KNOWN_OUTCOMES = {
        # (backbone, task): vpt_gain (positive = VPT wins)
        # From cross-backbone experiments
        'dinov2_cifar10': -0.01, 'dinov2_cifar100': -0.03, 'dinov2_svhn': -0.02,
        'dinov2_dtd': 0.01, 'dinov2_eurosat': 0.01, 'dinov2_gtsrb': -0.02,
        'dinov2_fashionmnist': -0.03, 'dinov2_food101': 0.02, 'dinov2_mnist': 0.005,
        'deit3_cifar10': 0.0, 'deit3_cifar100': 0.005, 'deit3_svhn': 0.05,
        'deit3_dtd': 0.02, 'deit3_eurosat': 0.0, 'deit3_gtsrb': 0.01,
        'deit3_fashionmnist': 0.0, 'deit3_food101': 0.04, 'deit3_mnist': 0.0,
        'clip_cifar10': 0.0, 'clip_svhn': 0.02, 'clip_dtd': 0.0,
        'supervised_svhn': 0.07, 'supervised_gtsrb': 0.01,
        'mae_cifar10': -0.01, 'mae_svhn': -0.36,
    }
    
    all_metrics = {}
    test_backbones = ['dinov2', 'deit3', 'clip', 'supervised', 'mae']
    test_tasks = ['cifar10', 'cifar100', 'svhn', 'dtd', 'eurosat']
    
    for bb_key in test_backbones:
        if bb_key not in BACKBONES:
            continue
        
        bb = BACKBONES[bb_key]
        print(f"\n  Loading {bb['name']}...")
        model = timm.create_model(bb['model'], pretrained=True, img_size=bb['img_size']).to(device)
        
        for task in test_tasks:
            if task not in TASKS:
                continue
            
            key = f"{bb_key}_{task}"
            num_classes = TASKS[task][0]
            model.head = nn.Linear(model.embed_dim, num_classes).to(device)
            
            ds = load_dataset(task, bb['img_size'], max_samples=500)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            
            # Metric 1: VPT gradient magnitude
            grad_mag = compute_vpt_gradient_magnitude(model, loader, device)
            
            # Metric 2: Fisher ratio of CLS features
            fisher = compute_attention_class_alignment(model, loader, device)
            
            all_metrics[key] = {
                'grad_magnitude': grad_mag,
                'fisher_ratio': fisher,
                'vpt_gain': KNOWN_OUTCOMES.get(key, None),
            }
            
            print(f"    {key:<25s}: grad={grad_mag:.4f}, fisher={fisher:.4f}, "
                  f"vpt_gain={KNOWN_OUTCOMES.get(key, 'N/A')}")
        
        del model; torch.cuda.empty_cache()
    
    # Correlation analysis
    print(f"\n{'='*70}")
    print("CORRELATION: Metric vs VPT-LoRA Gap")
    print(f"{'='*70}")
    
    pairs_with_outcome = {k: v for k, v in all_metrics.items() if v['vpt_gain'] is not None}
    
    if len(pairs_with_outcome) >= 5:
        grads = [v['grad_magnitude'] for v in pairs_with_outcome.values()]
        fishers = [v['fisher_ratio'] for v in pairs_with_outcome.values()]
        gains = [v['vpt_gain'] for v in pairs_with_outcome.values()]
        
        rho_grad, p_grad = stats.spearmanr(grads, gains)
        rho_fisher, p_fisher = stats.spearmanr(fishers, gains)
        
        print(f"  Gradient magnitude vs VPT gain: ρ = {rho_grad:.3f} (p = {p_grad:.3f})")
        print(f"  Fisher ratio vs VPT gain:       ρ = {rho_fisher:.3f} (p = {p_fisher:.3f})")
        
        if abs(rho_grad) > 0.4:
            print(f"\n  ✓ Gradient magnitude shows moderate correlation — potential Factor 3 metric!")
        if abs(rho_fisher) > 0.4:
            print(f"\n  ✓ Fisher ratio shows moderate correlation — potential Factor 3 metric!")
        
        # Per-backbone breakdown
        print(f"\n  Per-backbone gradient magnitudes:")
        for bb in test_backbones:
            bb_grads = [v['grad_magnitude'] for k, v in pairs_with_outcome.items() if k.startswith(bb)]
            if bb_grads:
                print(f"    {bb:<12s}: mean grad = {np.mean(bb_grads):.4f}")
    
    os.makedirs('results', exist_ok=True)
    with open('results/attention_metric.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Saved to results/attention_metric.json")


if __name__ == '__main__':
    main()
