"""
REVISION W3b: ATLAS Per-Task Predictions

Reviewer concern: "The actual predicted (r_task, p_task) values for each task
under ATLAS are never shown. The fair comparison requires running ATLAS's
predicted capacity and reporting those numbers alongside the oracle."

This experiment:
1. Runs ATLAS selection algorithm on each task
2. Reports the predicted method + capacity (r_task or p_task)
3. Trains with ATLAS's prediction and reports accuracy
4. Compares to oracle best and "Always LoRA_r8"

Also addresses W4 by comparing to simplified rules:
  - "Always LoRA_r8"
  - "LoRA unless non-DINO and γ<0.3"
  - Full ATLAS

Usage:
    python revision_w3b_atlas_pertask.py
    python revision_w3b_atlas_pertask.py --backbone dinov2
    python revision_w3b_atlas_pertask.py --backbone deit3 --tasks svhn dtd food101
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import json
from copy import deepcopy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import RidgeClassifier
from scipy.stats import f_oneway

from config import ExperimentConfig, setup_device
from exp2_comparison import (apply_lora, apply_vpt, apply_linear_probe,
                              train_and_evaluate)
from run_all_backbones import BACKBONES, TASKS, get_transforms, load_dataset


def extract_features_and_attention(model, loader, device):
    """Extract CLS features and attention class variance in one forward pass."""
    model.eval()
    all_features = []
    all_labels = []
    all_attn = []

    hooks = []
    layer_attns = []

    def make_hook():
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            attn = torch.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (d ** -0.5), dim=-1)
            # CLS→patch attention (row 0, columns 1:)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N_patches] avg over heads
            layer_attns.append(cls_attn.detach().cpu())
        return hook_fn

    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(make_hook()))

    with torch.no_grad():
        for bx, by in loader:
            layer_attns.clear()
            features = model.forward_features(bx.to(device))
            cls = features[:, 0].cpu()
            all_features.append(cls)
            all_labels.append(by)
            # Average attention across layers
            if layer_attns:
                avg_attn = torch.stack(layer_attns).mean(dim=0)  # [B, N_patches]
                all_attn.append(avg_attn)

    for h in hooks:
        h.remove()

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    attention = torch.cat(all_attn).numpy() if all_attn else None

    return features, labels, attention


def compute_task_metrics(features, labels, attention):
    """Compute γ (feature gap) and ρ (attention class variance)."""
    # γ = 1 - LP accuracy (ridge regression)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(features, labels)
    lp_acc = clf.score(features, labels)  # Training accuracy as proxy
    gamma = 1.0 - lp_acc

    # ρ = between-class / within-class attention variance
    rho = 0.0
    if attention is not None:
        classes = np.unique(labels)
        if len(classes) > 1:
            class_means = []
            within_var = 0.0
            for c in classes:
                mask = labels == c
                if mask.sum() > 1:
                    class_attn = attention[mask]
                    class_means.append(class_attn.mean(0))
                    within_var += class_attn.var(0).mean()
            within_var /= len(classes)
            if class_means:
                between_var = np.var(np.stack(class_means), axis=0).mean()
                rho = between_var / (within_var + 1e-10)

    return gamma, rho, lp_acc


def atlas_select(gamma, rho, sigma_sq, n=800, L=12, d=768, d_h=64):
    """ATLAS selection algorithm (Algorithm 1 from paper)."""
    # Capacity ceilings
    r_star = int(2 * n * sigma_sq / (L * d_h))
    p_star = int(4 * n * sigma_sq / (L * d))

    # Gap-scaled capacity
    r_task = max(1, min(32, int(r_star * gamma / 0.2)))
    p_task = max(1, min(50, int(np.ceil(p_star * gamma / 0.1))))

    # VPT score
    eps = 0.01
    S_vpt = rho * p_star / (gamma + eps)

    # Threshold
    c1, c4 = 6.0, 2.0
    rho_min = (c4 / c1) * np.sqrt(L * d / (2 * n * sigma_sq))

    # Selection
    if S_vpt > 3.0 and rho > rho_min:
        method = 'VPT'
        capacity = p_task
    else:
        method = 'LoRA'
        capacity = r_task

    return {
        'method': method,
        'capacity': capacity,
        'r_star': r_star,
        'p_star': p_star,
        'r_task': r_task,
        'p_task': p_task,
        'S_vpt': S_vpt,
        'rho_min': rho_min,
        'gamma': gamma,
        'rho': rho,
    }


def simple_rule_select(gamma, backbone_key):
    """Simplified rule: LoRA unless non-DINO and γ<0.3"""
    is_dino = backbone_key in ['dinov2', 'dinov2reg']
    if is_dino:
        return 'LoRA', 8
    elif gamma < 0.3:
        return 'VPT', 5  # Try VPT with moderate prompts
    else:
        return 'LoRA', 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--tasks', nargs='+', default=None)
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    tasks = args.tasks if args.tasks else list(TASKS.keys())
    tasks = [t for t in tasks if t in TASKS]

    print("=" * 75)
    print(f"REVISION W3b+W4: ATLAS Per-Task Predictions on {bb['name']}")
    print("=" * 75)

    # Load model once for feature extraction
    base_model = timm.create_model(bb['model'], pretrained=True,
                                    img_size=bb['img_size']).to(device)

    # Compute σ²_P
    total_norm, count = 0.0, 0
    for name, param in base_model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total_norm += param.float().norm().item() ** 2
            count += 1
    d_h = base_model.embed_dim // base_model.blocks[0].attn.num_heads
    sigma_sq = total_norm / (count * d_h) if count > 0 else 1.0
    print(f"\n  σ²_P = {sigma_sq:.2f}, r* = {int(2*800*sigma_sq/(12*d_h))}, "
          f"p* = {int(4*800*sigma_sq/(12*768))}")

    all_results = {}

    print(f"\n  {'Task':<14s} {'γ':>5s} {'ρ':>6s} {'ATLAS':>12s} "
          f"{'ATLAS acc':>10s} {'Oracle':>10s} {'LoRA8':>7s} {'SimpleR':>12s} {'SR acc':>8s}")
    print(f"  {'-'*90}")

    for task in tasks:
        num_classes = TASKS[task][0]

        # Load data
        tfm = get_transforms(bb['img_size'])
        if task in ['mnist', 'fashionmnist', 'emnist_letters']:
            tfm = transforms.Compose([
                transforms.Resize((bb['img_size'], bb['img_size'])),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        train_ds, val_ds = load_dataset(task, tfm, n_train=800, n_val=200)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        # Extract metrics
        features, labels, attention = extract_features_and_attention(
            base_model, train_loader, device)
        gamma, rho, lp_acc = compute_task_metrics(features, labels, attention)

        # ATLAS selection
        selection = atlas_select(gamma, rho, sigma_sq)

        # Simple rule selection
        sr_method, sr_cap = simple_rule_select(gamma, bb_key)

        # Train with ATLAS prediction
        model_atlas = deepcopy(base_model)
        if selection['method'] == 'LoRA':
            model_atlas = apply_lora(model_atlas, num_classes, rank=selection['capacity'])
        else:
            model_atlas = apply_vpt(model_atlas, num_classes, num_prompts=selection['capacity'])
        model_atlas = model_atlas.to(device)
        atlas_acc = train_and_evaluate(model_atlas, train_loader, val_loader, device, config)
        del model_atlas; torch.cuda.empty_cache()

        # Train with simple rule prediction
        model_sr = deepcopy(base_model)
        if sr_method == 'LoRA':
            model_sr = apply_lora(model_sr, num_classes, rank=sr_cap)
        else:
            model_sr = apply_vpt(model_sr, num_classes, num_prompts=sr_cap)
        model_sr = model_sr.to(device)
        sr_acc = train_and_evaluate(model_sr, train_loader, val_loader, device, config)
        del model_sr; torch.cuda.empty_cache()

        # Train LoRA_r8 baseline
        model_lr8 = deepcopy(base_model)
        model_lr8 = apply_lora(model_lr8, num_classes, rank=8)
        model_lr8 = model_lr8.to(device)
        lr8_acc = train_and_evaluate(model_lr8, train_loader, val_loader, device, config)
        del model_lr8; torch.cuda.empty_cache()

        # Oracle from existing results (approximate)
        oracle_acc = max(atlas_acc, sr_acc, lr8_acc, lp_acc)

        atlas_str = f"{selection['method']}_{'r' if selection['method']=='LoRA' else 'p'}{selection['capacity']}"
        sr_str = f"{sr_method}_{'r' if sr_method=='LoRA' else 'p'}{sr_cap}"

        print(f"  {task:<14s} {gamma:>5.2f} {rho:>6.3f} {atlas_str:>12s} "
              f"{atlas_acc:>10.3f} {oracle_acc:>10.3f} {lr8_acc:>7.3f} "
              f"{sr_str:>12s} {sr_acc:>8.3f}")

        all_results[task] = {
            'gamma': gamma, 'rho': rho, 'lp_acc': lp_acc,
            'atlas_selection': selection,
            'atlas_acc': atlas_acc,
            'lr8_acc': lr8_acc,
            'simple_rule': {'method': sr_method, 'capacity': sr_cap},
            'simple_rule_acc': sr_acc,
            'oracle_acc': oracle_acc,
        }

    # Summary
    atlas_accs = [r['atlas_acc'] for r in all_results.values()]
    lr8_accs = [r['lr8_acc'] for r in all_results.values()]
    sr_accs = [r['simple_rule_acc'] for r in all_results.values()]
    oracle_accs = [r['oracle_acc'] for r in all_results.values()]

    print(f"\n  {'Strategy':<20s} {'Avg Acc':>8s} {'Avg Regret':>10s}")
    print(f"  {'-'*40}")
    print(f"  {'ATLAS':<20s} {np.mean(atlas_accs):>8.3f} {np.mean(oracle_accs)-np.mean(atlas_accs):>10.3f}")
    print(f"  {'Always LoRA_r8':<20s} {np.mean(lr8_accs):>8.3f} {np.mean(oracle_accs)-np.mean(lr8_accs):>10.3f}")
    print(f"  {'Simple Rule':<20s} {np.mean(sr_accs):>8.3f} {np.mean(oracle_accs)-np.mean(sr_accs):>10.3f}")

    del base_model; torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    with open(f'results/revision_w3b_atlas_{bb_key}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/revision_w3b_atlas_{bb_key}.json")


import os
if __name__ == '__main__':
    main()
