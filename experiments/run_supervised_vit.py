"""
Cross-Backbone Validation: Supervised ViT-B/16

Runs key experiments on supervised ViT-B/16 (ImageNet-21K) to show
the theory generalizes beyond DINOv2. Tests 7 tasks covering all
regions of the prediction space.

Experiments:
  1. Task structure (exp5): feature_gap + attn_var for each task
  2. PEFT comparison (exp2): LP, LoRA, VPT, Adapter at n=800
  3. Selection algorithm: theory-derived thresholds with new σ_P²

Usage:
    python run_supervised_vit.py
    python run_supervised_vit.py --exp task_structure
    python run_supervised_vit.py --exp comparison
    python run_supervised_vit.py --exp all
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
import os
import time
from copy import deepcopy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.linear_model import LogisticRegression

from config import ExperimentConfig, setup_device, ensure_dirs
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


# Supervised ViT-B/16 config
SUP_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
SUP_IMG_SIZE = 224
SUP_EMBED_DIM = 768
SUP_NUM_LAYERS = 12
SUP_NUM_HEADS = 12
SUP_HEAD_DIM = 64

TASKS = {
    'cifar10':       (10,  'natural'),
    'cifar100':      (100, 'natural'),
    'svhn':          (10,  'structured'),
    'gtsrb':         (43,  'structured'),
    'mnist':         (10,  'structured'),
    'fashionmnist':  (10,  'structured'),
    'eurosat':       (10,  'specialized'),
}


def get_transforms():
    rgb = transforms.Compose([
        transforms.Resize((SUP_IMG_SIZE, SUP_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gray = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((SUP_IMG_SIZE, SUP_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return rgb, gray


def load_dataset(task_name, max_samples=2000):
    rgb, gray = get_transforms()
    loaders = {
        'cifar10': lambda: datasets.CIFAR10('./data', True, download=True, transform=rgb),
        'cifar100': lambda: datasets.CIFAR100('./data', True, download=True, transform=rgb),
        'svhn': lambda: datasets.SVHN('./data', 'train', download=True, transform=rgb),
        'gtsrb': lambda: datasets.GTSRB('./data', 'train', download=True, transform=rgb),
        'mnist': lambda: datasets.MNIST('./data', True, download=True, transform=gray),
        'fashionmnist': lambda: datasets.FashionMNIST('./data', True, download=True, transform=gray),
        'eurosat': lambda: datasets.EuroSAT('./data', download=True, transform=rgb),
    }
    ds = loaders[task_name]()
    if len(ds) > max_samples:
        indices = torch.randperm(len(ds))[:max_samples].tolist()
        ds = Subset(ds, indices)
    return ds


def compute_sigma_p_sq(model):
    total, count = 0.0, 0
    for name, param in model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total += param.float().norm().item() ** 2
            count += 1
    return total / (count * SUP_HEAD_DIM) if count > 0 else 1.0


def extract_features_and_attention(model, dataloader, device, max_batches=30):
    model.eval()
    all_features, all_labels = [], []
    attn_per_layer = {}
    hooks = []
    capture = [True]

    def make_hook(idx):
        def fn(module, inp, out):
            if not capture[0]:
                return
            B, N, C = inp[0].shape
            H = module.num_heads
            d = C // H
            qkv = module.qkv(inp[0]).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            attn = F.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (d ** -0.5), dim=-1)
            if idx not in attn_per_layer:
                attn_per_layer[idx] = []
            attn_per_layer[idx].append(attn.detach().cpu())
        return fn

    for idx, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for bi, (bx, by) in enumerate(dataloader):
            if bi >= max_batches:
                capture[0] = False
            bx = bx.to(device)
            out = model.forward_features(bx)
            all_features.append(out[:, 0].cpu())
            all_labels.append(by)

    for h in hooks:
        h.remove()
    for k in attn_per_layer:
        attn_per_layer[k] = torch.cat(attn_per_layer[k], dim=0)

    return torch.cat(all_features), torch.cat(all_labels).numpy(), attn_per_layer


def run_task_structure(model, device):
    """Exp5 equivalent: task structure metrics on supervised ViT-B."""
    print("\n" + "=" * 70)
    print("TASK STRUCTURE — Supervised ViT-B/16")
    print("=" * 70)

    results = {}
    for task_name, (n_classes, category) in TASKS.items():
        print(f"\n  {task_name} ({n_classes} classes)...")

        try:
            ds = load_dataset(task_name, max_samples=2000)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

            features, labels, attn = extract_features_and_attention(
                model, dl, device, max_batches=30)

            # LP accuracy
            n = len(labels)
            split = int(0.8 * n)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(features[:split].numpy(), labels[:split])
            lp_acc = float(clf.score(features[split:].numpy(), labels[split:]))

            # Attention class variance
            n_attn = attn[0].shape[0] if 0 in attn else 0
            attn_labels = labels[:n_attn]
            classes = np.unique(attn_labels)

            ratios = []
            for l in range(SUP_NUM_LAYERS):
                if l not in attn:
                    continue
                cls_attn = attn[l][:, :, 0, 1:]
                cms = []
                for c in classes:
                    m = attn_labels == c
                    if m.sum() > 0:
                        cms.append(cls_attn[m].mean(0))
                if len(cms) < 2:
                    continue
                cm = torch.stack(cms)
                gm = cm.mean(0, keepdim=True)
                bv = ((cm - gm) ** 2).mean().item()
                wvs = []
                for ci, c in enumerate(classes):
                    m = attn_labels == c
                    if m.sum() > 1:
                        wvs.append(((cls_attn[m] - cm[ci:ci+1]) ** 2).mean().item())
                wv = np.mean(wvs) if wvs else 1e-10
                ratios.append(bv / (wv + 1e-10))

            attn_var = float(np.mean(ratios)) if ratios else 0
            gap = 1.0 - lp_acc

            print(f"    LP={lp_acc:.3f}  gap={gap:.3f}  attn_var={attn_var:.4f}")

            results[task_name] = {
                'category': category,
                'lp_accuracy': lp_acc,
                'feature_gap': gap,
                'attention_variance': attn_var,
            }

            del features, labels, attn
            import gc; gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print(f"\n  {'Task':<15s} {'LP Acc':>7s} {'Gap':>6s} {'Attn Var':>9s}")
    print(f"  {'-'*40}")
    for t, r in sorted(results.items(), key=lambda x: -x[1]['attention_variance']):
        print(f"  {t:<15s} {r['lp_accuracy']:>7.3f} {r['feature_gap']:>6.3f} "
              f"{r['attention_variance']:>9.4f}")

    return results


def run_comparison(model, config, device):
    """Exp2 equivalent: PEFT comparison on supervised ViT-B."""
    print("\n" + "=" * 70)
    print("PEFT COMPARISON — Supervised ViT-B/16")
    print("=" * 70)

    base_model = model
    results = {}

    for task_name, (n_classes, category) in TASKS.items():
        print(f"\n{'='*55}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*55}")

        try:
            ds = load_dataset(task_name, max_samples=1000)
            n_total = len(ds)
            n_val = min(200, n_total // 5)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                      shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                    shuffle=False, num_workers=2)

            print(f"  {n_train} train / {n_val} val")
            task_results = {'category': category, 'methods': {}}

            methods = [
                ('LP', lambda m: apply_linear_probe(m, config)),
                ('LoRA_r1', lambda m: apply_lora(m, 1, config)),
                ('LoRA_r4', lambda m: apply_lora(m, 4, config)),
                ('LoRA_r8', lambda m: apply_lora(m, 8, config)),
                ('LoRA_r16', lambda m: apply_lora(m, 16, config)),
                ('VPT_p1', lambda m: apply_vpt(m, 1, config)),
                ('VPT_p5', lambda m: apply_vpt(m, 5, config)),
                ('VPT_p10', lambda m: apply_vpt(m, 10, config)),
                ('VPT_p20', lambda m: apply_vpt(m, 20, config)),
                ('Adapter_r8', lambda m: apply_adapter(m, 8, config)),
            ]

            for method_name, apply_fn in methods:
                mdl = deepcopy(base_model).to(device)
                mdl.head = nn.Linear(SUP_EMBED_DIM, n_classes).to(device)
                mdl = apply_fn(mdl)
                acc = train_and_evaluate(mdl, train_loader, val_loader,
                                         config, device, method_name)
                task_results['methods'][method_name] = {'accuracy': acc}
                del mdl

            # Summary
            m = task_results['methods']
            best = max(m, key=lambda k: m[k]['accuracy'])
            bl = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
            bv = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
            w = "LoRA" if bl > bv + 0.01 else "VPT" if bv > bl + 0.01 else "TIE"

            print(f"\n  Best: {best} ({m[best]['accuracy']:.3f})")
            print(f"  LoRA best: {bl:.3f}, VPT best: {bv:.3f} → {w}")

            task_results['best_lora'] = bl
            task_results['best_vpt'] = bv
            task_results['winner'] = w
            results[task_name] = task_results

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    # Final table
    print(f"\n{'='*70}")
    print(f"SUPERVISED ViT-B/16 — LoRA vs VPT")
    print(f"{'='*70}")
    print(f"\n  {'Task':<15s} {'LP':>6s} {'LoRA':>6s} {'VPT':>6s} {'Winner':>8s}")
    print(f"  {'-'*45}")
    for t, r in results.items():
        m = r['methods']
        print(f"  {t:<15s} {m['LP']['accuracy']:>6.3f} {r['best_lora']:>6.3f} "
              f"{r['best_vpt']:>6.3f} {r['winner']:>8s}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['task_structure', 'comparison', 'all'])
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    # Override config for supervised ViT-B
    config.model_name = SUP_MODEL
    config.img_size = SUP_IMG_SIZE
    config.embed_dim = SUP_EMBED_DIM
    config.num_layers = SUP_NUM_LAYERS
    config.num_heads = SUP_NUM_HEADS
    config.head_dim = SUP_HEAD_DIM

    print(f"Loading {SUP_MODEL}...")
    model = timm.create_model(SUP_MODEL, pretrained=True, img_size=SUP_IMG_SIZE).to(device)

    sigma_p_sq = compute_sigma_p_sq(model)
    print(f"σ_P² = {sigma_p_sq:.4f}")

    # Theory-derived thresholds for this model
    c1, c4 = 6.0, 2.0
    n = 800
    gamma_lp = (1/c1) * np.sqrt(SUP_NUM_LAYERS * SUP_HEAD_DIM / (n * sigma_p_sq))
    gamma_vpt = (1/c1) * np.sqrt(SUP_NUM_LAYERS * SUP_EMBED_DIM / (2 * n * sigma_p_sq))
    rho_min = c4 * gamma_vpt
    r_star = max(1, int(2 * n * sigma_p_sq / (SUP_NUM_LAYERS * SUP_HEAD_DIM)))
    p_star = max(1, int(4 * n * sigma_p_sq / (SUP_NUM_LAYERS * SUP_EMBED_DIM)))

    print(f"\nTheory thresholds (supervised ViT-B):")
    print(f"  γ_LP  = {gamma_lp:.4f}")
    print(f"  γ_VPT = {gamma_vpt:.4f}")
    print(f"  ρ_min = {rho_min:.4f}")
    print(f"  r*    = {r_star}")
    print(f"  p*    = {p_star}")

    print(f"\nDINOv2 thresholds (for comparison):")
    dino_sp = 5.08
    print(f"  γ_LP  = {(1/c1) * np.sqrt(12*64/(800*dino_sp)):.4f}")
    print(f"  γ_VPT = {(1/c1) * np.sqrt(12*768/(2*800*dino_sp)):.4f}")

    all_results = {'model': SUP_MODEL, 'sigma_p_sq': sigma_p_sq}

    if args.exp in ('task_structure', 'all'):
        ts_results = run_task_structure(model, device)
        all_results['task_structure'] = ts_results

    if args.exp in ('comparison', 'all'):
        comp_results = run_comparison(model, config, device)
        all_results['comparison'] = comp_results

    # Compare with DINOv2 if both available
    if 'task_structure' in all_results and 'comparison' in all_results:
        print(f"\n{'='*70}")
        print(f"CROSS-BACKBONE: DINOv2 vs Supervised ViT-B")
        print(f"{'='*70}")

        ts = all_results['task_structure']
        comp = all_results['comparison']

        print(f"\n  {'Task':<15s} {'Sup LP':>7s} {'Sup LoRA>VPT?':>14s} "
              f"{'DINOv2 LP':>10s} {'Same winner?':>13s}")
        print(f"  {'-'*60}")

        # DINOv2 results for comparison
        dinov2_winners = {
            'cifar10': 'TIE', 'cifar100': 'LoRA', 'svhn': 'LoRA',
            'gtsrb': 'LoRA', 'mnist': 'LoRA', 'fashionmnist': 'LoRA',
            'eurosat': 'LoRA',
        }
        dinov2_lp = {
            'cifar10': 0.980, 'cifar100': 0.790, 'svhn': 0.400,
            'gtsrb': 0.680, 'mnist': 0.950, 'fashionmnist': 0.885,
            'eurosat': 0.925,
        }

        same_count = 0
        total = 0
        for t in TASKS:
            if t in ts and t in comp:
                sup_w = comp[t]['winner']
                dino_w = dinov2_winners.get(t, '?')
                same = "✓" if sup_w == dino_w or 'TIE' in [sup_w, dino_w] else "✗"
                if same == "✓":
                    same_count += 1
                total += 1
                print(f"  {t:<15s} {ts[t]['lp_accuracy']:>7.3f} {sup_w:>14s} "
                      f"{dinov2_lp.get(t, 0):>10.3f} {same:>13s}")

        print(f"\n  Same winner: {same_count}/{total} tasks")
        print(f"  {'✓ THEORY GENERALIZES' if same_count >= total * 0.7 else '⚠ PARTIAL GENERALIZATION'}")

    # Save
    save_path = os.path.join(config.output_dir, 'supervised_vit_results.json')
    os.makedirs(config.output_dir, exist_ok=True)

    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=make_serializable)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
