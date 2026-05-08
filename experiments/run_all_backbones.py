"""
Run a single task across ALL backbones for cross-backbone comparison.

Usage:
    python run_all_backbones.py --task svhn
    python run_all_backbones.py --task mnist
    python run_all_backbones.py --task eurosat --backbones dinov2 clip deit3
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from copy import deepcopy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset

from config import ExperimentConfig, setup_device
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


BACKBONES = {
    'dinov2': {
        'model': 'vit_base_patch14_dinov2.lvd142m',
        'name': 'DINOv2 ViT-B/14',
        'img_size': 224,
    },
    'dinov2reg': {
        'model': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'name': 'DINOv2-reg ViT-B/14',
        'img_size': 224,
    },
    'clip': {
        'model': 'vit_base_patch16_clip_224',
        'name': 'CLIP ViT-B/16',
        'img_size': 224,
    },
    'supervised': {
        'model': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'name': 'Supervised ViT-B/16',
        'img_size': 224,
    },
    'mae': {
        'model': 'vit_base_patch16_224.mae',
        'name': 'MAE ViT-B/16',
        'img_size': 224,
    },
    'deit3': {
        'model': 'deit3_base_patch16_224',
        'name': 'DeiT-III ViT-B/16',
        'img_size': 224,
    },
}

TASKS = {
    'cifar10':      (10,  'natural'),
    'cifar100':     (100, 'natural'),
    'dtd':          (47,  'natural'),
    'fgvc_aircraft':(100, 'natural'),
    'oxford_iiit_pet': (37, 'natural'),
    'food101':      (101, 'natural'),
    'stl10':        (10,  'natural'),
    'svhn':         (10,  'structured'),
    'gtsrb':        (43,  'structured'),
    'mnist':        (10,  'structured'),
    'fashionmnist': (10,  'structured'),
    'rendered_sst2':(2,   'structured'),
    'eurosat':      (10,  'specialized'),
}


def get_transforms(img_size=224):
    rgb = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gray = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return rgb, gray


def load_dataset(task_name, img_size=224, max_samples=1000):
    rgb, gray = get_transforms(img_size)
    loaders = {
        'cifar10': lambda: datasets.CIFAR10('./data', True, download=True, transform=rgb),
        'cifar100': lambda: datasets.CIFAR100('./data', True, download=True, transform=rgb),
        'svhn': lambda: datasets.SVHN('./data', 'train', download=True, transform=rgb),
        'gtsrb': lambda: datasets.GTSRB('./data', 'train', download=True, transform=rgb),
        'mnist': lambda: datasets.MNIST('./data', True, download=True, transform=gray),
        'fashionmnist': lambda: datasets.FashionMNIST('./data', True, download=True, transform=gray),
        'eurosat': lambda: datasets.EuroSAT('./data', download=True, transform=rgb),
        'dtd': lambda: datasets.DTD('./data', split='train', download=True, transform=rgb),
        'fgvc_aircraft': lambda: datasets.FGVCAircraft('./data', split='train', download=True, transform=rgb),
        'oxford_iiit_pet': lambda: datasets.OxfordIIITPet('./data', split='trainval', download=True, transform=rgb),
        'food101': lambda: datasets.Food101('./data', split='train', download=True, transform=rgb),
        'stl10': lambda: datasets.STL10('./data', split='train', download=True, transform=rgb),
        'rendered_sst2': lambda: datasets.RenderedSST2('./data', split='train', download=True, transform=rgb),
    }
    ds = loaders[task_name]()
    if len(ds) > max_samples:
        indices = torch.randperm(len(ds))[:max_samples].tolist()
        ds = Subset(ds, indices)
    return ds


def compute_sigma_p_sq(model):
    total, count = 0.0, 0
    for name, param in model.named_parameters():
        if any(t in name for t in ['qkv.weight', 'proj.weight',
                                     'k_proj.weight', 'v_proj.weight',
                                     'q_proj.weight', 'out_proj.weight']):
            total += param.float().norm().item() ** 2
            count += 1
    if count == 0:
        for name, param in model.named_parameters():
            if 'attn' in name and 'weight' in name and param.dim() == 2:
                total += param.float().norm().item() ** 2
                count += 1
    d_h = model.embed_dim // model.blocks[0].attn.num_heads if hasattr(model, 'blocks') else 64
    return total / (count * d_h) if count > 0 else 1.0


def run_task_on_backbone(backbone_key, task_name, n_classes, device):
    """Run all PEFT methods for one task on one backbone."""
    bb = BACKBONES[backbone_key]
    
    print(f"\n  Loading {bb['name']}...")
    try:
        model = timm.create_model(bb['model'], pretrained=True,
                                   img_size=bb['img_size']).to(device)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    sigma = compute_sigma_p_sq(model)
    embed_dim = model.embed_dim
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    head_dim = embed_dim // num_heads
    
    # Capacity
    n = 800
    r_star = max(1, int(2 * n * sigma / (num_layers * head_dim)))
    p_star = max(1, int(4 * n * sigma / (num_layers * embed_dim)))

    print(f"  σ_P²={sigma:.2f}, r*={r_star}, p*={p_star}")

    # Load data
    ds = load_dataset(task_name, bb['img_size'])
    n_total = len(ds)
    n_val = min(200, n_total // 5)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    
    config = ExperimentConfig()
    config.img_size = bb['img_size']
    config.embed_dim = embed_dim
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.head_dim = head_dim

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                             shuffle=False, num_workers=2)

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

    results = {'sigma_p_sq': sigma, 'r_star': r_star, 'p_star': p_star}

    for method_name, apply_fn in methods:
        try:
            mdl = deepcopy(model).to(device)
            mdl.head = nn.Linear(embed_dim, n_classes).to(device)
            mdl = apply_fn(mdl)
            acc = train_and_evaluate(mdl, train_loader, val_loader,
                                      config, device, method_name)
            results[method_name] = acc
            print(f"    {method_name}: {acc:.3f}")
            del mdl
        except Exception as e:
            print(f"    {method_name}: ERROR ({str(e)[:60]})")
            results[method_name] = None
            torch.cuda.empty_cache()

    # Compute winner
    lora_best = max((results.get(f'LoRA_r{r}', 0) or 0) for r in [1,4,8,16])
    vpt_best = max((results.get(f'VPT_p{p}', 0) or 0) for p in [1,5,10,20])
    lp = results.get('LP', 0) or 0

    if lora_best > vpt_best + 0.01:
        winner = 'LoRA'
    elif vpt_best > lora_best + 0.01:
        winner = 'VPT'
    else:
        winner = 'TIE'

    results['best_lora'] = lora_best
    results['best_vpt'] = vpt_best
    results['lp'] = lp
    results['winner'] = winner

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=list(TASKS.keys()))
    parser.add_argument('--backbones', nargs='+', default=None,
                        choices=list(BACKBONES.keys()),
                        help='Subset of backbones (default: all)')
    args = parser.parse_args()

    device = setup_device()
    task_name = args.task
    n_classes, category = TASKS[task_name]

    backbone_keys = args.backbones or list(BACKBONES.keys())

    print(f"{'='*70}")
    print(f"CROSS-BACKBONE: {task_name} ({n_classes} classes, {category})")
    print(f"Backbones: {', '.join(backbone_keys)}")
    print(f"{'='*70}")

    # Load existing results
    save_path = f'./results/cross_backbone_{task_name}.json'
    all_results = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")

    for bb_key in backbone_keys:
        if bb_key in all_results:
            print(f"\n  {BACKBONES[bb_key]['name']}: already done, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  {BACKBONES[bb_key]['name']}")
        print(f"{'='*50}")

        result = run_task_on_backbone(bb_key, task_name, n_classes, device)
        if result:
            all_results[bb_key] = result

            # Save after each backbone (crash-safe)
            os.makedirs('./results', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"RESULTS: {task_name}")
    print(f"{'='*70}")
    print(f"\n  {'Backbone':<25s} {'σ_P²':>6s} {'p*':>4s} {'LP':>6s} {'LoRA':>6s} "
          f"{'VPT':>6s} {'Winner':>7s}")
    print(f"  {'-'*65}")

    for bb_key in BACKBONES:
        if bb_key not in all_results:
            continue
        r = all_results[bb_key]
        print(f"  {BACKBONES[bb_key]['name']:<25s} {r['sigma_p_sq']:>6.1f} "
              f"{r['p_star']:>4d} {r['lp']:>6.3f} {r['best_lora']:>6.3f} "
              f"{r['best_vpt']:>6.3f} {r['winner']:>7s}")

    print(f"\n  Saved to {save_path}")


if __name__ == '__main__':
    main()
