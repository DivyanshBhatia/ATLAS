"""
Validate Assumption A: Position-Aware Attention Heads

Assumption A (Theorem 1a) requires that the pretrained ViT has attention
heads where attention patterns are position-dependent — i.e., some heads
attend locally, others globally, and different classes activate different
spatial attention patterns.

We validate three properties:
1. HEAD DIVERSITY: Different heads have different attention patterns
2. POSITION AWARENESS: Attention depends on spatial distance
3. CLASS-DEPENDENT STEERING: Different classes induce different patterns

Usage:
    python validate_assumption_a.py
    python validate_assumption_a.py --task mnist
"""
import sys, os
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn.functional as F
import timm
import numpy as np
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from config import ExperimentConfig, setup_device


def extract_attention_maps(model, dataloader, device, max_batches=15):
    model.eval()
    all_attn = {}
    all_labels = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d_h = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = F.softmax((q @ k.transpose(-2, -1)) * (d_h ** -0.5), dim=-1)
            if layer_idx not in all_attn:
                all_attn[layer_idx] = []
            all_attn[layer_idx].append(attn.detach().cpu())
        return hook_fn

    for idx, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for batch_idx, (bx, by) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            model.forward_features(bx.to(device))
            all_labels.append(by)

    for h in hooks:
        h.remove()
    for li in all_attn:
        all_attn[li] = torch.cat(all_attn[li], dim=0)
    return all_attn, torch.cat(all_labels).numpy()


def validate_head_diversity(attn_maps, num_layers):
    print("\n  PROPERTY 1: Head Diversity")
    print("  " + "-" * 50)
    results = {}
    for l in range(num_layers):
        if l not in attn_maps:
            continue
        attn = attn_maps[l]
        cls_attn = attn[:, :, 0, 1:].mean(0)  # [H, S-1]
        cls_norm = F.normalize(cls_attn, dim=1)
        sim = (cls_norm @ cls_norm.T).numpy()
        H = sim.shape[0]
        mask = ~np.eye(H, dtype=bool)
        results[l] = {'avg_similarity': float(sim[mask].mean())}

    for l in [0, num_layers // 2, num_layers - 1]:
        if l in results:
            s = results[l]['avg_similarity']
            print(f"    Layer {l:2d}: avg head similarity = {s:.3f} "
                  f"{'✓' if s < 0.8 else '✗'}")

    diversity = 1.0 - np.mean([r['avg_similarity'] for r in results.values()])
    print(f"    Overall diversity: {diversity:.3f} "
          f"{'✓ SUPPORTED' if diversity > 0.1 else '✗ NOT SUPPORTED'}")
    return diversity


def validate_position_awareness(attn_maps, num_layers, img_size=224, patch_size=14):
    print("\n  PROPERTY 2: Position Awareness")
    print("  " + "-" * 50)
    n_p = img_size // patch_size
    total = n_p * n_p
    pos = np.array([(i, j) for i in range(n_p) for j in range(n_p)], dtype=np.float32)
    dist = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(-1)).flatten()

    total_local, total_heads = 0, 0
    for l in range(num_layers):
        if l not in attn_maps:
            continue
        attn = attn_maps[l]
        if attn.shape[2] - 1 != total:
            continue
        patch_attn = attn[:, :, 1:, 1:]
        H = patch_attn.shape[1]
        n_local = 0
        for h in range(H):
            mean_a = patch_attn[:, h].mean(0).numpy().flatten()
            corr = np.corrcoef(mean_a, dist)[0, 1]
            if corr < -0.1:
                n_local += 1
        total_local += n_local
        total_heads += H

        if l in [0, num_layers // 2, num_layers - 1]:
            print(f"    Layer {l:2d}: {n_local}/{H} local heads")

    frac = total_local / max(total_heads, 1)
    print(f"    Total: {total_local}/{total_heads} ({frac:.0%}) position-aware "
          f"{'✓ SUPPORTED' if frac > 0.1 else '✗ NOT SUPPORTED'}")
    return frac


def validate_class_steering(attn_maps, labels, num_layers):
    print("\n  PROPERTY 3: Class-Dependent Attention")
    print("  " + "-" * 50)
    classes = np.unique(labels)

    ratios = []
    for l in range(num_layers):
        if l not in attn_maps:
            continue
        cls_attn = attn_maps[l][:, :, 0, 1:]
        class_means = []
        for c in classes:
            m = labels == c
            if m.sum() > 0:
                class_means.append(cls_attn[m].mean(0))
        if len(class_means) < 2:
            continue
        cm = torch.stack(class_means)
        gm = cm.mean(0, keepdim=True)
        bv = ((cm - gm) ** 2).mean().item()
        wvs = []
        for ci, c in enumerate(classes):
            m = labels == c
            if m.sum() > 1:
                wvs.append(((cls_attn[m] - cm[ci:ci+1]) ** 2).mean().item())
        wv = np.mean(wvs) if wvs else 1e-10
        ratio = bv / (wv + 1e-10)
        ratios.append(ratio)

        if l in [0, num_layers // 2, num_layers - 1]:
            print(f"    Layer {l:2d}: between/within = {ratio:.4f}")

    mean_r = np.mean(ratios) if ratios else 0
    print(f"    Mean steering ratio: {mean_r:.4f} "
          f"{'✓ SUPPORTED' if mean_r > 0.01 else '✗ NOT SUPPORTED'}")
    return mean_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()

    config = ExperimentConfig()
    device = setup_device()

    print("Loading DINOv2 ViT-B...")
    model = timm.create_model(config.model_name, pretrained=True,
                               img_size=config.img_size).to(device)

    gray_tfm = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb_tfm = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    task_configs = {
        'mnist': (lambda: datasets.MNIST('./data', True, download=True, transform=gray_tfm), 10),
        'fashionmnist': (lambda: datasets.FashionMNIST('./data', True, download=True, transform=gray_tfm), 10),
        'cifar10': (lambda: datasets.CIFAR10('./data', True, download=True, transform=rgb_tfm), 10),
        'svhn': (lambda: datasets.SVHN('./data', 'train', download=True, transform=rgb_tfm), 10),
        'eurosat': (lambda: datasets.EuroSAT('./data', download=True, transform=rgb_tfm), 10),
        'gtsrb': (lambda: datasets.GTSRB('./data', 'train', download=True, transform=rgb_tfm), 43),
        'pcam': (lambda: datasets.PCAM('./data', 'train', download=True, transform=rgb_tfm), 2),
    }

    if args.task:
        if args.task in task_configs:
            task_configs = {args.task: task_configs[args.task]}
        else:
            print(f"Available: {list(task_configs.keys())}")
            return

    all_results = {}

    for task_name, (loader_fn, n_classes) in task_configs.items():
        print(f"\n{'='*60}")
        print(f"ASSUMPTION A: {task_name} ({n_classes} classes)")
        print(f"{'='*60}")

        try:
            ds = loader_fn()
            indices = torch.randperm(len(ds))[:1500].tolist()
            ds = Subset(ds, indices)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

            attn_maps, labels = extract_attention_maps(model, dl, device)
            n = attn_maps[0].shape[0]

            diversity = validate_head_diversity(attn_maps, config.num_layers)
            pos_frac = validate_position_awareness(attn_maps, config.num_layers)
            steer = validate_class_steering(attn_maps, labels[:n], config.num_layers)

            holds = diversity > 0.1 and pos_frac > 0.1 and steer > 0.01
            print(f"\n  → ASSUMPTION A {'HOLDS ✓' if holds else 'PARTIAL ~'} for {task_name}")

            all_results[task_name] = {
                'head_diversity': round(diversity, 4),
                'position_aware_frac': round(pos_frac, 4),
                'class_steering': round(steer, 4),
                'holds': holds,
            }

            del attn_maps, labels
            import gc; gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Task':<15s} {'Diversity':>10s} {'Pos-Aware':>10s} {'Steering':>10s} {'Holds?':>8s}")
    print(f"  {'-'*55}")
    for t, r in sorted(all_results.items(), key=lambda x: -x[1]['class_steering']):
        h = "✓" if r['holds'] else "~"
        print(f"  {t:<15s} {r['head_diversity']:>10.3f} {r['position_aware_frac']:>9.0%} "
              f"{r['class_steering']:>10.4f} {h:>8s}")

    save_path = os.path.join(config.output_dir, 'assumption_a_validation.json')
    os.makedirs(config.output_dir, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {save_path}")


if __name__ == '__main__':
    main()
