"""
Two checks requested by reviewer:

1. SEED VERIFICATION: Do seeds 42/123/456 actually produce different data splits?
   If not, the "3-seed" evaluation is fake and ±0.0 stds are a bug.

2. LP AT HIGH LR: Does the linear probe crash at LR=1e-2 on low-σ²_P backbones?
   If LP also crashes → the VPT "crash" is head divergence, not prompt-specific.
   If LP doesn't crash → the crash is VPT-specific, σ²_P story holds.

Usage:
    cd /content/ATLAS
    python experiments/reviewer_checks.py
    python experiments/reviewer_checks.py --ibot_checkpoint /content/checkpoint_teacher.pth
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

DATA_ROOT = os.path.expanduser('~/data')
SEEDS = [42, 123, 456]

BACKBONES = {
    'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
    'CLIP': 'vit_base_patch16_clip_224.openai',
    'DeiT-III': 'deit3_base_patch16_224',
    'MoCo-v3': None,
}

LP_LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_dataset(task_name):
    transform = get_transform()
    if task_name == 'cifar100':
        return datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
    elif task_name == 'svhn':
        return datasets.SVHN(root=DATA_ROOT, split='train', transform=transform, download=True)


def load_model(name, device):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name], pretrained=True, img_size=224).to(device)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        feats = model.forward_features(x)
        if len(feats.shape) == 3:
            cls = feats[:, 0]
        else:
            cls = feats
        all_feats.append(cls.cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_lp_at_lr(train_feats, train_labels, test_feats, test_labels, num_classes, lr, epochs=100):
    d = train_feats.shape[1]
    head = nn.Linear(d, num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        head.train()
        logits = head(train_feats)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            acc = (head(test_feats).argmax(1) == test_labels).float().mean().item()
        best_acc = max(best_acc, acc)

    # Also get last-epoch accuracy
    head.eval()
    with torch.no_grad():
        last_acc = (head(test_feats).argmax(1) == test_labels).float().mean().item()

    return best_acc, last_acc


def check_seeds():
    """Check 1: Do different seeds produce different data splits?"""
    print("=" * 60)
    print("  CHECK 1: SEED VERIFICATION")
    print("  Do seeds 42/123/456 produce different data splits?")
    print("=" * 60)

    for task in ['cifar100', 'svhn']:
        full_train = get_train_dataset(task)
        print(f"\n  Task: {task} (full train: {len(full_train)} examples)")

        all_indices = {}
        for seed in SEEDS:
            indices = torch.randperm(len(full_train),
                generator=torch.Generator().manual_seed(seed))[:1000]
            subset = Subset(full_train, indices.tolist())
            train_sub, val_sub = random_split(subset, [800, 200],
                generator=torch.Generator().manual_seed(seed))

            # Get actual indices
            train_indices = set([subset.indices[i] for i in train_sub.indices])
            val_indices = set([subset.indices[i] for i in val_sub.indices])
            all_indices[seed] = {'train': train_indices, 'val': val_indices}

            print(f"    Seed {seed}: train={len(train_indices)}, val={len(val_indices)}, "
                  f"first 5 train indices: {sorted(list(train_indices))[:5]}")

        # Compare overlap
        for s1, s2 in [(42, 123), (42, 456), (123, 456)]:
            train_overlap = len(all_indices[s1]['train'] & all_indices[s2]['train'])
            val_overlap = len(all_indices[s1]['val'] & all_indices[s2]['val'])
            total_overlap = len(
                (all_indices[s1]['train'] | all_indices[s1]['val']) &
                (all_indices[s2]['train'] | all_indices[s2]['val'])
            )
            print(f"    Seeds {s1} vs {s2}: "
                  f"train overlap={train_overlap}/800, "
                  f"val overlap={val_overlap}/200, "
                  f"total sample overlap={total_overlap}/1000")

        # Are they identical?
        if all_indices[42]['train'] == all_indices[123]['train']:
            print("    *** BUG: Seeds 42 and 123 have IDENTICAL training splits! ***")
        else:
            print("    ✓ Seeds produce different splits")


def check_lp_at_high_lr():
    """Check 2: Does LP crash at LR=1e-2 on low-σ²_P backbones?"""
    print("\n" + "=" * 60)
    print("  CHECK 2: LP AT HIGH LR")
    print("  Does LP crash at LR=1e-2 on low-σ²_P backbones?")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = 'cifar100'

    full_train = get_train_dataset(task)
    test_ds = datasets.CIFAR100(root=DATA_ROOT, train=False,
                                 transform=get_transform(), download=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    results = {}

    for bb_name in ['DINOv2', 'CLIP', 'DeiT-III', 'MoCo-v3']:
        print(f"\n  {bb_name}:")
        model = load_model(bb_name, device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Extract test features
        test_feats, test_labels = extract_features(model, test_loader, device)

        # Extract train features (seed 42, n=800)
        torch.manual_seed(42); np.random.seed(42)
        indices = torch.randperm(len(full_train),
            generator=torch.Generator().manual_seed(42))[:1000]
        subset = Subset(full_train, indices.tolist())
        train_sub, _ = random_split(subset, [800, 200],
            generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2)
        train_feats, train_labels = extract_features(model, train_loader, device)

        # Sweep LP LRs
        lr_results = {}
        for lr in LP_LRS:
            best_acc, last_acc = train_lp_at_lr(
                train_feats, train_labels, test_feats, test_labels, 100, lr
            )
            lr_results[f'{lr}'] = {'best': float(best_acc), 'last': float(last_acc)}
            crash = "CRASH" if best_acc < 0.15 else "degraded" if best_acc < 0.3 else ""
            print(f"    LR={lr:.0e}: best={best_acc:.3f} last={last_acc:.3f} {crash}")

        results[bb_name] = lr_results
        del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n  {'='*50}")
    print("  LP AT LR=1e-2: CRASH TEST")
    print(f"  {'='*50}")
    for bb, lrs in results.items():
        best_low = max(lrs[k]['best'] for k in lrs if float(k) <= 2e-3)
        at_1e2 = lrs['0.01']['best']
        drop = (at_1e2 - best_low) * 100
        print(f"  {bb:>10s}: LP@1e-2={at_1e2:.3f}, LP@best≤2e-3={best_low:.3f}, "
              f"drop={drop:+.1f}pts {'← CRASH' if drop < -10 else ''}")

    os.makedirs('results', exist_ok=True)
    with open('results/reviewer_checks.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n  If low-σ²_P backbones crash → head-LR effect, not VPT-specific")
    print("  If they don't crash → VPT-specific, σ²_P story holds")


if __name__ == '__main__':
    check_seeds()
    check_lp_at_high_lr()
