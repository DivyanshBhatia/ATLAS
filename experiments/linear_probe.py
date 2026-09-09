"""
Linear Probe Baseline: How good are pretrained features without any adaptation?

Freezes the entire backbone and trains only the classification head.
This contextualizes LoRA/VPT: if LP = 0.82 and LoRA = 0.83, PEFT barely helped.

Usage:
    cd /content/ATLAS
    python experiments/linear_probe.py
    python experiments/linear_probe.py --backbones DINOv2 CLIP MoCo-v3
    python experiments/linear_probe.py --backbones DINOv2 --tasks cifar100 svhn
    python experiments/linear_probe.py --resume
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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

SAVE_PATH = 'results/linear_probe.json'
SEEDS = [42, 123, 456]
LP_LRS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
DATA_ROOT = os.path.expanduser('~/data')

BACKBONES = {
    'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
    'iBOT': None,
    'DINOv1': 'vit_base_patch16_224.dino',
    'CLIP': 'vit_base_patch16_clip_224.openai',
    'DeiT-III': 'deit3_base_patch16_224',
    'Supervised': 'vit_base_patch16_224.augreg_in1k',
    'MoCo-v3': None,
    'MAE': 'vit_base_patch16_224.mae',
}

TASKS = {
    'cifar100': 100, 'svhn': 10, 'gtsrb': 43, 'eurosat': 10, 'dtd': 47,
}


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_dataset(task_name, img_size=224):
    transform = get_transform(img_size)
    if task_name == 'cifar100':
        return datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
    elif task_name == 'svhn':
        return datasets.SVHN(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'gtsrb':
        return datasets.GTSRB(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'eurosat':
        ds = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=True)
        n = len(ds); n_test = n // 5
        train_ds, _ = random_split(ds, [n - n_test, n_test],
                                    generator=torch.Generator().manual_seed(0))
        return train_ds
    elif task_name == 'dtd':
        return datasets.DTD(root=DATA_ROOT, split='train', transform=transform, download=True)


def get_test_dataset(task_name, img_size=224):
    transform = get_transform(img_size)
    if task_name == 'cifar100':
        return datasets.CIFAR100(root=DATA_ROOT, train=False, transform=transform, download=True)
    elif task_name == 'svhn':
        return datasets.SVHN(root=DATA_ROOT, split='test', transform=transform, download=True)
    elif task_name == 'gtsrb':
        return datasets.GTSRB(root=DATA_ROOT, split='test', transform=transform, download=True)
    elif task_name == 'eurosat':
        ds = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=True)
        n = len(ds); n_test = n // 5
        _, test_ds = random_split(ds, [n - n_test, n_test],
                                   generator=torch.Generator().manual_seed(0))
        return test_ds
    elif task_name == 'dtd':
        return datasets.DTD(root=DATA_ROOT, split='test', transform=transform, download=True)


def load_model(name, device, ibot_checkpoint=None):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'iBOT':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        candidates = [ibot_checkpoint, '/content/ibot/checkpoint_teacher.pth',
                      '/content/checkpoint_teacher.pth']
        ckpt_path = next((x for x in candidates if x and os.path.isfile(x)), None)
        if ckpt_path is None:
            raise FileNotFoundError("iBOT checkpoint not found")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint.get('teacher', checkpoint.get('state_dict', checkpoint))
        cleaned = {}
        for key, value in sd.items():
            new_key = key
            for prefix in ('module.', 'teacher.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if not new_key.startswith(('head.', 'last_layer.')):
                cleaned[new_key] = value
        model.load_state_dict(cleaned, strict=False)
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name], pretrained=True, img_size=224).to(device)


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract CLS token features from frozen backbone."""
    all_feats, all_labels = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        feats = model.forward_features(x)
        # CLS token is the first token
        if hasattr(feats, 'shape') and len(feats.shape) == 3:
            cls = feats[:, 0]
        else:
            cls = feats
        all_feats.append(cls.cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                       test_feats, test_labels, num_classes, lr, epochs=100):
    """Train a linear head on extracted features."""
    d = train_feats.shape[1]
    head = nn.Linear(d, num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

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
            val_acc = (head(val_feats).argmax(1) == val_labels).float().mean().item()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Evaluate on test
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_acc = (head(test_feats).argmax(1) == test_labels).float().mean().item()

    return best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=list(BACKBONES.keys()))
    parser.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    parser.add_argument('--ibot_checkpoint', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for bb_name in args.backbones:
        if bb_name not in BACKBONES: continue

        print(f"\n{'='*60}")
        print(f"  Linear Probe: {bb_name}")
        print(f"{'='*60}")

        model = load_model(bb_name, device, args.ibot_checkpoint)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task_name in args.tasks:
            if task_name not in TASKS: continue
            key = task_name
            if key in all_results.get(bb_name, {}):
                print(f"  {bb_name} x {task_name}: already done"); continue

            num_classes = TASKS[task_name]
            print(f"\n  --- {bb_name} x {task_name} ---")

            # Extract test features once
            test_ds = get_test_dataset(task_name)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
            test_feats, test_labels = extract_features(model, test_loader, device)
            print(f"    Test: {len(test_ds)} examples, features {test_feats.shape}")

            seed_results = {}
            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed)
                full_train = get_train_dataset(task_name)
                indices = torch.randperm(len(full_train),
                    generator=torch.Generator().manual_seed(seed))[:1000]
                subset = Subset(full_train, indices.tolist())
                train_sub, val_sub = random_split(subset, [800, 200],
                    generator=torch.Generator().manual_seed(seed))

                train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2)
                val_loader = DataLoader(val_sub, batch_size=128, shuffle=False, num_workers=2)

                train_feats, train_labels = extract_features(model, train_loader, device)
                val_feats, val_labels = extract_features(model, val_loader, device)

                # Sweep LR
                best_lr, best_val, best_test = None, -1, -1
                for lr in LP_LRS:
                    val_acc, test_acc = train_linear_head(
                        train_feats, train_labels, val_feats, val_labels,
                        test_feats, test_labels, num_classes, lr
                    )
                    if val_acc > best_val:
                        best_val = val_acc
                        best_test = test_acc
                        best_lr = lr

                seed_results[str(seed)] = {
                    'val': float(best_val), 'test': float(best_test), 'lr': float(best_lr)
                }
                print(f"    Seed {seed}: val={best_val:.3f} test={best_test:.3f} @ LR={best_lr:.0e}")

            vals = [v['val'] for v in seed_results.values()]
            tests = [v['test'] for v in seed_results.values()]
            all_results[bb_name][key] = {
                'seeds': seed_results,
                'val_mean': float(np.mean(vals)),
                'val_std': float(np.std(vals)),
                'test_mean': float(np.mean(tests)),
                'test_std': float(np.std(tests)),
            }
            print(f"    Mean: val={np.mean(vals):.3f}±{np.std(vals):.3f} test={np.mean(tests):.3f}±{np.std(tests):.3f}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("LINEAR PROBE SUMMARY")
    print(f"{'='*60}")
    for bb in sorted(all_results.keys()):
        print(f"\n  {bb}:")
        for task in TASKS:
            if task not in all_results.get(bb, {}): continue
            d = all_results[bb][task]
            print(f"    {task:>8s}: test={d['test_mean']:.3f}±{d['test_std']:.3f}")


if __name__ == '__main__':
    main()
