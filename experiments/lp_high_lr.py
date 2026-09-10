"""
LP at high LR for remaining 4 backbones (DINOv1, iBOT, Supervised, MAE).
Completes the head-divergence control for all 8 backbones.

Usage:
    cd /content/ATLAS
    python experiments/lp_high_lr.py
    python experiments/lp_high_lr.py --ibot_checkpoint /content/checkpoint_teacher.pth
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import argparse

DATA_ROOT = os.path.expanduser('~/data')
LP_LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

BACKBONES = {
    'DINOv1': None,
    'Supervised': 'vit_base_patch16_224',
    'MAE': None,
    'iBOT': None,
}


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(name, device, ibot_checkpoint=None):
    if name == 'DINOv1':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'MAE':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'model' in sd:
            sd = sd['model']
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
    model.eval()
    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        feats = model.forward_features(x)
        cls = feats[:, 0] if len(feats.shape) == 3 else feats
        all_feats.append(cls.cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_lp(train_feats, train_labels, test_feats, test_labels, num_classes, lr, epochs=100):
    d = train_feats.shape[1]
    head = nn.Linear(d, num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        head.train()
        loss = criterion(head(train_feats), train_labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        head.eval()
        with torch.no_grad():
            acc = (head(test_feats).argmax(1) == test_labels).float().mean().item()
        best_acc = max(best_acc, acc)

    head.eval()
    with torch.no_grad():
        last_acc = (head(test_feats).argmax(1) == test_labels).float().mean().item()
    return best_acc, last_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ibot_checkpoint', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform()

    full_train = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_ds = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    torch.manual_seed(42); np.random.seed(42)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42))[:1000]
    subset = Subset(full_train, indices.tolist())
    train_sub, _ = random_split(subset, [800, 200], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2)

    print("=" * 60)
    print("  LP AT HIGH LR — REMAINING 4 BACKBONES")
    print("=" * 60)

    for bb_name in BACKBONES:
        print(f"\n  {bb_name}:")
        model = load_model(bb_name, device, args.ibot_checkpoint)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        test_feats, test_labels = extract_features(model, test_loader, device)
        train_feats, train_labels = extract_features(model, train_loader, device)

        for lr in LP_LRS:
            best_acc, last_acc = train_lp(train_feats, train_labels, test_feats, test_labels, 100, lr)
            crash = " CRASH" if best_acc < 0.15 else " degraded" if best_acc < 0.3 else ""
            print(f"    LR={lr:.0e}: best={best_acc:.3f} last={last_acc:.3f}{crash}")

        del model; torch.cuda.empty_cache()

    print(f"\n  {'='*50}")
    print("  LP AT LR=1e-2: CRASH TEST (ALL 8 BACKBONES)")
    print(f"  {'='*50}")
    print("  Already done: DINOv2 +1.3, CLIP +0.5, DeiT-III +0.9, MoCo-v3 +41.5")
    print("  New results above ^^^")


if __name__ == '__main__':
    main()
