"""
LoRA at 1e-2: check if extending the grid improves any of the 13 edge cells.
Single seed, val split. If none improves by >1%, the LoRA-side asymmetry is closed.

Usage:
    cd /content/ATLAS
    python experiments/lora_grid_check.py
    python experiments/lora_grid_check.py --ibot_checkpoint /content/checkpoint_teacher.pth
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import os
from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora

DATA_ROOT = os.path.expanduser('~/data')

# The 13 cells at LoRA LR=5e-3 edge (from Table 10)
EDGE_CELLS = [
    ('DINOv2', 'svhn'), ('CLIP', 'svhn'),
    ('DeiT-III', 'cifar100'), ('DeiT-III', 'svhn'), ('DeiT-III', 'gtsrb'),
    ('Supervised', 'svhn'),
    ('MoCo-v3', 'cifar100'), ('MoCo-v3', 'svhn'), ('MoCo-v3', 'gtsrb'), ('MoCo-v3', 'eurosat'),
    ('MAE', 'cifar100'), ('MAE', 'svhn'), ('MAE', 'dtd'),
]

TASKS = {'cifar100': 100, 'svhn': 10, 'gtsrb': 43, 'eurosat': 10, 'dtd': 47}

BACKBONES = {
    'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
    'CLIP': 'vit_base_patch16_clip_224.openai',
    'DeiT-III': 'deit3_base_patch16_224',
    'Supervised': 'vit_base_patch16_224',
    'MoCo-v3': None,
    'MAE': None,
}


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_val(task_name, seed=42):
    transform = get_transform()
    if task_name == 'cifar100':
        ds = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
    elif task_name == 'svhn':
        ds = datasets.SVHN(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'gtsrb':
        ds = datasets.GTSRB(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'eurosat':
        full = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=True)
        n = len(full); n_test = n // 5
        ds, _ = random_split(full, [n - n_test, n_test],
                              generator=torch.Generator().manual_seed(0))
    elif task_name == 'dtd':
        ds = datasets.DTD(root=DATA_ROOT, split='train', transform=transform, download=True)

    torch.manual_seed(seed); np.random.seed(seed)
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:1000]
    subset = Subset(ds, indices.tolist())
    train_ds, val_ds = random_split(subset, [800, 200],
                                     generator=torch.Generator().manual_seed(seed))
    return DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2), \
           DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)


def load_model(name, device, ibot_checkpoint=None):
    if name == 'DINOv2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        return model.to(device)
    elif name == 'CLIP':
        try:
            model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, img_size=224)
        except RuntimeError:
            model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=224)
        return model.to(device)
    elif name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'MAE':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'model' in sd: sd = sd['model']
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'DeiT-III':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'model' in sd: sd = sd['model']
        # Handle pos_embed size mismatch (196 vs 197)
        if 'pos_embed' in sd and sd['pos_embed'].shape[1] != model.pos_embed.shape[1]:
            cls_pos = model.pos_embed[:, :1, :]
            sd['pos_embed'] = torch.cat([cls_pos, sd['pos_embed']], dim=1)
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'Supervised':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=224)
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name], pretrained=True, img_size=224).to(device)


def train_and_eval(model, train_loader, val_loader, lr, device, epochs=100):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_val = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        best_val = max(best_val, correct / total)
    return best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ibot_checkpoint', default=None)
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Filter to specific backbones, e.g. --backbones DeiT-III MoCo-v3')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()

    cells = EDGE_CELLS
    if args.backbones:
        cells = [(bb, task) for bb, task in EDGE_CELLS if bb in args.backbones]

    print("=" * 60)
    print("  LoRA at 1e-2: Grid Edge Check (13 cells)")
    print("=" * 60)

    loaded = {}
    for bb_name, task_name in cells:
        if bb_name not in loaded:
            loaded[bb_name] = load_model(bb_name, device, args.ibot_checkpoint)

        base_model = loaded[bb_name]
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads
        config.num_classes = TASKS[task_name]

        train_loader, val_loader = get_train_val(task_name)

        # LoRA at 5e-3 (current best)
        m = deepcopy(base_model)
        m.head = nn.Linear(config.embed_dim, TASKS[task_name]).to(device)
        m = apply_lora(m, 8, config); m = m.to(device)
        val_5e3 = train_and_eval(m, train_loader, val_loader, 5e-3, device)
        del m; torch.cuda.empty_cache()

        # LoRA at 1e-2 (extended)
        m = deepcopy(base_model)
        m.head = nn.Linear(config.embed_dim, TASKS[task_name]).to(device)
        m = apply_lora(m, 8, config); m = m.to(device)
        val_1e2 = train_and_eval(m, train_loader, val_loader, 1e-2, device)
        del m; torch.cuda.empty_cache()

        diff = (val_1e2 - val_5e3) * 100
        flag = " ← IMPROVED" if diff > 1 else ""
        print(f"  {bb_name:>10s} x {task_name:>8s}: 5e-3={val_5e3:.3f} 1e-2={val_1e2:.3f} Δ={diff:+.1f}%{flag}")

    print("\n  If no cell improved by >1%, LoRA-side asymmetry is closed.")


if __name__ == '__main__':
    main()
