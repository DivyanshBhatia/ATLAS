"""
MAE VPT at LR=1e-3: Does VPT recover like DINOv2 did?

MAE VPT CIFAR-100 = 0.127 at LR=1e-2. That's the same collapse
signature as DINOv2 at wrong LR (Figure 1). If MAE VPT recovers at
LR=1e-3, the sigma rule has a second exception. If not, Factor 2
(weak features) holds.

Quick test: ~30 min on A100.

Usage:
    python revision2_mae_lr_check.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt, apply_lora, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

device = setup_device()
config = ExperimentConfig()
config.epochs = 100

print("=" * 60)
print("MAE VPT LR CHECK: Does VPT recover at LR=1e-3?")
print("=" * 60)

model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, img_size=224).to(device)
config.embed_dim = model.embed_dim
config.num_layers = len(model.blocks)
config.num_heads = model.blocks[0].attn.num_heads
config.head_dim = config.embed_dim // config.num_heads

tasks = ['cifar100', 'svhn', 'dtd']
lrs = [1e-3, 2e-3, 5e-3, 1e-2]

for task in tasks:
    num_classes = TASKS[task][0]
    config.num_classes = num_classes

    print(f"\n  MAE x {task}:")

    # LP baseline
    torch.manual_seed(42); np.random.seed(42)
    ds = load_dataset(task, 224, max_samples=1000)
    n_val = min(200, len(ds) // 5)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    m = deepcopy(model)
    m.head = nn.Linear(config.embed_dim, num_classes).to(device)
    for p in m.parameters(): p.requires_grad_(False)
    for p in m.head.parameters(): p.requires_grad_(True)
    config.lr = 1e-2
    lp_acc = train_and_evaluate(m, train_loader, val_loader, config, device)
    del m; torch.cuda.empty_cache()
    print(f"    LP: {lp_acc:.3f}")

    for lr in lrs:
        torch.manual_seed(42); np.random.seed(42)
        ds = load_dataset(task, 224, max_samples=1000)
        n_val = min(200, len(ds) // 5)
        train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
            generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

        m = deepcopy(model)
        m.head = nn.Linear(config.embed_dim, num_classes).to(device)
        m = apply_vpt(m, 5, config)
        m = m.to(device)
        config.lr = lr
        acc = train_and_evaluate(m, train_loader, val_loader, config, device)
        del m; torch.cuda.empty_cache()
        marker = " <-- RECOVERY!" if acc > 0.3 and task == 'cifar100' else ""
        print(f"    VPT LR={lr:.0e}: {acc:.3f}{marker}")

    print(f"    (Compare: VPT at 1e-2 was 0.127 on CIFAR-100)")

del model; torch.cuda.empty_cache()
print("\nDone.")
