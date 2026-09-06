"""
DINOv1 SVHN Fix: Rerun at stable LR=2e-3 (5e-3 collapsed on 2/3 seeds)

Usage:
    cd /content/ATLAS
    python experiments/dinov1_svhn_fix.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora, apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SEEDS = [42, 123, 456]

device = setup_device()
config = ExperimentConfig()
config.epochs = 100

print("=" * 60)
print("DINOv1 SVHN: Stable LR Fix (2e-3 instead of 5e-3)")
print("=" * 60)

model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, img_size=224).to(device)
config.embed_dim = model.embed_dim
config.num_layers = len(model.blocks)
config.num_heads = model.blocks[0].attn.num_heads
config.head_dim = config.embed_dim // config.num_heads
config.num_classes = TASKS['svhn'][0]

lora_accs = []
vpt_accs = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = load_dataset('svhn', 224, max_samples=1000)
    nv = min(200, len(ds) // 5)
    tds, vds = random_split(ds, [len(ds) - nv, nv],
        generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
    vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

    # LoRA at stable LR=2e-3
    m = deepcopy(model)
    m.head = nn.Linear(config.embed_dim, config.num_classes).to(device)
    m = apply_lora(m, 8, config)
    m = m.to(device)
    config.lr = 2e-3
    la = train_and_evaluate(m, tl, vl, config, device)
    del m; torch.cuda.empty_cache()

    # VPT at LR=1e-3
    m = deepcopy(model)
    m.head = nn.Linear(config.embed_dim, config.num_classes).to(device)
    m = apply_vpt(m, 5, config)
    m = m.to(device)
    config.lr = 1e-3
    va = train_and_evaluate(m, tl, vl, config, device)
    del m; torch.cuda.empty_cache()

    lora_accs.append(la)
    vpt_accs.append(va)
    print(f"  Seed {seed}: LoRA={la:.3f} VPT={va:.3f}")

lm, ls = np.mean(lora_accs), np.std(lora_accs)
vm, vs = np.mean(vpt_accs), np.std(vpt_accs)
gap = lm - vm
w = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'

print(f"\n  RESULT: LoRA={lm:.3f}±{ls:.3f} VPT={vm:.3f}±{vs:.3f} gap={gap:+.3f} -> {w}")
print(f"  Stable: std(LoRA)={ls:.3f} (was 0.291 at 5e-3)")
print(f"\n  Previous (5e-3): collapsed on seeds 123,456")
print(f"  This run (2e-3): {'STABLE' if ls < 0.05 else 'STILL UNSTABLE'}")

del model; torch.cuda.empty_cache()
