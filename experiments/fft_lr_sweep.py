"""
FFT LR Sweep: Is full fine-tuning truly broken at n=1000, or just undertuned?

Reviewer: "DINOv2 C-100 FFT (0.498) below LP (0.745) is far worse than expected.
Was FFT's LR swept? If not, 'PEFT is essential' is a one-sided comparison."

Usage:
    cd /content/ATLAS
    python experiments/fft_lr_sweep.py
    python experiments/fft_lr_sweep.py --backbones DINOv2 DeiT-III MoCo-v3
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

from config import ExperimentConfig, setup_device
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

FFT_LRS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
TASKS_DEFAULT = ['cifar100', 'svhn']
SAVE_PATH = 'results/fft_lr_sweep.json'

BACKBONES = {
    'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
    'DeiT-III': 'deit3_base_patch16_224',
    'MoCo-v3': None,
}


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


def train_fft(model, train_loader, val_loader, lr, device, epochs=100):
    """Full fine-tuning: all parameters trainable."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / total
        best_acc = max(best_acc, acc)
    
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'DeiT-III'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    
    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for bb_name in args.backbones:
        print(f"\n{'='*60}")
        print(f"  FFT LR Sweep: {bb_name}")
        print(f"{'='*60}")
        
        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task in args.tasks:
            if task not in TASKS: continue
            key = task
            if key in all_results.get(bb_name, {}):
                print(f"  {bb_name} x {task}: already done"); continue

            num_classes = TASKS[task][0]
            print(f"\n  --- {bb_name} x {task} (FFT) ---")

            lr_results = {}
            for lr in FFT_LRS:
                torch.manual_seed(42); np.random.seed(42)
                base_model = load_model(bb_name, device)
                base_model.head = nn.Linear(base_model.embed_dim, num_classes).to(device)
                
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                acc = train_fft(base_model, tl, vl, lr, device)
                lr_results[f'{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")
                
                del base_model; torch.cuda.empty_cache()

            best_lr = max(FFT_LRS, key=lambda x: lr_results[f'{x}'])
            best_acc = lr_results[f'{best_lr}']
            
            all_results[bb_name][key] = {
                'lr_sweep': lr_results,
                'best_lr': float(best_lr),
                'best_acc': float(best_acc),
            }
            print(f"    Best: LR={best_lr:.0e} acc={best_acc:.3f}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("FFT LR SWEEP SUMMARY")
    print(f"{'='*60}")
    for bb, tasks in all_results.items():
        print(f"\n  {bb}:")
        for task, data in tasks.items():
            print(f"    {task}: FFT best = {data['best_acc']:.3f} @ {data['best_lr']:.0e}")


if __name__ == '__main__':
    main()
