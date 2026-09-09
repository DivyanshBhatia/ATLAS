"""
Test-Split Evaluation: Re-evaluate all 40 main-comparison cells on official test splits.

Protocol:
  - Train on n=800 random subset (same as before: n=1000 minus 200 val)
  - Select best-epoch checkpoint on 200-example val split (same as before)
  - Report accuracy on the OFFICIAL TEST SPLIT (new)
  - Also report last-epoch accuracy on test split (insurance)
  - 3 seeds, paired gap std

This resolves the #1 reviewer concern: "validation = evaluation on the same 200 examples."

Usage:
    cd /content/ATLAS
    python experiments/test_split_eval.py
    python experiments/test_split_eval.py --backbones DINOv2 MoCo-v3
    python experiments/test_split_eval.py --backbones DINOv2 --tasks cifar100 --seeds 42
    python experiments/test_split_eval.py --resume
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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt, apply_lora, train_and_evaluate

SAVE_PATH = 'results/test_split_eval.json'
SEEDS = [42, 123, 456]

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
    'cifar100':  {'num_classes': 100, 'dataset': 'CIFAR100'},
    'svhn':      {'num_classes': 10,  'dataset': 'SVHN'},
    'gtsrb':     {'num_classes': 43,  'dataset': 'GTSRB'},
    'eurosat':   {'num_classes': 10,  'dataset': 'EuroSAT'},
    'dtd':       {'num_classes': 47,  'dataset': 'DTD'},
}

# LR grids (same as paper protocol)
LORA_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
VPT_LRS_LOW = [5e-4, 1e-3, 2e-3]        # σ²_P < 0.7
VPT_LRS_HIGH = [2e-3, 5e-3, 1e-2]       # σ²_P ≥ 0.7

SIGMA_P = {
    'DINOv2': 0.22, 'iBOT': 0.15, 'DINOv1': 0.19, 'CLIP': 0.18,
    'DeiT-III': 1.04, 'Supervised': 1.60, 'MoCo-v3': 2.31, 'MAE': 1.76,
}


DATA_ROOT = os.path.expanduser('~/data')


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_dataset(task_name, img_size=224):
    """Load the official train split for each dataset."""
    transform = get_transform(img_size)

    if task_name == 'cifar100':
        return datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
    elif task_name == 'svhn':
        return datasets.SVHN(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'gtsrb':
        return datasets.GTSRB(root=DATA_ROOT, split='train', transform=transform, download=True)
    elif task_name == 'eurosat':
        ds = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=True)
        n = len(ds)
        n_test = n // 5
        train_ds, _ = random_split(ds, [n - n_test, n_test],
                                    generator=torch.Generator().manual_seed(0))
        return train_ds
    elif task_name == 'dtd':
        return datasets.DTD(root=DATA_ROOT, split='train', transform=transform, download=True)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def get_test_dataset(task_name, img_size=224):
    """Load the official test split for each dataset."""
    transform = get_transform(img_size)

    if task_name == 'cifar100':
        return datasets.CIFAR100(root=DATA_ROOT, train=False, transform=transform, download=True)
    elif task_name == 'svhn':
        return datasets.SVHN(root=DATA_ROOT, split='test', transform=transform, download=True)
    elif task_name == 'gtsrb':
        return datasets.GTSRB(root=DATA_ROOT, split='test', transform=transform, download=True)
    elif task_name == 'eurosat':
        ds = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=True)
        n = len(ds)
        n_test = n // 5
        _, test_ds = random_split(ds, [n - n_test, n_test],
                                   generator=torch.Generator().manual_seed(0))
        return test_ds
    elif task_name == 'dtd':
        return datasets.DTD(root=DATA_ROOT, split='test', transform=transform, download=True)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def get_train_val_split(task_name, img_size=224, n_total=1000, n_val=200, seed=42):
    """Sample n_total from train set, split into train/val."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_train = get_train_dataset(task_name, img_size)

    # Subsample to n_total
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))[:n_total]
    subset = Subset(full_train, indices.tolist())

    # Split into train and val
    n_train = n_total - n_val
    train_ds, val_ds = random_split(subset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    return train_ds, val_ds


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


def train_and_eval_with_test(model, train_loader, val_loader, test_loader,
                              config, device, epochs=100):
    """Train model, select best epoch on val, report both best-epoch and last-epoch on test."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch_state = None
    last_epoch_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val accuracy (for checkpoint selection)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        val_acc = correct / total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save last-epoch state
    last_epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def eval_on_test(state_dict):
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        return correct / total

    # Evaluate on test split
    test_best = eval_on_test(best_epoch_state)
    test_last = eval_on_test(last_epoch_state)
    val_best = best_val_acc

    return val_best, test_best, test_last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=list(BACKBONES.keys()))
    parser.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--ibot_checkpoint', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for bb_name in args.backbones:
        if bb_name not in BACKBONES:
            print(f"Unknown backbone: {bb_name}"); continue

        print(f"\n{'='*60}")
        print(f"  Test-Split Evaluation: {bb_name}")
        print(f"{'='*60}")

        base_model = load_model(bb_name, device, args.ibot_checkpoint)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for task_name in args.tasks:
            if task_name not in TASKS: continue
            key = task_name
            if key in all_results.get(bb_name, {}) and \
               len(all_results[bb_name][key].get('seeds', {})) >= len(args.seeds):
                print(f"  {bb_name} x {task_name}: already done"); continue

            num_classes = TASKS[task_name]['num_classes']
            config.num_classes = num_classes

            print(f"\n  --- {bb_name} x {task_name} ---")

            # Load test set once
            test_ds = get_test_dataset(task_name)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
            print(f"    Test set: {len(test_ds)} examples")

            # ===== PHASE 1: LR sweep on seed 42 (val-based selection) =====
            vpt_lrs = VPT_LRS_LOW if SIGMA_P[bb_name] < 0.7 else VPT_LRS_HIGH
            print(f"    Phase 1: LR sweep (seed 42)")
            print(f"      LoRA LRs: {LORA_LRS}")
            print(f"      VPT LRs:  {vpt_lrs} (σ²_P={SIGMA_P[bb_name]})")

            sweep_seed = 42
            torch.manual_seed(sweep_seed); np.random.seed(sweep_seed)
            train_ds_sweep, val_ds_sweep = get_train_val_split(task_name, seed=sweep_seed)
            tl_sweep = DataLoader(train_ds_sweep, batch_size=64, shuffle=True, num_workers=2)
            vl_sweep = DataLoader(val_ds_sweep, batch_size=64, shuffle=False, num_workers=2)

            # Sweep LoRA
            best_lora_lr, best_lora_val = None, -1
            for lr in LORA_LRS:
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_lora(m, 8, config)
                m = m.to(device)
                config.lr = lr
                val_acc, _, _ = train_and_eval_with_test(m, tl_sweep, vl_sweep, test_loader, config, device)
                if val_acc > best_lora_val:
                    best_lora_val = val_acc
                    best_lora_lr = lr
                del m; torch.cuda.empty_cache()
                print(f"      LoRA LR={lr:.0e}: val={val_acc:.3f}")


            print(f"      → Best LoRA LR: {best_lora_lr:.0e} (val={best_lora_val:.3f})")

            # Sweep VPT
            best_vpt_lr, best_vpt_val = None, -1
            for lr in vpt_lrs:
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config)
                m = m.to(device)
                config.lr = lr
                val_acc, _, _ = train_and_eval_with_test(m, tl_sweep, vl_sweep, test_loader, config, device)
                if val_acc > best_vpt_val:
                    best_vpt_val = val_acc
                    best_vpt_lr = lr
                del m; torch.cuda.empty_cache()
                print(f"      VPT  LR={lr:.0e}: val={val_acc:.3f}")
            print(f"      → Best VPT LR:  {best_vpt_lr:.0e} (val={best_vpt_val:.3f})")

            # ===== PHASE 2: 3-seed evaluation at chosen LRs =====
            print(f"    Phase 2: 3-seed eval (LoRA@{best_lora_lr:.0e}, VPT@{best_vpt_lr:.0e})")

            seed_results = {}
            for seed in args.seeds:
                print(f"    Seed {seed}:")
                train_ds, val_ds = get_train_val_split(task_name, seed=seed)
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

                # --- LoRA ---
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_lora(m, 8, config)
                m = m.to(device)
                config.lr = best_lora_lr
                lora_val, lora_test_best, lora_test_last = \
                    train_and_eval_with_test(m, train_loader, val_loader, test_loader, config, device)
                del m; torch.cuda.empty_cache()

                # --- VPT ---
                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config)
                m = m.to(device)
                config.lr = best_vpt_lr
                vpt_val, vpt_test_best, vpt_test_last = \
                    train_and_eval_with_test(m, train_loader, val_loader, test_loader, config, device)
                del m; torch.cuda.empty_cache()

                gap_val = lora_val - vpt_val
                gap_test_best = lora_test_best - vpt_test_best
                gap_test_last = lora_test_last - vpt_test_last

                seed_results[str(seed)] = {
                    'lora_lr': float(best_lora_lr),
                    'vpt_lr': float(best_vpt_lr),
                    'lora_val': float(lora_val),
                    'lora_test_best': float(lora_test_best),
                    'lora_test_last': float(lora_test_last),
                    'vpt_val': float(vpt_val),
                    'vpt_test_best': float(vpt_test_best),
                    'vpt_test_last': float(vpt_test_last),
                    'gap_val': float(gap_val),
                    'gap_test_best': float(gap_test_best),
                    'gap_test_last': float(gap_test_last),
                }

                print(f"      LoRA: val={lora_val:.3f} test(best)={lora_test_best:.3f} test(last)={lora_test_last:.3f}")
                print(f"      VPT:  val={vpt_val:.3f} test(best)={vpt_test_best:.3f} test(last)={vpt_test_last:.3f}")
                print(f"      Gap:  val={gap_val:+.3f} test(best)={gap_test_best:+.3f} test(last)={gap_test_last:+.3f}")

            # Aggregate
            gaps_val = [seed_results[s]['gap_val'] for s in seed_results]
            gaps_test = [seed_results[s]['gap_test_best'] for s in seed_results]
            gaps_last = [seed_results[s]['gap_test_last'] for s in seed_results]

            all_results[bb_name][key] = {
                'seeds': seed_results,
                'gap_val_mean': float(np.mean(gaps_val)),
                'gap_val_std': float(np.std(gaps_val, ddof=1)),
                'gap_test_best_mean': float(np.mean(gaps_test)),
                'gap_test_best_std': float(np.std(gaps_test, ddof=1)),
                'gap_test_last_mean': float(np.mean(gaps_last)),
                'gap_test_last_std': float(np.std(gaps_last, ddof=1)),
            }

            d = all_results[bb_name][key]
            print(f"\n    Summary ({bb_name} x {task_name}):")
            print(f"      Gap (val):        {d['gap_val_mean']:+.1%} ± {d['gap_val_std']:.1%}")
            print(f"      Gap (test, best): {d['gap_test_best_mean']:+.1%} ± {d['gap_test_best_std']:.1%}")
            print(f"      Gap (test, last): {d['gap_test_last_mean']:+.1%} ± {d['gap_test_last_std']:.1%}")

            # Does the winner change?
            val_winner = "LoRA" if d['gap_val_mean'] > 0.02 else "VPT" if d['gap_val_mean'] < -0.02 else "Tie"
            test_winner = "LoRA" if d['gap_test_best_mean'] > 0.02 else "VPT" if d['gap_test_best_mean'] < -0.02 else "Tie"
            if val_winner != test_winner:
                print(f"      *** WINNER CHANGED: val={val_winner} → test={test_winner} ***")
            else:
                print(f"      Winner unchanged: {val_winner}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print("TEST-SPLIT EVALUATION SUMMARY")
    print(f"{'='*60}")
    ltv_val = [0, 0, 0]
    ltv_test = [0, 0, 0]
    flips = 0

    for bb in sorted(all_results.keys()):
        print(f"\n  {bb}:")
        for task in TASKS:
            if task not in all_results[bb]: continue
            d = all_results[bb][task]
            gv = d['gap_val_mean']
            gt = d['gap_test_best_mean']
            vw = "L" if gv > 0.02 else "V" if gv < -0.02 else "T"
            tw = "L" if gt > 0.02 else "V" if gt < -0.02 else "T"
            flip = " FLIP!" if vw != tw else ""

            ltv_val[0 if vw == "L" else 1 if vw == "T" else 2] += 1
            ltv_test[0 if tw == "L" else 1 if tw == "T" else 2] += 1
            if vw != tw: flips += 1

            print(f"    {task:>8s}: val={gv:+.1%}({vw}) test={gt:+.1%}({tw}){flip}")

    print(f"\n  Val:  {ltv_val[0]}L / {ltv_val[1]}T / {ltv_val[2]}V")
    print(f"  Test: {ltv_test[0]}L / {ltv_test[1]}T / {ltv_test[2]}V")
    print(f"  Flips: {flips}")
    if flips == 0:
        print("  Test-split evaluation confirms all val-split results.")
    else:
        print(f"  {flips} cells changed winner under test-split evaluation.")


if __name__ == '__main__':
    main()
