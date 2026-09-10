"""
VPT Warmup Control: Is the 60-point crash a backbone property or a training recipe artifact?

If VPT at LR=1e-2 with 10-epoch warmup still crashes on DINOv2/CLIP → backbone property (σ²_P story holds)
If warmup fixes it → the practical recommendation simplifies to "use warmup for VPT"

Usage:
    cd /content/ATLAS
    python experiments/warmup_control.py
    python experiments/warmup_control.py --backbones DINOv2 CLIP DeiT-III
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

SAVE_PATH = 'results/warmup_control.json'
SEEDS = [42, 123, 456]
DATA_ROOT = os.path.expanduser('~/data')

BACKBONES = {
    'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
    'CLIP': 'vit_base_patch16_clip_224.openai',
    'DeiT-III': 'deit3_base_patch16_224',
    'DINOv1': 'vit_base_patch16_224.dino',
    'Supervised': 'vit_base_patch16_224.augreg_in1k',
    'MAE': 'vit_base_patch16_224.mae',
    'iBOT': None,
    'MoCo-v3': None,
}

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_data(task_name, seed, img_size=224):
    transform = get_transform(img_size)
    if task_name == 'cifar100':
        full_train = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform, download=True)
        test_ds = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    torch.manual_seed(seed); np.random.seed(seed)
    indices = torch.randperm(len(full_train),
        generator=torch.Generator().manual_seed(seed))[:1000]
    subset = Subset(full_train, indices.tolist())
    train_ds, val_ds = random_split(subset, [800, 200],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train_vpt_with_warmup(model, train_loader, val_loader, test_loader,
                           lr, warmup_epochs, total_epochs, device):
    """Train VPT with linear warmup then cosine decay."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    # Linear warmup + cosine decay scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup from 0 to lr
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(total_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val accuracy
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    test_acc = correct / total

    return best_val_acc, test_acc


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=['DINOv2', 'CLIP', 'DeiT-III'])
    parser.add_argument('--task', default='cifar100')
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

    conditions = [
        ('no_warmup', 0),
        ('warmup_10', 10),
    ]

    for bb_name in args.backbones:
        if bb_name not in BACKBONES: continue

        print(f"\n{'='*60}")
        print(f"  Warmup Control: {bb_name} x {args.task}")
        print(f"{'='*60}")

        base_model = load_model(bb_name, device, args.ibot_checkpoint)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads
        config.num_classes = 100

        if bb_name not in all_results:
            all_results[bb_name] = {}

        for cond_name, warmup_epochs in conditions:
            key = f"{cond_name}_lr1e-2"
            if key in all_results.get(bb_name, {}):
                print(f"  {bb_name} {cond_name}: already done"); continue

            print(f"\n  --- {bb_name}, VPT p=5, LR=1e-2, warmup={warmup_epochs} epochs ---")

            seed_results = {}
            for seed in SEEDS:
                train_loader, val_loader, test_loader = get_data(args.task, seed)

                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, 100).to(device)
                m = apply_vpt(m, 5, config)
                m = m.to(device)

                val_acc, test_acc = train_vpt_with_warmup(
                    m, train_loader, val_loader, test_loader,
                    lr=1e-2, warmup_epochs=warmup_epochs,
                    total_epochs=100, device=device
                )
                del m; torch.cuda.empty_cache()

                seed_results[str(seed)] = {
                    'val': float(val_acc), 'test': float(test_acc)
                }
                print(f"    Seed {seed}: val={val_acc:.3f} test={test_acc:.3f}")

            vals = [v['val'] for v in seed_results.values()]
            tests = [v['test'] for v in seed_results.values()]

            all_results[bb_name][key] = {
                'warmup_epochs': warmup_epochs,
                'lr': 1e-2,
                'seeds': seed_results,
                'val_mean': float(np.mean(vals)),
                'test_mean': float(np.mean(tests)),
                'test_std': float(np.std(tests)),
            }
            print(f"    Mean: val={np.mean(vals):.3f} test={np.mean(tests):.3f}±{np.std(tests):.3f}")

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("WARMUP CONTROL SUMMARY")
    print(f"{'='*60}")
    for bb in sorted(all_results.keys()):
        print(f"\n  {bb}:")
        for key, data in sorted(all_results[bb].items()):
            warmup = data['warmup_epochs']
            test = data['test_mean']
            print(f"    warmup={warmup:>2d}: test={test:.3f}±{data['test_std']:.3f}")

        no_warmup = all_results[bb].get('no_warmup_lr1e-2', {}).get('test_mean', 0)
        with_warmup = all_results[bb].get('warmup_10_lr1e-2', {}).get('test_mean', 0)
        if no_warmup and with_warmup:
            diff = with_warmup - no_warmup
            print(f"    Δ(warmup): {diff:+.3f} {'← WARMUP FIXES IT' if diff > 0.1 else '← still crashes' if with_warmup < 0.5 else ''}")


if __name__ == '__main__':
    main()
