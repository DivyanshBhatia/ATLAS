"""
Compute attention entropy across backbones.

Hypothesis: DINO models have LOWER attention entropy (more peaked,
more confident) → harder for VPT to steer → explains DINO exceptionalism.

If confirmed, attention entropy becomes a training-free "attention quality"
metric that completes the 3-factor framework.

Usage:
    python compute_attention_entropy.py
    python compute_attention_entropy.py --task cifar10
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn.functional as F
import timm
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from config import setup_device


BACKBONES = {
    'dinov2':     ('vit_base_patch14_dinov2.lvd142m', 'DINOv2 ViT-B/14', 224),
    'dinov2reg':  ('vit_base_patch14_reg4_dinov2.lvd142m', 'DINOv2-reg ViT-B/14', 224),
    'clip':       ('vit_base_patch16_clip_224', 'CLIP ViT-B/16', 224),
    'supervised': ('vit_base_patch16_224.augreg2_in21k_ft_in1k', 'Supervised ViT-B/16', 224),
    'mae':        ('vit_base_patch16_224.mae', 'MAE ViT-B/16', 224),
    'deit3':      ('deit3_base_patch16_224', 'DeiT-III ViT-B/16', 224),
}


def compute_attention_entropy(model, dataloader, device, max_batches=10):
    """Compute mean attention entropy across all heads and layers."""
    model.eval()
    all_entropies = []  # per layer
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            attn = F.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (d ** -0.5), dim=-1)
            # Entropy: -sum(p * log(p)), averaged over batch, heads, query positions
            entropy = -(attn * (attn + 1e-10).log()).sum(-1).mean(dim=(0, 2))  # [H]
            while len(all_entropies) <= layer_idx:
                all_entropies.append([])
            all_entropies[layer_idx].append(entropy.detach().cpu())
        return hook_fn

    for idx, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for bi, (bx, _) in enumerate(dataloader):
            if bi >= max_batches:
                break
            model.forward_features(bx.to(device))

    for h in hooks:
        h.remove()

    # Aggregate
    layer_entropies = []
    for layer_data in all_entropies:
        if layer_data:
            mean_entropy = torch.stack(layer_data).mean(0)  # [H]
            layer_entropies.append(mean_entropy)

    if not layer_entropies:
        return None, None, None

    all_h = torch.stack(layer_entropies)  # [L, H]
    mean_entropy = all_h.mean().item()
    std_entropy = all_h.std().item()
    
    # Also compute "attention concentration" = max attention weight (mean)
    # Lower entropy = more concentrated = harder to steer
    return mean_entropy, std_entropy, all_h.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cifar10')
    args = parser.parse_args()

    device = setup_device()

    # Load one dataset for attention extraction
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3) if args.task in ['mnist', 'fashionmnist'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    task_loaders = {
        'cifar10': lambda: datasets.CIFAR10('./data', True, download=True, transform=tfm),
        'svhn': lambda: datasets.SVHN('./data', 'train', download=True, transform=tfm),
        'gtsrb': lambda: datasets.GTSRB('./data', 'train', download=True, transform=tfm),
    }

    if args.task not in task_loaders:
        print(f"Available: {list(task_loaders.keys())}")
        return

    ds = task_loaders[args.task]()
    ds = Subset(ds, torch.randperm(len(ds))[:500].tolist())
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"Attention Entropy on {args.task} (500 samples)")
    print(f"{'='*65}")
    print(f"\n  {'Backbone':<25s} {'Mean Entropy':>13s} {'Std':>8s} {'σ_P²':>7s} {'VPT wins':>10s}")
    print(f"  {'-'*63}")

    vpt_wins = {
        'dinov2': '0/9', 'dinov2reg': '0/9', 'clip': '2/9',
        'supervised': '2/7', 'mae': '1/8', 'deit3': '2/8'
    }

    results = {}
    for key, (model_name, display_name, img_size) in BACKBONES.items():
        try:
            model = timm.create_model(model_name, pretrained=True,
                                       img_size=img_size).to(device)
            mean_ent, std_ent, _ = compute_attention_entropy(model, dl, device)

            if mean_ent is not None:
                # Compute σ_P²
                total, count = 0.0, 0
                for name, param in model.named_parameters():
                    if any(t in name for t in ['qkv.weight', 'proj.weight']):
                        total += param.float().norm().item() ** 2
                        count += 1
                d_h = model.embed_dim // model.blocks[0].attn.num_heads
                sigma = total / (count * d_h) if count > 0 else 0

                print(f"  {display_name:<25s} {mean_ent:>13.4f} {std_ent:>8.4f} "
                      f"{sigma:>7.1f} {vpt_wins.get(key, '?'):>10s}")
                results[key] = {'entropy': mean_ent, 'sigma': sigma}

            del model
            torch.cuda.empty_cache()
            import gc; gc.collect()

        except Exception as e:
            print(f"  {display_name:<25s} ERROR: {str(e)[:40]}")

    # Analysis
    print(f"\n  {'='*60}")
    dino_ent = [results[k]['entropy'] for k in ['dinov2', 'dinov2reg'] if k in results]
    non_dino_ent = [results[k]['entropy'] for k in ['clip', 'supervised', 'mae', 'deit3'] if k in results]

    if dino_ent and non_dino_ent:
        print(f"\n  DINO family mean entropy:     {np.mean(dino_ent):.4f}")
        print(f"  Non-DINO family mean entropy: {np.mean(non_dino_ent):.4f}")
        if np.mean(dino_ent) < np.mean(non_dino_ent):
            print(f"  → DINO has LOWER entropy (more concentrated attention)")
            print(f"  → Supports hypothesis: concentrated attention resists VPT steering")
        else:
            print(f"  → DINO does NOT have lower entropy")
            print(f"  → Attention quality difference is not captured by entropy alone")


if __name__ == '__main__':
    main()
