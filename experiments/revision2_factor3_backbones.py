"""
Factor 3 Generalization: Non-DINO Self-Supervised Backbones — R2.2

R2: "Testing the rule on self-supervised backbones outside the DINO family 
(iBOT or DINOv1 would be natural candidates) would substantially strengthen 
the framework."

Tests whether Factor 3 (DINO exceptionalism) is:
- Specific to self-distillation (DINOv1/v2 fail, others work)
- General to self-supervised models (all SSL fail)
- DINO-architecture-specific (only DINOv2 fails)

Usage:
    python revision2_factor3_backbones.py
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora, apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader

# Self-supervised backbones beyond DINO
SSL_BACKBONES = {
    # DINO family (expected to resist VPT)
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 518),
    # Non-DINO self-supervised (the test cases)
    'MAE': ('vit_base_patch16_mae', 224),
    'BEiTv2': ('beitv2_base_patch16_224.in1k_ft_in22k', 224),
    # Supervised controls
    'DeiT-III': ('deit3_base_patch16_224.fb_in1k', 224),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 224),
}

# Try to add iBOT if available
try:
    _ = timm.create_model('vit_base_patch16_224.ibot_in1k', pretrained=False)
    SSL_BACKBONES['iBOT'] = ('vit_base_patch16_224.ibot_in1k', 224)
except:
    pass

TEST_TASKS = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']


def compute_gradient_metric(model, data_loader, device, num_prompts=5, config=None):
    """Compute linearized VPT gradient magnitude."""
    from copy import deepcopy

    model_vpt = deepcopy(model).to(device)
    model_vpt = apply_vpt(model_vpt, num_prompts, config)
    model_vpt = model_vpt.to(device)
    model_vpt.eval()

    prompt_params = []
    for name, param in model_vpt.named_parameters():
        if 'prompt' in name:
            param.requires_grad_(True)
            prompt_params.append(param)
        else:
            param.requires_grad_(False)

    if not prompt_params:
        return 0.0

    total_grad_norm = 0.0
    n_batches = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model_vpt(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        model_vpt.zero_grad()
        loss.backward()

        batch_grad = sum(p.grad.float().norm().item()**2 for p in prompt_params if p.grad is not None)
        total_grad_norm += batch_grad
        n_batches += 1
        if n_batches >= 5:
            break

    del model_vpt
    torch.cuda.empty_cache()
    return total_grad_norm / max(n_batches, 1)


def main():
    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("Factor 3 Generalization: Self-Supervised Backbone Test")
    print("=" * 70)

    all_results = {}

    for bb_name, (model_name, img_size) in SSL_BACKBONES.items():
        print(f"\n{'='*55}")
        print(f"  Backbone: {bb_name} ({model_name})")
        print(f"{'='*55}")

        try:
            base_model = timm.create_model(model_name, pretrained=True,
                                            img_size=img_size).to(device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        # Compute σ²_P
        sigma_p = 0
        count = 0
        for block in base_model.blocks:
            W = block.attn.qkv.weight.float()
            d = base_model.embed_dim
            for i in range(3):
                sigma_p += W[i*d:(i+1)*d].norm()**2
                count += 1
            sigma_p += block.attn.proj.weight.float().norm()**2
            count += 1
        sigma_p = (sigma_p / (count * config.head_dim)).item()

        bb_results = {'sigma_p': sigma_p, 'tasks': {}}
        print(f"  σ²_P = {sigma_p:.2f}")

        for task in TEST_TASKS:
            if task not in TASKS:
                continue
            num_classes = TASKS[task][0]

            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = torch.utils.data.random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            # LoRA_r8
            model = deepcopy(base_model)
            model.head = nn.Linear(config.embed_dim, num_classes).to(device)
            model = apply_lora(model, 8, config)
            model = model.to(device)
            config.lr = 1e-3
            lora_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            del model; torch.cuda.empty_cache()

            # VPT_p5
            model = deepcopy(base_model)
            model.head = nn.Linear(config.embed_dim, num_classes).to(device)
            model = apply_vpt(model, 5, config)
            model = model.to(device)
            config.lr = 1e-2
            vpt_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            del model; torch.cuda.empty_cache()

            # Gradient metric
            model_tmp = deepcopy(base_model)
            model_tmp.head = nn.Linear(config.embed_dim, num_classes).to(device)
            grad_mag = compute_gradient_metric(model_tmp, val_loader, device,
                                                config=config)
            del model_tmp; torch.cuda.empty_cache()

            winner = 'L' if lora_acc > vpt_acc + 0.02 else 'V' if vpt_acc > lora_acc + 0.02 else 'T'

            bb_results['tasks'][task] = {
                'lora': lora_acc, 'vpt': vpt_acc,
                'winner': winner, 'grad_mag': grad_mag
            }

            print(f"  {task:<12s}: LoRA={lora_acc:.3f} VPT={vpt_acc:.3f} "
                  f"→ {winner}  grad={grad_mag:.2f}")

        # Summary for this backbone
        wins = {'L': 0, 'T': 0, 'V': 0}
        for t, r in bb_results['tasks'].items():
            wins[r['winner']] += 1
        bb_results['wins'] = wins
        mean_grad = np.mean([r['grad_mag'] for r in bb_results['tasks'].values()])
        bb_results['mean_grad'] = mean_grad

        print(f"\n  L/T/V = {wins['L']}/{wins['T']}/{wins['V']}, "
              f"mean grad = {mean_grad:.2f}")

        all_results[bb_name] = bb_results
        del base_model; torch.cuda.empty_cache()

    # Cross-backbone summary
    print(f"\n{'='*70}")
    print("CROSS-BACKBONE FACTOR 3 ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'Backbone':<15s} {'σ²_P':>8s} {'Grad':>8s} {'L/T/V':>8s} {'VPT viable?'}")
    for bb, r in sorted(all_results.items(), key=lambda x: x[1].get('mean_grad', 0), reverse=True):
        w = r.get('wins', {})
        viable = "Yes" if w.get('V', 0) > 0 else "No"
        print(f"  {bb:<15s} {r['sigma_p']:>8.2f} {r.get('mean_grad', 0):>8.2f} "
              f"{w.get('L',0)}/{w.get('T',0)}/{w.get('V',0):>8s} {viable}")

    os.makedirs('results', exist_ok=True)
    with open('results/revision2_factor3.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_factor3.json")


if __name__ == '__main__':
    main()
