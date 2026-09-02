"""
Non-DINO Self-Supervised Control Experiment

The critical missing experiment: does VPT resistance come from
self-distillation specifically, or self-supervised pretraining in general?

Tests MAE (masked autoencoder) and MoCo-v3 (contrastive without distillation)
as non-DINO SSL controls. If they allow VPT (like CLIP), the DINO effect is
self-distillation-specific. If they resist VPT (like DINOv2), it's a general
SSL property.

Model loading strategy:
1. Try timm first
2. Fall back to torch.hub (Facebook Research repos)
3. Fall back to direct URL download

Usage:
    python revision2_ssl_control.py
    python revision2_ssl_control.py --models mae moco
    python revision2_ssl_control.py --resume --tasks svhn gtsrb
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
from exp2_comparison import apply_lora, apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SAVE_PATH = 'results/revision2_ssl_control.json'
TEST_TASKS = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']


def try_load_model(model_candidates, img_size=224):
    """Try loading a model from multiple sources."""
    
    # Strategy 1: timm
    for name in model_candidates.get('timm', []):
        try:
            model = timm.create_model(name, pretrained=True, img_size=img_size)
            print(f"    Loaded from timm: {name}")
            return model, name
        except Exception as e:
            print(f"    timm {name}: {str(e)[:60]}")
    
    # Strategy 2: torch.hub
    for repo, name in model_candidates.get('hub', []):
        try:
            model = torch.hub.load(repo, name, pretrained=True)
            print(f"    Loaded from hub: {repo}/{name}")
            return model, f"hub:{repo}/{name}"
        except Exception as e:
            print(f"    hub {repo}/{name}: {str(e)[:60]}")
    
    # Strategy 3: direct checkpoint URL
    for url, arch_fn in model_candidates.get('url', []):
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                # Remove prefix if present
                state_dict = {k.replace('module.', '').replace('base_encoder.', ''): v 
                             for k, v in state_dict.items()}
            
            model = arch_fn()
            # Try loading, ignoring size mismatches
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"    Loaded from URL (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
            return model, f"url:{url.split('/')[-1]}"
        except Exception as e:
            print(f"    URL load: {str(e)[:60]}")
    
    return None, None


def get_vit_base():
    """Create a ViT-B/16 architecture without pretrained weights."""
    return timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)


SSL_MODELS = {
    'MAE': {
        'timm': [
            'vit_base_patch16_224.mae',
            'vit_base_patch16_mae',
        ],
        'hub': [
            ('facebookresearch/mae', 'mae_vit_base_patch16'),
        ],
        'url': [
            ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
             get_vit_base),
        ],
    },
    'MoCo-v3': {
        'timm': [
            'vit_base_patch16_224.moco_v3',
            'vit_base_patch16_moco_v3',
        ],
        'hub': [
            ('facebookresearch/moco-v3', 'vit_base'),
        ],
        'url': [
            ('https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar',
             get_vit_base),
        ],
    },
    # Controls (should already be in Factor 3 results)
    'DINOv2': {
        'timm': ['vit_base_patch14_dinov2.lvd142m'],
        'hub': [],
        'url': [],
    },
    'CLIP': {
        'timm': ['vit_base_patch16_clip_224.openai'],
        'hub': [],
        'url': [],
    },
    'DeiT-III': {
        'timm': ['deit3_base_patch16_224.fb_in1k'],
        'hub': [],
        'url': [],
    },
}


def compute_sigma_invariant(model):
    """Compute reparameterization-invariant sigma_P."""
    d = model.embed_dim
    per_layer = []
    for block in model.blocks:
        W_qkv = block.attn.qkv.weight.float()
        W_q = W_qkv[:d]
        W_k = W_qkv[d:2*d]
        W_v = W_qkv[2*d:]
        W_o = block.attn.proj.weight.float()
        
        geom_attn = W_q.norm().item() * W_k.norm().item()
        geom_val = W_v.norm().item() * W_o.norm().item()
        per_layer.append((geom_attn + geom_val) / (2 * d))
    
    return np.mean(per_layer)


def compute_gradient_metric(model, data_loader, device, num_prompts=5, config=None):
    """Compute linearized VPT gradient magnitude."""
    model_vpt = deepcopy(model).to(device)
    try:
        model_vpt = apply_vpt(model_vpt, num_prompts, config)
    except Exception:
        del model_vpt; torch.cuda.empty_cache()
        return -1.0
    
    model_vpt = model_vpt.to(device)
    model_vpt.eval()

    prompt_params = [p for n, p in model_vpt.named_parameters() if 'prompt' in n]
    for p in prompt_params:
        p.requires_grad_(True)
    for n, p in model_vpt.named_parameters():
        if 'prompt' not in n:
            p.requires_grad_(False)

    if not prompt_params:
        del model_vpt; torch.cuda.empty_cache()
        return 0.0

    total_grad_norm = 0.0
    n_batches = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model_vpt(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        model_vpt.zero_grad()
        loss.backward()

        batch_grad = sum(p.grad.float().norm().item()**2 
                        for p in prompt_params if p.grad is not None)
        total_grad_norm += batch_grad
        n_batches += 1
        if n_batches >= 5:
            break

    del model_vpt; torch.cuda.empty_cache()
    return total_grad_norm / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['MAE', 'MoCo-v3'])
    parser.add_argument('--tasks', nargs='+', default=TEST_TASKS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    print("=" * 70)
    print("NON-DINO SSL CONTROL EXPERIMENT")
    print(f"Models: {args.models}, Tasks: {args.tasks}")
    print("=" * 70)

    # Load existing results
    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"  Resuming: {list(all_results.keys())}")

    img_size = 224

    for model_key in args.models:
        if model_key not in SSL_MODELS:
            print(f"  Unknown model: {model_key}")
            continue
        
        # Check what's done
        done_tasks = list(all_results.get(model_key, {}).get('tasks', {}).keys())
        pending = [t for t in args.tasks if t in TASKS and t not in done_tasks]
        
        if not pending:
            print(f"\n  {model_key}: all tasks done, skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  Loading {model_key}...")
        print(f"  Pending tasks: {pending}")
        print(f"{'='*55}")

        base_model, source = try_load_model(SSL_MODELS[model_key], img_size)
        
        if base_model is None:
            print(f"  FAILED to load {model_key} from any source")
            continue

        # Verify ViT structure
        if not hasattr(base_model, 'blocks') or not hasattr(base_model, 'embed_dim'):
            print(f"  SKIP: no ViT structure (blocks/embed_dim)")
            del base_model; torch.cuda.empty_cache()
            continue

        base_model = base_model.to(device)
        
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        # Compute sigma_P
        sigma_p = compute_sigma_invariant(base_model)
        
        # LR based on sigma_P
        vpt_lr = 1e-3 if sigma_p < 0.5 else 1e-2
        
        # Determine if DINO-like
        is_ssl = model_key in ['MAE', 'MoCo-v3', 'BYOL', 'SimCLR']
        is_dino = 'dino' in model_key.lower()

        if model_key not in all_results:
            all_results[model_key] = {
                'sigma_p': float(sigma_p),
                'source': source,
                'vpt_lr': float(vpt_lr),
                'is_dino': is_dino,
                'is_ssl': is_ssl,
                'tasks': {}
            }

        print(f"  sigma_P = {sigma_p:.4f}, VPT LR = {vpt_lr:.0e}")
        print(f"  SSL: {is_ssl}, DINO: {is_dino}")

        for task in pending:
            num_classes = TASKS[task][0]

            ds = load_dataset(task, img_size, max_samples=1000)
            n_val = min(200, len(ds) // 5)
            train_ds, val_ds = random_split(
                ds, [len(ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

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
            try:
                model = apply_vpt(model, 5, config)
                model = model.to(device)
                config.lr = vpt_lr
                vpt_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            except Exception as e:
                print(f"    VPT failed: {str(e)[:60]}")
                vpt_acc = -1.0
            del model; torch.cuda.empty_cache()

            # Gradient metric
            model_tmp = deepcopy(base_model)
            model_tmp.head = nn.Linear(config.embed_dim, num_classes).to(device)
            grad_mag = compute_gradient_metric(model_tmp, val_loader, device, config=config)
            del model_tmp; torch.cuda.empty_cache()

            if vpt_acc < 0:
                winner = 'L (VPT incompatible)'
            else:
                winner = 'L' if lora_acc > vpt_acc + 0.02 else \
                         'V' if vpt_acc > lora_acc + 0.02 else 'T'

            all_results[model_key]['tasks'][task] = {
                'lora': float(lora_acc),
                'vpt': float(vpt_acc),
                'winner': winner,
                'grad_mag': float(grad_mag),
            }

            print(f"    {task:<12s}: LoRA={lora_acc:.3f} VPT={vpt_acc:.3f} "
                  f"-> {winner}  grad={grad_mag:.2f}")

            # Incremental save
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        # Summary
        tasks_data = all_results[model_key]['tasks']
        wins = {'L': 0, 'T': 0, 'V': 0}
        grad_vals = []
        for t, r in tasks_data.items():
            w = r['winner']
            if w.startswith('L'): wins['L'] += 1
            elif w.startswith('V'): wins['V'] += 1
            else: wins['T'] += 1
            if r['grad_mag'] > 0:
                grad_vals.append(r['grad_mag'])
        
        all_results[model_key]['wins'] = wins
        all_results[model_key]['mean_grad'] = float(np.mean(grad_vals)) if grad_vals else -1

        print(f"\n  {model_key}: L/T/V = {wins['L']}/{wins['T']}/{wins['V']}, "
              f"mean grad = {all_results[model_key]['mean_grad']:.2f}")

        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        del base_model; torch.cuda.empty_cache()

    # Cross-model summary
    print(f"\n{'='*70}")
    print("SSL CONTROL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<15s} {'sigma_P':>8s} {'Grad':>8s} {'L/T/V':>10s} {'SSL?':>5s} {'DINO?':>6s}")
    for m, r in sorted(all_results.items(), key=lambda x: x[1].get('mean_grad', 0), reverse=True):
        w = r.get('wins', {})
        ltv = f"{w.get('L',0)}/{w.get('T',0)}/{w.get('V',0)}"
        print(f"  {m:<15s} {r['sigma_p']:>8.4f} {r.get('mean_grad', -1):>8.2f} "
              f"{ltv:>10s} {'Y' if r.get('is_ssl') else 'N':>5s} {'Y' if r.get('is_dino') else 'N':>6s}")

    print(f"\n  KEY QUESTION: Is VPT resistance specific to self-distillation?")
    dino_models = [m for m, r in all_results.items() if r.get('is_dino')]
    ssl_non_dino = [m for m, r in all_results.items() if r.get('is_ssl') and not r.get('is_dino')]
    non_ssl = [m for m, r in all_results.items() if not r.get('is_ssl') and not r.get('is_dino')]

    for group_name, group in [('DINO', dino_models), ('SSL non-DINO', ssl_non_dino), ('Non-SSL', non_ssl)]:
        if not group:
            continue
        total_tasks = sum(len(all_results[m].get('tasks', {})) for m in group)
        vpt_wins = sum(all_results[m].get('wins', {}).get('V', 0) for m in group if 'wins' in all_results[m])
        ties = sum(all_results[m].get('wins', {}).get('T', 0) for m in group if 'wins' in all_results[m])
        comp = (vpt_wins + ties) / total_tasks * 100 if total_tasks > 0 else 0
        print(f"    {group_name:<15s}: {vpt_wins + ties}/{total_tasks} VPT competitive ({comp:.0f}%)")


if __name__ == '__main__':
    main()
