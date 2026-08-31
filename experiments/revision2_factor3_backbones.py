"""
Factor 3 Generalization: Non-DINO Self-Supervised Backbones — R2.2

R2: "Testing the rule on self-supervised backbones outside the DINO family 
(iBOT or DINOv1 would be natural candidates) would substantially strengthen 
the framework."

Tests whether Factor 3 (DINO exceptionalism) is:
- Specific to self-distillation (DINOv1/v2 fail, others work)
- General to self-supervised models (all SSL fail)
- DINO-architecture-specific (only DINOv2 fails)

IMPORTANT: Before running, verify model names with:
  python -c "import timm; print(timm.list_models('*mae*', pretrained=True))"
  python -c "import timm; print(timm.list_models('*beit*', pretrained=True))"

Usage:
    python revision2_factor3_backbones.py
    python revision2_factor3_backbones.py --backbones DINOv2 MAE DeiT-III
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


def discover_backbones():
    """Discover available backbone models in timm."""
    candidates = {
        # DINO family (expected to resist VPT)
        'DINOv2': [
            'vit_base_patch14_dinov2.lvd142m',
        ],
        'DINOv1': [
            'vit_base_patch16_224.dino',
            'vit_base_patch8_224.dino',
        ],
        # Non-DINO self-supervised
        'BEiT': [
            'beit_base_patch16_224.in22k_ft_in22k',
            'beit_base_patch16_224.in22k_ft_in22k_in1k',
        ],
        'BEiT3': [
            'beit3_base_patch16_224.in22k_ft_in1k',
            'beit3_base_patch16_224.indomain_in22k_ft_in1k',
        ],
        # Supervised controls
        'DeiT-III': [
            'deit3_base_patch16_224.fb_in1k',
            'deit3_base_patch16_224',
        ],
        'Supervised': [
            'vit_base_patch16_224.augreg_in1k',
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        ],
    }
    
    # Try to find iBOT and DINOv1
    for name_pattern in ['*ibot*', '*dino*']:
        try:
            models = timm.list_models(name_pattern, pretrained=True)
            if 'ibot' in name_pattern:
                base_models = [m for m in models if 'base' in m]
                if base_models:
                    candidates['iBOT'] = base_models[:3]
            elif 'dino' in name_pattern:
                # DINOv1 only (exclude v2)
                v1_models = [m for m in models if 'base' in m and 'v2' not in m and 'dinov2' not in m]
                if v1_models:
                    candidates['DINOv1'] = v1_models[:3]
        except:
            pass
    
    # Try MoCo
    try:
        moco_models = timm.list_models('*moco*', pretrained=True)
        base_moco = [m for m in moco_models if 'base' in m]
        if base_moco:
            candidates['MoCo-v3'] = base_moco[:3]
    except:
        pass
    
    # Resolve: try each candidate, keep first that works
    available = {}
    for bb_name, model_list in candidates.items():
        for model_name in model_list:
            try:
                # Quick check — don't download, just verify it exists
                timm.create_model(model_name, pretrained=False)
                available[bb_name] = model_name
                break
            except:
                continue
    
    return available


TEST_TASKS = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']

# Use best LRs from fair comparison experiment
BEST_LRS = {
    'lora': 1e-3,      # default, works well across backbones
    'vpt_default': 1e-2,
    'vpt_dinov2': 1e-3,  # DINOv2 needs lower VPT LR
}


def compute_gradient_metric(model, data_loader, device, num_prompts=5, config=None):
    """Compute linearized VPT gradient magnitude."""
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

    del model_vpt
    torch.cuda.empty_cache()
    return total_grad_norm / max(n_batches, 1)


def compute_sigma_invariant(model):
    """Compute reparameterization-invariant σ²_P (geometric mean)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Subset of backbones to test (e.g., DINOv1 BEiT DeiT-III)')
    parser.add_argument('--tasks', nargs='+', default=TEST_TASKS)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved results, skip completed backbone-task pairs')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    SAVE_PATH = 'results/revision2_factor3.json'

    print("=" * 70)
    print("Factor 3 Generalization: Self-Supervised Backbone Test")
    print("=" * 70)

    # Load existing results if resuming
    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)
        print(f"\n  Resuming: loaded {len(all_results)} backbones from {SAVE_PATH}")
        for bb, r in all_results.items():
            tasks_done = list(r.get('tasks', {}).keys())
            print(f"    {bb}: {tasks_done}")

    # Discover available backbones
    print("\n  Discovering available backbones...")
    available = discover_backbones()
    for bb, model in available.items():
        print(f"    {bb:<15s}: {model}")
    
    if not available:
        print("  ERROR: No backbones found! Check timm installation.")
        return

    # Filter if user specified
    if args.backbones:
        available = {k: v for k, v in available.items() if k in args.backbones}

    img_size = 224  # consistent across all backbones

    for bb_name, model_name in available.items():
        # Check which tasks are already done for this backbone
        done_tasks = []
        if bb_name in all_results and 'tasks' in all_results[bb_name]:
            done_tasks = list(all_results[bb_name]['tasks'].keys())
        
        pending_tasks = [t for t in args.tasks if t in TASKS and t not in done_tasks]
        
        if not pending_tasks:
            print(f"\n  {bb_name}: all tasks done, skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  Backbone: {bb_name} ({model_name})")
        print(f"  Pending tasks: {pending_tasks}")
        print(f"{'='*55}")

        try:
            base_model = timm.create_model(model_name, pretrained=True,
                                            img_size=img_size).to(device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Check model has expected structure
        if not hasattr(base_model, 'blocks') or not hasattr(base_model, 'embed_dim'):
            print(f"  SKIP: model doesn't have expected ViT structure")
            del base_model; torch.cuda.empty_cache()
            continue

        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = base_model.embed_dim // base_model.blocks[0].attn.num_heads

        # Compute invariant σ²_P
        sigma_p = compute_sigma_invariant(base_model)
        
        # Determine VPT LR based on backbone type
        is_dino = 'dino' in bb_name.lower() or 'dino' in model_name.lower()
        # DINOv1 uses self-distillation too — should be DINO family
        vpt_lr = BEST_LRS['vpt_dinov2'] if is_dino else BEST_LRS['vpt_default']

        # Initialize or resume backbone results
        if bb_name not in all_results:
            all_results[bb_name] = {
                'sigma_p': float(sigma_p), 'model': model_name,
                'vpt_lr': float(vpt_lr), 'is_dino': is_dino, 'tasks': {}
            }
        
        print(f"  σ²_P (invariant) = {sigma_p:.4f}, VPT LR = {vpt_lr:.0e}")

        for task in pending_tasks:
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
            config.lr = BEST_LRS['lora']
            lora_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            del model; torch.cuda.empty_cache()

            # VPT_p5 at backbone-appropriate LR
            model = deepcopy(base_model)
            model.head = nn.Linear(config.embed_dim, num_classes).to(device)
            model = apply_vpt(model, 5, config)
            model = model.to(device)
            config.lr = vpt_lr
            vpt_acc = train_and_evaluate(model, train_loader, val_loader, config, device)
            del model; torch.cuda.empty_cache()

            # Gradient metric
            model_tmp = deepcopy(base_model)
            model_tmp.head = nn.Linear(config.embed_dim, num_classes).to(device)
            grad_mag = compute_gradient_metric(model_tmp, val_loader, device,
                                                config=config)
            del model_tmp; torch.cuda.empty_cache()

            winner = 'L' if lora_acc > vpt_acc + 0.02 else \
                     'V' if vpt_acc > lora_acc + 0.02 else 'T'

            all_results[bb_name]['tasks'][task] = {
                'lora': float(lora_acc), 'vpt': float(vpt_acc),
                'winner': winner, 'grad_mag': float(grad_mag)
            }

            print(f"    {task:<12s}: LoRA={lora_acc:.3f} VPT={vpt_acc:.3f} "
                  f"→ {winner}  grad={grad_mag:.2f}")

            # INCREMENTAL SAVE after each task
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        # Update summary for this backbone
        wins = {'L': 0, 'T': 0, 'V': 0}
        for t, r in all_results[bb_name]['tasks'].items():
            wins[r['winner']] += 1
        all_results[bb_name]['wins'] = wins
        
        grad_values = [r['grad_mag'] for r in all_results[bb_name]['tasks'].values()]
        all_results[bb_name]['mean_grad'] = float(np.mean(grad_values)) if grad_values else 0

        print(f"\n  L/T/V = {wins['L']}/{wins['T']}/{wins['V']}, "
              f"mean grad = {all_results[bb_name]['mean_grad']:.2f}")

        # Save after backbone complete
        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        del base_model; torch.cuda.empty_cache()

    # Cross-backbone summary
    print(f"\n{'='*70}")
    print("CROSS-BACKBONE FACTOR 3 ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'Backbone':<15s} {'σ²_P':>8s} {'Grad':>8s} {'L/T/V':>10s} {'DINO?':>6s} {'VPT viable?'}")
    for bb, r in sorted(all_results.items(), 
                         key=lambda x: x[1].get('mean_grad', 0), reverse=True):
        w = r.get('wins', {})
        viable = "Yes" if w.get('V', 0) > 0 else "No"
        dino = "Yes" if r.get('is_dino', False) else "No"
        ltv = f"{w.get('L',0)}/{w.get('T',0)}/{w.get('V',0)}"
        print(f"  {bb:<15s} {r['sigma_p']:>8.4f} {r.get('mean_grad', 0):>8.2f} "
              f"{ltv:>10s} {dino:>6s} {viable}")

    # Key question
    print(f"\n  KEY QUESTION: Is VPT resistance specific to DINO self-distillation?")
    dino_bbs = [bb for bb, r in all_results.items() if r.get('is_dino', False)]
    ssl_non_dino = [bb for bb, r in all_results.items() 
                    if not r.get('is_dino', False) and bb in ['MAE', 'BEiT', 'BEiTv2', 'BEiT3', 'iBOT', 'MoCo-v3', 'DINOv1']]
    sup_bbs = [bb for bb, r in all_results.items()
               if bb in ['DeiT-III', 'Supervised']]
    
    for group_name, group in [('DINO family', dino_bbs), 
                               ('SSL non-DINO', ssl_non_dino),
                               ('Supervised', sup_bbs)]:
        if not group:
            continue
        vpt_wins = sum(all_results[bb]['wins'].get('V', 0) for bb in group)
        total = sum(sum(all_results[bb]['wins'].values()) for bb in group)
        print(f"    {group_name:<20s}: {vpt_wins}/{total} VPT wins")

    os.makedirs('results', exist_ok=True)
    with open('results/revision2_factor3.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/revision2_factor3.json")


if __name__ == '__main__':
    main()
