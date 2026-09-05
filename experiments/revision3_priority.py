"""
Pre-Submission Required Experiments

1. LoRA LR sweep on CLIP, DeiT-III, Supervised (Priority 1)
2. σ²_P with LN-folded weights (Priority 2)  
3. iBOT backbone test (Priority 3)

Usage:
    # All three:
    python revision3_priority.py

    # Just LoRA sweep:
    python revision3_priority.py --mode lora_sweep

    # Just LN-folded sigma:
    python revision3_priority.py --mode sigma_ln

    # Just iBOT:
    python revision3_priority.py --mode ibot
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

SEEDS = [42, 123, 456]
LORA_LRS = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
VPT_LRS_LOW = [5e-4, 1e-3, 2e-3]
VPT_LRS_HIGH = [5e-3, 1e-2, 2e-2]
TASKS_5 = ['cifar100', 'svhn', 'gtsrb', 'eurosat', 'dtd']

BACKBONES = {
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DeiT-III': ('deit3_base_patch16_224', 1.04),
    'Supervised': ('vit_base_patch16_224.augreg_in1k', 1.60),
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
}


# ============================================================
# EXPERIMENT 1: LoRA LR Sweep on All Backbones
# ============================================================
def run_lora_sweep(device, config, backbone_filter=None, task_filter=None):
    """Sweep LoRA LR on CLIP, DeiT-III, Supervised. 3 seeds at best LR."""
    SAVE_PATH = 'results/revision3_lora_sweep.json'
    
    all_results = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    backbones_to_run = {k: v for k, v in BACKBONES.items() if k in (backbone_filter or BACKBONES.keys())}

    for bb_name, (model_name, sigma_p) in backbones_to_run.items():
        done_tasks = list(all_results.get(bb_name, {}).get('tasks', {}).keys())
        pending = [t for t in (task_filter or TASKS_5) if t in TASKS and t not in done_tasks]
        if not pending:
            print(f"\n  {bb_name}: all done"); continue

        print(f"\n{'='*55}")
        print(f"  {bb_name} ({model_name}, sigma={sigma_p:.2f})")
        print(f"  Pending: {pending}")
        print(f"{'='*55}")

        base_model = timm.create_model(model_name, pretrained=True, img_size=224).to(device)
        config.embed_dim = base_model.embed_dim
        config.num_layers = len(base_model.blocks)
        config.num_heads = base_model.blocks[0].attn.num_heads
        config.head_dim = config.embed_dim // config.num_heads

        vpt_lrs = VPT_LRS_LOW if sigma_p < 0.5 else VPT_LRS_HIGH

        if bb_name not in all_results:
            all_results[bb_name] = {'sigma_p': sigma_p, 'tasks': {}}

        for task in pending:
            num_classes = TASKS[task][0]
            config.num_classes = num_classes
            print(f"\n  --- {bb_name} x {task} ---")

            task_results = {}

            # LoRA sweep (seed 42)
            print(f"  LoRA LR sweep:")
            best_lora_lr, best_lora_acc = 1e-3, 0
            for lr in LORA_LRS:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds)//5)
                tds, vds = random_split(ds, [len(ds)-nv, nv], generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)
                m = deepcopy(base_model); m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_lora(m, 8, config); m = m.to(device); config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()
                task_results[f'lora_lr{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")
                if acc > best_lora_acc: best_lora_acc = acc; best_lora_lr = lr

            # VPT sweep (seed 42)
            print(f"  VPT LR sweep:")
            best_vpt_lr, best_vpt_acc = vpt_lrs[1], 0
            for lr in vpt_lrs:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds)//5)
                tds, vds = random_split(ds, [len(ds)-nv, nv], generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)
                m = deepcopy(base_model); m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config); m = m.to(device); config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()
                task_results[f'vpt_lr{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}")
                if acc > best_vpt_acc: best_vpt_acc = acc; best_vpt_lr = lr

            # 3-seed at best LRs
            print(f"\n  3-seed (LoRA lr={best_lora_lr:.0e}, VPT lr={best_vpt_lr:.0e}):")
            lora_seeds, vpt_seeds = [], []
            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds)//5)
                tds, vds = random_split(ds, [len(ds)-nv, nv], generator=torch.Generator().manual_seed(seed))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)
                
                m = deepcopy(base_model); m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_lora(m, 8, config); m = m.to(device); config.lr = best_lora_lr
                la = train_and_evaluate(m, tl, vl, config, device); del m; torch.cuda.empty_cache()
                
                m = deepcopy(base_model); m.head = nn.Linear(config.embed_dim, num_classes).to(device)
                m = apply_vpt(m, 5, config); m = m.to(device); config.lr = best_vpt_lr
                va = train_and_evaluate(m, tl, vl, config, device); del m; torch.cuda.empty_cache()
                
                lora_seeds.append(la); vpt_seeds.append(va)
                print(f"    Seed {seed}: LoRA={la:.3f} VPT={va:.3f}")

            lm, ls = np.mean(lora_seeds), np.std(lora_seeds)
            vm, vs = np.mean(vpt_seeds), np.std(vpt_seeds)
            gap = lm - vm
            task_results['best_lora_lr'] = float(best_lora_lr)
            task_results['best_vpt_lr'] = float(best_vpt_lr)
            task_results['lora_3seed'] = {'mean': float(lm), 'std': float(ls), 'seeds': [float(x) for x in lora_seeds]}
            task_results['vpt_3seed'] = {'mean': float(vm), 'std': float(vs), 'seeds': [float(x) for x in vpt_seeds]}
            task_results['gap'] = float(gap)
            
            winner = 'LoRA' if gap > 0.02 else 'VPT' if gap < -0.02 else 'TIE'
            print(f"  RESULT: LoRA={lm:.3f}+/-{ls:.3f} VPT={vm:.3f}+/-{vs:.3f} gap={gap:+.3f} -> {winner}")

            all_results[bb_name]['tasks'][task] = task_results
            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        del base_model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("LORA SWEEP SUMMARY")
    print(f"{'='*60}")
    for bb, data in all_results.items():
        print(f"\n  {bb}:")
        for task, tdata in data.get('tasks', {}).items():
            l3 = tdata.get('lora_3seed', {})
            v3 = tdata.get('vpt_3seed', {})
            if l3 and v3:
                gap = l3['mean'] - v3['mean']
                w = 'L' if gap > 0.02 else 'V' if gap < -0.02 else 'T'
                print(f"    {task}: LoRA={l3['mean']:.3f} VPT={v3['mean']:.3f} gap={gap:+.3f} {w} "
                      f"(LoRA lr={tdata['best_lora_lr']:.0e}, VPT lr={tdata['best_vpt_lr']:.0e})")


# ============================================================
# EXPERIMENT 2: σ²_P with LayerNorm-Folded Weights
# ============================================================
def run_sigma_ln(device, config):
    """Compute σ²_P with and without LN folding, test invariance."""
    print(f"\n{'='*60}")
    print("σ²_P WITH LAYERNORM-FOLDED WEIGHTS")
    print(f"{'='*60}")

    all_models = {
        'CLIP': 'vit_base_patch16_clip_224.openai',
        'DINOv1': 'vit_base_patch16_224.dino',
        'DINOv2': 'vit_base_patch14_dinov2.lvd142m',
        'DeiT-III': 'deit3_base_patch16_224.fb_in1k',
        'Supervised': 'vit_base_patch16_224.augreg_in1k',
        'MAE': 'vit_base_patch16_224.mae',
    }

    results = {}
    for bb_name, model_name in all_models.items():
        model = timm.create_model(model_name, pretrained=True, img_size=224)
        d = model.embed_dim

        # Without LN folding (current definition)
        per_layer_raw = []
        per_layer_ln = []
        for block in model.blocks:
            W_qkv = block.attn.qkv.weight.float()
            W_q, W_k, W_v = W_qkv[:d], W_qkv[d:2*d], W_qkv[2*d:]
            W_o = block.attn.proj.weight.float()

            # Raw
            g_attn = W_q.norm().item() * W_k.norm().item()
            g_val = W_v.norm().item() * W_o.norm().item()
            per_layer_raw.append((g_attn + g_val) / (2 * d))

            # LN-folded: W_tilde = diag(gamma) @ W
            ln_weight = block.norm1.weight.float()  # gamma
            W_q_ln = (ln_weight.unsqueeze(0) * W_q)  # broadcast: (d, d) * (1, d)
            W_k_ln = (ln_weight.unsqueeze(0) * W_k)
            W_v_ln = (ln_weight.unsqueeze(0) * W_v)
            # W_o is not affected by norm1's LN

            g_attn_ln = W_q_ln.norm().item() * W_k_ln.norm().item()
            g_val_ln = W_v_ln.norm().item() * W_o.norm().item()
            per_layer_ln.append((g_attn_ln + g_val_ln) / (2 * d))

        sigma_raw = np.mean(per_layer_raw)
        sigma_ln = np.mean(per_layer_ln)

        # Test LN-gain invariance
        # Rescale: gamma -> c*gamma (equivalent to W_Q -> W_Q/c etc after folding)
        deviations = []
        for c in [0.5, 2.0, 5.0, 0.1, 10.0]:
            per_layer_scaled = []
            for block in model.blocks:
                W_qkv = block.attn.qkv.weight.float()
                W_q, W_k, W_v = W_qkv[:d], W_qkv[d:2*d], W_qkv[2*d:]
                W_o = block.attn.proj.weight.float()
                ln_weight = block.norm1.weight.float() * c  # scaled gamma

                W_q_ln = (ln_weight.unsqueeze(0) * W_q)
                W_k_ln = (ln_weight.unsqueeze(0) * W_k)
                W_v_ln = (ln_weight.unsqueeze(0) * W_v)

                g_attn_ln = W_q_ln.norm().item() * W_k_ln.norm().item()
                g_val_ln = W_v_ln.norm().item() * W_o.norm().item()
                per_layer_scaled.append((g_attn_ln + g_val_ln) / (2 * d))

            sigma_scaled = np.mean(per_layer_scaled)
            dev = abs(sigma_scaled - sigma_ln) / sigma_ln
            deviations.append(dev)

        max_dev = max(deviations)
        
        # Test Q/K rescaling invariance on LN-folded version
        qk_devs = []
        for c in [0.5, 2.0, 5.0, 0.1, 10.0]:
            per_layer_qk = []
            for block in model.blocks:
                W_qkv = block.attn.qkv.weight.float()
                W_q, W_k, W_v = W_qkv[:d], W_qkv[d:2*d], W_qkv[2*d:]
                W_o = block.attn.proj.weight.float()
                ln_weight = block.norm1.weight.float()

                W_q_ln = (ln_weight.unsqueeze(0) * W_q) * c
                W_k_ln = (ln_weight.unsqueeze(0) * W_k) / c
                W_v_ln = (ln_weight.unsqueeze(0) * W_v)

                g_attn = W_q_ln.norm().item() * W_k_ln.norm().item()
                g_val = W_v_ln.norm().item() * W_o.norm().item()
                per_layer_qk.append((g_attn + g_val) / (2 * d))

            sigma_qk = np.mean(per_layer_qk)
            qk_devs.append(abs(sigma_qk - sigma_ln) / sigma_ln)

        qk_max = max(qk_devs)

        results[bb_name] = {
            'sigma_raw': float(sigma_raw),
            'sigma_ln': float(sigma_ln),
            'ratio': float(sigma_ln / sigma_raw),
            'ln_gain_invariant': max_dev < 0.01,
            'ln_gain_max_dev': float(max_dev),
            'qk_rescale_invariant': qk_max < 1e-4,
            'qk_rescale_max_dev': float(qk_max),
        }

        print(f"  {bb_name:<12s}: raw={sigma_raw:.4f} LN-folded={sigma_ln:.4f} "
              f"ratio={sigma_ln/sigma_raw:.2f} "
              f"LN-inv={'Y' if max_dev<0.01 else 'N'}(dev={max_dev:.1e}) "
              f"QK-inv={'Y' if qk_max<1e-4 else 'N'}(dev={qk_max:.1e})")

        del model

    # Check ordering preserved
    raw_order = sorted(results.keys(), key=lambda x: results[x]['sigma_raw'])
    ln_order = sorted(results.keys(), key=lambda x: results[x]['sigma_ln'])
    print(f"\n  Raw ordering:       {' < '.join(raw_order)}")
    print(f"  LN-folded ordering: {' < '.join(ln_order)}")
    print(f"  Same ordering: {raw_order == ln_order}")

    os.makedirs('results', exist_ok=True)
    with open('results/revision3_sigma_ln.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================
# EXPERIMENT 3: iBOT Backbone Test
# ============================================================
def run_ibot(device, config):
    """Test iBOT (self-distillation + MIM) as a control."""
    print(f"\n{'='*60}")
    print("iBOT BACKBONE TEST")
    print(f"{'='*60}")

    # Try to find iBOT in timm
    import timm
    ibot_candidates = [
        'vit_base_patch16_224.ibot',
        'vit_base_patch16_ibot',
    ]
    
    # Also check torch hub
    model = None
    source = None
    
    for name in ibot_candidates:
        try:
            model = timm.create_model(name, pretrained=True, img_size=224)
            source = f"timm:{name}"
            break
        except:
            pass

    if model is None:
        # Try torch hub
        try:
            model = torch.hub.load('facebookresearch/dino', 'dino_vitb16', pretrained=True)
            source = "hub:facebookresearch/dino/dino_vitb16"
            print(f"  Note: loaded DINOv1 from hub instead of iBOT (iBOT not available)")
            print(f"  iBOT is not available in timm or torch hub.")
            print(f"  Consider downloading from: https://github.com/bytedance/ibot")
            return
        except:
            pass

    if model is None:
        print("  iBOT not available. Try downloading manually from bytedance/ibot")
        print("  or use DINOv1 as the closest available self-distillation control.")
        return

    print(f"  Loaded: {source}")
    # Run same protocol as Factor 3...
    # (Similar to revision2_ssl_control.py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all', choices=['all', 'lora_sweep', 'sigma_ln', 'ibot'])
    parser.add_argument('--tasks', nargs='+', default=TASKS_5)
    parser.add_argument('--backbones', nargs='+', default=list(BACKBONES.keys()))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    if args.mode in ['all', 'lora_sweep']:
        run_lora_sweep(device, config, args.backbones, args.tasks)

    if args.mode in ['all', 'sigma_ln']:
        run_sigma_ln(device, config)

    if args.mode in ['all', 'ibot']:
        run_ibot(device, config)


if __name__ == '__main__':
    main()
