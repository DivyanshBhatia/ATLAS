"""
Experiment 2: PEFT Method Comparison & Task Characterization

Purpose: Validate Theorems 1 and 2 by comparing LoRA, VPT, Adapter, and LP
across VTAB-1K tasks, and measuring the task descriptors (S_attn, S_feat, α)
that the theory predicts determine the winner.

What we test:
1. Theorem 1 predictions:
   - VPT > LoRA on structured tasks (attention-steering)
   - LoRA > VPT on natural tasks (feature-discrimination)
2. Theorem 2 rate predictions:
   - LoRA accuracy vs rank follows O(r^{1-2α})
   - VPT accuracy vs prompts saturates (sin²θ plateau)
3. Task characterization:
   - Measure S_attn, S_feat^{(c)}, sin²θ for each task
   - Correlate with which method wins
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (ExperimentConfig, setup_device, ensure_dirs, save_results,
                    compute_svd_profile, fit_spectral_decay,
                    compute_attention_shift, compute_feature_shift,
                    compute_quantization_error, compute_sigma_p_sq)

from exp1_spectral import SyntheticVTABDataset, load_real_dataset


# ============================================================================
# PEFT Method Implementations
# ============================================================================

class LoRALayer(nn.Module):
    """LoRA adapter for a linear layer."""
    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        self.original = original
        self.rank = rank
        d_in, d_out = original.in_features, original.out_features
        # Create on the SAME device as the original layer
        device = original.weight.device
        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, d_out, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.randn(d_in, rank, device=device, dtype=dtype) * 0.01)
        # Freeze original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_B @ self.lora_A)


class AdapterLayer(nn.Module):
    """Bottleneck adapter inserted after a transformer block."""
    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def apply_lora(model, rank: int, config: ExperimentConfig):
    """Apply LoRA to attention Q and V projections."""
    for block in model.blocks:
        # timm ViT uses a combined qkv projection
        # We apply LoRA to the full qkv and mask for Q, V only
        d = config.embed_dim
        qkv = block.attn.qkv

        # Create LoRA for the full qkv (simpler, captures Q and V)
        lora = LoRALayer(qkv, rank)
        block.attn.qkv = lora

    # Freeze all except LoRA parameters and head
    for name, param in model.named_parameters():
        if 'lora_' not in name and 'head' not in name:
            param.requires_grad_(False)

    return model


def apply_vpt(model, n_prompts: int, config: ExperimentConfig):
    """Apply VPT-Deep (learnable prompts at every layer)."""
    d = config.embed_dim

    # Detect device from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Store prompts for each layer — on the SAME device as the model
    model.vpt_prompts = nn.ParameterList([
        nn.Parameter(torch.randn(1, n_prompts, d, device=device, dtype=dtype) * 0.02)
        for _ in range(len(model.blocks))
    ])

    # Wrap each block to inject prompts
    original_forwards = []
    for i, block in enumerate(model.blocks):
        original_forward = block.forward
        original_forwards.append(original_forward)

        prompt_param = model.vpt_prompts[i]

        def make_new_forward(orig_fwd, prompt_p):
            def new_forward(x, **kwargs):
                B = x.shape[0]
                prompts = prompt_p.expand(B, -1, -1)
                # Prepend prompts to the sequence
                x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
                x = orig_fwd(x, **kwargs)
                # Remove prompts from output
                x = torch.cat([x[:, :1], x[:, 1+n_prompts:]], dim=1)
                return x
            return new_forward

        block.forward = make_new_forward(original_forward, prompt_param)

    # Freeze all except VPT prompts and head
    for name, param in model.named_parameters():
        if 'vpt_prompts' not in name and 'head' not in name:
            param.requires_grad_(False)

    return model


def apply_adapter(model, bottleneck_dim: int, config: ExperimentConfig):
    """Apply adapter layers after each transformer block."""
    device = next(model.parameters()).device

    # Register adapters as a ModuleList so params are found by named_parameters()
    model.adapters = nn.ModuleList()

    for block in model.blocks:
        adapter = AdapterLayer(config.embed_dim, bottleneck_dim).to(device)
        model.adapters.append(adapter)
        original_forward = block.forward

        def make_new_forward(orig_fwd, adapt):
            def new_forward(x, **kwargs):
                x = orig_fwd(x, **kwargs)
                x = adapt(x)
                return x
            return new_forward

        block.forward = make_new_forward(original_forward, adapter)

    # Freeze all except adapter parameters and head
    for name, param in model.named_parameters():
        is_adapter = 'adapters' in name
        if not is_adapter and 'head' not in name:
            param.requires_grad_(False)

    return model


def apply_linear_probe(model, config: ExperimentConfig):
    """Linear probing: freeze everything except the classification head."""
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad_(False)
    return model


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_and_evaluate(model, train_loader, val_loader, config, device,
                       method_name='', max_epochs=None, lr=None):
    """Train a PEFT model and return accuracy."""
    epochs = max_epochs or config.epochs
    learning_rate = lr or config.lr

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {method_name}: {trainable:,} trainable / {total:,} total params "
          f"({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct, total_samples = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total_samples += batch_y.shape[0]

        acc = correct / total_samples
        best_acc = max(best_acc, acc)

    print(f"    Best accuracy: {best_acc:.4f}")
    return best_acc


# ============================================================================
# Attention Map Extraction
# ============================================================================

def extract_attention_maps(model, dataloader, device, max_batches=5):
    """Extract CLS attention maps from all layers and heads."""
    model.eval()
    all_attn = []
    hooks = []

    attn_storage = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d_h = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = F.softmax((q @ k.transpose(-2, -1)) * (d_h ** -0.5), dim=-1)
            attn_storage[layer_idx] = attn.detach().cpu()
        return hook_fn

    for idx, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        for i, (batch_x, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            batch_x = batch_x.to(device)
            _ = model(batch_x)
            # Collect per-image attention maps
            for b in range(batch_x.shape[0]):
                img_attn = {}
                for l in attn_storage:
                    img_attn[l] = attn_storage[l][b]  # [H, N, N]
                all_attn.append(img_attn)

    for h in hooks:
        h.remove()

    return all_attn


# ============================================================================
# Main Comparison Experiment
# ============================================================================

def run_comparison(config: ExperimentConfig):
    """Run PEFT method comparison on all datasets at n=1000.

    Tests Theorem 1: which method wins on which task?
    Tests Theorem 2: how does accuracy scale with rank/prompts?
    Cross-reference with Exp5 task structure metrics.
    """
    print("=" * 70)
    print("EXPERIMENT 2: PEFT METHOD COMPARISON")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    print("\nLoading pretrained model...")
    try:
        base_model = timm.create_model(config.model_name, pretrained=True, img_size=config.img_size)
    except Exception:
        print("Cannot download model. Using random init.")
        base_model = timm.create_model(config.model_name, pretrained=False, img_size=config.img_size)

    # All 20 datasets (matching exp5 task structure)
    tasks = {
        # Natural
        'cifar10': (10, 'natural'), 'cifar100': (100, 'natural'),
        'dtd': (47, 'natural'), 'oxford_flowers102': (102, 'natural'),
        'oxford_iiit_pet': (37, 'natural'), 'food101': (101, 'natural'),
        'stl10': (10, 'natural'), 'fgvc_aircraft': (100, 'natural'),
        # Specialized
        'eurosat': (10, 'specialized'), 'pcam': (2, 'specialized'),
        'country211': (211, 'specialized'),
        # Structured
        'svhn': (10, 'structured'), 'gtsrb': (43, 'structured'),
        'mnist': (10, 'structured'), 'fashionmnist': (10, 'structured'),
        'kmnist': (10, 'structured'), 'emnist_letters': (26, 'structured'),
        'rendered_sst2': (2, 'structured'),
        'clevr_count': (8, 'structured'), 'dsprites_loc': (16, 'structured'),
    }

    n_samples = config.n_train  # 1000 by default (VTAB-1K)
    all_results = {}

    for task_name, (n_classes, category) in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*60}")

        try:
            # Load dataset
            dataset = load_real_dataset(task_name, n_classes, config,
                                        max_samples=n_samples)
            if dataset is None:
                print(f"  SKIPPED — no real dataset")
                continue

            print(f"  Loaded: {len(dataset)} samples")

            # Train/val split
            from torch.utils.data import random_split
            n_total = len(dataset)
            n_val = min(200, n_total // 5)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                      shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                    shuffle=False, num_workers=2)

            print(f"  Split: {n_train} train / {n_val} val")

            task_results = {'category': category, 'n_train': n_train, 'methods': {}}

            # --- LP ---
            model = deepcopy(base_model).to(device)
            model.head = nn.Linear(config.embed_dim, n_classes).to(device)
            model = apply_linear_probe(model, config)
            acc = train_and_evaluate(model, train_loader, val_loader, config,
                                     device, 'LP')
            task_results['methods']['LP'] = {'accuracy': acc}
            del model

            # --- LoRA (multiple ranks for rate curve) ---
            for r in [1, 2, 4, 8, 16, 32]:
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                model = apply_lora(model, r, config)
                acc = train_and_evaluate(model, train_loader, val_loader, config,
                                         device, f'LoRA(r={r})')
                task_results['methods'][f'LoRA_r{r}'] = {'accuracy': acc, 'rank': r}
                del model

            # --- VPT (multiple prompts to detect plateau) ---
            for p in [1, 5, 10, 20, 50]:
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                model = apply_vpt(model, p, config)
                acc = train_and_evaluate(model, train_loader, val_loader, config,
                                         device, f'VPT(p={p})')
                task_results['methods'][f'VPT_p{p}'] = {'accuracy': acc, 'n_prompts': p}
                del model

            # --- Adapter ---
            for r_a in [8, 32, 64]:
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                model = apply_adapter(model, r_a, config)
                acc = train_and_evaluate(model, train_loader, val_loader, config,
                                         device, f'Adapter(r_a={r_a})')
                task_results['methods'][f'Adapter_r{r_a}'] = {'accuracy': acc, 'bottleneck': r_a}
                del model

            # Print task summary
            best = max(task_results['methods'],
                      key=lambda k: task_results['methods'][k]['accuracy'])
            best_acc = task_results['methods'][best]['accuracy']
            print(f"\n  Task results (n={n_train}):")
            for name, res in sorted(task_results['methods'].items(),
                                    key=lambda x: -x[1]['accuracy']):
                marker = " ←" if name == best else ""
                print(f"    {name:15s}: {res['accuracy']:.4f}{marker}")

            all_results[task_name] = task_results

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # Summary Tables
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Table 1: Best method per task
    print(f"\n  {'Task':<18s} {'Cat':<12s} {'Best Method':<15s} {'Acc':>6s}  "
          f"{'LP':>6s} {'LoRA4':>6s} {'LoRA16':>6s} {'VPT10':>6s} {'VPT50':>6s}")
    print(f"  {'-'*95}")

    for task_name, res in all_results.items():
        m = res['methods']
        best = max(m, key=lambda k: m[k]['accuracy'])
        lp = m.get('LP', {}).get('accuracy', 0)
        l4 = m.get('LoRA_r4', {}).get('accuracy', 0)
        l16 = m.get('LoRA_r16', {}).get('accuracy', 0)
        v10 = m.get('VPT_p10', {}).get('accuracy', 0)
        v50 = m.get('VPT_p50', {}).get('accuracy', 0)
        print(f"  {task_name:<18s} {res['category']:<12s} {best:<15s} "
              f"{m[best]['accuracy']:>6.3f}  {lp:>6.3f} {l4:>6.3f} "
              f"{l16:>6.3f} {v10:>6.3f} {v50:>6.3f}")

    # Table 2: LoRA vs VPT winners
    print(f"\n  THEOREM 1 VALIDATION: LoRA vs VPT")
    print(f"  {'Task':<18s} {'Cat':<12s} {'Best LoRA':>10s} {'Best VPT':>10s} {'Winner':>8s}")
    print(f"  {'-'*60}")

    lora_wins, vpt_wins, ties = 0, 0, 0
    for task_name, res in all_results.items():
        m = res['methods']
        best_lora = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
        best_vpt = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
        if best_lora > best_vpt + 0.01:
            winner = "LoRA"
            lora_wins += 1
        elif best_vpt > best_lora + 0.01:
            winner = "VPT"
            vpt_wins += 1
        else:
            winner = "TIE"
            ties += 1
        print(f"  {task_name:<18s} {res['category']:<12s} "
              f"{best_lora:>10.3f} {best_vpt:>10.3f} {winner:>8s}")

    print(f"\n  LoRA wins: {lora_wins}, VPT wins: {vpt_wins}, Ties: {ties}")

    # ========================================================================
    # Cross-reference with Exp5 task structure predictions
    # ========================================================================
    exp5_path = os.path.join(config.output_dir, 'task_structure_analysis.json')
    if os.path.exists(exp5_path):
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION: Exp5 Predictions vs Exp2 Actual Results")
        print("=" * 70)

        with open(exp5_path) as f:
            exp5_data = json.load(f)

        correct, total = 0, 0
        print(f"\n  {'Task':<18s} {'Gap':>5s} {'AtnV':>6s} {'Exp5 Prediction':<30s} "
              f"{'Exp2 Winner':<15s} {'Match':>5s}")
        print(f"  {'-'*85}")

        for task_name, res in all_results.items():
            if task_name not in exp5_data:
                continue

            e5 = exp5_data[task_name]
            gap = e5.get('feature_gap', 0)
            attn_var = e5.get('attention_class_variance_ratio', 0)

            # Exp5 prediction logic
            if gap < 0.10:
                prediction = "LP"
            elif attn_var > 0.3 and gap > 0.2:
                prediction = "VPT"
            elif attn_var > 0.2:
                prediction = "VPT or LoRA"
            else:
                prediction = "LoRA"

            # Exp2 actual winner
            m = res['methods']
            best_lp = m.get('LP', {}).get('accuracy', 0)
            best_lora = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
            best_vpt = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
            best_adapter = max((m[k]['accuracy'] for k in m if 'Adapter' in k), default=0)

            all_best = max(best_lp, best_lora, best_vpt, best_adapter)
            if best_lp >= all_best - 0.01:
                actual = "LP"
            elif best_vpt >= best_lora + 0.01:
                actual = "VPT"
            elif best_lora >= best_vpt + 0.01:
                actual = "LoRA"
            else:
                actual = "TIE(LoRA/VPT)"

            # Check match
            match = False
            if prediction == "LP" and actual == "LP":
                match = True
            elif prediction == "LoRA" and actual in ("LoRA", "TIE(LoRA/VPT)"):
                match = True
            elif prediction == "VPT" and actual in ("VPT", "TIE(LoRA/VPT)"):
                match = True
            elif prediction == "VPT or LoRA":
                match = actual in ("VPT", "LoRA", "TIE(LoRA/VPT)")

            if match:
                correct += 1
            total += 1

            symbol = "✓" if match else "✗"
            print(f"  {task_name:<18s} {gap:>5.2f} {attn_var:>6.3f} "
                  f"{prediction:<30s} {actual:<15s} {symbol:>5s}")

        if total > 0:
            acc = correct / total
            print(f"\n  Prediction accuracy: {correct}/{total} = {acc:.1%}")
            if acc >= 0.7:
                print(f"  THEORY VALIDATED — training-free metrics predict PEFT winner")
            elif acc >= 0.5:
                print(f"  PARTIAL VALIDATION — metrics have predictive power but need refinement")
            else:
                print(f"  THEORY NEEDS REVISION — metrics don't predict PEFT winner")
    else:
        print(f"\n  Run exp5 (task_structure) first to enable cross-validation.")

    save_results(all_results, 'experiment2_comparison.json', config)
    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    run_comparison(config)
