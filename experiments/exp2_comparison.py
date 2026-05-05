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
                       method_name='', max_epochs=None):
    """Train a PEFT model and return accuracy."""
    epochs = max_epochs or config.epochs

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {method_name}: {trainable:,} trainable / {total:,} total params "
          f"({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
    """Run PEFT method comparison across multiple data scales.

    Validates:
    - Theorem 1: VPT wins structured tasks, LoRA wins natural tasks
    - Theorem 2: Accuracy vs capacity follows predicted rates
    - Theorem 3: Optimal method changes with n
    """
    print("=" * 70)
    print("EXPERIMENT 2: PEFT METHOD COMPARISON (multi-scale)")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    print("\nLoading pretrained model...")
    try:
        base_model = timm.create_model(config.model_name, pretrained=True, img_size=config.img_size)
    except Exception:
        print("Cannot download model. Using random init.")
        base_model = timm.create_model(config.model_name, pretrained=False, img_size=config.img_size)

    pretrained_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    tasks_to_run = {
        'natural': ['cifar100', 'dtd', 'svhn'],
        'specialized': ['eurosat'],
        'structured': ['gtsrb', 'clevr_count', 'dsprites_loc'],
        # gtsrb: guaranteed via torchvision
        # clevr_count, dsprites_loc: via HuggingFace (fallback to synthetic)
    }
    n_classes_map = {
        'cifar100': 100, 'dtd': 47, 'svhn': 10, 'eurosat': 10,
        'gtsrb': 43, 'clevr_count': 8, 'dsprites_loc': 16,
    }

    all_results = {}

    for category, task_list in tasks_to_run.items():
        for task_name in task_list:
            print(f"\n{'='*60}")
            print(f"Task: {task_name} ({category})")
            print(f"{'='*60}")

            n_classes = n_classes_map[task_name]
            task_type = 'structured' if category == 'structured' else 'natural'

            # Load FULL dataset once
            full_dataset = load_real_dataset(task_name, n_classes, config,
                                             max_samples=0)
            if full_dataset is None:
                print(f"  Falling back to synthetic data")
                full_dataset = SyntheticVTABDataset(5000, n_classes,
                                                    config.img_size, task_type)
            else:
                print(f"  Loaded full dataset: {len(full_dataset)} samples")

            task_results = {'category': category, 'scales': {}}

            for n_scale in config.n_train_scales:
                n_use = len(full_dataset) if n_scale == 0 else min(n_scale, len(full_dataset))
                n_val = min(config.n_val, n_use // 5)
                n_train = n_use - n_val

                print(f"\n  --- n_train = {n_train}, n_val = {n_val} ---")

                from torch.utils.data import random_split, Subset
                if n_use < len(full_dataset):
                    indices = torch.randperm(len(full_dataset))[:n_use].tolist()
                    subset = Subset(full_dataset, indices)
                else:
                    subset = full_dataset

                train_ds, val_ds = random_split(subset, [n_train, n_val])
                train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                          shuffle=True, num_workers=2)
                val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                        shuffle=False, num_workers=2)

                scale_results = {}

                # LP
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                model = apply_linear_probe(model, config)
                acc = train_and_evaluate(model, train_loader, val_loader, config,
                                         device, 'LP')
                scale_results['LP'] = {'accuracy': acc}
                del model

                # LoRA — expanded grid for rate curve fitting (Theorem 2b)
                for r in [1, 2, 4, 8, 16, 32]:
                    model = deepcopy(base_model).to(device)
                    model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                    model = apply_lora(model, r, config)
                    acc = train_and_evaluate(model, train_loader, val_loader, config,
                                             device, f'LoRA(r={r})')
                    scale_results[f'LoRA_r{r}'] = {'accuracy': acc, 'rank': r}
                    del model

                # VPT — expanded grid to detect plateau (Theorem 2c)
                for p in [1, 5, 10, 20, 50]:
                    model = deepcopy(base_model).to(device)
                    model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                    model = apply_vpt(model, p, config)
                    acc = train_and_evaluate(model, train_loader, val_loader, config,
                                             device, f'VPT(p={p})')
                    scale_results[f'VPT_p{p}'] = {'accuracy': acc, 'n_prompts': p}
                    del model

                # Adapter
                for r_a in [8, 32, 64]:
                    model = deepcopy(base_model).to(device)
                    model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                    model = apply_adapter(model, r_a, config)
                    acc = train_and_evaluate(model, train_loader, val_loader, config,
                                             device, f'Adapter(r_a={r_a})')
                    scale_results[f'Adapter_r{r_a}'] = {'accuracy': acc, 'bottleneck': r_a}
                    del model

                task_results['scales'][str(n_train)] = scale_results
                best = max(scale_results, key=lambda k: scale_results[k]['accuracy'])
                print(f"  Best at n={n_train}: {best} ({scale_results[best]['accuracy']:.4f})")

            # ==============================================================
            # Task Characterization (run ONCE per task on full data)
            # Measures S_attn, S_feat, sin²θ, α to correlate with winners
            # ==============================================================
            print(f"\n  [Task Characterization] FFT on full data...")
            model_fft = deepcopy(base_model).to(device)
            model_fft.head = nn.Linear(config.embed_dim, n_classes).to(device)

            # Full dataset for FFT
            from torch.utils.data import random_split as rs2
            n_fft_total = len(full_dataset)
            n_fft_val = min(config.n_val, n_fft_total // 5)
            fft_train, fft_val = rs2(full_dataset, [n_fft_total - n_fft_val, n_fft_val])
            fft_train_loader = DataLoader(fft_train, batch_size=config.batch_size,
                                          shuffle=True, num_workers=2)
            fft_val_loader = DataLoader(fft_val, batch_size=config.batch_size,
                                        shuffle=False, num_workers=2)

            # Get pretrained attention maps
            pre_attn = extract_attention_maps(model_fft, fft_val_loader, device,
                                              max_batches=3)

            # Run FFT
            for param in model_fft.parameters():
                param.requires_grad_(True)
            _ = train_and_evaluate(model_fft, fft_train_loader, fft_val_loader,
                                   config, device, 'FFT')

            # Get fine-tuned attention maps
            ft_attn = extract_attention_maps(model_fft, fft_val_loader, device,
                                              max_batches=3)

            # Compute S_attn
            s_attn = 0.0
            if pre_attn and ft_attn:
                n_imgs = min(len(pre_attn), len(ft_attn))
                s_attn_vals = []
                for i in range(n_imgs):
                    for l in pre_attn[i]:
                        if l in ft_attn[i]:
                            cls_diff = ft_attn[i][l][:, 0, 1:] - pre_attn[i][l][:, 0, 1:]
                            s_attn_vals.append((cls_diff ** 2).mean().item())
                s_attn = float(np.mean(s_attn_vals)) if s_attn_vals else 0.0

            # Compute spectral profile and T(r)
            fft_state = {k: v.cpu() for k, v in model_fft.state_dict().items()}
            alphas = []
            spectral_tails = {str(r): 0.0 for r in [1, 2, 4, 8, 16, 32]}
            for name in pretrained_state:
                if 'qkv.weight' in name or 'proj.weight' in name:
                    delta = fft_state[name].float() - pretrained_state[name].float()
                    sv = compute_svd_profile(delta)
                    alpha_val, C_val = fit_spectral_decay(sv)
                    alphas.append(alpha_val)
                    for r in [1, 2, 4, 8, 16, 32]:
                        spectral_tails[str(r)] += float((sv[r:] ** 2).sum())

            task_results['task_descriptors'] = {
                'S_attn': s_attn,
                'mean_alpha': float(np.mean(alphas)) if alphas else 0.0,
                'spectral_tails': spectral_tails,
                'category': category,
            }

            print(f"  Task descriptors: S_attn={s_attn:.6f}, "
                  f"α={task_results['task_descriptors']['mean_alpha']:.3f}")

            del model_fft

            all_results[task_name] = task_results

    # Summary
    print("\n" + "=" * 70)
    print("THEOREM 3 VALIDATION: Optimal Method vs. Data Scale")
    print("=" * 70)

    for task_name, res in all_results.items():
        print(f"\n  {task_name} ({res['category']}):")
        for n_str in sorted(res['scales'].keys(), key=lambda x: int(x)):
            scale_res = res['scales'][n_str]
            best = max(scale_res, key=lambda k: scale_res[k]['accuracy'])
            acc = scale_res[best]['accuracy']
            print(f"    n={int(n_str):>6d}: {best:15s} ({acc:.4f})")

    print("\n  Expected: LP at small n → LoRA at moderate n → higher capacity at large n")

    # ============================================================================
    # Cross-experiment: Correlate task descriptors with method winners
    # ============================================================================
    print("\n" + "=" * 70)
    print("THEOREM 1 VALIDATION: Task Descriptors vs. Method Winners")
    print("=" * 70)

    for task_name, res in all_results.items():
        td = res.get('task_descriptors', {})
        cat = res['category']
        s_attn = td.get('S_attn', 0)
        alpha = td.get('mean_alpha', 0)

        # Get best LoRA vs VPT at the largest scale
        largest_scale = max(res['scales'].keys(), key=lambda x: int(x))
        scale_res = res['scales'][largest_scale]

        lora_best = max((v['accuracy'] for k, v in scale_res.items() if 'LoRA' in k), default=0)
        vpt_best = max((v['accuracy'] for k, v in scale_res.items() if 'VPT' in k), default=0)
        winner = "LoRA" if lora_best >= vpt_best else "VPT"

        print(f"  {task_name:15s}: S_attn={s_attn:.6f}, α={alpha:.3f} → "
              f"LoRA={lora_best:.3f} vs VPT={vpt_best:.3f} → {winner}")

    print("\n  Theory predicts: high S_attn → VPT wins; high α → LoRA needs low rank")

    # ============================================================================
    # Cross-experiment: T(r) vs LoRA accuracy (Theorem 2b validation)
    # ============================================================================
    print("\n" + "=" * 70)
    print("THEOREM 2 VALIDATION: Spectral Tail T(r) vs. LoRA Accuracy")
    print("=" * 70)

    for task_name, res in all_results.items():
        td = res.get('task_descriptors', {})
        tails = td.get('spectral_tails', {})
        largest_scale = max(res['scales'].keys(), key=lambda x: int(x))
        scale_res = res['scales'][largest_scale]

        print(f"\n  {task_name} (α={td.get('mean_alpha', 0):.3f}):")
        print(f"    {'Rank':>6s}  {'T(r)':>12s}  {'LoRA Acc':>10s}  {'1-Acc':>10s}")
        for r in [1, 2, 4, 8, 16, 32]:
            tail = tails.get(str(r), 0)
            acc = scale_res.get(f'LoRA_r{r}', {}).get('accuracy', 0)
            print(f"    {r:>6d}  {tail:>12.2f}  {acc:>10.4f}  {1-acc:>10.4f}")

    print("\n  Theory predicts: (1 - accuracy) ∝ T(r) ∝ r^{1-2α}")

    save_results(all_results, 'experiment2_comparison.json', config)
    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    run_comparison(config)
