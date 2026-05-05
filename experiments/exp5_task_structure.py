"""
Task Structure Analysis — Training-Free Metrics

Measures task properties that predict PEFT method performance WITHOUT
requiring fine-tuning. These are more reliable than FFT-based spectral
analysis because they don't suffer from overfitting noise.

Metrics:
1. Linear Probe Accuracy — how good are pretrained features already?
2. Attention Entropy — how uncertain is pretrained attention on this data?
3. Feature Discriminability — can pretrained features separate classes?
4. Gradient Rank — what rank of update does the task require?
5. Attention Consistency — does attention vary across classes?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
from torch.utils.data import DataLoader, random_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from config import ExperimentConfig, setup_device, ensure_dirs, save_results
from exp1_spectral import load_real_dataset, SyntheticVTABDataset


def extract_features_and_attention(model, dataloader, device, max_batches=None):
    """
    Single forward pass: extract CLS features and attention maps.
    Returns:
        features: [n_samples, embed_dim]
        labels: [n_samples]
        attention_maps: list of [n_samples, n_heads, seq_len, seq_len] per layer
    """
    model.eval()
    all_features = []
    all_labels = []
    attn_per_layer = {l: [] for l in range(len(model.blocks))}

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            B, N, C = input[0].shape
            H = module.num_heads
            d_h = C // H
            qkv = module.qkv(input[0]).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = F.softmax((q @ k.transpose(-2, -1)) * (d_h ** -0.5), dim=-1)
            attn_per_layer[layer_idx].append(attn.detach().cpu())
        return hook_fn

    for idx, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_x = batch_x.to(device)
            out = model.forward_features(batch_x)
            # CLS token features
            cls_features = out[:, 0].cpu()
            all_features.append(cls_features)
            all_labels.append(batch_y)

    for h in hooks:
        h.remove()

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Concatenate attention maps per layer
    attention = {}
    for l in attn_per_layer:
        if attn_per_layer[l]:
            attention[l] = torch.cat(attn_per_layer[l], dim=0)

    return features.numpy(), labels.numpy(), attention


def measure_linear_probe_accuracy(features, labels, n_classes):
    """Quick linear probe accuracy using sklearn (no training loop needed)."""
    from sklearn.linear_model import LogisticRegression

    n = len(labels)
    n_train = int(0.8 * n)

    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    try:
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
    except Exception:
        acc = 0.0

    return float(acc)


def measure_knn_accuracy(features, labels, k=5):
    """k-NN accuracy — measures feature discriminability without training."""
    n = len(labels)
    n_train = int(0.8 * n)

    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)

    return float(acc)


def measure_attention_entropy(attention_maps, n_layers):
    """
    Mean attention entropy of CLS token across layers and samples.
    High entropy = uniform attention (model doesn't know where to look)
    Low entropy = focused attention (model has clear spatial strategy)
    """
    entropies = []

    for l in range(n_layers):
        if l not in attention_maps:
            continue
        # CLS attention to patches: [n_samples, n_heads, 1→patches]
        attn = attention_maps[l][:, :, 0, 1:]  # [n, H, N_patches]
        # Entropy per head per sample
        attn_clamped = attn.clamp(min=1e-10)
        ent = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [n, H]
        entropies.append(ent.mean().item())

    max_entropy = np.log(attention_maps[0].shape[-1] - 1)  # log(N_patches)
    mean_entropy = np.mean(entropies) if entropies else 0.0

    return float(mean_entropy), float(max_entropy), float(mean_entropy / max_entropy)


def measure_attention_class_variance(attention_maps, labels, n_layers):
    """
    How much does attention vary BETWEEN classes vs WITHIN classes?

    High between/within ratio = attention is class-dependent
    → task needs attention steering → VPT might help

    Low ratio = attention is similar for all classes
    → task is about features, not attention → LoRA better
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0, 0.0, 0.0

    between_var_total = 0.0
    within_var_total = 0.0
    n_layers_used = 0

    for l in range(n_layers):
        if l not in attention_maps:
            continue
        # CLS attention: [n_samples, n_heads, N_patches]
        attn = attention_maps[l][:, :, 0, 1:].numpy()
        # Average over heads: [n_samples, N_patches]
        attn_avg = attn.mean(axis=1)

        # Global mean attention
        global_mean = attn_avg.mean(axis=0)  # [N_patches]

        # Between-class variance
        between_var = 0.0
        within_var = 0.0
        for c in unique_labels:
            mask = labels == c
            if mask.sum() < 2:
                continue
            class_mean = attn_avg[mask].mean(axis=0)
            between_var += mask.sum() * np.sum((class_mean - global_mean) ** 2)
            within_var += np.sum((attn_avg[mask] - class_mean[None, :]) ** 2)

        between_var /= len(labels)
        within_var /= len(labels)

        between_var_total += between_var
        within_var_total += within_var
        n_layers_used += 1

    if n_layers_used == 0 or within_var_total < 1e-12:
        return 0.0, 0.0, 0.0

    between_var_avg = between_var_total / n_layers_used
    within_var_avg = within_var_total / n_layers_used
    ratio = between_var_avg / (within_var_avg + 1e-10)

    return float(between_var_avg), float(within_var_avg), float(ratio)


def measure_gradient_rank(model, dataloader, device, n_classes, max_batches=3):
    """
    Estimate the effective rank of the gradient ∂L/∂W for the task.
    High effective rank = task needs high-rank adaptation = LoRA needs more rank
    Low effective rank = task is simple = low-rank LoRA suffices

    Uses the gradient covariance across samples.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Collect gradients for qkv weights of last 3 layers
    target_params = []
    target_names = []
    for name, param in model.named_parameters():
        if 'qkv.weight' in name and any(f'blocks.{i}.' in name
                                         for i in range(9, 12)):
            target_params.append(param)
            target_names.append(name)
            param.requires_grad_(True)

    if not target_params:
        return 0.0, 0.0

    all_grads = []

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        for i in range(min(8, batch_x.shape[0])):  # per-sample gradients
            model.zero_grad()
            out = model(batch_x[i:i+1])
            loss = criterion(out, batch_y[i:i+1])
            loss.backward()

            # Concatenate gradients from target params
            grad_vec = torch.cat([p.grad.flatten() for p in target_params
                                  if p.grad is not None])
            all_grads.append(grad_vec.detach().cpu())

    # Reset requires_grad
    for p in target_params:
        p.requires_grad_(False)

    if len(all_grads) < 5:
        return 0.0, 0.0

    grad_matrix = torch.stack(all_grads)  # [n_samples, total_grad_dim]

    # SVD of gradient matrix
    # Use a subset of dimensions for speed
    d = grad_matrix.shape[1]
    if d > 5000:
        idx = torch.randperm(d)[:5000]
        grad_matrix = grad_matrix[:, idx]

    try:
        sv = torch.linalg.svdvals(grad_matrix.float())
        sv = sv.numpy()

        # Effective rank = exp(entropy of normalized singular values)
        sv_norm = sv / sv.sum()
        sv_norm = sv_norm[sv_norm > 1e-10]
        effective_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm)))

        # Fraction of variance in top-4
        total_var = (sv ** 2).sum()
        top4_var = (sv[:4] ** 2).sum()
        top4_fraction = top4_var / total_var if total_var > 0 else 0

        return float(effective_rank), float(top4_fraction)
    except Exception:
        return 0.0, 0.0


def run_task_structure_analysis(config: ExperimentConfig):
    """Main task structure analysis — training-free."""
    print("=" * 70)
    print("TASK STRUCTURE ANALYSIS (Training-Free)")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    print("\nLoading pretrained model...")
    try:
        model = timm.create_model(config.model_name, pretrained=True,
                                   img_size=config.img_size)
    except Exception:
        model = timm.create_model(config.model_name, pretrained=False,
                                   img_size=config.img_size)
    model = model.to(device)

    tasks = {
        'cifar100':     (100, 'natural'),
        'dtd':          (47,  'natural'),
        'svhn':         (10,  'natural'),
        'eurosat':      (10,  'specialized'),
        'gtsrb':        (43,  'structured'),
        'clevr_count':  (8,   'structured'),
        'dsprites_loc': (16,  'structured'),
    }

    all_results = {}

    for task_name, (n_classes, category) in tasks.items():
        print(f"\n{'='*55}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*55}")

        # Load data (1000 samples is plenty for feature extraction)
        dataset = load_real_dataset(task_name, n_classes, config,
                                    max_samples=1000)
        if dataset is None:
            task_type = 'structured' if category == 'structured' else 'natural'
            dataset = SyntheticVTABDataset(1000, n_classes,
                                           config.img_size, task_type)
            print(f"  Using synthetic data")
        else:
            print(f"  Loaded: {len(dataset)} samples")

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        # --- Extract features and attention ---
        print("  Extracting features and attention maps...")
        features, labels, attention = extract_features_and_attention(
            model, loader, device, max_batches=20)
        print(f"  Extracted: {features.shape[0]} samples, {features.shape[1]}d features")

        # --- Metric 1: Linear Probe Accuracy ---
        lp_acc = measure_linear_probe_accuracy(features, labels, n_classes)
        print(f"  Linear probe accuracy:    {lp_acc:.4f}")

        # --- Metric 2: k-NN Accuracy ---
        knn_acc = measure_knn_accuracy(features, labels, k=5)
        print(f"  5-NN accuracy:            {knn_acc:.4f}")

        # --- Metric 3: Attention Entropy ---
        mean_ent, max_ent, norm_ent = measure_attention_entropy(
            attention, config.num_layers)
        print(f"  Attention entropy:        {mean_ent:.4f} / {max_ent:.4f} "
              f"(normalized: {norm_ent:.4f})")

        # --- Metric 4: Attention Class Variance ---
        between_var, within_var, attn_ratio = measure_attention_class_variance(
            attention, labels, config.num_layers)
        print(f"  Attention class variance: between={between_var:.6f}, "
              f"within={within_var:.6f}, ratio={attn_ratio:.4f}")

        # --- Metric 5: Gradient Rank ---
        print("  Computing gradient rank...")
        eff_rank, top4_frac = measure_gradient_rank(
            model, loader, device, n_classes, max_batches=3)
        print(f"  Gradient effective rank:  {eff_rank:.1f}")
        print(f"  Gradient top-4 fraction:  {top4_frac:.4f}")

        # --- Derived predictions ---
        # Feature gap = 1 - LP accuracy (how much the task needs beyond pretrained features)
        feature_gap = 1.0 - lp_acc

        # Predicted best method based on our metrics
        if feature_gap < 0.1:
            predicted = "LP (features already sufficient)"
        elif attn_ratio > 0.5 and feature_gap > 0.3:
            predicted = "VPT (attention-steering task)"
        elif top4_frac > 0.8:
            predicted = "LoRA low-rank (gradient is low-rank)"
        else:
            predicted = "LoRA or Adapter (feature adaptation needed)"

        print(f"\n  Feature gap (1-LP_acc):   {feature_gap:.4f}")
        print(f"  Predicted best method:    {predicted}")

        all_results[task_name] = {
            'category': category,
            'n_classes': n_classes,
            'linear_probe_accuracy': lp_acc,
            'knn_accuracy': knn_acc,
            'attention_entropy_normalized': norm_ent,
            'attention_class_variance_ratio': attn_ratio,
            'gradient_effective_rank': eff_rank,
            'gradient_top4_fraction': top4_frac,
            'feature_gap': feature_gap,
            'predicted_method': predicted,
        }

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("TASK STRUCTURE SUMMARY")
    print("=" * 70)
    print(f"\n  {'Task':<15s} {'Cat':<12s} {'LP Acc':>8s} {'Gap':>6s} "
          f"{'Attn Ent':>9s} {'Attn Var':>9s} {'Grad Rank':>10s} {'Prediction'}")
    print(f"  {'-'*85}")

    for task_name, r in all_results.items():
        print(f"  {task_name:<15s} {r['category']:<12s} "
              f"{r['linear_probe_accuracy']:>8.3f} "
              f"{r['feature_gap']:>6.3f} "
              f"{r['attention_entropy_normalized']:>9.4f} "
              f"{r['attention_class_variance_ratio']:>9.4f} "
              f"{r['gradient_effective_rank']:>10.1f} "
              f"  {r['predicted_method'][:30]}")

    print(f"\n  Theory predictions:")
    print(f"  - Natural tasks: high LP acc, low feature gap, low attn variance")
    print(f"  - Structured tasks: low LP acc, high feature gap, high attn variance")
    print(f"  - High attn variance → VPT advantage (Theorem 1a)")
    print(f"  - High feature gap → LoRA advantage (Theorem 1b)")

    save_results(all_results, 'task_structure_analysis.json', config)
    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    run_task_structure_analysis(config)
