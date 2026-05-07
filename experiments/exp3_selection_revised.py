"""
Experiment 3 (Revised): Selection Algorithm using Training-Free Metrics

The original Theorem 5 algorithm used pilot FFT + spectral scoring.
Problem: KL/(2n) dominates at small n, always selects LoRA_r1.

Fix: Use training-free metrics (feature gap, attention class variance)
that achieve 93% prediction accuracy across 16+ tasks.

Selection Rule (from Exp5 validation):
  1. Compute feature_gap and attn_var from ONE forward pass
  2. Select method family:
     - gap < 0.05               → LP
     - attn_var > 0.25 AND gap < 0.15 → VPT
     - else                     → LoRA
  3. Select capacity based on n:
     - LoRA: r* = clip(n / (2 * L * d_head), 1, 32)
     - VPT:  p* = clip(n / (L * d), 1, 50)
     - Adapter: r_a* = clip(n / (2 * L * d), 4, 64)
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
import time
from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Subset
from scipy import stats

from config import ExperimentConfig, setup_device, ensure_dirs, save_results
from exp1_spectral import load_real_dataset
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)
from exp5_task_structure import (extract_features_and_attention,
                                  measure_linear_probe_accuracy,
                                  measure_attention_class_variance)


class TrainingFreeSelector:
    """
    PEFT method selection using training-free task characterization.

    Phase 1: One forward pass → feature_gap + attn_var (~30 seconds)
    Phase 2: Rule-based selection with capacity formula (~0 seconds)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.task_metrics = {}
        self.sigma_p_sq = None

    def compute_sigma_p_sq(self, model):
        """Compute prior variance from pretrained weights (one-time per model)."""
        total_norm_sq = 0.0
        count = 0
        for name, param in model.named_parameters():
            if any(t in name for t in ['qkv.weight', 'proj.weight']):
                total_norm_sq += param.float().norm().item() ** 2
                count += 1
        self.sigma_p_sq = total_norm_sq / (count * self.config.head_dim) if count > 0 else 1.0
        print(f"  [Selection] σ_P² = {self.sigma_p_sq:.4f} (from {count} weight matrices)")

    def characterize_task(self, model, dataloader, device, n_classes):
        """Phase 1: Extract training-free metrics."""
        # Compute σ_P² if not already done
        if self.sigma_p_sq is None:
            self.compute_sigma_p_sq(model)

        print("  [Selection] Extracting training-free metrics...")
        t0 = time.time()

        features, labels, attention = extract_features_and_attention(
            model, dataloader, device, max_attn_batches=30)

        # Feature gap
        lp_acc = measure_linear_probe_accuracy(features, labels, n_classes)
        feature_gap = 1.0 - lp_acc

        # Attention class variance
        n_attn = attention[0].shape[0] if 0 in attention else 0
        attn_labels = labels[:n_attn]
        _, _, attn_var = measure_attention_class_variance(
            attention, attn_labels, self.config.num_layers)

        elapsed = time.time() - t0
        self.task_metrics = {
            'feature_gap': feature_gap,
            'lp_accuracy': lp_acc,
            'attention_variance': attn_var,
            'characterization_time': elapsed,
        }

        print(f"  [Selection] feature_gap={feature_gap:.3f}, "
              f"attn_var={attn_var:.4f} ({elapsed:.1f}s)")

        return self.task_metrics

    def select(self, n_train, n_classes=None):
        """
        Phase 2: Select method + capacity.

        Changes from v1:
        1. Removed LP branch (LoRA_r1 is equally cheap and usually better)
        2. Capacity scales with gap (hard tasks need more rank)
        3. VPT score-based selection (ratio of attn_var to gap)
        """
        gap = self.task_metrics['feature_gap']
        attn_var = self.task_metrics['attention_variance']

        L = self.config.num_layers
        d = self.config.embed_dim
        d_h = self.config.head_dim
        sigma_p_sq = self.sigma_p_sq

        # PAC-Bayes constants
        c1 = 6.0
        c4 = 2.0

        # Theory-derived CEILING (max affordable capacity)
        r_max = max(1, int(2 * n_train * sigma_p_sq / (L * d_h)))
        p_max = max(1, int(4 * n_train * sigma_p_sq / (L * d)))

        # Task-dependent capacity: scale by gap
        # Higher gap → needs more adaptation → use more of the capacity budget
        # gap=0.02 → use 10% of budget, gap=0.6 → use 300% (beyond ceiling for hard tasks)
        r_task = max(1, round(r_max * gap / 0.2))
        r_task = min(r_task, 32)  # hardware ceiling
        p_task = max(1, round(p_max * max(gap, 0.05) / 0.1))
        p_task = min(p_task, 50)

        # VPT score: ratio of attention steering potential to feature gap cost
        # High score = VPT has advantage; Low score = LoRA has advantage
        vpt_score = attn_var / (gap + 0.01)

        # Theory-derived VPT threshold (from Theorem 1 crossover)
        gamma_vpt = (1.0 / c1) * np.sqrt(L * d / (2 * n_train * sigma_p_sq))
        rho_min = c4 * gamma_vpt

        print(f"  [Thresholds] r_max={r_max}, p_max={p_max}")
        print(f"  [Capacity]   r_task={r_task}, p_task={p_task}")
        print(f"  [Task]       γ={gap:.4f}, ρ={attn_var:.4f}, VPT_score={vpt_score:.2f}")

        # Method selection
        if attn_var > rho_min and gap < gamma_vpt and vpt_score > 3.0:
            method = 'VPT'
            capacity = p_task
            reason = (f"VPT_score={vpt_score:.1f} > 3.0 AND ρ={attn_var:.3f} > ρ_min={rho_min:.3f}"
                     f" AND γ={gap:.3f} < γ_VPT={gamma_vpt:.3f} → VPT(p={capacity})")
        else:
            method = 'LoRA'
            capacity = r_task
            reason = f"LoRA(r={capacity}) [gap-scaled from r_max={r_max}]"

        # Round to nearest available config
        if method == 'VPT':
            available_p = [1, 5, 10, 20, 50]
            capacity = min(available_p, key=lambda p: abs(p - capacity))
            method_name = f'VPT_p{capacity}'
        else:
            available_r = [1, 2, 4, 8, 16, 32]
            capacity = min(available_r, key=lambda r: abs(r - capacity))
            method_name = f'LoRA_r{capacity}'

        print(f"  [Selection]  → {method_name} ({reason})")

        return method_name, {
            'method': method,
            'capacity': capacity,
            'reason': reason,
            'feature_gap': gap,
            'attention_variance': attn_var,
            'vpt_score': vpt_score,
            'r_max': r_max,
            'r_task': r_task,
            'p_max': p_max,
            'p_task': p_task,
            'sigma_p_sq': sigma_p_sq,
        }


def run_selection_benchmark(config: ExperimentConfig):
    """Benchmark the training-free selection algorithm."""
    print("=" * 70)
    print("EXPERIMENT 3: SELECTION ALGORITHM (Training-Free)")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    print("\nLoading pretrained model...")
    try:
        base_model = timm.create_model(config.model_name, pretrained=True,
                                        img_size=config.img_size)
    except Exception:
        base_model = timm.create_model(config.model_name, pretrained=False,
                                        img_size=config.img_size)

    tasks = {
        'cifar10': (10, 'natural'),
        'cifar100': (100, 'natural'),
        'dtd': (47, 'natural'),
        'oxford_iiit_pet': (37, 'natural'),
        'fgvc_aircraft': (100, 'natural'),
        'eurosat': (10, 'specialized'),
        'pcam': (2, 'specialized'),
        'svhn': (10, 'structured'),
        'gtsrb': (43, 'structured'),
        'mnist': (10, 'structured'),
        'fashionmnist': (10, 'structured'),
        'emnist_letters': (26, 'structured'),
        'clevr_count': (8, 'structured'),
    }

    all_results = {}

    for task_name, (n_classes, category) in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*60}")

        try:
            # Load 1000 samples
            dataset = load_real_dataset(task_name, n_classes, config,
                                        max_samples=1000)
            if dataset is None:
                print("  SKIPPED — no data")
                continue

            n_total = len(dataset)
            n_val = min(200, n_total // 5)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                      shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                    shuffle=False, num_workers=2)

            # Also need a loader for feature extraction (from full dataset)
            char_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                                     num_workers=2)

            print(f"  {n_train} train / {n_val} val")

            # === Phase 1: Training-free characterization ===
            selector = TrainingFreeSelector(config)
            char_model = deepcopy(base_model).to(device)
            metrics = selector.characterize_task(char_model, char_loader,
                                                  device, n_classes)
            del char_model

            # === Phase 2: Select method ===
            selected_name, selection_info = selector.select(n_train)
            selection_time = metrics['characterization_time']

            # === Phase 3: Train selected method ===
            print(f"\n  Training selected method: {selected_name}")
            model = deepcopy(base_model).to(device)
            model.head = nn.Linear(config.embed_dim, n_classes).to(device)

            if 'LP' in selected_name:
                model = apply_linear_probe(model, config)
            elif 'LoRA' in selected_name:
                r = int(selected_name.split('_r')[1])
                model = apply_lora(model, r, config)
            elif 'VPT' in selected_name:
                p = int(selected_name.split('_p')[1])
                model = apply_vpt(model, p, config)

            selected_acc = train_and_evaluate(model, train_loader, val_loader,
                                              config, device, selected_name)
            del model

            # === Phase 4: Exhaustive search (oracle) ===
            print(f"\n  EXHAUSTIVE SEARCH (oracle):")
            exhaustive = {}

            methods_to_try = [('LP', lambda m: apply_linear_probe(m, config))]
            for r in [1, 2, 4, 8, 16, 32]:
                methods_to_try.append(
                    (f'LoRA_r{r}', lambda m, _r=r: apply_lora(m, _r, config)))
            for p in [1, 5, 10, 20, 50]:
                methods_to_try.append(
                    (f'VPT_p{p}', lambda m, _p=p: apply_vpt(m, _p, config)))
            for r_a in [8, 32, 64]:
                methods_to_try.append(
                    (f'Adapter_r{r_a}', lambda m, _r=r_a: apply_adapter(m, _r, config)))

            for method_name, apply_fn in methods_to_try:
                model = deepcopy(base_model).to(device)
                model.head = nn.Linear(config.embed_dim, n_classes).to(device)
                model = apply_fn(model)
                acc = train_and_evaluate(model, train_loader, val_loader,
                                         config, device, method_name)
                exhaustive[method_name] = acc
                del model

            # === Analysis ===
            oracle_name = max(exhaustive, key=exhaustive.get)
            oracle_acc = exhaustive[oracle_name]
            worst_name = min(exhaustive, key=exhaustive.get)
            worst_acc = exhaustive[worst_name]
            mean_acc = np.mean(list(exhaustive.values()))

            regret = oracle_acc - selected_acc
            lift_over_mean = selected_acc - mean_acc
            lift_over_worst = selected_acc - worst_acc

            # Percentile
            n_below = sum(1 for v in exhaustive.values() if v < selected_acc)
            percentile = n_below / len(exhaustive) * 100

            # Compute savings
            n_methods = len(methods_to_try)
            compute_ratio = n_methods  # selection does 1 forward pass vs n_methods full trainings

            print(f"\n  RESULTS:")
            print(f"    Selected:  {selected_name} = {selected_acc:.4f}")
            print(f"    Oracle:    {oracle_name} = {oracle_acc:.4f}")
            print(f"    Worst:     {worst_name} = {worst_acc:.4f}")
            print(f"    Mean:      {mean_acc:.4f}")
            print(f"    ---")
            print(f"    Regret:         {regret:.4f} ({regret*100:.1f}%)")
            print(f"    Lift over mean: {lift_over_mean:.4f} ({lift_over_mean*100:.1f}%)")
            print(f"    Lift over worst:{lift_over_worst:.4f} ({lift_over_worst*100:.1f}%)")
            print(f"    Percentile:     {percentile:.0f}th")
            print(f"    Compute:        {selection_time:.1f}s selection vs "
                  f"~{n_methods * 100}s exhaustive ({compute_ratio}× savings)")

            all_results[task_name] = {
                'category': category,
                'selected_method': selected_name,
                'selected_accuracy': selected_acc,
                'oracle_method': oracle_name,
                'oracle_accuracy': oracle_acc,
                'worst_method': worst_name,
                'worst_accuracy': worst_acc,
                'mean_accuracy': mean_acc,
                'regret': regret,
                'lift_over_mean': lift_over_mean,
                'lift_over_worst': lift_over_worst,
                'percentile': percentile,
                'selection_time': selection_time,
                'compute_savings': compute_ratio,
                'task_metrics': metrics,
                'selection_info': selection_info,
                'exhaustive': exhaustive,
            }

            save_results(all_results, 'exp3_selection_revised.json', config)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    # === Final Summary ===
    print("\n" + "=" * 70)
    print("SELECTION ALGORITHM SUMMARY")
    print("=" * 70)

    print(f"\n  {'Task':<18s} {'Selected':<12s} {'Sel Acc':>7s} {'Oracle':>7s} "
          f"{'Regret':>7s} {'vs Mean':>7s} {'vs Worst':>8s} {'%ile':>5s}")
    print(f"  {'-'*80}")

    regrets = []
    lifts = []
    for task_name, r in all_results.items():
        print(f"  {task_name:<18s} {r['selected_method']:<12s} "
              f"{r['selected_accuracy']:>7.3f} {r['oracle_accuracy']:>7.3f} "
              f"{r['regret']:>7.3f} {r['lift_over_mean']:>+7.3f} "
              f"{r['lift_over_worst']:>+8.3f} {r['percentile']:>4.0f}th")
        regrets.append(r['regret'])
        lifts.append(r['lift_over_mean'])

    print(f"\n  Mean regret:      {np.mean(regrets):.4f} ({np.mean(regrets)*100:.1f}%)")
    print(f"  Max regret:       {np.max(regrets):.4f} ({np.max(regrets)*100:.1f}%)")
    print(f"  Mean lift (mean): {np.mean(lifts):.4f} ({np.mean(lifts)*100:.1f}%)")
    print(f"  Method selection time: ~30s per task (vs ~{len(methods_to_try)*100}s exhaustive)")

    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    run_selection_benchmark(config)
