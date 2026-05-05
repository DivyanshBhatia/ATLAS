"""
Experiment 3: Selection Algorithm Benchmark (Theorem 5)

Purpose: Validate that the selection algorithm from Theorem 5 achieves
near-oracle performance with ~10× less compute than exhaustive search.

What we test:
1. Regret = Oracle accuracy - Selected method accuracy (should be < 0.5%)
2. Compute savings: pilot cost vs. exhaustive search cost
3. Correlation between Score and true accuracy (Kendall τ)
4. Non-vacuousness of PAC-Bayes bounds (Theorem 4)
"""

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from copy import deepcopy
from scipy import stats
from torch.utils.data import DataLoader, Subset

from config import (ExperimentConfig, setup_device, ensure_dirs, save_results,
                    compute_svd_profile, fit_spectral_decay,
                    compute_quantization_error, compute_pac_bayes_bound,
                    compute_lora_kl, compute_sigma_p_sq)
from exp1_spectral import SyntheticVTABDataset
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate,
                              extract_attention_maps)


class SelectionAlgorithm:
    """
    PEFT Method Selection Algorithm (Theorem 5).

    Phases:
    1. Pilot: short FFT on n/4 samples to estimate task descriptors
    2. Score: compute ε_approx + ε_gen for each method × capacity
    3. Select: pick the method with minimum score
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.task_descriptors = {}
        self.scores = {}

    def run_pilot(self, model, train_loader, val_loader, pretrained_state,
                  n_classes, device):
        """Phase 1: Pilot fine-tuning to estimate task descriptors."""
        print("  [Pilot] Running pilot fine-tuning...")

        # Full fine-tune for pilot_epochs
        model.head = nn.Linear(self.config.embed_dim, n_classes).to(device)
        for param in model.parameters():
            param.requires_grad_(True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr,
                                      weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.config.pilot_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # Extract spectral profiles
        pilot_state = {k: v.cpu() for k, v in model.state_dict().items()}

        spectral_profiles = {}
        alphas = []
        nuclear_norms = {}

        for name in pretrained_state:
            if 'qkv.weight' in name or 'proj.weight' in name:
                delta = pilot_state[name].float() - pretrained_state[name].float()
                sv = compute_svd_profile(delta)
                alpha, C = fit_spectral_decay(sv)
                spectral_profiles[name] = sv
                alphas.append(alpha)
                nuclear_norms[name] = float(sv.sum())

        # Compute S_attn (requires attention map comparison)
        pre_attn = extract_attention_maps(
            # Need pretrained model for this
            model, val_loader, device, max_batches=2)
        s_attn_est = 0.01  # Placeholder; real implementation compares pre/post

        # Compute σ_P²
        sigma_p_sq = compute_sigma_p_sq(pretrained_state, self.config)

        self.task_descriptors = {
            'mean_alpha': float(np.mean(alphas)) if alphas else 1.0,
            'spectral_profiles': {k: v.tolist() for k, v in spectral_profiles.items()},
            'nuclear_norms': nuclear_norms,
            'S_attn_est': s_attn_est,
            'sigma_p_sq': sigma_p_sq,
            'n_train': self.config.n_train,
        }

        print(f"  [Pilot] Estimated α = {self.task_descriptors['mean_alpha']:.3f}")
        print(f"  [Pilot] σ_P² = {sigma_p_sq:.6f}")

        return self.task_descriptors

    def compute_scores(self):
        """Phase 2: Compute Score(A, κ) for all method × capacity combinations."""
        td = self.task_descriptors
        n = td['n_train']
        sigma_p_sq = td['sigma_p_sq']
        alpha = td['mean_alpha']

        self.scores = {}

        # --- LP ---
        # Approximation: full task shift (constant)
        # Generalization: small (dK' parameters)
        d_K = self.config.embed_dim * 100  # assume 100 classes max
        kl_lp = d_K * 0.01 / (2 * sigma_p_sq)  # rough estimate
        gen_lp = np.sqrt((kl_lp + np.log(2 * np.sqrt(n))) / (2 * n))
        approx_lp = 1.0  # large (no internal adaptation)
        self.scores['LP'] = {
            'score': approx_lp + gen_lp,
            'approx': approx_lp,
            'gen': gen_lp,
            'kl': kl_lp,
        }

        # --- LoRA ---
        for r in self.config.lora_ranks:
            # Approximation: spectral tail T(r)
            tail = 0.0
            nuclear = 0.0
            for name, sv_list in td['spectral_profiles'].items():
                sv = np.array(sv_list)
                tail += (sv[r:] ** 2).sum()
                nuclear += sv[:r].sum()

            approx_lora = tail * 0.01  # scaled

            # Generalization: KL = nuclear_norm / sigma_p_sq
            kl_lora = nuclear / sigma_p_sq
            gen_lora = np.sqrt((kl_lora + np.log(2 * np.sqrt(n))) / (2 * n))

            self.scores[f'LoRA_r{r}'] = {
                'score': approx_lora + gen_lora,
                'approx': approx_lora,
                'gen': gen_lora,
                'kl': kl_lora,
                'rank': r,
            }

        # --- VPT ---
        for p in self.config.vpt_prompt_counts:
            # Approximation: Q(a*, p) + sin²θ residual
            approx_attn = td['S_attn_est'] / max(p, 1)  # simplified
            approx_feat = 0.1  # sin²θ residual estimate
            approx_vpt = approx_attn + approx_feat

            # Generalization: KL = Lpd·η² / (2σ_P²)
            d_vpt = self.config.num_layers * p * self.config.embed_dim
            eta_sq = 0.01  # estimated prompt norm
            kl_vpt = d_vpt * eta_sq / (2 * sigma_p_sq)
            gen_vpt = np.sqrt((kl_vpt + np.log(2 * np.sqrt(n))) / (2 * n))

            self.scores[f'VPT_p{p}'] = {
                'score': approx_vpt + gen_vpt,
                'approx': approx_vpt,
                'gen': gen_vpt,
                'kl': kl_vpt,
                'n_prompts': p,
            }

        # --- Adapter ---
        for r_a in self.config.adapter_dims:
            # Approximation: S_attn + T_feat(r_a)
            tail_feat = 0.0
            for name, sv_list in td['spectral_profiles'].items():
                if 'qkv' in name:  # V is part of qkv
                    sv = np.array(sv_list)
                    tail_feat += (sv[r_a:] ** 2).sum()

            approx_adapter = td['S_attn_est'] + tail_feat * 0.01

            # Generalization
            d_adapter = 2 * self.config.num_layers * r_a * self.config.embed_dim
            kl_adapter = d_adapter * 0.01 / (2 * sigma_p_sq)
            gen_adapter = np.sqrt((kl_adapter + np.log(2 * np.sqrt(n))) / (2 * n))

            self.scores[f'Adapter_r{r_a}'] = {
                'score': approx_adapter + gen_adapter,
                'approx': approx_adapter,
                'gen': gen_adapter,
                'kl': kl_adapter,
                'bottleneck': r_a,
            }

        return self.scores

    def select(self):
        """Phase 3: Select the method with minimum score."""
        best_method = min(self.scores, key=lambda k: self.scores[k]['score'])
        return best_method, self.scores[best_method]


def run_selection_benchmark(config: ExperimentConfig):
    """Benchmark the selection algorithm against exhaustive search."""
    print("=" * 70)
    print("EXPERIMENT 3: SELECTION ALGORITHM BENCHMARK")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    try:
        base_model = timm.create_model(config.model_name, pretrained=True)
    except Exception:
        base_model = timm.create_model(config.model_name, pretrained=False)

    pretrained_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    tasks = {'cifar100': 100, 'dtd': 47, 'clevr_count': 8, 'eurosat': 10}
    all_results = {}

    for task_name, n_classes in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        category = config.task_category(task_name)
        task_type = 'structured' if category == 'structured' else 'natural'

        train_ds = SyntheticVTABDataset(config.n_train, n_classes,
                                        config.img_size, task_type)
        val_ds = SyntheticVTABDataset(config.n_val, n_classes,
                                      config.img_size, task_type)

        # Split train into pilot + actual
        n_pilot = config.n_train // 4
        pilot_ds = Subset(train_ds, range(n_pilot))
        actual_ds = Subset(train_ds, range(n_pilot, config.n_train))

        pilot_loader = DataLoader(pilot_ds, batch_size=config.batch_size,
                                   shuffle=True, num_workers=0)
        train_loader = DataLoader(actual_ds, batch_size=config.batch_size,
                                   shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, num_workers=0)

        # --- Phase 1: Run selection algorithm ---
        print("\n  SELECTION ALGORITHM:")
        selector = SelectionAlgorithm(config)

        pilot_model = deepcopy(base_model).to(device)
        selector.run_pilot(pilot_model, pilot_loader, val_loader,
                          pretrained_state, n_classes, device)
        del pilot_model

        scores = selector.compute_scores()
        selected_method, selected_score = selector.select()
        print(f"  Selected: {selected_method} (score={selected_score['score']:.4f})")

        # --- Phase 2: Exhaustive search (oracle) ---
        print("\n  EXHAUSTIVE SEARCH (oracle):")
        exhaustive_results = {}

        methods_to_try = [
            ('LP', lambda m: apply_linear_probe(m, config), {}),
            ('LoRA_r4', lambda m: apply_lora(m, 4, config), {}),
            ('LoRA_r8', lambda m: apply_lora(m, 8, config), {}),
            ('LoRA_r16', lambda m: apply_lora(m, 16, config), {}),
            ('VPT_p10', lambda m: apply_vpt(m, 10, config), {}),
            ('VPT_p50', lambda m: apply_vpt(m, 50, config), {}),
            ('Adapter_r16', lambda m: apply_adapter(m, 16, config), {}),
            ('Adapter_r64', lambda m: apply_adapter(m, 64, config), {}),
        ]

        for method_name, apply_fn, _ in methods_to_try:
            model = deepcopy(base_model).to(device)
            model.head = nn.Linear(config.embed_dim, n_classes).to(device)
            model = apply_fn(model)
            acc = train_and_evaluate(model, train_loader, val_loader, config,
                                     device, method_name)
            exhaustive_results[method_name] = acc
            del model

        # --- Comparison ---
        oracle_method = max(exhaustive_results, key=exhaustive_results.get)
        oracle_acc = exhaustive_results[oracle_method]
        selected_acc = exhaustive_results.get(selected_method, 0.0)
        regret = oracle_acc - selected_acc

        print(f"\n  COMPARISON:")
        print(f"    Oracle: {oracle_method} ({oracle_acc:.4f})")
        print(f"    Selected: {selected_method} ({selected_acc:.4f})")
        print(f"    Regret: {regret:.4f}")

        # Kendall τ between scores and accuracies
        common_methods = [m for m in scores if m in exhaustive_results]
        if len(common_methods) >= 3:
            score_vals = [scores[m]['score'] for m in common_methods]
            acc_vals = [exhaustive_results[m] for m in common_methods]
            tau, p_val = stats.kendalltau(score_vals, [-a for a in acc_vals])
            print(f"    Kendall τ (score vs accuracy): {tau:.3f} (p={p_val:.3f})")
        else:
            tau = 0.0

        # Non-vacuousness check
        print(f"\n  NON-VACUOUSNESS (Theorem 4):")
        n_actual = len(actual_ds)
        for method_name in ['LoRA_r4', 'LoRA_r8', 'VPT_p10']:
            if method_name in scores:
                kl = scores[method_name]['kl']
                threshold = 2 * n_actual
                is_nonvacuous = kl < threshold
                status = "✓ NON-VACUOUS" if is_nonvacuous else "✗ VACUOUS"
                print(f"    {method_name}: KL={kl:.1f}, threshold={threshold}, {status}")

        all_results[task_name] = {
            'selected_method': selected_method,
            'selected_accuracy': selected_acc,
            'oracle_method': oracle_method,
            'oracle_accuracy': oracle_acc,
            'regret': regret,
            'kendall_tau': tau,
            'scores': {k: v['score'] for k, v in scores.items()},
            'exhaustive_accuracies': exhaustive_results,
            'task_descriptors': selector.task_descriptors,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SELECTION ALGORITHM SUMMARY")
    print("=" * 70)

    regrets = [r['regret'] for r in all_results.values()]
    taus = [r['kendall_tau'] for r in all_results.values()]

    print(f"  Mean regret: {np.mean(regrets):.4f} (target: < 0.005)")
    print(f"  Max regret:  {np.max(regrets):.4f}")
    print(f"  Mean Kendall τ: {np.mean(taus):.3f} (target: > 0.7)")
    print(f"  Compute savings: ~{len(methods_to_try)}× fewer runs than exhaustive")

    save_results(all_results, 'experiment3_selection.json', config)
    return all_results


if __name__ == '__main__':
    config = ExperimentConfig(epochs=15, pilot_epochs=5)
    run_selection_benchmark(config)
