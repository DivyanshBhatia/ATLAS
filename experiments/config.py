"""
Experimental Codebase for:
"A Unified Theory of Adaptation Efficiency in Vision Foundation Models"

Requirements:
    pip install torch torchvision timm peft datasets scipy matplotlib seaborn tqdm

Hardware: 1× GPU with ≥16GB VRAM (A100/V100/RTX 3090)
Estimated runtime: ~8 hours for all experiments

Usage:
    python run_all.py                    # Run everything
    python run_all.py --exp spectral     # Run only spectral analysis
    python run_all.py --exp comparison   # Run only PEFT comparison
    python run_all.py --exp selection    # Run only selection algorithm
    python run_all.py --exp assumption_a # Run only Assumption A verification
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Central configuration for all experiments."""

    # Model
    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    head_dim: int = 64
    patch_size: int = 14
    img_size: int = 518
    num_patches: int = 1369  # (518/14)^2 = 37^2

    # VTAB-1K tasks (grouped by category)
    vtab_natural: List[str] = field(default_factory=lambda: [
        "cifar100", "caltech101", "dtd", "oxford_flowers102",
        "oxford_iiit_pet", "svhn", "sun397",
    ])
    vtab_specialized: List[str] = field(default_factory=lambda: [
        "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy",
    ])
    vtab_structured: List[str] = field(default_factory=lambda: [
        "clevr_count", "clevr_dist", "dmlab", "kitti",
        "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele",
        "gtsrb",  # German Traffic Signs — available via torchvision
    ])

    # PEFT configurations
    lora_ranks: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    vpt_prompt_counts: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    adapter_dims: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])

    # Training
    n_train: int = 1000        # Default for single-scale experiments
    n_train_fft: int = 0       # 0 = use ALL available data for FFT spectral analysis
    n_train_scales: list = field(default_factory=lambda: [200, 500, 1000, 5000, 0])
    # ^ 0 = full dataset. Tests Theorem 3 prediction: optimal method changes with n
    n_val: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 100          # Standard for VTAB-1K (cosine schedule)
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    pilot_epochs: int = 5      # For selection algorithm

    # Paths
    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"

    @property
    def all_tasks(self):
        return self.vtab_natural + self.vtab_specialized + self.vtab_structured

    def task_category(self, task_name):
        if task_name in self.vtab_natural:
            return "natural"
        elif task_name in self.vtab_specialized:
            return "specialized"
        elif task_name in self.vtab_structured:
            return "structured"
        return "unknown"


# ============================================================================
# Shared Utilities
# ============================================================================

def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU. Experiments will be very slow.")
    return device


def ensure_dirs(config: ExperimentConfig):
    """Create output directories."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def save_results(results: dict, filename: str, config: ExperimentConfig):
    """Save results as JSON."""
    path = os.path.join(config.output_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def compute_svd_profile(delta_W: torch.Tensor) -> np.ndarray:
    """Compute singular values of a weight shift matrix."""
    with torch.no_grad():
        S = torch.linalg.svdvals(delta_W.float())
    return S.cpu().numpy()


def fit_spectral_decay(singular_values: np.ndarray, max_k: int = 50
                       ) -> Tuple[float, float]:
    """
    Fit σ_k = C · k^{-α} by linear regression on log-log scale.
    Returns (alpha, C).
    """
    sv = singular_values[:max_k]
    sv = sv[sv > 1e-8]  # Remove near-zero values
    if len(sv) < 3:
        return 0.0, 0.0

    k = np.arange(1, len(sv) + 1)
    log_k = np.log(k)
    log_sv = np.log(sv)

    # Linear regression: log σ = log C - α log k
    coeffs = np.polyfit(log_k, log_sv, 1)
    alpha = -coeffs[0]
    C = np.exp(coeffs[1])

    return float(alpha), float(C)


def compute_attention_shift(attn_pretrained: torch.Tensor,
                            attn_finetuned: torch.Tensor) -> float:
    """
    Compute S_attn = mean squared attention change.
    Inputs: [n_images, n_heads, seq_len, seq_len]
    """
    # CLS-to-patch attention only (row 0, columns 1:)
    a0 = attn_pretrained[:, :, 0, 1:]  # [n_images, H, N_patches]
    a1 = attn_finetuned[:, :, 0, 1:]

    s_attn = ((a1 - a0) ** 2).mean().item()
    return s_attn


def compute_feature_shift(W_V_pretrained: torch.Tensor,
                          W_V_finetuned: torch.Tensor,
                          readout_c: torch.Tensor) -> Tuple[float, float]:
    """
    Compute S_feat^{(c)} and sin²θ_avg.

    Args:
        W_V_pretrained: [num_layers, d, d] pretrained value projections
        W_V_finetuned: [num_layers, d, d] fine-tuned value projections
        readout_c: [d] readout vector (classification head direction)

    Returns:
        (S_feat_c, sin2_theta_avg)
    """
    L = W_V_pretrained.shape[0]
    s_feat_c = 0.0
    sin2_theta_sum = 0.0

    for l in range(L):
        delta_WV = W_V_finetuned[l] - W_V_pretrained[l]

        # c-projected feature shift
        c_delta = readout_c @ delta_WV  # [d]
        s_feat_c += c_delta.norm().item() ** 2

        # Angle between c^T W_V* and c^T W_V^0
        eta_0 = readout_c @ W_V_pretrained[l]  # [d]
        eta_star = readout_c @ W_V_finetuned[l]  # [d]

        cos_theta = (eta_0 @ eta_star).item() / (
            eta_0.norm().item() * eta_star.norm().item() + 1e-10)
        cos_theta = np.clip(cos_theta, -1, 1)
        sin2_theta = 1 - cos_theta ** 2
        sin2_theta_sum += sin2_theta

    s_feat_c /= L
    sin2_theta_avg = sin2_theta_sum / L

    return float(s_feat_c), float(sin2_theta_avg)


def compute_quantization_error(attn_target: np.ndarray, p: int) -> float:
    """
    Compute Q(a*, p) = k-means quantization error of attention pattern.

    Args:
        attn_target: [N_patches] target attention vector
        p: number of groups

    Returns:
        Quantization error (sum of within-group variances)
    """
    from scipy.cluster.vq import kmeans2

    if p >= len(attn_target):
        return 0.0

    # Run k-means
    attn_reshaped = attn_target.reshape(-1, 1)
    centroids, labels = kmeans2(attn_reshaped.astype(np.float64), p,
                                minit='points', iter=50)

    # Compute quantization error
    error = 0.0
    for k in range(p):
        mask = labels == k
        if mask.sum() > 0:
            group_vals = attn_target[mask]
            error += np.sum((group_vals - group_vals.mean()) ** 2)

    return float(error)


# ============================================================================
# PAC-Bayes Bound Computation
# ============================================================================

def compute_pac_bayes_bound(
    empirical_risk: float,
    kl_divergence: float,
    n: int,
    delta: float = 0.05
) -> float:
    """
    Compute PAC-Bayes generalization bound.

    Risk ≤ empirical_risk + sqrt((KL + ln(2√n/δ)) / (2n))
    """
    complexity = (kl_divergence + np.log(2 * np.sqrt(n) / delta)) / (2 * n)
    if complexity < 0:
        complexity = 0
    gen_penalty = np.sqrt(complexity)
    return empirical_risk + gen_penalty


def compute_lora_kl(singular_values_per_layer: Dict[str, np.ndarray],
                    rank: int,
                    sigma_p_sq: float) -> float:
    """
    KL for LoRA = Σ_{l,m} Σ_{k=1}^r σ_k / σ_P²  (nuclear norm / σ_P²).
    """
    total_nuclear = 0.0
    for key, sv in singular_values_per_layer.items():
        total_nuclear += sv[:rank].sum()
    return total_nuclear / sigma_p_sq


def compute_vpt_kl(prompt_norms_sq: float,
                   num_layers: int,
                   num_prompts: int,
                   embed_dim: int,
                   sigma_p_sq: float) -> float:
    """
    KL for VPT = Σ_l ‖P̂^{(l)}‖_F² / (2σ_P²).
    """
    return prompt_norms_sq / (2 * sigma_p_sq)


def compute_sigma_p_sq(model_state_dict: dict,
                       config: ExperimentConfig) -> float:
    """
    Compute prior variance σ_P² from pretrained weights.
    σ_P² = (1 / (L·M·d_h)) · Σ_{l,m} ‖W_0^{(l,m)}‖_F²
    """
    total_norm_sq = 0.0
    count = 0

    for name, param in model_state_dict.items():
        if any(t in name for t in ['qkv.weight', 'proj.weight']):
            total_norm_sq += param.float().norm().item() ** 2
            count += 1

    if count == 0:
        return 1.0

    return total_norm_sq / (count * config.head_dim)


if __name__ == "__main__":
    config = ExperimentConfig()
    print("Experiment Configuration:")
    print(json.dumps(asdict(config), indent=2, default=str))
