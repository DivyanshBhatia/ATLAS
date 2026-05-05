# Experimental Codebase

## Paper: "A Unified Theory of Adaptation Efficiency in Vision Foundation Models"

---

## Setup

```bash
pip install torch torchvision timm datasets scipy matplotlib seaborn tqdm
```

**Hardware:** 1× GPU with ≥16GB VRAM (A100, V100, RTX 3090/4090)

---

## Quick Start

```bash
# Run all experiments (full, ~8 hours)
python run_all.py

# Quick test run (~30 minutes)
python run_all.py --fast

# Run individual experiments
python run_all.py --exp spectral      # ~1 hour
python run_all.py --exp comparison    # ~4 hours
python run_all.py --exp selection     # ~3 hours
```

---

## Experiment → Theorem Mapping

| Experiment | File | Validates | Key Prediction |
|---|---|---|---|
| **1. Spectral Profiles** | `exp1_spectral.py` | Theorem 2(b) | LoRA rate = O(r^{1-2α}); α varies by task category |
| **2. PEFT Comparison** | `exp2_comparison.py` | Theorems 1, 2 | VPT wins on structured tasks, LoRA wins on natural tasks |
| **3. Selection Benchmark** | `exp3_selection.py` | Theorems 4, 5 | Selection algorithm regret < 0.5%, Kendall τ > 0.7 |
| **4. Assumption A** | `validate_assumption_a.py` | Theorem 1(a) | Pretrained ViTs have position-steerable heads (γ > 5) |

---

## What Each Experiment Measures

### Experiment 1: Spectral Profile Analysis

For each VTAB task, we fine-tune a DINOv2-ViT-B model, compute ΔW = W_fft − W_0,
and analyze the SVD:

- **Singular value decay rate α** (fit σ_k = C·k^{-α})
- **Spectral tail T(r)** = Σ_{k>r} σ_k² for each rank r
- **Nuclear norm** Σ_k σ_k (determines LoRA's KL divergence)
- **Rank for 90% energy** (minimum rank capturing 90% of ‖ΔW‖_F²)

**Expected results:**
- Natural tasks: α ≈ 1.5–2.5 (fast decay → low-rank LoRA suffices)
- Structured tasks: α ≈ 0.5–1.5 (slow decay → higher rank or VPT needed)

### Experiment 2: PEFT Method Comparison

Trains LoRA (r=4,8,16), VPT (p=10,50), Adapter (r_a=16,64), and LP
on each VTAB task. Also measures task descriptors:

- **S_attn**: attention shift complexity (comparing pre/post attention maps)
- **S_feat^{(c)}**: c-projected feature shift (from ΔW_V)
- **sin²θ_avg**: feature direction change angle

**Expected results:**
- High S_attn / low S_feat tasks (structured): VPT ≥ LoRA
- High S_feat / low S_attn tasks (natural): LoRA > VPT
- Accuracy vs. rank follows the O(r^{1-2α}) prediction from Theorem 2

### Experiment 3: Selection Algorithm

Implements the full selection procedure from Theorem 5:
1. Pilot FFT on n/4 samples (5 epochs)
2. Estimate task descriptors
3. Score each method using Theorems 2+3
4. Select the best
5. Compare to oracle (exhaustive search)

**Expected results:**
- Regret < 0.5% (accuracy gap vs. oracle)
- Kendall τ > 0.7 (score ranking correlates with accuracy ranking)
- ~10× compute savings over exhaustive search
- Non-vacuous PAC-Bayes bounds for LoRA(4) at n=1000

### Experiment 4: Assumption A

Verifies that pretrained ViTs satisfy the position-steerability condition
required for Theorem 1(a). Measures QK asymmetry, cross/self attention ratio,
and directly tests prompt steering capability.

**Expected results:**
- QK asymmetry ≈ 1/√2 ≈ 0.707 (proven analytically)
- Cross/self ratio > 8 (proven: O(√d_h))
- Pretrained model γ > 5 in position-aware heads (to be verified with GPU)

---

## Output Files

All results are saved as JSON in `./results/`:

```
results/
├── experiment1_spectral.json     # Spectral profiles per task
├── experiment2_comparison.json   # PEFT accuracies + task descriptors
├── experiment3_selection.json    # Selection vs. oracle comparison
└── assumption_a_results.json    # Assumption A verification
```

---

## Adapting for Real VTAB-1K

The codebase uses synthetic data by default (for environments without
dataset access). To run on real VTAB-1K:

1. Install datasets: `pip install datasets`
2. The code automatically attempts to load from HuggingFace datasets
3. If unavailable, falls back to synthetic data

For the final paper, replace `SyntheticVTABDataset` with proper VTAB-1K
data loaders using the tensorflow_datasets or HuggingFace backends.

---

## Key Implementation Notes

- **LoRA:** Applied to the combined QKV projection in timm's ViT (not separate Q, K, V)
- **VPT-Deep:** Prompts prepended at every layer, removed from output to maintain sequence length
- **Adapter:** Bottleneck adapter after each transformer block (down-project, GELU, up-project)
- **Selection:** Uses pilot SVD profiles to compute theoretical scores; no hyperparameter tuning needed
