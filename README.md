# ATLAS: Adaptation Theory — Limits, Approximation, and Selection

> **A Unified Theory of Adaptation Efficiency in Vision Foundation Models**

[![Paper](https://img.shields.io/badge/Paper-TPAMI%20(under%20review)-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

---

## Overview

ATLAS provides a **unified theoretical framework** for understanding and selecting parameter-efficient fine-tuning (PEFT) methods for Vision Transformers. We prove that LoRA, VPT, and Adapters have fundamentally different expressiveness characteristics, derive their approximation rates as continuous functions of measurable task properties, and provide a principled selection algorithm that achieves near-oracle performance with 10× less compute.

### Key Contributions

| Theorem | Result | Practical Impact |
|---------|--------|-----------------|
| **Separation (Thm 1)** | LoRA and VPT have provably different expressiveness on different task types | Explains *why* no single PEFT method dominates |
| **Rates (Thm 2)** | Closed-form approximation rates parameterized by spectral decay α | Predicts how accuracy scales with rank/prompt count |
| **Bounds (Thm 3)** | PAC-Bayes generalization bounds with method-specific priors | LoRA's KL = nuclear norm (structural advantage) |
| **Selection (Thm 5)** | Near-optimal method selection from a 2-epoch pilot | 10× compute savings over exhaustive search |

### Core Insight

Every downstream task decomposes into an **attention shift** (which patches to attend to) and a **feature shift** (what features to extract). Different PEFT methods address different components:

| Method | Attention Shift | Feature Shift | Capacity |
|--------|:-:|:-:|:-:|
| **LoRA** | ✓ (via W_Q, W_K) | ✓ (via W_V) | rank r |
| **VPT** | ✓ (via prompt steering) | ✗ (frozen W_V) | prompt count p |
| **Adapter** | ✗ (frozen attention) | ✓ (via bottleneck) | bottleneck dim r_a |
| **Linear Probe** | ✗ | ✗ | — |

---

## Repository Structure

```
ATLAS/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
│
├── theory/                         # Formal proofs and analysis
│   ├── paper_framework.md          # Problem formulation & notation
│   ├── theorem1_proof.md           # Expressiveness separation proof
│   ├── softmax_analysis.md         # Softmax Lipschitz analysis
│   ├── lemma3_and_review.md        # VPT relay construction & critical review
│   ├── theorem2_proof.md           # Approximation rates proof
│   ├── theorem2_review_and_theorem3.md  # Theorem 2 errata + Theorem 3
│   ├── consolidated_review_and_theorem5.md  # Final review + Theorem 5
│   └── assumption_a_analysis.md    # Assumption A validation results
│
├── experiments/                    # Experimental code
│   ├── README.md                   # Experiment documentation
│   ├── config.py                   # Configuration & shared utilities
│   ├── exp1_spectral.py            # Spectral profile analysis
│   ├── exp2_comparison.py          # PEFT method comparison
│   ├── exp3_selection.py           # Selection algorithm benchmark
│   ├── validate_assumption_a.py    # Assumption A verification
│   └── run_all.py                  # Main experiment runner
│
├── paper/                          # LaTeX manuscript (forthcoming)
│   └── .gitkeep
│
└── figures/                        # Generated figures
    └── .gitkeep
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/ATLAS.git
cd ATLAS
pip install -r requirements.txt
```

### Run Experiments

```bash
# Quick test run (~30 minutes, synthetic data)
cd experiments
python run_all.py --fast

# Full experiments (~8 hours, requires GPU)
python run_all.py

# Individual experiments
python run_all.py --exp spectral      # Spectral decay analysis
python run_all.py --exp comparison    # LoRA vs VPT vs Adapter comparison
python run_all.py --exp selection     # Selection algorithm benchmark
```

### Use the Selection Algorithm

```python
from experiments.exp3_selection import SelectionAlgorithm
from experiments.config import ExperimentConfig

config = ExperimentConfig()
selector = SelectionAlgorithm(config)

# Phase 1: Run a 5-epoch pilot fine-tuning
selector.run_pilot(model, train_loader, val_loader, pretrained_state, n_classes, device)

# Phase 2: Score all methods
scores = selector.compute_scores()

# Phase 3: Get recommendation
best_method, best_score = selector.select()
print(f"Recommended: {best_method} (score: {best_score['score']:.4f})")
```

---

## Theoretical Results

### Theorem 1: Expressiveness Separation

There exist task distributions where:

**(a) VPT beats LoRA by Ω(N):** On attention-steering tasks (counting, spatial reasoning), VPT with √N prompts achieves O(1/N²) error while LoRA with rank r achieves Ω(1/N) error — a gap of ~196× for ViT-B.

**(b) LoRA beats VPT by ∞:** On feature-discrimination tasks (fine-grained classification), LoRA achieves zero error with sufficient rank, while VPT has **irreducible error** ≥ sin²θ > 0 for any number of prompts.

### Theorem 2: Approximation Rates

For a task with spectral decay σ_k ≤ C·k^{-α}:

- **LoRA(r):** ε = O(r^{1−2α}) — decays to zero, rate controlled by α
- **VPT(p):** ε = O(Q(a*,p)) + S_feat·sin²θ — decays then **plateaus**
- **Adapter(r_a):** ε = S_attn + O(r_a^{1−2α}) — **floor** from attention

### Theorem 3: Generalization

LoRA has a structural advantage: its KL divergence equals the **nuclear norm** of the weight shift (Σ σ_k), not the parameter count. For fast-decaying spectra, this is much smaller, explaining LoRA's strong generalization at low data counts.

### Theorem 5: Selection Algorithm

A 2-epoch pilot fine-tuning suffices to estimate task descriptors and select the near-optimal PEFT method with regret O(1/√n) over the oracle, saving ~10× compute.

---

## Key Assumptions

| Assumption | Content | Justification |
|---|---|---|
| **A** (Steerability) | Pretrained ViT has position-aware attention heads | Empirical: all standard ViTs (DINOv2, CLIP, MAE) develop position-specialized heads |
| **B** (Well-conditioned) | Residual stream Lipschitz constant O(1) | Standard for ViTs with LayerNorm |

---

## Citation

```bibtex
@article{atlas2026,
  title={A Unified Theory of Adaptation Efficiency in Vision Foundation Models: 
         Expressiveness, Generalization, and Optimal Method Selection},
  author={},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
