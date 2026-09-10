# ATLAS: Not a Fair Fight — Learning-Rate Tuning Tips the LoRA vs VPT Balance

> **An empirical study showing that conflicting LoRA-vs-VPT conclusions largely stem from asymmetric learning-rate tuning**

[![Paper](https://img.shields.io/badge/Paper-TMLR%20(under%20review)-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

---

## Overview

Published comparisons between LoRA and Visual Prompt Tuning (VPT) on Vision Transformers disagree on which method is better. We show that a plausible source of this disagreement is a **two-sided learning-rate interaction**: both methods are sensitive to per-backbone LR tuning, but in opposite directions.

After sweeping both methods' LRs across **8 ViT-B backbones × 5 tasks** (3 seeds, official test splits):

- **LoRA wins or ties on 37/40 pairs** (20 wins, 17 ties, 3 VPT wins)
- The 3 VPT wins are on MoCo-v3 (CIFAR-100, DTD) and iBOT (CIFAR-100)
- Threshold sensitivity: 36–38/40 at any threshold in [1%, 3%]

### Key Findings

| Finding | Evidence |
|---------|----------|
| **Two-sided LR interaction** | DINOv1 DTD: VPT "wins" by 6% → tie after sweeping LoRA's LR |
| **Warmup fixes VPT crashes** | DINOv2 at LR=1e-2: 0.386±0.224 → 0.781±0.017 with 10-epoch warmup |
| **σ²_P correlates with instability** | All drops >10 pts occur at σ²_P < 0.7 (44/44 informative sweeps) |
| **LP confirms prompt-specificity** | LP at LR=1e-2 shows no crash on any backbone; VPT drops 60 pts |
| **Capacity is task-dependent** | p=50 helps on GTSRB (+1 to +3 pts), hurts on CIFAR-100 (up to -37 pts) |
| **Regret analysis** | Always picking swept LoRA costs only 0.3 pts vs oracle |

### Practical Recommendation

**Use warmup for VPT, sweep LR for LoRA.** (Algorithm 1 in the paper)

---

## Backbones

| Backbone | Pretraining | σ²_P | L/T/V |
|----------|-------------|------|-------|
| DINOv2 | Self-distillation | 0.22 | 2/3/0 |
| iBOT | Self-distillation | 0.15 | 2/2/1 |
| DINOv1 | Self-distillation | 0.19 | 3/2/0 |
| CLIP | Contrastive | 0.18 | 2/3/0 |
| DeiT-III | Supervised | 1.04 | 3/2/0 |
| Supervised | Supervised | 1.60 | 2/3/0 |
| MoCo-v3 | Contrastive | 2.31 | 2/1/2 |
| MAE | Masked autoencoding | 1.76 | 4/1/0 |

## Tasks

CIFAR-100, SVHN, GTSRB, EuroSAT, DTD — all at n=1000 per task.

---

## Repository Structure

```
ATLAS/
├── README.md
├── experiments/
│   ├── test_split_eval.py          # Main comparison (40/40, test splits, 3 seeds)
│   ├── linear_probe.py             # LP baseline (8 backbones)
│   ├── warmup_control.py           # Warmup vs no-warmup (8 backbones)
│   ├── reviewer_checks.py          # Seed verification + LP at high LR
│   ├── lora_grid_check.py          # LoRA grid-edge check (13 cells)
│   ├── lp_high_lr.py               # LP at 1e-2 (head-divergence control)
│   ├── warmup_rerun.py             # With-warmup full comparison (future)
│   ├── revision_vpt_full_sweep.py  # 7-point VPT LR sweeps
│   ├── revision_capacity_lr_sweep.py  # Per-p LR tuning
│   ├── sigma_ablation.py           # σ²_P ablation (8 statistics)
│   ├── lora_small_rank.py          # LoRA r=1,2
│   ├── dinov2_reg_sweep.py         # Register-prompt interference
│   ├── weight_decay_sweep.py       # WD sensitivity check
│   ├── grid_edge_check.py          # VPT grid-edge extension
│   └── fft_lr_sweep.py             # Full fine-tuning baseline
│
├── paper/
│   └── tmlr/
│       ├── atlas_tmlr.tex          # Main manuscript
│       ├── atlas_tmlr.bib          # References
│       └── figures/                # Paper figures
│
├── config.py                       # Shared configuration
└── exp2_comparison.py              # LoRA/VPT application utilities
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/ATLAS.git
cd ATLAS
pip install -r requirements.txt
```

### Run the Main Comparison

```bash
# Full 40-cell test-split evaluation (all 8 backbones, 3 seeds)
python experiments/test_split_eval.py --resume

# Specific backbones
python experiments/test_split_eval.py --backbones DINOv2 CLIP --resume

# Single cell
python experiments/test_split_eval.py --backbones DINOv2 --tasks cifar100
```

### Run Controls

```bash
# Linear probe baseline (all 8 backbones)
python experiments/linear_probe.py --resume

# Warmup control (all 8 backbones)
python experiments/warmup_control.py --resume

# Seed verification + LP at high LR
python experiments/reviewer_checks.py

# LoRA grid-edge check
python experiments/lora_grid_check.py --backbones DeiT-III MoCo-v3 MAE Supervised
```

### Run Secondary Experiments

```bash
# σ²_P validation (7-point VPT sweeps)
python experiments/revision_vpt_full_sweep.py

# Capacity sweep (per-p LR tuning)
python experiments/revision_capacity_lr_sweep.py

# LoRA r=1,2
python experiments/lora_small_rank.py

# Register-prompt interference
python experiments/dinov2_reg_sweep.py
```

---

## Results Summary

### Main Comparison (Table 3, test splits)

LoRA r=8 Q/V (295K params) vs VPT p=5 deep (46K params), evaluated on official test splits:

| | LoRA wins | Ties | VPT wins |
|---|---|---|---|
| **Count** | 20 | 17 | 3 |
| **At ±1%** | 24 | 12 | 4 |
| **At ±3%** | 17 | 21 | 2 |

### Regret Analysis

| Strategy | Mean regret | Max regret |
|----------|-------------|------------|
| Always LoRA (swept) | **0.30 pts** | 4.1 pts |
| Always VPT (swept) | 3.89 pts | 23.8 pts |

### Warmup Control (CIFAR-100, LR=1e-2)

| Backbone | No warmup | 10-ep warmup | Δ |
|----------|-----------|-------------|---|
| DINOv2 | 0.386±0.224 | 0.781±0.017 | **+39.6** |
| DINOv1 | 0.406±0.092 | 0.619±0.002 | **+21.3** |
| Others | modest or none | | +2.8 to +5.9 |

---

## Citation

```bibtex
@article{atlas2026,
  title={Not a Fair Fight: Learning-Rate Tuning Tips the 
         {LoRA} vs {VPT} Balance on Vision Transformers},
  author={},
  journal={Transactions on Machine Learning Research},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
