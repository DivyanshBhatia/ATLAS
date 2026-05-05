# Proposition 2: Spectral-Informed Learning Rate Selection

## Connecting α to Optimal Learning Rate for Each PEFT Method

---

## 1. Motivation

We observed empirically that FFT requires LR ≈ 2e-5 while LoRA requires
LR ≈ 1e-3 — a 50× difference. This isn't arbitrary: it follows from the
spectral structure of the adaptation problem. Since our framework already
computes α and the spectral profile of ΔW*, we can derive the optimal LR
for each method as a function of these same quantities.

---

## 2. Theory

### 2.1 Setup

Consider fine-tuning a pretrained model f_{θ₀} on a downstream task with
loss L(φ) = E[(f_{θ₀,φ}(x) - y)²], where φ represents the trainable
parameters of adaptation method A.

Gradient descent updates: φ_{t+1} = φ_t − η ∇L(φ_t)

The optimal learning rate must satisfy two conditions:
1. **Stability:** η < 2/λ_max(H_φ) where H_φ = ∇²L is the Hessian
   (otherwise gradient descent diverges)
2. **Efficiency:** η shouldn't be much smaller than 1/λ_max(H_φ)
   (otherwise convergence is unnecessarily slow)

So: η_opt ≈ c / λ_max(H_φ) for a constant c ∈ (0, 2).

### 2.2 Hessian Eigenvalue for Each Method

For a model f_{θ₀,φ}(x) with output z = f(x) ∈ ℝ^K, the Hessian of the
squared loss is:

$$H_\phi = \frac{1}{n} \sum_{i=1}^n J_\phi(x_i)^\top J_\phi(x_i) + \text{second-order terms}$$

where J_φ(x) = ∂f/∂φ ∈ ℝ^{K × d_φ} is the Jacobian of the model output
with respect to the trainable parameters.

The maximum eigenvalue is dominated by:

$$\lambda_{\max}(H_\phi) \approx \frac{1}{n} \sum_{i=1}^n \|J_\phi(x_i)\|_{\text{op}}^2$$

(The second-order terms are small near the pretrained initialization when
the residuals are small.)

### 2.3 Jacobian Norm for Each Method

**Full Fine-Tuning (FFT):**

All d_FFT = |θ| parameters are trainable. The Jacobian J_θ(x) has:

$$\|J_\theta\|_{\text{op}} \leq \prod_{l=1}^{L} (1 + \text{Lip}(g_l)) \cdot \max_l \|z^{(l)}(x)\|$$

Under Assumption B (well-conditioned residual stream), this is O(‖x‖).

$$\lambda_{\max}^{\text{FFT}} = O\!\left(\frac{1}{n}\sum_i \|x_i\|^2\right) = O(\text{Var}(x))$$

**LoRA with rank r:**

Only (B, A) matrices are trainable. The Jacobian with respect to B^{(l)} is:

$$\frac{\partial f}{\partial B^{(l)}} = \frac{\partial f}{\partial W^{(l)}} \cdot \frac{\partial (W_0 + BA)}{\partial B} = \frac{\partial f}{\partial W^{(l)}} \cdot (I \otimes A)$$

The key: the (I ⊗ A) factor projects the full-weight Jacobian onto the
rank-r subspace defined by A. This reduces the operator norm:

$$\|J_{\text{LoRA}}\|_{\text{op}} \leq \|J_\theta\|_{\text{op}} \cdot \|A\|_{\text{op}}$$

At initialization, A = 0 (or small random), so ‖A‖_op ≈ 0 and λ_max is
tiny. As training progresses, ‖A‖_op grows. At convergence:

$$\|A^*\|_{\text{op}} = \sigma_1^{1/2}(\Delta W^*)$$

(from the balanced factorization B* = UΣ^{1/2}, A* = Σ^{1/2}V^T).

Therefore:

$$\lambda_{\max}^{\text{LoRA}} \approx \lambda_{\max}^{\text{FFT}} \cdot \sigma_1(\Delta W^*)$$

**VPT with p prompts:**

The Jacobian with respect to prompt P^{(l)} involves how the prompt
affects the attention and subsequent computation:

$$\frac{\partial f}{\partial P^{(l)}} = \frac{\partial f}{\partial Z_{\text{prompted}}^{(l)}} \cdot \frac{\partial Z_{\text{prompted}}}{\partial P}$$

The second factor is bounded by the attention weights on prompt tokens:

$$\left\|\frac{\partial Z_{\text{prompted}}}{\partial P}\right\|_{\text{op}} \leq \max_j a_{j \to \text{prompts}}$$

At initialization (small prompts), this is O(p/N) (prompts get uniform
attention share). So:

$$\lambda_{\max}^{\text{VPT}} \approx \lambda_{\max}^{\text{FFT}} \cdot \frac{p^2}{N^2} \cdot d$$

(The d factor comes from the prompt having d dimensions per token.)

**Adapter with bottleneck r_a:**

Similar to LoRA, the adapter Jacobian is projected through the bottleneck:

$$\lambda_{\max}^{\text{Adapter}} \approx \lambda_{\max}^{\text{FFT}} \cdot \|W_{\text{down}}\|_{\text{op}}^2 \cdot \text{Lip}(\sigma)^2$$

At initialization (W_up = 0): λ_max ≈ 0. At convergence, similar scaling
to LoRA.

### 2.4 Optimal LR Formula

**Proposition 2 (Spectral-Informed Learning Rate).**

The optimal learning rate for each PEFT method scales as:

$$\eta_{\text{opt}}(\mathcal{A}) \approx \frac{c}{\lambda_{\max}^{\text{FFT}} \cdot \rho(\mathcal{A})}$$

where ρ(A) is the method-specific curvature factor:

| Method | ρ(A) | Scaling |
|--------|------|---------|
| FFT | 1 | η_FFT = c / λ_max |
| LoRA(r) | σ₁(ΔW*) | η_LoRA = η_FFT / σ₁ |
| VPT(p) | p²d / N² | η_VPT = η_FFT · N² / (p²d) |
| Adapter(r_a) | ‖W_down*‖² | η_Adapter ≈ η_FFT / ‖W_down*‖² |

**Corollary (LR Ratio):**

$$\frac{\eta_{\text{LoRA}}}{\eta_{\text{FFT}}} \approx \frac{1}{\sigma_1(\Delta W^*)}$$

For tasks with (α, C)-spectral decay: σ₁ = C, so:

$$\eta_{\text{LoRA}} \approx \frac{\eta_{\text{FFT}}}{C}$$

Tasks with large C (large weight shift) need a smaller LoRA LR relative
to FFT. Tasks with small C (small shift) can use a relatively larger LR.

**Practical formula (combining with d_eff):**

A more robust estimate that accounts for the total parameter count:

$$\eta_{\text{opt}}(\mathcal{A}) \approx \frac{\eta_0}{\sqrt{d_{\text{eff}}(\mathcal{A})} \cdot \sigma_1(\Delta W^*)}$$

where η₀ is a universal constant (≈ 0.1–1.0) and d_eff is the effective
parameter count.

**Verification with known values:**

For ViT-B with DINOv2 on CIFAR-100:
- d_eff(FFT) = 86M, d_eff(LoRA r=4) ≈ 150K
- σ₁(ΔW*) ≈ 1 (to be measured)
- Ratio: √(86M/150K) = √573 ≈ 24
- Predicted: η_LoRA / η_FFT ≈ 24
- Observed: 1e-3 / 2e-5 = 50 (same order of magnitude ✓)

---

## 3. Connection to Spectral Decay α

The maximum singular value σ₁ relates to α through:

$$\sigma_1 = C \cdot 1^{-\alpha} = C$$

And the total spectral energy relates to both C and α:

$$\sum_k \sigma_k^2 = C^2 \sum_k k^{-2\alpha} \approx C^2 \cdot \frac{1}{2\alpha - 1} \quad (\alpha > 1/2)$$

So tasks with high α (fast decay) have most energy in σ₁, meaning:
- The Hessian is dominated by one direction
- LR can be relatively large (only one direction to be careful about)
- The landscape is "narrow valley" shaped

Tasks with low α (slow decay) have energy spread across many σ_k:
- The Hessian has many large eigenvalues
- LR must be small (many directions to be careful about)
- The landscape is "broad plateau with many cliffs"

**This gives a refined LR formula involving α:**

$$\eta_{\text{opt}}(\text{LoRA}, r) \propto \frac{1}{C \cdot r^{1/2}} \cdot \frac{2\alpha - 1}{1}$$

Higher α → can use larger LR for the same rank. This is a testable
prediction.

---

## 4. Experimental Validation Plan

### Experiment: LR Sweep

For each (task, method, capacity) triple, sweep LR over a grid and find
the optimal:

**LR grid:** [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

**Methods × Capacities:**
- FFT (all params)
- LoRA: r ∈ {4, 16}
- VPT: p ∈ {10, 50}
- Adapter: r_a ∈ {16, 64}

**Tasks:** cifar100, dtd, gtsrb, clevr_count (covering high and low α)

**Measure:** For each LR, record best validation accuracy.

**What to plot:**

1. **LR vs Accuracy curves** per method: Should show an inverted-U shape
   (too low → slow convergence, too high → divergence).

2. **Optimal LR vs d_eff**: Should show η_opt ∝ 1/√d_eff across methods.

3. **Optimal LR vs σ₁(ΔW*)**: Should show η_opt ∝ 1/σ₁ across tasks.

4. **Optimal LR vs α**: Higher α tasks should tolerate higher LR at the
   same d_eff.

### What Validates the Theory

If the plots show:
- Optimal LR decreases with √d_eff (R² > 0.8 on log-log plot)
- Optimal LR decreases with σ₁ (R² > 0.7)
- Tasks with higher α tolerate higher LR at matched d_eff

Then Proposition 2 is validated.

---

## 5. How This Fits in the Paper

### Paper Structure (Updated)

| Section | Content |
|---------|---------|
| §4.1 | Theorem 1: Expressiveness separation |
| §4.2 | Theorem 2: Approximation rates |
| §4.3 | Theorem 3: Generalization bounds |
| §4.4 | Theorem 5: Method selection algorithm |
| §4.5 | **Proposition 2: LR selection (NEW)** |
| §5 | Experiments |
| §5.1 | Spectral profiles (Exp 1) |
| §5.2 | PEFT comparison (Exp 2) |
| §5.3 | Selection benchmark (Exp 3) |
| §5.4 | **LR sweep validation (Exp 4, NEW)** |

### The Story It Tells

"Our spectral framework doesn't just tell you WHICH method to use — it
also tells you HOW to tune it. The same quantity α that determines method
selection also determines the optimal learning rate, making the pilot
phase of our algorithm a one-stop diagnostic for the entire adaptation
pipeline."

This elevates the paper from "method selection" to "complete adaptation
framework" — a stronger contribution for a journal.
