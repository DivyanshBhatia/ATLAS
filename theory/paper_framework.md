# A Unified Theory of Adaptation Efficiency in Vision Foundation Models

## Paper Foundation Document — Problem Formulation & Mathematical Framework

---

## 1. Literature Gap Analysis

### 1.1 What Exists

Based on a thorough survey, the following theoretical results exist for individual PEFT methods:

**LoRA Expressiveness:**
- **Zeng & Lee (2024)** — "The Expressive Power of Low-Rank Adaptation" (ICLR 2024). Proved that for fully-connected networks, rank ≥ (width × depth_target / depth_source) suffices for LoRA to express any smaller target model. For Transformers, rank-(d/2) adapters suffice, where d = embedding dimension. **Limitation:** This is a worst-case existence result. It does not characterize *which tasks* need *which rank*, nor does it compare LoRA to other methods.
- **Jang, Lee & Ryu (2024)** — "LoRA Training in the NTK Regime has No Spurious Local Minima." Proved that LoRA with rank Ω(√N) has no spurious local minima in the NTK regime. **Limitation:** NTK linearization may not reflect practical LoRA training at low ranks.
- **Rank-Accuracy Trade-off (2025)** — Recent gradient-flow analysis deriving loss as a closed-form function of rank r, using spectral properties. **Limitation:** Single-method analysis, no cross-method comparison.

**Prompt Tuning Expressiveness:**
- **Wang, Chauhan, Wang & Hsieh (2023)** — "Universality and Limitations of Prompt Tuning" (NeurIPS 2023). Proved universality for "strong" transformers (sufficient depth), and showed that single-layer transformers with prompts of any length cannot memorize certain datasets. Also provided a lower bound on prompt parameters vs. LoRA parameters for single-layer setting. **Limitation:** The comparison to LoRA is only for a single-layer, and concerns parameter counts, not task-dependent expressiveness.
- **Le et al. (2025)** — "Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts." Reinterpreted VPT through MoE lens, showed prompt experts have restricted expressiveness (static), and proposed VAPT. **Limitation:** Primarily empirical; the MoE framing is an interpretation, not a formal function-class characterization.
- **"Fundamental Limits of Prompt Tuning" (2024)** — Established universality for single-layer, single-head attention. **Limitation:** Does not address multi-method comparison.

**Generalization Bounds for PEFT:**
- **Song et al. (2024)** — "Sparse is Enough in Fine-tuning Pre-trained LLMs." Used PAC-Bayes bounds to view pre-training as a prior shift. Derived bounds for sparse fine-tuning. **Limitation:** Only analyzes sparsity-based PEFT, not LoRA/VPT/Adapters.
- **PACE (NeurIPS 2024 Spotlight)** — Connected gradient norms and dataset size to generalization in PEFT. **Limitation:** Provides a regularization technique, not comparative generalization bounds across methods.
- **PAC-tuning (2023)** — Directly minimized PAC-Bayes bounds during fine-tuning. **Limitation:** NLP-focused, not vision; does not compare across PEFT methods.

**Empirical Comparisons:**
- **"Lessons and Insights from a Unifying Study of PEFT in Visual Recognition" (2024)** — Systematic empirical comparison of 13+ PEFT methods on VTAB-1K. **Limitation:** Purely empirical, no theory.
- **He et al. (2022)** — "Towards a Unified View of PETL." Showed mathematical connections (prefix-tuning ≈ adapter in attention computation). **Limitation:** Structural equivalence analysis, not expressiveness/generalization theory.
- **GLoRA (2023)** — Proposed a unified adapter formulation. **Limitation:** Empirical unification, not theoretical.

### 1.2 The Gap (Our Contribution)

**Nobody has done ALL of the following in one paper:**

1. **Cross-method expressiveness theory for ViTs:** Existing work analyzes LoRA OR prompt tuning individually. No paper proves a formal separation theorem characterizing *which task structures* favor which method.

2. **Unified generalization bounds:** PAC-Bayes bounds exist for generic neural networks and for sparse fine-tuning, but no paper derives *method-specific* bounds for LoRA vs. VPT vs. Adapters that account for each method's parameterization geometry within a ViT.

3. **Principled method selection:** There is no existing theory that takes measurable task properties + data budget → recommends the optimal PEFT method with provable guarantees.

---

## 2. Formal Setup & Notation

### 2.1 Vision Transformer Architecture

We consider a standard ViT (Dosovitskiy et al., 2020) with the following components:

**Input:** An image x ∈ ℝ^(H×W×C) is divided into N patches, embedded as a token sequence:

$$\mathbf{Z}^{(0)} = [\mathbf{z}_{\text{cls}}; \, \mathbf{E} \mathbf{x}_1 + \mathbf{e}_1; \, \ldots; \, \mathbf{E} \mathbf{x}_N + \mathbf{e}_N] \in \mathbb{R}^{(N+1) \times d}$$

where **E** ∈ ℝ^(d×P²C) is the patch embedding, **e**_i are positional embeddings, and d is the token dimension.

**Transformer block l ∈ {1, ..., L}:** Each block applies multi-head self-attention (MHSA) followed by a feed-forward network (FFN):

$$\mathbf{Z}^{(l)} = \text{FFN}^{(l)}\!\left(\text{MHSA}^{(l)}(\mathbf{Z}^{(l-1)})\right)$$

**Multi-Head Self-Attention:** For head h ∈ {1, ..., H}:

$$\text{head}_h = \text{Softmax}\!\left(\frac{\mathbf{Z} \mathbf{W}_Q^{(l,h)} (\mathbf{Z} \mathbf{W}_K^{(l,h)})^\top}{\sqrt{d_h}}\right) \mathbf{Z} \mathbf{W}_V^{(l,h)}$$

where W_Q^{(l,h)}, W_K^{(l,h)}, W_V^{(l,h)} ∈ ℝ^(d × d_h), with d_h = d/H.

**Output projection:**
$$\text{MHSA}^{(l)}(\mathbf{Z}) = \mathbf{Z} + \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O^{(l)}$$

**FFN:**
$$\text{FFN}^{(l)}(\mathbf{Z}) = \mathbf{Z} + \sigma(\mathbf{Z} \mathbf{W}_1^{(l)} + \mathbf{b}_1^{(l)}) \mathbf{W}_2^{(l)} + \mathbf{b}_2^{(l)}$$

where σ is GELU activation, W_1^{(l)} ∈ ℝ^(d × d_ff), W_2^{(l)} ∈ ℝ^(d_ff × d).

**Notation summary:**

| Symbol | Meaning |
|--------|---------|
| d | Token/embedding dimension |
| L | Number of transformer blocks |
| H | Number of attention heads |
| d_h = d/H | Per-head dimension |
| d_ff | FFN hidden dimension (typically 4d) |
| N | Number of image patches |
| θ₀ | All frozen pretrained parameters |

**The full pretrained model** is denoted f_{θ₀} : ℝ^(H×W×C) → ℝ^K (classification head on CLS token), mapping images to K classes.

### 2.2 The Adaptation Problem

**Definition 1 (Downstream Task).** A downstream task is a distribution 𝒟 over (x, y) ∈ ℝ^(H×W×C) × {1, ..., K'} with K' classes. We are given n i.i.d. samples S = {(x_i, y_i)}_{i=1}^n.

**Definition 2 (Adaptation Method).** An adaptation method 𝒜 defines:
1. A *parameter space* Φ_𝒜 ⊆ ℝ^(d_𝒜) of dimension d_𝒜 (the trainable parameters).
2. A *function class* ℱ_𝒜(θ₀) = { f_{θ₀, φ} : ℝ^(H×W×C) → ℝ^(K') | φ ∈ Φ_𝒜 } of all models reachable from θ₀ by tuning φ.

**Goal:** Given a target function f* : ℝ^(H×W×C) → ℝ^(K') (the Bayes-optimal classifier for 𝒟), find φ* ∈ Φ_𝒜 minimizing:

$$\mathcal{R}(f_{θ₀, φ^*}) = \mathbb{E}_{(x,y) \sim \mathcal{D}} [\ell(f_{θ₀, φ^*}(x), y)]$$

The excess risk decomposes as:

$$\underbrace{\mathcal{R}(f_{θ₀, \hat{φ}}) - \mathcal{R}(f^*)}_{\text{excess risk}} = \underbrace{\inf_{φ \in Φ_𝒜} \mathcal{R}(f_{θ₀, φ}) - \mathcal{R}(f^*)}_{\text{approximation error } ε_{\text{approx}}(𝒜)} + \underbrace{\mathcal{R}(f_{θ₀, \hat{φ}}) - \inf_{φ \in Φ_𝒜} \mathcal{R}(f_{θ₀, φ})}_{\text{estimation error } ε_{\text{est}}(𝒜, n)}$$

**Key Insight:** Different methods 𝒜 trade off ε_approx and ε_est differently. Methods with larger function classes (lower ε_approx) tend to have higher ε_est due to more parameters. Our theory makes this precise.

---

## 3. Formal Definition of Each PEFT Method's Function Class

### 3.1 Linear Probing (LP)

**Parameters:** φ_LP = (W_head, b_head) where W_head ∈ ℝ^(d × K'), b_head ∈ ℝ^(K').

**Function class:**
$$\mathcal{F}_{\text{LP}}(\theta_0) = \{ x \mapsto \mathbf{W}_{\text{head}} \cdot g_{\theta_0}(x) + \mathbf{b}_{\text{head}} \}$$

where g_{θ₀}(x) ∈ ℝ^d is the frozen CLS token representation.

**Dimensionality:** d_LP = dK' + K'.

**Interpretation:** LP can only learn linear decision boundaries in the pretrained representation space.

### 3.2 Low-Rank Adaptation (LoRA)

**Parameters:** For each adapted weight matrix W^{(l)} ∈ ℝ^(d_in × d_out) in layer l, LoRA introduces:
$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)}_0 + \mathbf{B}^{(l)} \mathbf{A}^{(l)}$$

where A^{(l)} ∈ ℝ^(r × d_out), B^{(l)} ∈ ℝ^(d_in × r), and r is the rank.

**Standard application:** LoRA is applied to {W_Q, W_K, W_V, W_O} in each layer. Total adapted matrices: 4L.

**Parameter space:**
$$Φ_{\text{LoRA}}(r) = \{ (B^{(l,m)}, A^{(l,m)}) \mid l \in [L], m \in \{Q,K,V,O\} \}$$

**Function class:** ℱ_LoRA(θ₀, r) = all models reachable by rank-r perturbations to attention projections.

**Dimensionality:** d_LoRA = 4L × (d_in + d_out) × r = 4L × 2d × r (when d_in = d_out = d).

**Key property (Weight-Space Perturbation):** LoRA directly modifies the projection matrices. The perturbation ΔW = BA has rank at most r, so LoRA explores a *rank-constrained neighborhood* in weight space around θ₀.

### 3.3 Visual Prompt Tuning (VPT)

**VPT-Shallow:** Learnable prompt tokens P ∈ ℝ^(p × d) are prepended to the input sequence at layer 1 only:
$$\mathbf{Z}_{\text{prompted}}^{(0)} = [\mathbf{P}; \, \mathbf{Z}^{(0)}] \in \mathbb{R}^{(N+1+p) \times d}$$

**VPT-Deep:** Independent learnable prompts P^{(l)} ∈ ℝ^(p × d) are inserted at every layer:
$$\mathbf{Z}_{\text{prompted}}^{(l)} = [\mathbf{P}^{(l)}; \, \tilde{\mathbf{Z}}^{(l)}]$$

where Z̃^{(l)} denotes the non-prompt tokens output from the previous layer.

**Parameter space:**
$$Φ_{\text{VPT-Deep}}(p) = \{ \mathbf{P}^{(l)} \in \mathbb{R}^{p \times d} \mid l \in [L] \}$$

**Dimensionality:** d_VPT = L × p × d.

**Key property (Input-Space Steering):** VPT does *not* modify the weight matrices. Instead, prompt tokens participate in self-attention, effectively *steering attention patterns* and *injecting information via cross-attention* between prompt tokens and image tokens. The attention matrix changes because:
$$\text{Attn}(\mathbf{Z}_{\text{prompted}}) = \text{Softmax}\left(\frac{[\mathbf{P}; \mathbf{Z}] \mathbf{W}_Q (\, [\mathbf{P}; \mathbf{Z}] \mathbf{W}_K)^\top}{\sqrt{d_h}}\right)$$

The prompt tokens create new key-value pairs that image tokens can attend to, but the *projection geometry* (W_Q, W_K, W_V) remains frozen.

### 3.4 Adapter Modules

**Structure:** A bottleneck module inserted after the FFN (or MHSA) in each layer:

$$\text{Adapter}^{(l)}(\mathbf{Z}) = \mathbf{Z} + \sigma(\mathbf{Z} \mathbf{W}_{\text{down}}^{(l)}) \mathbf{W}_{\text{up}}^{(l)}$$

where W_down^{(l)} ∈ ℝ^(d × r_a), W_up^{(l)} ∈ ℝ^(r_a × d), and r_a is the bottleneck dimension.

**Parameter space:**
$$Φ_{\text{Adapter}}(r_a) = \{ (\mathbf{W}_{\text{down}}^{(l)}, \mathbf{W}_{\text{up}}^{(l)}) \mid l \in [L] \}$$

**Dimensionality:** d_Adapter = 2L × d × r_a (if inserted after both MHSA and FFN: 4L × d × r_a).

**Key property (Representation-Space Nonlinear Perturbation):** Adapters add a *nonlinear, low-rank* residual to the representation at each layer. Unlike LoRA (which is linear in the input), the adapter applies a nonlinearity σ, allowing it to make input-dependent perturbations.

### 3.5 Full Fine-Tuning (FFT)

**Parameters:** All parameters θ are trainable.

**Dimensionality:** d_FFT = |θ| (typically tens to hundreds of millions).

**Function class:** ℱ_FFT = all representable functions of the ViT architecture.

---

## 4. Key Theoretical Quantities to Define

### 4.1 Task-Weight Shift Matrix

**Definition 3 (Optimal Weight Shift).** For a target task 𝒟, define the *optimal weight shift* for layer l, weight matrix m as:
$$\Delta \mathbf{W}^{*(l,m)} = \mathbf{W}^{*(l,m)} - \mathbf{W}_0^{(l,m)}$$

where W*^{(l,m)} are the weights obtained by full fine-tuning to convergence.

**Definition 4 (Spectral Profile).** The spectral profile of the task at layer l is the ordered singular values:
$$\sigma_1^{(l,m)} \geq \sigma_2^{(l,m)} \geq \ldots \geq \sigma_{\min(d_{\text{in}}, d_{\text{out}})}^{(l,m)}$$

of ΔW*^{(l,m)}.

**Definition 5 (Spectral Decay Rate).** We say the task has (α, C)-spectral decay at layer (l, m) if:
$$\sigma_k^{(l,m)} \leq C \cdot k^{-\alpha} \quad \forall k \geq 1$$

**Interpretation:** Tasks with fast spectral decay (large α) are "intrinsically low-rank" — LoRA with small r should suffice. Tasks with slow decay require higher rank.

### 4.2 Attention Shift Complexity

**Definition 6 (Attention Shift).** For layer l, head h, let:
- A₀^{(l,h)}(x) = attention matrix under pretrained model for input x.
- A*^{(l,h)}(x) = attention matrix under fully fine-tuned model for input x.

The *attention shift complexity* for the task is:
$$\mathcal{S}_{\text{attn}} = \frac{1}{LH} \sum_{l,h} \mathbb{E}_{x \sim \mathcal{D}_X}\!\left[\| \mathbf{A}_*^{(l,h)}(x) - \mathbf{A}_0^{(l,h)}(x) \|_F^2 \right]$$

**Interpretation:** High S_attn means the task requires substantially different attention patterns from the pretrained model. We hypothesize VPT is better suited for high-S_attn tasks.

### 4.3 Feature Discrimination Complexity

**Definition 7 (Feature Shift).** Let g_{θ₀}(x) be the CLS representation under the pretrained model, and g_{θ*}(x) under the fully fine-tuned model. The *feature discrimination complexity* is:
$$\mathcal{S}_{\text{feat}} = \mathbb{E}_{x \sim \mathcal{D}_X}\!\left[\| g_{\theta^*}(x) - g_{\theta_0}(x) \|^2 \right]$$

**Interpretation:** High S_feat means the task requires the model to produce fundamentally different representations, not just route attention differently. LoRA's direct weight modification should be better suited here.

---

## 5. Theorem Sketches

### Theorem 1 (Expressiveness Separation — Informal)

*There exist task distributions 𝒟_attn and 𝒟_feat such that:*

*(a) For 𝒟_attn (attention-shift-dominant task): VPT-Deep with p prompt tokens achieves ε_approx = O(1/p), while LoRA of rank r achieves ε_approx = Ω(1) for r < r₀ (a threshold depending on how much the attention pattern change requires weight-space modification).*

*(b) For 𝒟_feat (feature-discrimination-dominant task): LoRA of rank r achieves ε_approx = O(σ_{r+1}(ΔW*)), while VPT with p tokens achieves ε_approx = Ω(σ_{r₀+1}(ΔW*)) regardless of p (when the attention patterns are already adequate but the projection geometry is wrong).*

**Proof Strategy:**
- Part (a): Construct a task where the pretrained attention is uniform, but the target requires attending to a specific subset of patches (e.g., counting objects in a specific region). Prompt tokens can directly create key-value pairs that steer attention. LoRA modifies W_Q, W_K globally but a low-rank perturbation cannot create the specific sparse attention pattern without high rank.
- Part (b): Construct a task where the pretrained attention is already correct (the model looks at the right patches) but the features extracted by W_V need to change (e.g., fine-grained texture discrimination). Prompt tokens can inject values but cannot change how existing token features are projected. LoRA directly modifies W_V.

### Theorem 2 (Approximation Rates — Informal)

*For a task with (α, C)-spectral decay:*

*LoRA with rank r achieves:*
$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r) = O\!\left(C^2 \sum_{l,m} \sum_{k>r} k^{-2\alpha}\right) = O(C^2 \cdot L \cdot r^{1-2\alpha})$$

*VPT-Deep with p prompts per layer, for a task with attention shift complexity S_attn, achieves:*
$$\varepsilon_{\text{approx}}^{\text{VPT}}(p) = O\!\left(\frac{S_{\text{attn}}}{p}\right) + \varepsilon_{\text{residual}}$$

*where ε_residual captures the error from not being able to modify weight matrices (irreducible for VPT).*

### Theorem 3 (PAC-Bayes Generalization Bound — Informal)

*For adaptation method 𝒜 ∈ {LoRA(r), VPT(p), Adapter(r_a), LP}, with probability ≥ 1 − δ over the draw of n training samples:*

$$\mathcal{R}(f_{θ_0, \hat{φ}}) \leq \hat{\mathcal{R}}_S(f_{θ_0, \hat{φ}}) + \sqrt{\frac{\text{KL}(Q_{\hat{φ}} \| P_{θ_0, 𝒜}) + \ln(2n/\delta)}{2(n-1)}}$$

*where:*

- *Q_φ̂ is the posterior (Gaussian centered at learned parameters)*
- *P_{θ₀, 𝒜} is the **method-specific prior** defined by the frozen model:*
  - *For LoRA: P is Gaussian on (B, A) matrices with variance calibrated by the pretrained weight spectrum*
  - *For VPT: P is Gaussian on prompt tokens with variance calibrated by the pretrained token distribution*
  - *For Adapters: P is centered at zero (identity residual)*

*The KL terms differ structurally:*
- *KL_LoRA = O(r × Σ_l ‖ΔW^{(l)}‖²_F / σ²_prior)*
- *KL_VPT = O(p × d × L × ‖P̂‖² / σ²_prior)*
- *KL_Adapter = O(r_a × d × L × ‖Ŵ_down, Ŵ_up‖² / σ²_prior)*

### Theorem 4 (Non-Vacuous Bounds — Informal)

*For practical configurations (LoRA rank 4–16, VPT 10–50 prompts), the PAC-Bayes bounds from Theorem 3 are non-vacuous (bound < 1) on VTAB-1K tasks with n = 1000 samples, and the predicted ranking of methods by bound value has Kendall-τ > 0.7 with the true ranking by test accuracy.*

**Proof Strategy:** Show that the effective dimensionality d_eff(𝒜) for each method is small relative to n, and that the pretrained model provides a strong prior (small KL) because adaptation only moves parameters slightly.

### Theorem 5 (Optimal Selection Criterion — Informal)

*Given:*
- *Estimated spectral decay rate α̂ (from a small unlabeled probe set)*
- *Estimated attention shift complexity Ŝ_attn (from comparing pretrained attention on source vs. target domain)*
- *Data budget n*

*The selection rule:*
- *If α̂ > α_threshold(n) and Ŝ_attn < S_threshold: use LoRA with r* = ⌈C^{1/α} · (n/log n)^{1/(2α)}⌉*
- *If Ŝ_attn > S_threshold and α̂ < α_threshold(n): use VPT with p* = ⌈Ŝ_attn · √n⌉*
- *If both are moderate: use Adapter*
- *If n is very small (< n_min): use Linear Probing*

*achieves excess risk within O(log n / √n) of the oracle method that knows the task perfectly.*

---

## 6. Proof Roadmap

### Phase 1: Expressiveness (Theorems 1–2) — Weeks 1–4

**Week 1–2: Construct separation examples.**
- For Theorem 1(a): Design a "spatial counting" task on synthetic images where the target attention pattern is sparse and patch-specific. Formalize as a specific distribution over token sequences. Show that VPT achieves the correct attention with p = O(1) prompts by explicit construction.
- For Theorem 1(b): Design a "texture discrimination" task where all patches need attention but the V-projection must change. Prove that VPT's approximation error is lower-bounded regardless of p.

**Week 3–4: Derive approximation rates.**
- Use Eckart-Young-Mirsky theorem for LoRA rates (rank-r best approximation in Frobenius norm).
- For VPT rates, analyze the effect of prompt tokens on the attention output as a function of p, using the softmax attention structure.

### Phase 2: Generalization (Theorems 3–4) — Weeks 5–8

**Week 5–6: Derive PAC-Bayes bounds.**
- Define method-specific priors. Key novelty: the prior for each method is *induced by the pretrained model* in different ways.
- Apply McAllester's PAC-Bayes theorem with the KL computed for each method's parameterization.

**Week 7–8: Compute bounds numerically.**
- Implement bound computation for real models (DINOv2 ViT-B, CLIP ViT-L) on VTAB-1K.
- Verify non-vacuousness and rank correlation.

### Phase 3: Selection Algorithm (Theorem 5) — Weeks 9–10

**Week 9:** Design and implement the spectral probe estimator and attention shift estimator.

**Week 10:** Prove the selection theorem by combining Theorems 2 and 3 — the selection rule chooses the method minimizing the sum of approximation error and generalization bound.

### Phase 4: Experiments — Weeks 11–16

Detailed experimental plan as outlined in the original proposal.

---

## 7. Key Mathematical Tools Required

1. **Matrix Perturbation Theory:** Weyl's inequality, Davis-Kahan sin(θ) theorem — for analyzing how low-rank weight perturbations affect the output.

2. **Softmax Attention Analysis:** Properties of softmax as a function of prompt tokens. The key identity:
   $$\frac{\partial \text{Softmax}(\mathbf{q}^\top \mathbf{K}^\top / \sqrt{d})}{\partial \mathbf{K}} = \ldots$$
   characterizes how prompt key-values shift attention weights.

3. **PAC-Bayes Theory:** McAllester's bound, Catoni's bound, data-dependent priors (Dziugaite & Roy, 2017).

4. **Spectral Analysis of Attention:** The connection between the spectrum of W_Q W_K^T and the rank of attention patterns.

5. **Rademacher Complexity / Covering Numbers:** For converting function-class size to generalization guarantees as an alternative/complement to PAC-Bayes.

---

## 8. Notation Reference Table

| Symbol | Definition |
|--------|-----------|
| f_{θ₀} | Pretrained ViT model |
| f_{θ₀, φ} | Adapted model with PEFT parameters φ |
| 𝒟 | Downstream task distribution |
| S | Training set of n samples |
| ℱ_𝒜(θ₀) | Function class of adaptation method 𝒜 |
| d_𝒜 | Number of trainable parameters for method 𝒜 |
| r | LoRA rank |
| p | Number of VPT prompt tokens per layer |
| r_a | Adapter bottleneck dimension |
| ΔW* | Optimal weight shift (FFT - pretrained) |
| σ_k(ΔW*) | k-th singular value of optimal weight shift |
| α | Spectral decay rate |
| S_attn | Attention shift complexity |
| S_feat | Feature discrimination complexity |
| ε_approx | Approximation error |
| ε_est | Estimation (generalization) error |
| KL(Q‖P) | KL divergence (posterior from prior) |
| 𝒜 | Adaptation method ∈ {LP, LoRA, VPT, Adapter, FFT} |

---

## 9. Immediate Next Steps

1. **Formalize Theorem 1 fully** — Write the exact construction of 𝒟_attn and 𝒟_feat, including the specific ViT configuration (number of heads, dimension, etc.). Derive the separation results.

2. **Implement spectral analysis** — On a real pretrained ViT (DINOv2-ViT-B), fine-tune on several VTAB-1K tasks and compute the spectral profiles of ΔW*. This validates our assumption that tasks have varying spectral decay rates.

3. **Implement attention shift measurement** — For the same tasks, compute S_attn and S_feat. Verify that there is meaningful variation across tasks (otherwise the separation is moot).
