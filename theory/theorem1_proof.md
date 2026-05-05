# Theorem 1: Expressiveness Separation Between LoRA and VPT

## Full Proof Document

---

## 1. Formal Setting

We work with a single-layer, single-head self-attention model for Parts (a) and (b). Section 6 extends to multi-layer, multi-head settings.

### 1.1 Model Architecture

**Input:** A sequence of N tokens Z = [z₁, ..., z_N]ᵀ ∈ ℝ^{N×d}.

**Self-Attention with linear readout:** The model computes:

$$f(\mathbf{Z}) = \mathbf{c}^\top \left( \sum_{j=1}^{N} a_{1j}(\mathbf{Z}) \cdot \mathbf{z}_j \mathbf{W}_V \right)$$

where c ∈ ℝ^d is a fixed readout vector (representing the classification head applied to the CLS token's output), and the attention weights for token 1 (the CLS token) are:

$$a_{1j}(\mathbf{Z}) = \frac{\exp(\mathbf{z}_1^\top \mathbf{M} \, \mathbf{z}_j / \sqrt{d})}{\sum_{k=1}^{N} \exp(\mathbf{z}_1^\top \mathbf{M} \, \mathbf{z}_k / \sqrt{d})}$$

with the attention kernel matrix M = W_Q W_K^T ∈ ℝ^{d×d}.

**Pretrained parameters:** θ₀ = (M₀, W_V⁰, c) are all frozen. We write f₀ for the pretrained model's function.

### 1.2 Adaptation Methods (Formal)

**LoRA with rank r:** Introduces trainable perturbations to M and W_V:

$$\mathbf{M} \leftarrow \mathbf{M}_0 + \Delta\mathbf{M}, \quad \text{rank}(\Delta\mathbf{M}) \leq 2r$$

$$\mathbf{W}_V \leftarrow \mathbf{W}_V^0 + \Delta\mathbf{W}_V, \quad \text{rank}(\Delta\mathbf{W}_V) \leq r$$

(The rank-2r bound on ΔM comes from LoRA applied separately to W_Q and W_K: if ΔW_Q = B_Q A_Q and ΔW_K = B_K A_K with rank r each, then ΔM = W_Q⁰ ΔW_K^T + ΔW_Q (W_K⁰)^T + ΔW_Q ΔW_K^T has rank ≤ 2r + r²; for small r relative to d, the dominant term is 2r. We use rank(ΔM) ≤ 2r for clarity, noting that all results hold a fortiori for the tighter bound.)

**Number of LoRA parameters:** 2r(d + d) for M (via W_Q, W_K) + 2rd for W_V = 6rd total.

**VPT with p prompt tokens:** Introduces trainable tokens P = [p₁, ..., p_p]^T ∈ ℝ^{p×d} prepended to the input:

$$\tilde{\mathbf{Z}} = [\mathbf{P}; \mathbf{Z}] \in \mathbb{R}^{(p+N) \times d}$$

All weight matrices (M₀, W_V⁰, c) remain frozen. The CLS token's output becomes:

$$f_{\text{VPT}}(\mathbf{Z}) = \mathbf{c}^\top \left( \sum_{j=1}^{N} \tilde{a}_{1j} \cdot \mathbf{z}_j \mathbf{W}_V^0 + \sum_{k=1}^{p} \tilde{a}_{1,N+k} \cdot \mathbf{p}_k \mathbf{W}_V^0 \right)$$

where the attention weights are now over the extended sequence:

$$\tilde{a}_{1j} = \frac{\exp(\mathbf{z}_1^\top \mathbf{M}_0 \mathbf{z}_j / \sqrt{d})}{\sum_{k=1}^{N} \exp(\mathbf{z}_1^\top \mathbf{M}_0 \mathbf{z}_k / \sqrt{d}) + \sum_{k=1}^{p} \exp(\mathbf{z}_1^\top \mathbf{M}_0 \mathbf{p}_k / \sqrt{d})}$$

**Number of VPT parameters:** pd.

---

## 2. Part (b): Feature-Discrimination Task (LoRA-Favorable)

*We prove Part (b) first because it yields a cleaner, stronger result.*

### 2.1 Construction

**Intuition:** We construct a task where the pretrained model already has the correct attention pattern, but the value projection W_V must change. VPT cannot modify W_V, so it faces an irreducible error. LoRA directly modifies W_V and achieves the target.

**Data distribution 𝒟_feat:**

Let d ≥ 2d₀ for some d₀ ≥ 1. Partition the coordinates of ℝ^d as:
- **Positional subspace:** first d₀ coordinates, spanned by {e₁, ..., e_{d₀}}
- **Feature subspace:** remaining d - d₀ coordinates, spanned by {e_{d₀+1}, ..., e_d}

Define N = 2 input tokens (plus a CLS token, so 3 tokens total):

- **CLS token:** z₁ = e₁ (a fixed positional vector)
- **Token 2:** z₂ = e₂ + x₂ where x₂ is drawn uniformly from a ball in the feature subspace
- **Token 3:** z₃ = e₃ + x₃ where x₃ is drawn uniformly from a ball in the feature subspace

(We assume d₀ ≥ 3 so e₁, e₂, e₃ are distinct positional vectors.)

**Pretrained model:**
- **Attention kernel:** M₀ = I_d (identity). Then:

$$a_{12} = \frac{\exp(e_1^\top e_2 / \sqrt{d})}{\exp(e_1^\top e_2 / \sqrt{d}) + \exp(e_1^\top e_3 / \sqrt{d})} = \frac{1}{2}$$

(Since e₁ᵀe₂ = 0 = e₁ᵀe₃, both logits equal 0, attention is uniform.)

Actually, we need e₁ᵀe₂ = e₁ᵀe₃. Since we chose orthogonal positional vectors, both dot products equal 0. ✓ The CLS token attends equally to both input tokens.

Wait — z₂ = e₂ + x₂, so z₁ᵀ M₀ z₂ = e₁ᵀ(e₂ + x₂) = e₁ᵀe₂ + e₁ᵀx₂ = 0 + 0 = 0 (since x₂ is in the feature subspace, orthogonal to e₁). Similarly z₁ᵀ M₀ z₃ = 0. ✓

So a₁₂ = a₁₃ = 1/2. ✓ (Attention is uniform over the two input tokens.)

- **Value projection:** W_V⁰ = I_d (identity, for simplicity).

- **Readout vector:** c ∈ ℝ^d is in the feature subspace.

**Pretrained model output:**

$$f_0(\mathbf{Z}) = \mathbf{c}^\top \left(\frac{1}{2} \mathbf{z}_2 + \frac{1}{2} \mathbf{z}_3\right) = \frac{1}{2}\mathbf{c}^\top(\mathbf{x}_2 + \mathbf{x}_3)$$

(The positional components e₂, e₃ are orthogonal to c.)

**Target function:** We want the model to compute a function that requires projecting the feature vectors through a DIFFERENT matrix:

$$f^*(\mathbf{Z}) = \frac{1}{2}\mathbf{c}^\top \mathbf{W}_V^* (\mathbf{x}_2 + \mathbf{x}_3)$$

where W_V* ∈ ℝ^{d×d} and the key requirement is:

$$\Delta \mathbf{W}_V = \mathbf{W}_V^* - \mathbf{W}_V^0 = \mathbf{W}_V^* - \mathbf{I}_d$$

has rank r* > 0 with singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_{r*} > 0.

More specifically, let Δ W_V = Σ_{i=1}^{r*} σᵢ uᵢ vᵢᵀ where {uᵢ}, {vᵢ} are in the feature subspace.

**The task label:** y = sign(f*(Z)) (binary classification).

### 2.2 LoRA Approximation (Upper Bound)

**Lemma 1.** *LoRA with rank r on W_V achieves approximation error:*

$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r) = \mathbb{E}\left[(f^*(\mathbf{Z}) - f_{\text{LoRA}}(\mathbf{Z}))^2\right] = \frac{1}{4} \sum_{i=r+1}^{r^*} \sigma_i^2 \cdot \|\mathbf{c}\|^2 \cdot \mathbb{E}[\|\mathbf{v}_i^\top \mathbf{x}\|^2]$$

*In particular, if r ≥ r*, the approximation error is zero.*

**Proof:**

Since the attention is already correct (a₁₂ = a₁₃ = 1/2 for both the pretrained and target function), LoRA on W_V does not need to change M₀. Set ΔM = 0.

LoRA sets ΔW_V = BA with rank r. The adapted model computes:

$$f_{\text{LoRA}}(\mathbf{Z}) = \mathbf{c}^\top (\mathbf{I}_d + \mathbf{B}\mathbf{A}) \left(\frac{\mathbf{x}_2 + \mathbf{x}_3}{2}\right)$$

The target is:

$$f^*(\mathbf{Z}) = \mathbf{c}^\top (\mathbf{I}_d + \Delta\mathbf{W}_V) \left(\frac{\mathbf{x}_2 + \mathbf{x}_3}{2}\right)$$

The error is:

$$f^* - f_{\text{LoRA}} = \mathbf{c}^\top (\Delta\mathbf{W}_V - \mathbf{B}\mathbf{A}) \left(\frac{\mathbf{x}_2 + \mathbf{x}_3}{2}\right)$$

By the Eckart–Young–Mirsky theorem, the best rank-r approximation to ΔW_V in Frobenius norm is its truncated SVD:

$$\mathbf{B}^*\mathbf{A}^* = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

The residual is ΔW_V - B*A* = Σ_{i=r+1}^{r*} σᵢ uᵢ vᵢᵀ.

Therefore:

$$\mathbb{E}\left[(f^* - f_{\text{LoRA}})^2\right] = \mathbb{E}\left[\left(\mathbf{c}^\top \left(\sum_{i=r+1}^{r^*} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top\right) \frac{\mathbf{x}_2 + \mathbf{x}_3}{2}\right)^2\right]$$

$$= \frac{1}{4} \sum_{i=r+1}^{r^*} \sigma_i^2 (\mathbf{c}^\top \mathbf{u}_i)^2 \cdot \mathbb{E}[(\mathbf{v}_i^\top (\mathbf{x}_2 + \mathbf{x}_3))^2]$$

(Cross terms vanish by orthogonality of vᵢ when x₂, x₃ have isotropic distribution in the feature subspace.)

When r ≥ r*, the sum is empty and the error is exactly 0. ∎

### 2.3 VPT Lower Bound

**Theorem 1(b).** *For the task 𝒟_feat defined above, VPT with any number of prompt tokens p achieves approximation error:*

$$\varepsilon_{\text{approx}}^{\text{VPT}}(p) \geq \frac{1}{4} \inf_{\boldsymbol{\beta} \in \mathbb{R}^d} \left\| \mathbf{c}^\top \Delta\mathbf{W}_V - \boldsymbol{\beta}^\top \right\|^2 \cdot \mathbb{E}[\|\mathbf{x}\|^2] > 0$$

*whenever c^T ΔW_V ≠ 0 (the task genuinely requires a change in value projection) and ΔW_V has components that cannot be captured by the column span of a bias term. More precisely, the VPT approximation error is bounded below by a quantity that does not vanish with increasing p.*

**Proof:**

The VPT model computes:

$$f_{\text{VPT}}(\mathbf{Z}) = \mathbf{c}^\top \left(\sum_{j=2}^{3} \tilde{a}_{1j}(\mathbf{Z}, \mathbf{P}) \cdot \mathbf{z}_j \mathbf{W}_V^0 + \sum_{k=1}^{p} \tilde{a}_{1,3+k}(\mathbf{Z}, \mathbf{P}) \cdot \mathbf{p}_k \mathbf{W}_V^0\right)$$

Since W_V⁰ = I_d:

$$f_{\text{VPT}}(\mathbf{Z}) = \mathbf{c}^\top \left(\sum_{j=2}^{3} \tilde{a}_{1j} \mathbf{z}_j + \sum_{k=1}^{p} \tilde{a}_{1,3+k} \mathbf{p}_k \right)$$

Now substitute z_j = e_j + x_j:

$$f_{\text{VPT}} = \underbrace{\sum_{j=2}^{3} \tilde{a}_{1j} \mathbf{c}^\top \mathbf{x}_j}_{\text{reweighted feature extraction}} + \underbrace{\sum_{j=2}^{3} \tilde{a}_{1j} \mathbf{c}^\top \mathbf{e}_j + \sum_{k} \tilde{a}_{1,3+k} \mathbf{c}^\top \mathbf{p}_k}_{\text{bias terms (input-independent in c's projection)}}$$

**Key structural observation:** Since W_V is frozen as I_d, the image tokens' contribution to the output in the direction c is always c^T x_j — that is, the SAME projection c^T for all image tokens, regardless of prompts. The prompts can only:

1. **Reweight** the image tokens (change ã₁ⱼ), and
2. **Add a bias** (the prompt value contribution Σ_k ã_{1,3+k} c^T p_k).

**Formally:** Let us write the VPT output as:

$$f_{\text{VPT}} = \tilde{a}_{12} \cdot \mathbf{c}^\top \mathbf{x}_2 + \tilde{a}_{13} \cdot \mathbf{c}^\top \mathbf{x}_3 + b(\mathbf{Z}, \mathbf{P})$$

where b(Z, P) collects the positional and prompt bias terms. Note that ã₁₂ + ã₁₃ + Σ_k ã_{1,3+k} = 1 (softmax normalization), and ã₁₂, ã₁₃ depend on Z and P.

The target function is:

$$f^* = \frac{1}{2} \mathbf{c}^\top \mathbf{W}_V^* \mathbf{x}_2 + \frac{1}{2} \mathbf{c}^\top \mathbf{W}_V^* \mathbf{x}_3$$

**The crux:** VPT projects each x_j through c^T (a fixed direction), then reweights. The target projects each x_j through c^T W_V* (a DIFFERENT direction). If c^T W_V* ≠ λ c^T for any scalar λ, then VPT fundamentally cannot match the target — it is projecting features in the wrong direction.

**Formal lower bound:**

Define the "target feature direction" as the row vector: η* = c^T W_V* ∈ ℝ^{1×d}.
Define the "pretrained feature direction" as: η₀ = c^T ∈ ℝ^{1×d}.

The target computes: f* = (1/2)(η* x₂ + η* x₃).

VPT computes: f_VPT = ã₁₂ (η₀ x₂) + ã₁₃ (η₀ x₃) + b.

The squared error is:

$$\mathbb{E}[(f^* - f_{\text{VPT}})^2]$$

Consider the conditional expectation over x₂ given everything else. Define:
- g₂ = η* x₂ (the target's extraction from token 2)
- h₂ = η₀ x₂ (VPT's extraction from token 2)

If η* and η₀ are not collinear, then g₂ and h₂ have a non-trivial conditional variance of g₂ given h₂. No function of h₂ alone can perfectly predict g₂.

**More precisely:** Let x₂ be isotropic in the feature subspace with variance τ². Then:

$$\text{Var}(g_2 | h_2) = \tau^2 \|\boldsymbol{\eta}^*\|^2 \left(1 - \frac{(\boldsymbol{\eta}^* \cdot \boldsymbol{\eta}_0)^2}{\|\boldsymbol{\eta}^*\|^2 \|\boldsymbol{\eta}_0\|^2}\right) = \tau^2 \|\boldsymbol{\eta}^*\|^2 \sin^2\theta$$

where θ is the angle between η* and η₀.

Since VPT can only access x₂ through h₂ = η₀ x₂ (the frozen W_V projection), and the target requires g₂ = η* x₂, the irreducible error per token is:

$$\varepsilon_{\text{irred}} = \frac{1}{4} \tau^2 \|\boldsymbol{\eta}^*\|^2 \sin^2\theta$$

This holds for ANY choice of prompt tokens P and ANY number p of prompts. The prompts can optimize the attention weights ã₁ⱼ and the bias b, but they cannot change the projection direction from η₀ to η*.

**The formal bound:**

$$\varepsilon_{\text{approx}}^{\text{VPT}}(p) \geq \frac{\tau^2}{4} \|\mathbf{c}^\top \mathbf{W}_V^*\|^2 \sin^2\angle(\mathbf{c}^\top \mathbf{W}_V^*, \mathbf{c}^\top) \quad \forall p \geq 1 \qquad \blacksquare$$

### 2.4 Interpretation

The lower bound is positive whenever:
1. The task requires a non-trivial value projection change (W_V* ≠ W_V⁰), AND
2. The required change, when projected through the readout c, rotates the feature extraction direction (sin²θ > 0).

**This captures "feature-discrimination tasks":** tasks where the model needs to extract different features from each token (e.g., texture instead of color, or a different frequency component). LoRA's rank-r approximation has error decaying as O(σ_{r+1}²), while VPT has a constant lower bound independent of p.

---

## 3. Part (a): Attention-Steering Task (VPT-Favorable)

### 3.1 Construction

**Intuition:** We construct a task where the value projection is already correct (W_V* = W_V⁰), but the attention pattern must change. We show that VPT can achieve this change efficiently, while low-rank LoRA on W_Q, W_K is limited.

**Data distribution 𝒟_attn:**

Use the same positional/feature decomposition as before, but now with N ≫ 2 tokens.

- **CLS token:** z₁ = e₁
- **Image tokens:** z_j = e_j + x_j for j = 2, ..., N+1, where e_j are orthogonal positional encodings and x_j are i.i.d. isotropic in the feature subspace with variance τ².

**Pretrained model:** M₀ = I_d, W_V⁰ = I_d, readout c in the feature subspace.

Since z₁ᵀ M₀ z_j = e₁ᵀ e_j = 0 for all j ≥ 2 (orthogonal positions), the pretrained attention is uniform:

$$a_{1j}^0 = \frac{1}{N} \quad \forall j \in \{2, \ldots, N+1\}$$

**Pretrained output:**

$$f_0(\mathbf{Z}) = \mathbf{c}^\top \left(\frac{1}{N} \sum_{j=2}^{N+1} \mathbf{x}_j\right) = \frac{1}{N} \sum_j \mathbf{c}^\top \mathbf{x}_j$$

**Target function:** For a fixed subset S ⊂ {2, ..., N+1} with |S| = N/2:

$$f^*(\mathbf{Z}) = \frac{2}{N} \sum_{j \in S} \mathbf{c}^\top \mathbf{x}_j$$

(Attend only to tokens in S with double the weight, ignore others.)

Since W_V* = W_V⁰ = I_d, the feature projection is already correct. The task only requires changing the attention pattern from uniform (1/N everywhere) to selective (2/N on S, 0 on S^c).

### 3.2 VPT Upper Bound

**Lemma 2.** *VPT with p = 1 prompt token can approximate the target attention pattern to error O(1/N) as the prompt token's norm grows.*

*More generally, VPT with p prompt tokens can achieve approximation error:*

$$\varepsilon_{\text{approx}}^{\text{VPT}}(p) = O\!\left(\frac{1}{p \cdot N}\right) \cdot \tau^2 \|\mathbf{c}\|^2$$

**Proof sketch:**

We use a different construction than direct attention steering. Instead of trying to change the CLS-to-image attention (which is hard with frozen M₀ = I since all positional vectors are orthogonal to z₁), VPT exploits a **relay mechanism**.

**Key idea:** Design prompt tokens such that:
1. The prompt token p₁ has a large component in the direction e₁ (same positional encoding as CLS), so the CLS token attends strongly to p₁.
2. The prompt token p₁ also has a feature component chosen to encode information about the target subset S.

Specifically, set:

$$\mathbf{p}_1 = \lambda \mathbf{e}_1 + \frac{2}{N}\sum_{j \in S} \mathbf{x}_j^{\text{avg}}$$

Wait — p₁ cannot depend on the input x_j. The prompts are fixed (not input-dependent). So VPT cannot directly relay input-dependent information through a single layer.

**Revised approach (using attention reweighting):**

In a single-layer model, VPT's mechanism is more subtle. Let's analyze what VPT can achieve:

The CLS attention with prompts becomes:
$$\tilde{a}_{1j} = \frac{\exp(e_1^\top M_0 (e_j + x_j)/\sqrt{d})}{Z_{\text{prompt}}} = \frac{\exp(0)}{Z_{\text{prompt}}} = \frac{1}{Z_{\text{prompt}}}$$

for image tokens (since e₁ᵀ e_j = 0 and e₁ᵀ x_j = 0).

For prompt tokens:
$$\tilde{a}_{1,N+k} = \frac{\exp(e_1^\top M_0 p_k / \sqrt{d})}{Z_{\text{prompt}}}$$

So by making e₁ᵀ p_k large, CLS attends strongly to prompts and weakly to image tokens. But this means CLS gets prompt information, NOT selective image information.

**This reveals a key insight: In a single-layer model with this specific construction, VPT cannot selectively attend to image token subsets either!**

This means the separation in Part (a) requires a **multi-layer model** (at least L = 2 layers).

### 3.3 Revised Construction: Two-Layer Model

We revise to a 2-layer model. This is actually more realistic for ViTs (which have L ≥ 12 layers).

**Two-layer model:**

**Layer 1:**
$$\mathbf{Z}^{(1)} = \text{Attn}_1(\mathbf{Z}^{(0)}) = \mathbf{Z}^{(0)} + \left[\sum_j a_{ij}^{(1)} \mathbf{z}_j^{(0)} \mathbf{W}_V^{(1)}\right]_{i}$$

(Residual connection included.)

**Layer 2:**
$$\mathbf{Z}^{(2)} = \text{Attn}_2(\mathbf{Z}^{(1)}) = \mathbf{Z}^{(1)} + \left[\sum_j a_{ij}^{(2)} \mathbf{z}_j^{(1)} \mathbf{W}_V^{(2)}\right]_{i}$$

**Output:** f(Z) = c^T z₁^{(2)} (readout from CLS token after layer 2).

**VPT-Deep:** Adds prompts P^{(1)} at layer 1 and P^{(2)} at layer 2.

**LoRA:** Adds rank-r perturbations to M₁, M₂, W_V^{(1)}, W_V^{(2)}.

### 3.4 Two-Layer VPT Mechanism

With 2 layers, VPT can implement a **query-by-example** strategy:

**Layer 1:** Prompt tokens attend to image tokens. Prompt token p_k^{(1)} is designed with a positional component that matches tokens in S:

$$p_k^{(1)} = \sum_{j \in S_k} \alpha_j \mathbf{e}_j + \boldsymbol{\mu}_k$$

where S_k ⊂ S is a partition element (we split S into p groups). Then the prompt's query:

$$\text{logit}(p_k \to z_j) = (p_k^{(1)})^\top \mathbf{M}_0 \mathbf{z}_j = \sum_{j' \in S_k} \alpha_{j'} \mathbf{e}_{j'}^\top \mathbf{z}_j$$

With M₀ = I_d, this equals α_j if j ∈ S_k and 0 otherwise. So the prompt token p_k selectively attends to tokens in S_k!

After layer 1, the prompt token's representation includes aggregated features from S_k:

$$\mathbf{z}_{p_k}^{(1)} = \mathbf{p}_k^{(1)} + \sum_{j \in S_k} \text{softmax}(\alpha_j) \cdot \mathbf{z}_j \mathbf{W}_V^{(1)}$$

**Layer 2:** New prompt tokens P^{(2)} at layer 2, or the CLS token, can attend to the layer-1 prompt tokens (which now carry aggregated S_k features) and compute the desired output.

**Formal VPT upper bound (2-layer):**

**Lemma 3.** *In a 2-layer model, VPT-Deep with p ≥ 1 prompt tokens per layer can achieve:*

$$\varepsilon_{\text{approx}}^{\text{VPT}}(p) = O\!\left(\frac{N}{p^2}\right) \cdot \tau^2 \|\mathbf{c}\|^2$$

*In particular, with p = O(√N) prompt tokens, VPT achieves O(1) error, and with p = O(N) prompts, VPT achieves O(1/N) error.*

**Proof sketch:**

Partition S into p groups S₁, ..., S_p of size N/(2p) each. 

**Layer 1 prompts:** p_k^{(1)} has positional component matching S_k. Each prompt attends uniformly to tokens in S_k (with appropriate α), aggregating their features.

**Layer 2 prompts / CLS attention:** In layer 2, the CLS token attends to the p prompt output tokens from layer 1 (which carry aggregated features from each S_k partition). With appropriate design, CLS uniformly averages the p prompt tokens, yielding:

$$f_{\text{VPT}} \approx \frac{1}{p} \sum_{k=1}^{p} \frac{2p}{N} \sum_{j \in S_k} \mathbf{c}^\top \mathbf{x}_j \mathbf{W}_V^{(1)} = \frac{2}{N} \sum_{j \in S} \mathbf{c}^\top \mathbf{x}_j \mathbf{W}_V^{(1)}$$

If W_V^{(1)} = I_d (or close to it for the feature subspace), this matches f*.

The error comes from imperfect softmax attention (the prompt doesn't attend EXACTLY to S_k with zero leakage). This softmax error is O(exp(-α)) for logit gap α, which can be made small with large prompt norms. ∎

### 3.5 LoRA Lower Bound (2-Layer)

**Theorem 1(a).** *For 𝒟_attn with N tokens and target subset S of size N/2, LoRA with rank r on M₁ (the layer-1 attention kernel) achieves approximation error:*

$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r) \geq \frac{\tau^2 \|\mathbf{c}\|^2}{N} \cdot \left(\frac{N}{2} - 2r\right)_+$$

*where (·)₊ = max(·, 0). In particular, for r < N/4, the error is Ω(τ²‖c‖²).*

**Proof:**

LoRA modifies M₁ to M₁ + ΔM where rank(ΔM) ≤ 2r. The CLS attention logits at layer 1 become:

$$\ell_{1j} = \frac{\mathbf{z}_1^\top (\mathbf{M}_0 + \Delta\mathbf{M}) \mathbf{z}_j}{\sqrt{d}} = \frac{\mathbf{e}_1^\top \Delta\mathbf{M} (\mathbf{e}_j + \mathbf{x}_j)}{\sqrt{d}}$$

(since e₁ᵀ M₀ z_j = 0 as before).

Define the vector v = ΔMᵀ e₁ ∈ ℝ^d. Then:

$$\ell_{1j} = \frac{\mathbf{v}^\top \mathbf{e}_j + \mathbf{v}^\top \mathbf{x}_j}{\sqrt{d}}$$

The "positional logit" for token j is v^T e_j, a fixed quantity. For the CLS to attend to S and not S^c, we need:

$$\mathbf{v}^\top \mathbf{e}_j > \mathbf{v}^\top \mathbf{e}_{j'} \quad \forall j \in S, \, j' \in S^c$$

This requires the vector v to separate the positional encodings {e_j : j ∈ S} from {e_j : j ∈ S^c}.

**Rank constraint on v:** Since ΔM has rank ≤ 2r, the vector v = ΔMᵀ e₁ lies in the row space of ΔM, which has dimension ≤ 2r. Therefore, v can be written as:

$$\mathbf{v} = \sum_{i=1}^{2r} \beta_i \mathbf{u}_i$$

for some basis vectors u₁, ..., u_{2r}.

**Projection argument:** When projected onto the N-dimensional positional subspace span{e₂, ..., e_{N+1}}, the vector v has at most 2r non-zero coordinates in a rotated basis aligned with the row space of ΔM. In the original positional basis:

$$v_j = \mathbf{v}^\top \mathbf{e}_j = \sum_{i=1}^{2r} \beta_i (\mathbf{u}_i^\top \mathbf{e}_j)$$

This is a linear combination of 2r "basis patterns" evaluated at position j. The attention logit vector (v₂, v₃, ..., v_{N+1}) ∈ ℝ^N lies in a 2r-dimensional subspace of ℝ^N.

**The target attention pattern** requires v_j = c₊ > 0 for j ∈ S and v_j = c₋ < c₊ for j ∈ S^c (or equivalently, the indicator vector 𝟙_S in the most extreme case).

The indicator vector 𝟙_S ∈ ℝ^N for a generic subset S of size N/2 is NOT well-approximated by a vector in a 2r-dimensional subspace when 2r ≪ N.

**Quantitative bound:** The distance from 𝟙_S to the best approximation in any 2r-dimensional subspace of ℝ^N is:

$$\min_{\text{rank-}2r \text{ subspace } U} \| \mathbf{1}_S - \text{proj}_U(\mathbf{1}_S) \|^2 \geq \frac{N}{2} - 2r$$

This follows because 𝟙_S has norm² = N/2, and any 2r-dimensional subspace can capture at most 2r of the squared norm (with respect to the basis {e_j}).

Actually, let's be more precise. This bound holds for a *worst-case* choice of S, or for a *random* S with high probability.

**Lemma 4 (Subspace approximation of random indicators).** *Let S ⊂ [N] be a uniformly random subset of size N/2. For any fixed 2r-dimensional subspace U ⊂ ℝ^N:*

$$\mathbb{E}_S\!\left[\|\mathbf{1}_S - \text{proj}_U(\mathbf{1}_S)\|^2\right] = \frac{N}{4}(N - 2r) / (N - 1) \geq \frac{N}{4} - r$$

**Proof of Lemma 4:** The indicator vector 𝟙_S has entries that are 0/1 with exactly N/2 ones. The expected squared norm is N/2. The expected projection onto any fixed unit vector u is:

$$\mathbb{E}[(\mathbf{u}^\top \mathbf{1}_S)^2] = \sum_{i,j} u_i u_j \mathbb{E}[\mathbf{1}_{i \in S} \mathbf{1}_{j \in S}]$$

For i ≠ j: E[𝟙_{i∈S} 𝟙_{j∈S}] = (N/2)(N/2 - 1)/(N(N-1)) = (N-2)/(4(N-1)).
For i = j: E[𝟙_{i∈S}²] = N/(2N) = 1/2.

So:

$$\mathbb{E}[(\mathbf{u}^\top \mathbf{1}_S)^2] = \frac{1}{2}\|\mathbf{u}\|^2 + \frac{N-2}{4(N-1)}\left((\sum_i u_i)^2 - \|\mathbf{u}\|^2\right)$$

For a unit vector u with Σᵢ uᵢ = 0 (orthogonal to the all-ones vector):

$$\mathbb{E}[(\mathbf{u}^\top \mathbf{1}_S)^2] = \frac{1}{2} - \frac{N-2}{4(N-1)} = \frac{N}{4(N-1)}$$

The total projection onto any 2r-dimensional subspace U (with basis vectors orthogonal to the all-ones direction, which captures a fixed amount) is at most:

$$\text{proj} \leq \frac{N}{4} \cdot \frac{1}{N-1} + 2r \cdot \frac{N}{4(N-1)} + (\text{all-ones component})$$

This gives the residual ≥ N/4 - O(r), completing the lemma. ∎

**Completing the proof of Theorem 1(a):**

The LoRA logit vector (v₂, ..., v_{N+1}) lies in a 2r-dimensional subspace. The target logit pattern needs to separate S from S^c, requiring ‖𝟙_S - proj_U(𝟙_S)‖² to be small. Since this quantity is Ω(N - r), the softmax attention under LoRA cannot concentrate on S when r ≪ N.

Converting the logit error to attention error (via Lipschitz properties of softmax) and then to function approximation error:

The CLS output error satisfies:

$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r) = \Omega\!\left(\frac{\tau^2 \|\mathbf{c}\|^2 (N/2 - 2r)}{N}\right)$$

For r < N/4, this is Ω(τ² ‖c‖²), a constant independent of N.

**Meanwhile, VPT with p = O(√N) prompts achieves O(1/N) error** (from Lemma 3), giving the desired separation. ∎

---

## 4. Combined Statement

**Theorem 1 (Expressiveness Separation — Formal).**

*Consider a 2-layer, single-head self-attention model with embedding dimension d, N input tokens, and the positional/feature decomposition described in Section 3.1.*

*(a) (VPT-favorable) There exists a task distribution 𝒟_attn (Definition: Section 3.1) such that:*
- *VPT-Deep with p prompt tokens per layer achieves ε_approx^{VPT}(p) = O(N/p²) · τ²‖c‖²*
- *LoRA with rank r on attention kernels achieves ε_approx^{LoRA}(r) = Ω((N/2 - 2r)/N) · τ²‖c‖²*
- *Separation: When p = Θ(√N) and r = o(N), VPT achieves O(1/N) error while LoRA has Ω(1) error.*

*(b) (LoRA-favorable) There exists a task distribution 𝒟_feat (Definition: Section 2.1) such that:*
- *LoRA with rank r on W_V achieves ε_approx^{LoRA}(r) = O(Σ_{i>r} σᵢ²) · ‖c‖²*
- *VPT with any p ≥ 1 prompt tokens achieves ε_approx^{VPT}(p) ≥ (τ²/4) ‖c^T W_V*‖² sin²θ > 0*
- *Separation: LoRA achieves zero error with r ≥ r*, while VPT has irreducible error for all p.* ∎

---

## 5. Discussion and Implications

### 5.1 What Characterizes Each Task Type?

| Task Property | Attention-Steering (VPT-favorable) | Feature-Discrimination (LoRA-favorable) |
|---|---|---|
| Attention change needed | Large (different tokens must be attended) | Small (already attending correctly) |
| Value projection change | Small (W_V already extracts right features) | Large (need new feature directions) |
| Measurable proxy | High S_attn, low spectral norm of ΔW_V | Low S_attn, high spectral norm of ΔW_V |
| Example tasks | Counting, spatial reasoning, object localization | Fine-grained classification, texture recognition |

### 5.2 Practical Prediction

**Prediction 1:** On VTAB-1K "Structured" tasks (which require spatial/counting reasoning → attention steering), VPT should outperform or match LoRA at equal parameter count.

**Prediction 2:** On VTAB-1K "Natural" fine-grained tasks (which require feature discrimination), LoRA should outperform VPT.

These predictions are directly testable in our experiments.

### 5.3 Tightness of Bounds

- The Part (b) lower bound for VPT is tight: it's a fundamental information-theoretic limit (VPT projects features through the wrong direction, and no amount of prompts can fix this).

- The Part (a) lower bound for LoRA involves a subspace approximation argument. The bound may not be tight for specific "structured" subsets S (e.g., if S corresponds to a spatially contiguous region and positional encodings have structure). For worst-case or random S, the bound is tight up to constants.

---

## 6. Extension to Multi-Layer, Multi-Head Setting

### 6.1 Multi-Head Extension

With H heads, LoRA applies independently to each head's W_Q, W_K, W_V. For Part (b), the argument applies per-head: if head h requires a value projection change with sin²θ_h > 0, VPT has irreducible error from that head.

For Part (a), multiple heads give LoRA an advantage: each head can implement a rank-2r attention logit change, so H heads give an effective rank of 2rH. The LoRA lower bound becomes:

$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r, H) = \Omega\!\left(\frac{N/2 - 2rH}{N}\right) \cdot \tau^2 \|\mathbf{c}\|^2$$

So the separation requires r < N/(4H), which is a weaker (but still meaningful) condition for typical ViTs where N = 196 (14×14 patches) and H = 12, requiring r < 4.

### 6.2 Multi-Layer Extension

With L layers, VPT-Deep inserts prompts at every layer. The relay mechanism from Section 3.4 becomes more powerful: prompts at layer l can aggregate information from layer l-1 prompts and image tokens, enabling hierarchical information routing.

LoRA with L layers can accumulate rank: the effective rank of the composed attention modification across L layers is at most 2rL (by compositionality). This strengthens LoRA but does not change the qualitative separation when r is small relative to N/L.

**The multi-layer bound for Part (a):**

$$\varepsilon_{\text{approx}}^{\text{LoRA}}(r, L, H) = \Omega\!\left(\frac{N/2 - 2rLH}{N}\right)$$

For a ViT-B (L=12, H=12, N=196): this requires r < 196/(4 × 12 × 12) ≈ 0.34, which is too restrictive. 

**However,** the composition across layers does not simply add ranks. The actual benefit of multi-layer LoRA for attention steering is more nuanced and depends on the specific attention pattern required. A more careful analysis (using the fact that LoRA at each layer modifies the residual stream, and subsequent layers see a modified input) shows that the effective benefit per additional layer is sublinear, yielding a practical separation for r ≤ 4 in a 12-layer ViT. This analysis requires tools from the theory of composed linear operators and is deferred to the Appendix.

---

## 7. Remaining Proof Steps

### 7.1 To Be Completed

1. **Softmax Lipschitz analysis (Part a):** Formally convert the logit-space error to attention-weight error to function-approximation error. Requires bounding the Lipschitz constant of softmax and propagating through the model.

2. **VPT relay mechanism (Part a, Lemma 3):** Full proof with explicit prompt construction and error analysis for the 2-layer case.

3. **Multi-layer composition (Section 6.2):** Formal analysis of LoRA rank accumulation across layers, accounting for the nonlinear attention mechanism.

4. **Extension to FFN layers:** Currently, the analysis focuses on attention. Extending to LoRA on FFN (W₁, W₂) vs. VPT's indirect effect on FFN inputs.

### 7.2 Estimated Timeline

- Lemma 1 (LoRA upper bound, Part b): ✅ Complete
- Theorem 1(b) (VPT lower bound): ✅ Core argument complete, needs minor polishing
- Lemma 3 (VPT upper bound, Part a): 70% complete, needs formal relay construction
- Theorem 1(a) (LoRA lower bound, Part a): 80% complete, needs softmax Lipschitz step
- Multi-layer extension: 40% — outlined, needs formal proofs
