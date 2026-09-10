# Reconstruction Assumptions Record

This document records all decisions where public sources were underspecified or silent, following the non-negotiable scientific rule: *Never invent an unspecified detail without explicit documentation.*

---

## Assumption 1: Transformer Positional Encoding
- **What is missing**: The paper does not state whether the baseline Transformer uses learned absolute 1D positional embeddings, sinusoidal positional embeddings, or rotary position embeddings (RoPE).
- **Where we searched**: `resonator_body.tex`, `resonator_results.tex`, the Axionic Labs website bundle, and author publications. Positional encoding is entirely omitted from the text.
- **Why it matters**: In long-context evaluations (Table 3), the Transformer experiences severe degradation when moving from sequence length 256 to 512 and 1024 (PPL rising from 5.06 to 8.70 and 11.02). If the baseline used fixed-length absolute learned positional embeddings or improperly interpolated embeddings, this degradation could be an artifact of out-of-context positional mismatch rather than an attention limitation.
- **Our Assumption**:
  1. For the *faithful paper reproduction*, we use standard learned absolute 1D positional embeddings initialized for context up to 256 (and extended/interpolated for 512/1024 to test for collapse).
  2. For the *adversarial audit* (Phase 7), we explicitly test a modernized Transformer with Rotary Position Embeddings (RoPE) to isolate whether the long-context collapse is caused by attention or flawed positional encoding.

---

## Assumption 2: Batch Size During Training
- **What is missing**: The exact batch size (number of sequences per optimization step) is omitted in §4 and §5. (Only the benchmark specifies batch size 1 for inference).
- **Where we searched**: `resonator_results.tex` (Section 4 and 5), Axionic Labs blog post.
- **Why it matters**: Batch size determines the total tokens processed per step and gradient variance across the 10,000 optimization steps.
- **Our Assumption**: We use a batch size of 32 sequences of length 256 (effective batch size 8,192 tokens/step). Over 10,000 steps, this corresponds to $81.92 \times 10^6$ training tokens (~82M tokens), which cleanly traverses WikiText-2 raw (~2M characters) for ~40 epochs, standard for 6M character language model pretraining. We also test batch size 64 in sensitivity analysis.

---

## Assumption 3: Recurrent Phase Sign Discrepancy
- **What is missing**: Equation (1) specifies $k_h[t] = \exp(-\alpha_h t)\cos(\omega_h t + \phi_h)$, while Equation (4) gives $\hat{y}_{t+1} = \Re(e^{-i\phi_h} s_{t+1})$.
- **Where we searched**: Mathematical unrolling of the recurrence $s_{t+1} = \sum_{\tau=0}^t e^{-\alpha(t-\tau)+i\omega(t-\tau)} u_\tau$.
- **Why it matters**: Computing $\Re(e^{-i\phi} s_{t+1})$ yields $\sum_\tau e^{-\alpha(t-\tau)} \cos(\omega(t-\tau) - \phi) u_\tau$, which has $-\phi$ rather than $+\phi$. This causes an error of up to $\sim 1.64$ between FFT and recurrent evaluation.
- **Our Assumption**: Equation (4) contains a typographical error in the paper. The mathematically sound equation is $\hat{y}_{t+1} = \Re(e^{+i\phi_h} s_{t+1})$. We implement both in unit tests to formally demonstrate the discrepancy, and use $e^{+i\phi_h}$ for model execution to ensure exact equivalence ($\Delta < 10^{-15}$ in float64, $< 10^{-6}$ in float32).

---

## Assumption 4: Local Depthwise Convolution Kernel Size $K$
- **What is missing**: The kernel size $K$ for the causal depthwise local lexical convolution in §3.4 is stated only as $K > 0$.
- **Where we searched**: `resonator_body.tex` §3.4, `resonator_results.tex`.
- **Why it matters**: Local convolution adds $K \times d_{\text{model}}$ parameters to the model.
- **Our Assumption**: We set $K = 3$ (standard for short-range n-gram / lexical capture). With $d_{\text{model}} = 256$, $K=3$ adds 768 parameters, fitting neatly within the 6.039M total parameter count.

---

## Assumption 5: Head Coupling Matrix Strength $\lambda_c$
- **What is missing**: The exact numeric value of $\lambda_c$ is not declared in §3.4, though Table 2 lists "Coupling radius 1.0".
- **Where we searched**: `resonator_body.tex` §3.4, `resonator_results.tex` Table 2.
- **Why it matters**: $\lambda_c$ scales the perturbation from the identity matrix $C = I + \frac{\lambda_c}{\sqrt{H}}\tanh(\widetilde{C})$.
- **Our Assumption**: We set $\lambda_c = 1.0$, matching Table 2's reported coupling radius. $\widetilde{C}$ is initialized to zero (so $C$ begins as identity $I$) with standard Gaussian scale $0.02$.

---

## Assumption 6: Optimizer Hyperparameters & LR Warmup
- **What is missing**: The paper mentions "same optimizer family, schedule" in §4 and "at learning rate $5\times 10^{-4}$ and zero dropout" in §5.2, but does not state $\beta_1, \beta_2$, weight decay, or warmup steps.
- **Where we searched**: LaTeX manuscripts and Axionic blog post.
- **Why it matters**: Optimization stability is critical for fair comparisons across 10,000 steps.
- **Our Assumption**: We use standard AdamW with $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-8}$, weight decay $0.01$, linear warmup for 500 steps (5% of 10k steps), followed by cosine decay to $0.1 \times \text{lr}_{\text{max}}$. Gradient norm clipping is set to $1.0$.

---

## Assumption 7: Seed Numbers
- **What is missing**: The integer values of the six seeds in Table 1 are not listed.
- **Where we searched**: LaTeX manuscripts.
- **Why it matters**: Seed selection must be predetermined to prevent cherry-picking.
- **Our Assumption**: In strict accordance with the prompt guidelines, we use predetermined seeds: `[0, 1, 2, 3, 4, 5]`.
