# Complete Paper Specification: ResonatorLM

**Paper**: *ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling*  
**Author**: Archie Chaudhury, Axionic Labs  
**Source**: arXiv:2607.05583v2 (July 9, 2026; camera-ready ICANN 2026 submission)

---

## Specification Extraction Table

| Item | Paper Value | Source / Location | Confidence | Missing / Underspecified? |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Dataset** | WikiText-2 (raw) | §4, l. 13; Tab. 1 | High | No |
| **Auxiliary Datasets** | WikiText-103 (char), TinyStories (char/byte) | §4, l. 39; §5.2, Tab. 3 | High | No |
| **Dataset Splits** | Standard train/val/test splits | §4, l. 18, 26 | High | Exact splits standard to Merity et al. |
| **Tokenization** | Character tokenizer (WikiText-2 char) | §4, l. 13-14 | High | Exact character vocab set unlisted in paper |
| **Primary Sequence Length** | 256 tokens | §4, l. 39 ("256-token matched-budget baseline") | High | No |
| **Long Context Lengths** | 512, 1024 (and up to 32768 in benchmarks) | §5.2, Tab. 3; §5.3, Tab. 4 | High | No |
| **Batch Size** | Unstated in §4 text; block benchmark uses batch size 1 | §4, l. 27; §5.3, l. 205 | Medium | Missing in training text; standard for 6M char LM is 32 or 64 sequences (see ASSUMPTIONS) |
| **Eval Batch Size** | Unstated in text (matches train batch size per protocol) | §4, l. 27 | Medium | Implicitly matched |
| **Optimization Steps** | 10,000 steps | §4, l. 18; §5.1 | High | No |
| **Optimizer Family** | AdamW / Adam family ("same optimizer family") | §4, l. 18, 27 | High | Missing exact $\beta_1, \beta_2, \epsilon$ in text |
| **Learning Rate** | $5 \times 10^{-4}$ | §5.2, l. 146-147 ("at learning rate $5\times 10^{-4}$") | High | Verified in parity sweep text |
| **LR Schedule** | Cosine decay / warmup schedule | §4, l. 18, 27 | High | Warmup step count unstated (standard is 500-1000 steps) |
| **Warmup Steps** | Unstated | §4, l. 18, 27 | Low | Missing (assume 500 steps, 5%) |
| **Gradient Clipping** | Unstated | §4 | Low | Missing (standard is 1.0) |
| **Dropout** | 0.0 in parity sweep; unstated default | §5.2, l. 147 ("and zero dropout") | High | Baseline tested with 0.0 dropout |
| **Weight Decay** | Unstated | §4 | Low | Missing (standard is 0.01 or 0.1) |
| **Initialization** | Damped resonator log-spaced modes; standard Gaussian for projections | §3.1, l. 64-65 | High | No |
| **Training Precision** | bfloat16 AMP (on CUDA); float32 in kernel bench | §5.3, l. 202-203 | High | No |
| **Loss Function** | Cross-entropy next-token prediction | §3.4, l. 131; §4 | High | No |
| **Parameter Tying** | Unstated (embedding and output LM head untied yields ~6.03M) | Model parameter analysis | High | Untied embedding matches 6.03M exactly |
| **Vocabulary Size** | Character set (~256 for raw characters/bytes) | §4, l. 13; §5.2 | High | Character vocab size ~256 |
| **Evaluation Metrics** | Test Perplexity ($\exp(\text{CE})$), Top-1 Accuracy (\%), Train tok/s | §4, l. 19-20; Tab. 1 | High | No |
| **Seeds** | 6 seeds for matched primary; 3 seeds for breadth/ablations | §4, l. 19-22; Tab. 1 | High | Specific integer seed values unlisted (standard 0..5) |
| **Hardware** | 1x NVIDIA L4 (24GB VRAM) | §5.3, l. 201 | High | No |
| **Software** | PyTorch 2.10.0+cu128, cuFFT, PyTorch SDPA | §5.3, l. 201-210 | High | No |
| **Transformer Layers** | 6 layers | §4, l. 15 | High | No |
| **Transformer $d_{\text{model}}$** | 248 | §4, l. 15 | High | No |
| **Transformer Heads** | 8 heads ($d_h = 31$) | §4, l. 15 | High | No |
| **Transformer FFN Dimension** | $4 \times d_{\text{model}} = 992$ (SwiGLU) | §3.4, l. 131; param count verification | High | $3 \times 248 \times 992 = 738,048$ params/layer |
| **Transformer Positional Encoding** | Unstated in text (no mention of learned vs RoPE) | §4 | Low | **CRITICAL OMISSION** in paper |
| **Transformer Attention** | Causal PyTorch scaled-dot-product attention (SDPA) | §5.3, l. 207-208 | High | No |
| **ResonatorLM Layers** | 6 layers | §4, l. 16 | High | No |
| **ResonatorLM $d_{\text{model}}$** | 256 | §4, l. 16 | High | No |
| **ResonatorLM Heads** | 8 heads ($d_h = 32$) | §4, l. 16 | High | No |
| **Resonator Projections** | Input: $d \to 2d$ ($u$ drive and $g$ gate); Output: $d \to d$ | §3.1, l. 66; §3.4, l. 110 | High | Matches 6.039M param count |
| **Normalization** | Pre-RMSNorm before mixer and before FFN | §3.4, l. 131 | High | No |
| **MLP Sublayer** | SwiGLU MLP ($d_{\text{ff}} = 4 \times d_{\text{model}} = 1024$) | §3.4, l. 131; param count verification | High | No |
| **Kernel Equation** | $k_h[t] = \exp(-\alpha_h t)\cos(\omega_h t + \phi_h)$ | §3.1, Eq. 1 | High | No |
| **$\alpha$ Range & Parameterization** | $\alpha_h = 10^{-4} + \text{softplus}(\tilde{\alpha}_h) > 10^{-4}$ | §3.1, l. 64 | High | No |
| **$\omega$ Range & Parameterization** | $\omega_h = \pi \cdot \sigma(\tilde{\omega}_h) \in (0, \pi)$ | §3.1, l. 64 | High | No |
| **$\phi$ Parameterization** | Unconstrained learned phase offset $\phi_h \in \mathbb{R}$ | §3.1, l. 64 | High | No |
| **Half-life Initialization** | Log-spaced across heads from $t_{1/2} = 2.0$ to $2048.0$ tokens | §3.1, l. 64; Tab. 2 | High | $\alpha_h = \ln 2 / t_{1/2}$ |
| **Frequency Initialization** | Log/frequency-spaced across $(0, \pi)$ | §3.1, l. 64 | High | Spaced from $\pi / 2048$ to $\pi / 2$ |
| **FFT Convolution** | $y_h = \mathcal{F}^{-1}(\mathcal{F}(u_h) \odot \mathcal{F}(k_h))$, zero-padded to next power of 2 ($\ge 2T$) | §3.2, Eq. 3, l. 79 | High | No |
| **Recurrent Step** | $s_{t+1} = \lambda_h s_t + u_t$, $\lambda_h = e^{-\alpha_h + i\omega_h}$ | §3.3, Eq. 4-5 | High | No |
| **Recurrent Output** | $\hat{y}_{t+1} = \Re(e^{+i\phi_h} s_{t+1})$ (paper writes $e^{-i\phi}$, proved sign typo) | §3.3, Eq. 4 | High | Math discrepancy diagnosed |
| **Recurrent State Size** | $B \times H \times d_h \times 2$ (real and imaginary tensors) | §3.3, l. 94 | High | No |
| **Head Coupling Matrix** | $C = I + \frac{\lambda_c}{\sqrt{H}}\tanh(\widetilde{C})$, with $\lambda_c = 1.0$ | §3.4, Eq. 6-7; Tab. 2 | High | No |
| **Local Lexical Path** | Depthwise causal 1D conv ($K=3$ or $4$) + learned channel scale $s$ | §3.4, Eq. 8 | High | Kernel size $K$ unstated (3 standard) |
| **Output Gating** | $z_t = \gamma_t \odot \operatorname{flatten}(\bar{y}_t) + s \odot l_t$, where $\gamma = \sigma(g)$ | §3.4, Eq. 8 | High | No |
| **Prefill Terminal State** | Weighted sum of past drives $u_t$ under complex decay factor | §3.2, l. 81-82 | High | No |
| **Practical Block Benchmark** | $B=1$, decode len 128, 5 timed after 2 warmups, device sync | §5.3, l. 204-207 | High | No |
| **Kernel-Tail Benchmark** | float32, chunk size 32, vs quadratic causal reference | §5.3, l. 203, 233-235 | High | No |
