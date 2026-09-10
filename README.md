# ResonatorLM: Independent Scientific Replication and Adversarial Audit

This repository contains an independent, from-scratch replication, mathematical verification, efficiency benchmark, and critical audit of:

> **ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling**  
> Archie Chaudhury, Axionic Labs  
> arXiv:2607.05583v2 (July 9, 2026)

---

## What We Found

1. **Replication Confirmed**: Under the paper's exact 6M-parameter budget on WikiText-2 (raw characters), ResonatorLM achieves a test perplexity of 3.322 and top-1 accuracy of 64.33%. This matches the paper's training curve within expected seed variance.
2. **Decoding Memory is Truly Constant**: Generation requires exactly 2.0 KiB of state cache across all 6 layers. That footprint remains unchanged whether the prompt has 2,048 tokens or 32,768 tokens.
3. **Transformer Collapse Refuted**: The paper claimed that TransformerLM collapses at 1024 context length (perplexity jumping from 3.40 to 20.00). We showed this happens only when using learned absolute positional embeddings beyond the training horizon ($T=256$). When equipped with modern Rotary Position Embeddings (RoPE), the Transformer achieves 3.210 PPL (beating ResonatorLM at length 256) and extrapolates smoothly to 1024 without blowing up.
4. **ResonatorLM is a Diagonal Complex State-Space Model**: Stripping away the physical terminology ("damped resonant fields", "eigenmodes") reveals an architecture mathematically and numerically identical to a 1D complex diagonal SSM (such as S4D-Lin or LRU) with single conjugate pole pairs, bounded identity-residual head coupling, and Mamba-style local gating.
5. **Phase Sign Error in Equation (4)**: Equation (4) defines $\hat{y}_{t+1} = \Re(e^{-i\phi} s_{t+1})$, which evaluates to $\cos(\omega t - \phi)$. This conflicts with the convolution kernel $\cos(\omega t + \phi)$ in Equation (1), causing a large numerical mismatch (~1.64). Changing the sign to $e^{+i\phi}$ resolves the error to machine precision ($< 10^{-14}$).
6. **1M Context Retention is Mathematically Impossible**: Head half-lives are capped at 2,048 tokens. Signal amplitude decays exponentially and drops to $10^{-147}$ at 1M tokens, meaning the model retains zero associative memory at that scale.

For the full breakdown written for researchers, see [**`FINDINGS.md`**](FINDINGS.md).

---

## Documents and Reports

- [**`FINDINGS.md`**](FINDINGS.md): Detailed narrative of the five core findings discovered during our audit.
- [**`REPRODUCTION_REPORT.md`**](REPRODUCTION_REPORT.md): Complete replication report with claims matrix, empirical results, and figures.
- [**`REVIEWER_REPORT.md`**](REVIEWER_REPORT.md): Formal peer review formatted for top ML conferences (NeurIPS/ICLR/ICML).
- [**`IRREDUCIBLE_CONTRIBUTION.md`**](IRREDUCIBLE_CONTRIBUTION.md): Mathematical de-branding and comparative analysis against S4, Mamba, and LRU.
- [**`RESEARCH_VERDICT.md`**](RESEARCH_VERDICT.md): Quantitative scores (Technical Correctness: 5.5/10, Empirical Strength: 4.0/10, Novelty: 3.0/10, Potential Importance: 6.5/10).
- [**`PAPER_SPEC.md`**](PAPER_SPEC.md): Full architectural, dataset, and hyperparameter specifications extracted from the paper's LaTeX source.
- [**`ASSUMPTIONS.md`**](ASSUMPTIONS.md): Ledger of implementation choices and hyperparameter resolutions.
- [**`ENVIRONMENT.md`**](ENVIRONMENT.md): Hardware, operating system, and PyTorch environment details.

---

## Results Summary

### In-Distribution and Context Extrapolation (WikiText-2 Character Level)

| Model | Positional Encoding | Length 256 PPL | Length 256 Acc | Length 512 PPL | Length 1024 PPL | State Cache |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TransformerLM (Paper Baseline)** | Learned Absolute | 3.401 | 63.60% | 11.169 | 20.001 | Grows linearly ($O(T)$) |
| **ResonatorLM (Paper Model)** | None (Causal Conv) | 3.322 | 64.33% | 3.301 | 3.293 | Constant 2.0 KiB ($O(1)$) |
| **TransformerLM (RoPE Control)** | Rotary (RoPE) | 3.210 | 65.11% | 3.536 | 5.588 | Grows linearly ($O(T)$) |

---

## Repository Structure

```text
├── LICENSE                     # MIT License for open scientific reproduction
├── FINDINGS.md                 # Core audit discoveries and technical recommendations
├── REPRODUCTION_REPORT.md      # Primary scientific reproduction report
├── REVIEWER_REPORT.md          # Academic peer review report
├── IRREDUCIBLE_CONTRIBUTION.md # Architecture de-branding & delta analysis
├── RESEARCH_VERDICT.md         # Final quantitative ratings
├── src/
│   ├── resonator.py            # Resonant field causal convolution & recurrent step
│   ├── resonator_lm.py         # Full 6-layer ResonatorLM architecture
│   ├── transformer.py          # Parameter-matched TransformerLM (Learned PE & RoPE)
│   ├── ssm.py                  # Mathematical equivalence to diagonal complex SSM
│   ├── data.py                 # WikiText-2 raw character dataset & tokenizer
│   ├── train.py                # Deterministic training & evaluation pipeline
│   ├── benchmark.py            # Latency, memory, and throughput profiling
│   └── synthetic_retrieval.py  # Half-life exponential decay & retrieval limits
├── tests/
│   ├── test_kernel.py          # FFT conv vs direct causal conv vs recurrent stepping
│   ├── test_model.py           # Parameter counts, layer shapes, and state cache size
│   └── test_ssm.py             # Numerical equivalence test against diagonal SSM
├── results/
│   ├── figures/                # Visualizations (context sweep, memory, half-life)
│   ├── context_sweep_results.csv
│   ├── benchmark_latency.csv
│   └── half_life_decay_audit.csv
└── scripts/                    # Profiling, evaluation, and plotting scripts
```

---

## Branches and Licensing

- **`main` (this branch)**: Contains the pure, independent scientific reproduction and audit of arXiv:2607.05583v2. Released under the permissive **MIT License** ([`LICENSE`](LICENSE)).
- **`improvements` branch**: Contains novel architectural extensions (Selective Resonator with dynamic input gating, Long-Context Carrier Modes with multi-million-token half-lives, and Hybrid Resonator-Attention LM). Released under the **Research Use & Anti-Scooping License (v1.0)**, requiring prior written consent from Kanishk Paul for derivative academic publications or commercial deployments.

---

## How to Run the Tests

```bash
# Run all unit tests (kernel correctness, SSM equivalence, parameter parity)
python -m unittest discover -s tests/ -v

# Run memory, latency, and decay benchmarks
python -m src.benchmark
python -m src.synthetic_retrieval

# Evaluate models across context lengths (256, 512, 1024)
python -m scripts.eval_context_sweep
```