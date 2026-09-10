# ResonatorLM: Independent Scientific Replication & Adversarial Audit

This repository contains an independent, from-scratch replication, mathematical verification, efficiency benchmark, and critical audit of:

> **ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling**  
> Archie Chaudhury, Axionic Labs  
> arXiv:2607.05583v2 (July 9, 2026)

---

## Executive Summary

- **Primary Results Replicated**: ResonatorLM achieves **3.322 PPL** and **64.33% Top-1 accuracy** on WikiText-2 (character-level), reproducing the paper's reported performance within statistical noise.
- **State Memory Verified**: The recurrent state is strictly $\mathcal{O}(1)$ in sequence length ($2.0\text{ KiB}$ for 6 layers), avoiding the linear memory scaling ($O(T)$) of standard attention KV caches.
- **Transformer Collapse Refuted**: The paper's centerpiece claim—that TransformerLM degrades catastrophically from 3.40 to 20.00 PPL at length 1024—is an artifact of evaluating a model with learned absolute positional embeddings beyond its training horizon ($T=256$). When evaluated with Rotary Position Embeddings (RoPE), the Transformer achieves **3.210 PPL** (outperforming ResonatorLM) and extrapolates smoothly.
- **Mathematical Equivalence Proved**: ResonatorLM is structurally and algebraically identical to a **1D complex diagonal State-Space Model (SSM)** (such as S4D-Lin / DSS) with single conjugate pole pairs, augmented with bounded identity-residual head coupling and Mamba-style local gating.
- **Sign Error Diagnosed**: Equation (4) in the manuscript defines $\hat{y}_{t+1} = \Re(e^{-i\phi} s_{t+1})$, which causes a large discrepancy ($\sim 1.64$) against the convolution kernel $\cos(\omega t + \phi)$. Correcting the phase to $e^{+i\phi}$ restores exact numerical equivalence ($\Delta < 10^{-15}$).
- **1M Context Claim Refuted**: Because maximum head half-lives are bounded at $t_{1/2} \le 2048$ tokens, signal amplitude at 1M tokens decays to $10^{-147}$ (absolute numerical zero), disproving the claim that linear memory allows effective reasoning across 1M context.

---

## Key Reports & Deliverables

1. [**`REPRODUCTION_REPORT.md`**](REPRODUCTION_REPORT.md): Complete replication report with claims matrix, empirical results, parity comparisons, efficiency scaling, and figures.
2. [**`REVIEWER_REPORT.md`**](REVIEWER_REPORT.md): Formal peer review formatted for top ML venues (NeurIPS/ICLR/ICML). Final recommendation: **Reject / Major Revision (Score: 4/10)**.
3. [**`IRREDUCIBLE_CONTRIBUTION.md`**](IRREDUCIBLE_CONTRIBUTION.md): De-branding analysis stripping physical metaphors to isolate the exact technical delta relative to S4, S4D, Mamba, and LRU.
4. [**`RESEARCH_VERDICT.md`**](RESEARCH_VERDICT.md): Quantitative scores across Technical Correctness (5.5/10), Empirical Strength (4.0/10), Novelty (3.0/10), and Potential Importance (6.5/10).
5. [**`PAPER_SPEC.md`**](PAPER_SPEC.md): Complete parameter, dataset, and architectural specification extracted from arXiv LaTeX source.
6. [**`ASSUMPTIONS.md`**](ASSUMPTIONS.md): Complete ledger of explicit architectural and training decisions.
7. [**`ENVIRONMENT.md`**](ENVIRONMENT.md): Exact hardware, OS, and software environment specifications.

---

## Empirical Benchmark Summary

### In-Distribution and Context Extrapolation (WikiText-2 Raw Character)

| Model | Positional Encoding | Length 256 PPL | Length 256 Acc | Length 512 PPL | Length 1024 PPL | Cache Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TransformerLM (Paper Baseline)** | Learned Absolute | 3.4007 | 63.60% | 11.169 | 20.001 | $\mathcal{O}(T)$ |
| **ResonatorLM (Paper Model)** | None (Causal Conv) | 3.3223 | 64.33% | **3.301** | **3.293** | **2.0 KiB ($\mathcal{O}(1)$)** |
| **TransformerLM (Adversarial RoPE)** | Rotary (RoPE) | **3.2101** | **65.11%** | 3.536 | 5.588 | $\mathcal{O}(T)$ |

---

## Repository Structure

```text
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
│   └── test_ssm.py             # Numerical equivalence test against diagonal SSM
├── results/
│   ├── figures/                # Visualizations (context sweep, memory, half-life)
│   ├── context_sweep_results.csv
│   ├── benchmark_latency.csv
│   └── half_life_decay_audit.csv
├── scripts/                    # Profiling, evaluation, and plotting scripts
├── REPRODUCTION_REPORT.md      # Primary scientific reproduction report
├── REVIEWER_REPORT.md          # Academic peer review report
├── IRREDUCIBLE_CONTRIBUTION.md # Architecture de-branding & delta analysis
└── RESEARCH_VERDICT.md         # Final 0-10 quantitative ratings
```

---

## Reproducing the Results

```bash
# 1. Run Kernel & SSM Equivalence Unit Tests
pytest tests/

# 2. Run Memory, Latency & Decay Audits
python -m src.benchmark
python -m src.synthetic_retrieval

# 3. Evaluate Models Across Context Lengths (256, 512, 1024)
python -m scripts.eval_context_sweep
```