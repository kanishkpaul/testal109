# ResonatorLM: Independent Scientific Replication and Adversarial Audit

This repository contains an independent, from-scratch replication, mathematical verification, efficiency benchmark, and critical audit of:

> **ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling**  
> Archie Chaudhury, Axionic Labs  
> arXiv:2607.05583v2 (July 9, 2026)

---

## What We Found

1. **Replication Confirmed (Directional)**: Under the paper's 6M-parameter budget on WikiText-2 (raw characters), ResonatorLM achieves a test perplexity of 3.322 and top-1 accuracy of 64.33%, ahead of the parameter-matched Transformer baseline (3.400 PPL, 63.62%). This reproduces the paper's *ordering*, not its absolute values: we train for 2,000 steps on a single seed against the paper's 10,000 steps over six seeds, and both of our models land well below the paper's reported perplexities (3.764 ResonatorLM, 4.617 Transformer). See [Reproduction Protocol](#reproduction-protocol) for the exact budget and what it does and does not license.
2. **Decoding Memory is Truly Constant**: The resonant state is 2.0 KiB per layer, matching the paper's $B 	imes H 	imes d_h 	imes 2$ figure. Decoding also carries a 3.0 KiB per-layer ring buffer for the $K=3$ depthwise local path, so the full decode cache is 5.0 KiB per layer and 30.0 KiB for the 6-layer model. Both parts are constant in sequence length, which is the claim that matters. That footprint remains unchanged whether the prompt has 2,048 tokens or 32,768 tokens.
3. **The Transformer Collapse is a Positional-Encoding Artifact**: The paper's collapse at 1024 context reproduces only with learned absolute positional embeddings evaluated beyond the training horizon ($T=256$). Swapping in Rotary Position Embeddings (RoPE) at the same parameter budget removes it: 5.59 PPL at 1024 instead of 20.00, a 3.6x difference. That effect is large and robust. Two things it does **not** show: at $T=256$ RoPE measures 3.210 against ResonatorLM's 3.322, but on a single seed that 0.112 gap is roughly 1.6 standard deviations of the paper's own reported Transformer seed spread ($\pm 0.070$), so we treat it as a tie rather than a win; and at 1024 ResonatorLM is still clearly ahead (3.293 vs 5.588). ResonatorLM's zero-shot length generalization is real. What the RoPE control removes is the *catastrophic* reading of the baseline, not ResonatorLM's extrapolation advantage.
4. **ResonatorLM is a Diagonal Complex State-Space Model**: Stripping away the physical terminology ("damped resonant fields", "eigenmodes") reveals an architecture mathematically and numerically identical to a 1D complex diagonal SSM (such as S4D-Lin or LRU) with single conjugate pole pairs, bounded identity-residual head coupling, and Mamba-style local gating.
5. **Phase Sign Error in Equation (4)**: Equation (4) defines $\hat{y}_{t+1} = \Re(e^{-i\phi} s_{t+1})$, which evaluates to $\cos(\omega t - \phi)$. This conflicts with the convolution kernel $\cos(\omega t + \phi)$ in Equation (1), causing an order-one numerical mismatch (we measure between 1.64 and 2.77 depending on the drive sequence; the magnitude is input-dependent, the discrepancy is not). Changing the sign to $e^{+i\phi}$ resolves the error to machine precision ($< 10^{-14}$) on every input we tried.
6. **Effective Memory Span is Far Shorter Than the Benchmarked Context**: The paper does not claim 1M context; its largest stated context is 32K, and it benchmarks speed there. The gap we flag is that the *speed* claims run to 32K while the *memory* cannot follow. The parameterization is $\alpha_h = 10^{-4} + \text{softplus}(\tilde{\alpha}_h)$, so $\alpha_h \ge 10^{-4}$ and no head can exceed $t_{1/2} = \ln 2 / 10^{-4} \approx 6{,}931$ tokens. (Softplus is strictly positive in exact arithmetic, but underflows to exactly $0$ in float32 for sufficiently negative inputs, so the bound is attained rather than merely approached.) The $2{,}048$ figure is the *initialization* ceiling, not a hard cap: the paper reports a learned maximum of $2{,}048.0$ and our run learns $1{,}526.4$. Either way the conclusion is unchanged: amplitude at 1M tokens is $1.03 \times 10^{-147}$ at the reported $2{,}048$ half-life, and still only $3.7 \times 10^{-44}$ at the $6{,}931$ architectural ceiling. The model retains no usable associative memory at that scale.

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
| **ResonatorLM (Paper Model)** | None (Causal Conv) | 3.322 | 64.33% | 3.301 | 3.293 | Constant 30.0 KiB ($O(1)$) |
| **TransformerLM (RoPE Control)** | Rotary (RoPE) | 3.210 | 65.11% | 3.536 | 5.588 | Grows linearly ($O(T)$) |

All three models are trained at $T=256$ and evaluated zero-shot at 512 and 1024. The 512 and 1024 columns therefore measure *context extrapolation*, not training at those lengths. The paper's own long-context rows appear to be separate training runs per length, so those numbers are not directly comparable to this table.

---

## Reproduction Protocol

Being explicit about the compute budget, because it bounds what these results support:

| Setting | Paper | This Replication |
| :--- | :---: | :---: |
| Optimization steps | 10,000 | **2,000** |
| Seeds | 6 (primary), 3 (breadth) | **1 (seed 0)** |
| Batch size | unstated | 32 |
| Sequence length | 256 | 256 |
| Optimizer | AdamW, lr $5\times10^{-4}$, cosine | identical |
| Hardware | 1x NVIDIA L4 (CUDA) | Apple M5 (MPS) |

**What this budget supports.** The architectural, numerical, and memory findings, all of which are seed-independent and exactly reproducible: the Equation (4) phase-sign error, the diagonal-SSM equivalence, the constant-state-memory result, the half-life decay limit, and the RoPE-versus-learned-PE diagnosis. Re-running `scripts/eval_context_sweep.py` against the trained checkpoints reproduces every digit of the table above, run after run on the same host.

**Note on artifacts.** The checkpoints (116 MB) and the WikiText-2 raw corpus are gitignored, so a fresh clone has neither. Regenerate them with `python -m src.train --model resonator --steps 2000 --seed 0` and the equivalent `--model transformer` runs (`--pos_encoding learned` and `--pos_encoding rope`) after placing `wiki.{train,valid,test}.raw` under `data/wikitext-2-raw/`. Only `data/char_vocab.json` is tracked, and `tests/test_data.py` asserts it still matches what the pipeline rebuilds from the corpus.

**An unexplained discrepancy, stated plainly.** Our absolute numbers come out *better* than the paper's on both models (3.32 vs 3.764 PPL for ResonatorLM, 3.40 vs 4.617 for the Transformer; 64.3% vs 61.31% and 63.6% vs 55.32% accuracy) despite one fifth of the optimization steps. Less training cannot produce better test loss, so our setup and the paper's differ somewhere beyond step count. Two candidates we can name: our character vocabulary is 284 symbols built from the raw training split, against the paper's stated ~256; and at batch 32 the paper's 10,000 steps is 7.6 epochs over WikiText-2 raw where our 2,000 steps is 1.5, so if their batch size was comparable, some of the gap may be overfitting on their side. We have not resolved this, and we flag it rather than presenting our numbers as a match. Every comparison in this repository is therefore *within* our own setup, never across to the paper's absolute values.

**What it does not support.** Any claim about matching the paper's absolute perplexities, or about the size of the ResonatorLM-versus-Transformer quality gap. With one seed we cannot estimate variance, so the 3.322 vs 3.400 gap should be read as a direction, not a measured effect. Reproducing the paper's headline numbers would require the full 10,000-step, six-seed protocol.

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

## Licensing

This repository contains an independent scientific reproduction and audit of arXiv:2607.05583v2, released under the permissive **MIT License** ([`LICENSE`](LICENSE)). Use it, fork it, and check our work.

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