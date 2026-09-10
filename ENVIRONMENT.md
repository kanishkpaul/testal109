# Environment Specification

## Local Host Machine
- **Architecture**: arm64 (Apple Silicon)
- **Processor**: Apple M5 (10 cores)
- **Memory**: 16 GB Unified Memory (17,179,869,184 bytes)
- **Operating System**: macOS 26.5.2 (Darwin 25.5.0, Build 25F84)

## Python & Deep Learning Stack
- **Python Runtime**: CPython 3.12.13 (managed via `uv 0.12.12`)
- **PyTorch**: 2.14.0
- **Accelerator Backend**: Apple Metal Performance Shaders (`mps`)
  - `torch.backends.mps.is_available()`: True
  - `torch.backends.mps.is_built()`: True
- **Core Numerical Libraries**:
  - `numpy`: 2.5.3
  - `scipy`: 1.18.1
  - `matplotlib`: 3.11.1

## Reference Machine (from arXiv:2607.05583v2)
- **GPU**: 1x NVIDIA L4 (24GB VRAM, Ada Lovelace)
- **PyTorch**: 2.10.0+cu128
- **Precision**: bfloat16 AMP for training; float32 for kernel benchmark
- **FFT Backend**: cuFFT via `torch.fft`
- **Attention Reference**: PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention-2 backend)

## Evaluation Protocol Alignment
In accordance with Rule 7:
- Local speed experiments on this Apple M5 host measure **scaling, relative ratios, and crossover trends** rather than direct reproduction of absolute NVIDIA L4 latency numbers.
- A dedicated L4-compatible benchmark script matching the exact protocol (bf16, device synchronization, 5 timed iterations after 2 warmups) is provided for external execution.
