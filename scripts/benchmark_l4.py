"""Exact NVIDIA L4-compatible long-context block benchmark script.

Hardware / Software Target:
- GPU: Single NVIDIA L4 (24GB VRAM)
- PyTorch: 2.10.0+cu128 / latest CUDA
- Dtype: bfloat16 AMP for training/prefill/decode; float32 for kernel tail
- Batch size: 1, Decode length: 128
- Iterations: 5 timed after 2 warmups
- Explicit CUDA device synchronization
"""

import csv
import os
import time
import torch
from src.resonator import ResonantFieldMixer
from src.transformer import CausalSelfAttention


def run_l4_benchmark():
    assert torch.cuda.is_available(), "CUDA device required for exact L4 benchmark"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_lengths = [2048, 4096, 8192, 16384, 32768]
    decode_len = 128
    n_iters = 5
    n_warmup = 2

    res_block = ResonantFieldMixer(d_model=256, n_heads=8).to(device=device, dtype=dtype)
    attn_block = CausalSelfAttention(d_model=248, n_heads=8).to(device=device, dtype=dtype)
    res_block.eval()
    attn_block.eval()

    results = []
    print("\n=== Exact NVIDIA L4 Long-Context Block Benchmark ===")
    print(f"{'Seq Len':<10} | {'Train Spd':<10} | {'Prefill Spd':<12} | {'Decode Spd':<12} | {'Res ms/tok':<12} | {'Attn ms/tok':<12}")
    print("-" * 75)

    for seq_len in seq_lengths:
        x_res = torch.randn(1, seq_len, 256, device=device, dtype=dtype)
        x_attn = torch.randn(1, seq_len, 248, device=device, dtype=dtype)

        # 1. Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = res_block(x_res)
                _ = attn_block(x_attn)
            torch.cuda.synchronize()

        # 2. Prefill Timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = res_block(x_res)
            torch.cuda.synchronize()
        res_prefill_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = attn_block(x_attn)
            torch.cuda.synchronize()
        attn_prefill_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

        prefill_spd = attn_prefill_ms / max(res_prefill_ms, 1e-6)

        # 3. Decode Timing
        with torch.no_grad():
            s_real, s_imag = res_block.compute_terminal_state(x_res)
            res_state = (s_real, s_imag)
            _, attn_cache = attn_block(x_attn, kv_cache=(torch.zeros(1, 8, 0, 31, device=device, dtype=dtype),
                                                         torch.zeros(1, 8, 0, 31, device=device, dtype=dtype)))
            torch.cuda.synchronize()

        tok_res = torch.randn(1, 256, device=device, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            st = res_state
            for _ in range(decode_len):
                _, st, _ = res_block.step(tok_res, st)
            torch.cuda.synchronize()
        res_ms_tok = ((time.perf_counter() - t0) / decode_len) * 1000.0

        tok_attn = torch.randn(1, 1, 248, device=device, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            c = attn_cache
            for i in range(decode_len):
                _, c = attn_block(tok_attn, kv_cache=c, start_pos=seq_len + i)
            torch.cuda.synchronize()
        attn_ms_tok = ((time.perf_counter() - t0) / decode_len) * 1000.0

        decode_spd = attn_ms_tok / max(res_ms_tok, 1e-6)

        print(f"{seq_len:<10} | {'-':<10} | {prefill_spd:<12.2f}x | {decode_spd:<12.2f}x | {res_ms_tok:<12.4f} | {attn_ms_tok:<12.4f}")

        results.append({
            "seq_len": seq_len,
            "res_prefill_ms": res_prefill_ms,
            "attn_prefill_ms": attn_prefill_ms,
            "prefill_speedup": prefill_spd,
            "res_ms_per_tok": res_ms_tok,
            "attn_ms_per_tok": attn_ms_tok,
            "decode_speedup": decode_spd,
        })

    os.makedirs("results", exist_ok=True)
    with open("results/l4_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_l4_benchmark()
