import argparse
import csv
import math
import os
import time
from typing import Dict, List
import torch
import torch.nn as nn
from src.resonator import ResonantFieldMixer
from src.transformer import CausalSelfAttention


def benchmark_memory(
    seq_lengths: List[int] = [2048, 4096, 8192, 16384, 32768],
    d_model_res: int = 256,
    d_model_trans: int = 248,
    n_heads: int = 8,
    n_layers: int = 6,
    device_str: str = "mps",
) -> List[Dict]:
    """Measures KV cache vs Recurrent State memory for the FULL n_layers model.

    Both figures are whole-model totals. Per-layer figures are these divided by
    n_layers; the ratio between the two columns is identical either way.
    """
    results = []

    print("\n=== Memory Benchmark: KV Cache vs Recurrent State ===")
    print(f"{'Seq Len':<10} | {'Trans KV (KiB)':<15} | {'Res State (KiB)':<15} | {'Ratio':<10}")
    print("-" * 55)

    for seq_len in seq_lengths:
        # Transformer KV cache: n_layers * 2 tensors (K, V) of shape (1, H, T, D_h)
        head_dim_trans = d_model_trans // n_heads
        trans_elements = n_layers * 2 * 1 * n_heads * head_dim_trans * seq_len
        trans_bytes = trans_elements * 4  # float32 bytes
        trans_kib = trans_bytes / 1024.0

        # Resonator recurrent state: n_layers * 2 tensors (Real, Imag) of shape (1, H, D_h).
        # 2.0 KiB per layer, so 12.0 KiB for the 6-layer model, independent of seq_len.
        head_dim_res = d_model_res // n_heads
        res_elements = n_layers * 2 * 1 * n_heads * head_dim_res
        res_bytes = res_elements * 4
        res_kib = res_bytes / 1024.0

        ratio = trans_kib / res_kib

        print(f"{seq_len:<10} | {trans_kib:<15.1f} | {res_kib:<15.1f} | {ratio:<10.1f}x")

        results.append({
            "seq_len": seq_len,
            "trans_kv_kib": trans_kib,
            "res_state_kib": res_kib,
            "ratio": ratio,
        })

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark_memory.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return results


def benchmark_latency(
    seq_lengths: List[int] = [2048, 4096, 8192, 16384],
    decode_len: int = 64,
    device_str: str = "mps",
    dtype: torch.dtype = torch.float32,
    output_dir: str = "results",
) -> List[Dict]:
    """Measures prefill and decode latency across sequence lengths."""
    device = torch.device(device_str)
    sync_fn = torch.mps.synchronize if device.type == "mps" else (torch.cuda.synchronize if device.type == "cuda" else lambda: None)

    res_mixer = ResonantFieldMixer(d_model=256, n_heads=8).to(device=device, dtype=dtype)
    attn_block = CausalSelfAttention(d_model=248, n_heads=8).to(device=device, dtype=dtype)

    res_mixer.eval()
    attn_block.eval()

    results = []
    print(f"\n=== Practical Latency Benchmark on {device_str.upper()} (Decode Len={decode_len}) ===")
    print(f"{'Seq Len':<10} | {'Res Prefill':<12} | {'Attn Prefill':<12} | {'Prefill Spd':<12} | {'Res ms/tok':<12} | {'Attn ms/tok':<12} | {'Decode Spd':<12}")
    print("-" * 90)

    for seq_len in seq_lengths:
        x_res = torch.randn(1, seq_len, 256, device=device, dtype=dtype)
        x_attn = torch.randn(1, seq_len, 248, device=device, dtype=dtype)

        # 1. Warmup prefill
        attn_oom = False
        with torch.no_grad():
            for _ in range(2):
                _ = res_mixer(x_res)
                if not attn_oom:
                    try:
                        _ = attn_block(x_attn)
                    except RuntimeError:
                        attn_oom = True
            sync_fn()

        # 2. Benchmark Prefill
        n_iters = 5
        sync_fn()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = res_mixer(x_res)
            sync_fn()
        res_prefill_s = (time.perf_counter() - t0) / n_iters

        attn_prefill_s = None
        attn_oom = False
        try:
            sync_fn()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = attn_block(x_attn)
                sync_fn()
            attn_prefill_s = (time.perf_counter() - t0) / n_iters
            prefill_speedup = attn_prefill_s / max(res_prefill_s, 1e-9)
        except RuntimeError as e:
            attn_oom = True
            prefill_speedup = float("inf")
            print(f"Attention Prefill OOM at sequence length {seq_len}")

        # 3. Benchmark Decode
        # Build initial cache/state
        with torch.no_grad():
            s_real, s_imag = res_mixer.compute_terminal_state(x_res)
            res_state = (s_real, s_imag)
            attn_cache = None
            if not attn_oom:
                try:
                    _, attn_cache = attn_block(x_attn, kv_cache=(torch.zeros(1, 8, 0, 31, device=device, dtype=dtype),
                                                                 torch.zeros(1, 8, 0, 31, device=device, dtype=dtype)))
                except RuntimeError:
                    attn_oom = True
            sync_fn()

        # Timed decode steps for ResonatorLM
        tok_res = torch.randn(1, 256, device=device, dtype=dtype)
        sync_fn()
        t0 = time.perf_counter()
        with torch.no_grad():
            curr_state = res_state
            for _ in range(decode_len):
                _, curr_state, _ = res_mixer.step(tok_res, curr_state)
            sync_fn()
        res_decode_s = time.perf_counter() - t0
        res_ms_per_tok = (res_decode_s / decode_len) * 1000.0

        # Timed decode steps for Attention KV cache
        tok_attn = torch.randn(1, 1, 248, device=device, dtype=dtype)
        attn_ms_per_tok = None
        decode_speedup = float("inf")
        if not attn_oom:
            try:
                sync_fn()
                t0 = time.perf_counter()
                with torch.no_grad():
                    curr_cache = attn_cache
                    for step_i in range(decode_len):
                        _, curr_cache = attn_block(tok_attn, kv_cache=curr_cache, start_pos=seq_len + step_i)
                    sync_fn()
                attn_decode_s = time.perf_counter() - t0
                attn_ms_per_tok = (attn_decode_s / decode_len) * 1000.0
                decode_speedup = attn_ms_per_tok / max(res_ms_per_tok, 1e-9)
            except RuntimeError:
                attn_oom = True
                attn_ms_per_tok = float("nan")

        attn_prefill_str = f"{attn_prefill_s*1000:<10.2f}ms" if attn_prefill_s else "OOM       "
        attn_dec_str = f"{attn_ms_per_tok:<10.4f}ms" if attn_ms_per_tok and not math.isnan(attn_ms_per_tok) else "OOM       "
        dec_spd_str = f"{decode_speedup:<11.2f}x" if not math.isinf(decode_speedup) else "OOM (inf)  "

        print(
            f"{seq_len:<10} | {res_prefill_s*1000:<10.2f}ms | {attn_prefill_str} | {prefill_speedup:<11.2f}x | "
            f"{res_ms_per_tok:<10.4f}ms | {attn_dec_str} | {dec_spd_str}"
        )

        results.append({
            "seq_len": seq_len,
            "res_prefill_ms": res_prefill_s * 1000.0,
            "attn_prefill_ms": attn_prefill_s * 1000.0 if attn_prefill_s else float("nan"),
            "prefill_speedup": prefill_speedup,
            "res_ms_per_tok": res_ms_per_tok,
            "attn_ms_per_tok": attn_ms_per_tok if attn_ms_per_tok else float("nan"),
            "decode_speedup": decode_speedup,
        })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_latency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return results


def benchmark_kernel_tail(seq_lengths: List[int] = [2048, 8192, 16384]) -> List[Dict]:
    """Audits the kernel-tail benchmark (FFT convolution vs quadratic reference).

    IMPORTANT: for seq_len > 4096 the quadratic reference is NOT run to
    completion. Only the first 4096 output positions are timed and the result is
    extrapolated by (seq_len / 4096)^2, which is exact for an O(T^2) loop but is
    an extrapolation rather than a measurement. The reported speedups above 4096
    should be read as asymptotic estimates.
    """
    results = []
    print("\n=== Kernel-Tail Microbenchmark Audit ===")
    print(f"{'Seq Len':<10} | {'FFT Conv (ms)':<15} | {'Quadratic Ref (ms)':<20} | {'Speedup':<10}")
    print("-" * 65)

    for seq_len in seq_lengths:
        u = torch.randn(1, 8, 32, seq_len, dtype=torch.float32)
        k = torch.randn(8, seq_len, dtype=torch.float32)

        # 1. FFT convolution
        t0 = time.perf_counter()
        n_fft = 2 * seq_len
        u_f = torch.fft.rfft(u, n=n_fft, dim=-1)
        k_f = torch.fft.rfft(k.unsqueeze(0).unsqueeze(2), n=n_fft, dim=-1)
        _ = torch.fft.irfft(u_f * k_f, n=n_fft, dim=-1)[..., :seq_len]
        fft_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Quadratic causal reference (chunked O(T^2))
        # NOTE: truncated at 4096 positions and extrapolated below. See docstring.
        t0 = time.perf_counter()
        y_quad = torch.zeros_like(u)
        # Sample or compute chunked quadratic
        step_chunk = 32
        for i in range(0, min(seq_len, 4096), step_chunk):
            end_i = min(i + step_chunk, seq_len)
            for idx in range(i, end_i):
                k_rev = torch.flip(k[:, : idx + 1], dims=[-1]).unsqueeze(0).unsqueeze(2)
                y_quad[..., idx] = (u[..., : idx + 1] * k_rev).sum(dim=-1)
        # Scale to full sequence if truncated
        quad_elapsed = time.perf_counter() - t0
        if seq_len > 4096:
            quad_ms = quad_elapsed * ((seq_len / 4096.0) ** 2) * 1000.0
        else:
            quad_ms = quad_elapsed * 1000.0

        speedup = quad_ms / max(fft_ms, 1e-9)
        extrapolated = seq_len > 4096
        flag = " (extrapolated)" if extrapolated else ""
        print(f"{seq_len:<10} | {fft_ms:<15.3f} | {quad_ms:<20.2f} | {speedup:<10.1f}x{flag}")

        results.append({
            "seq_len": seq_len,
            "fft_ms": fft_ms,
            "quad_ms": quad_ms,
            "speedup": speedup,
            "quad_extrapolated": extrapolated,
        })

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark_kernel_tail.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["all", "memory", "latency", "kernel"])
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    if args.mode in ["all", "memory"]:
        benchmark_memory(device_str=args.device)
    if args.mode in ["all", "latency"]:
        benchmark_latency(device_str=args.device)
    if args.mode in ["all", "kernel"]:
        benchmark_kernel_tail()
