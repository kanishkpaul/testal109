import argparse
import csv
import json
import math
import os
import time
from typing import Dict, Optional
import torch
import torch.nn as nn
from src.data import CharacterTokenizer, CharDataStream, load_wikitext_split, compute_metrics
from src.evaluate import evaluate_stream
from src.resonator_lm import ResonatorLM
from src.transformer import TransformerLM


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step) / float(max(1, warmup_steps))
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def extract_physics_diagnostics(model: nn.Module) -> Dict[str, float]:
    """Extracts half-life ranges and causality prefix errors from ResonatorLM."""
    if not isinstance(model, ResonatorLM):
        return {}
    all_half_lives = []
    for layer in model.layers:
        hl = layer.mixer.half_lives.detach().cpu()
        all_half_lives.extend(hl.tolist())

    min_hl = min(all_half_lives)
    max_hl = max(all_half_lives)

    # Measure numerical causality on first layer
    first_mixer = model.layers[0].mixer
    x = torch.randn(1, 64, first_mixer.d_model, device=next(first_mixer.parameters()).device)
    y1 = first_mixer(x)
    x_pert = x.clone()
    x_pert[:, 32:] += torch.randn_like(x_pert[:, 32:])
    y2 = first_mixer(x_pert)
    max_prefix_err = (y1[:, :32] - y2[:, :32]).abs().max().item()

    return {
        "min_half_life": min_hl,
        "max_half_life": max_hl,
        "max_prefix_error": max_prefix_err,
    }


def train_model(
    model_type: str = "resonator",
    preset: str = "balanced",
    pos_encoding: str = "learned",
    seed: int = 0,
    steps: int = 10000,
    batch_size: int = 32,
    seq_len: int = 256,
    lr: float = 5e-4,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_interval: int = 1000,
    device_str: Optional[str] = None,
    output_dir: str = "results",
) -> Dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    if device_str is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Training on device: {device} | Model: {model_type} ({preset}) | Seed: {seed} | Steps: {steps}")

    # Load data and tokenizer
    tokenizer = CharacterTokenizer.load("data/char_vocab.json")
    vocab_size = tokenizer.vocab_size

    train_text = load_wikitext_split("train")
    valid_text = load_wikitext_split("valid")
    test_text = load_wikitext_split("test")

    train_stream = CharDataStream(tokenizer.encode(train_text), batch_size=batch_size, seq_len=seq_len, device=device)
    valid_stream = CharDataStream(tokenizer.encode(valid_text), batch_size=batch_size, seq_len=seq_len, device=device)
    test_stream = CharDataStream(tokenizer.encode(test_text), batch_size=batch_size, seq_len=seq_len, device=device)

    # Build model
    if model_type == "resonator":
        use_coupling = (preset in ["balanced", "no_local"])
        use_local = (preset in ["balanced", "no_coupling"])
        model = ResonatorLM(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            lambda_c=1.0,
            local_kernel_size=3,
            use_coupling=use_coupling,
            use_local=use_local,
        ).to(device)
    elif model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=248,
            n_layers=6,
            n_heads=8,
            d_ff=992,
            max_seq_len=max(seq_len, 256),
            pos_encoding=pos_encoding,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    param_info = model.count_parameters()
    total_params = param_info["total"]
    print(f"Model {model_type} parameter count: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    running_loss = 0.0
    running_tokens = 0
    t0 = time.perf_counter()
    tokens_per_step = batch_size * seq_len

    val_history = []

    while step < steps:
        step += 1
        current_lr = get_cosine_lr(step, steps, warmup_steps, lr, lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x, y = train_stream.get_batch(step - 1)
        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(x) if hasattr(model, "pos_embed") else (model(x), None)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        running_tokens += tokens_per_step

        if step % eval_interval == 0 or step == steps:
            val_metrics = evaluate_stream(model, valid_stream)
            model.train()
            elapsed = time.perf_counter() - t0
            throughput = running_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:5d}/{steps} | Train Loss: {loss.item():.4f} | "
                f"Val PPL: {val_metrics['ppl']:.3f} | Val Acc: {val_metrics['acc']:.2f}% | "
                f"Speed: {throughput:.1f} tok/s | LR: {current_lr:.6f}"
            )
            val_history.append((step, val_metrics))

    total_time = time.perf_counter() - t0
    avg_throughput = (steps * tokens_per_step) / total_time

    # Final evaluation on Test Set
    test_metrics = evaluate_stream(model, test_stream)
    diagnostics = extract_physics_diagnostics(model)

    results = {
        "model": model_type,
        "preset": preset,
        "pos_encoding": pos_encoding,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "params": total_params,
        "test_loss": test_metrics["loss"],
        "test_ppl": test_metrics["ppl"],
        "test_bpc": test_metrics["bpc"],
        "test_acc": test_metrics["acc"],
        "train_tok_s": avg_throughput,
        "wall_clock_s": total_time,
        "min_half_life": diagnostics.get("min_half_life", 0.0),
        "max_half_life": diagnostics.get("max_half_life", 0.0),
        "max_prefix_error": diagnostics.get("max_prefix_error", 0.0),
        "device": str(device),
    }

    print(f"\nFinal Test Results for {model_type} (seed {seed}):")
    print(f"  Test PPL : {results['test_ppl']:.4f}")
    print(f"  Test BPC : {results['test_bpc']:.4f}")
    print(f"  Test Acc : {results['test_acc']:.2f}%")
    print(f"  Throughput: {results['train_tok_s']:.1f} tok/s")
    if diagnostics:
        print(f"  Half-life span: [{results['min_half_life']:.1f}, {results['max_half_life']:.1f}] tokens")
        print(f"  Max prefix error: {results['max_prefix_error']:.2e}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{model_type}_{preset}_s{seed}.pt"
    torch.save({"model_state": model.state_dict(), "results": results}, ckpt_path)

    # Append to master results.csv
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    # Append to results.jsonl
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resonator", choices=["resonator", "transformer"])
    parser.add_argument("--preset", type=str, default="balanced", choices=["balanced", "no_coupling", "no_local", "no_both"])
    parser.add_argument("--pos_encoding", type=str, default="learned", choices=["learned", "rope"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        preset=args.preset,
        pos_encoding=args.pos_encoding,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        eval_interval=args.eval_interval,
        device_str=args.device,
        output_dir=args.output_dir,
    )
