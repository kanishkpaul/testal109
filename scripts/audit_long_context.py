"""Adversarial Audit: Long-Context Investigation (256, 512, 1024).

Investigates whether the paper's reported Transformer collapse (PPL jumping from 5.06 to 8.70 at 512 and 11.02 at 1024)
is caused by fundamental attention limitations, or by evaluating learned absolute positional embeddings beyond
their trained context length, or undertuned baseline positional encoding.
"""

import csv
import os
import torch
from src.data import CharacterTokenizer, CharDataStream, load_wikitext_split
from src.evaluate import evaluate_stream
from src.resonator_lm import ResonatorLM
from src.train import train_model
from src.transformer import TransformerLM


def run_long_context_audit(device_str: str = "mps", steps: int = 1500):
    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== Long-Context Audit: Evaluating Transformer Degradation on {device} ===")

    tokenizer = CharacterTokenizer.load("data/char_vocab.json")
    test_text = load_wikitext_split("test")

    results = []

    # 1. Train Paper Baseline: Transformer with learned absolute PE (seq_len=256)
    print("\n--- 1. Training Transformer (Learned PE, seq_len=256) ---")
    trans_learned = train_model(
        model_type="transformer",
        pos_encoding="learned",
        seed=0,
        steps=steps,
        batch_size=32,
        seq_len=256,
        output_dir="results/audit_lc",
    )

    # 2. Train Modern Baseline: Transformer with RoPE (seq_len=256)
    print("\n--- 2. Training Transformer (RoPE, seq_len=256) ---")
    trans_rope = train_model(
        model_type="transformer",
        pos_encoding="rope",
        seed=0,
        steps=steps,
        batch_size=32,
        seq_len=256,
        output_dir="results/audit_lc",
    )

    # 3. Train ResonatorLM (seq_len=256)
    print("\n--- 3. Training ResonatorLM (seq_len=256) ---")
    res_model = train_model(
        model_type="resonator",
        preset="balanced",
        seed=0,
        steps=steps,
        batch_size=32,
        seq_len=256,
        output_dir="results/audit_lc",
    )

    # Load saved checkpoints for multi-context evaluation
    ckpt_trans_learned = torch.load("checkpoints/transformer_balanced_learned_s0.pt", map_location=device)
    model_tl = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=248, max_seq_len=256, pos_encoding="learned").to(device)
    model_tl.load_state_dict(ckpt_trans_learned["model_state"])

    ckpt_trans_rope = torch.load("checkpoints/transformer_balanced_rope_s0.pt", map_location=device)
    model_tr = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=248, max_seq_len=256, pos_encoding="rope").to(device)
    model_tr.load_state_dict(ckpt_trans_rope["model_state"])

    ckpt_res = torch.load("checkpoints/resonator_balanced_learned_s0.pt", map_location=device)
    model_res = ResonatorLM(vocab_size=tokenizer.vocab_size, d_model=256).to(device)
    model_res.load_state_dict(ckpt_res["model_state"])

    print("\n=== Context Extrapolation Evaluation (Trained on 256) ===")
    print(f"{'Model':<22} | {'Pos Enc':<10} | {'Eval Len':<10} | {'Test PPL':<12} | {'Test Acc (%)':<12}")
    print("-" * 75)

    test_tokens = tokenizer.encode(test_text)
    eval_lengths = [256, 512, 1024]

    eval_configs = [
        ("Transformer (Paper)", "learned", model_tl),
        ("Transformer (RoPE)", "rope", model_tr),
        ("ResonatorLM", "none", model_res),
    ]

    for name, pos_type, m in eval_configs:
        for l in eval_lengths:
            test_stream = CharDataStream(test_tokens, batch_size=16, seq_len=l, device=device)
            metrics = evaluate_stream(m, test_stream)
            print(f"{name:<22} | {pos_type:<10} | {l:<10} | {metrics['ppl']:<12.4f} | {metrics['acc']:<12.2f}%")
            results.append({
                "model": name,
                "pos_encoding": pos_type,
                "eval_seq_len": l,
                "test_ppl": metrics["ppl"],
                "test_acc": metrics["acc"],
            })

    os.makedirs("results", exist_ok=True)
    with open("results/long_context_audit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_long_context_audit()
