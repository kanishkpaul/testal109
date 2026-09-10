import csv
import os
import torch
from src.data import CharacterTokenizer, CharDataStream, load_wikitext_split
from src.evaluate import evaluate_stream
from src.resonator_lm import ResonatorLM
from src.transformer import TransformerLM


def eval_context_sweep():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = CharacterTokenizer.load("data/char_vocab.json")
    test_text = load_wikitext_split("test")
    test_tokens = tokenizer.encode(test_text)

    # 1. Load Transformer (Learned PE)
    ckpt_tl = torch.load("checkpoints/transformer_balanced_learned_s0.pt", map_location=device)
    model_tl = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=248, max_seq_len=256, pos_encoding="learned").to(device)
    model_tl.load_state_dict(ckpt_tl["model_state"])

    # 2. Load Transformer (RoPE)
    ckpt_tr = torch.load("checkpoints/transformer_balanced_rope_s0.pt", map_location=device)
    model_tr = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=248, max_seq_len=256, pos_encoding="rope").to(device)
    model_tr.load_state_dict(ckpt_tr["model_state"])

    # 3. Load ResonatorLM
    ckpt_res = torch.load("checkpoints/resonator_balanced_learned_s0.pt", map_location=device)
    model_res = ResonatorLM(vocab_size=tokenizer.vocab_size, d_model=256).to(device)
    model_res.load_state_dict(ckpt_res["model_state"])

    models = [
        ("Transformer (Learned PE - Paper)", model_tl),
        ("Transformer (RoPE - Strong Baseline)", model_tr),
        ("ResonatorLM (Paper Model)", model_res),
    ]

    seq_lengths = [256, 512, 1024]
    results = []

    print("\n=== Multi-Context Evaluation on Test Set (All Trained on seq_len=256) ===")
    print(f"{'Model':<38} | {'Context':<8} | {'Test PPL':<12} | {'Test BPC':<10} | {'Test Acc (%)':<12}")
    print("-" * 88)

    for name, m in models:
        for seq_len in seq_lengths:
            stream = CharDataStream(test_tokens, batch_size=16, seq_len=seq_len, device=device)
            m_res = evaluate_stream(m, stream)
            print(f"{name:<38} | {seq_len:<8} | {m_res['ppl']:<12.4f} | {m_res['bpc']:<10.4f} | {m_res['acc']:<12.2f}%")
            results.append({"model": name, "context": seq_len, "ppl": m_res["ppl"], "bpc": m_res["bpc"], "acc": m_res["acc"]})

    os.makedirs("results", exist_ok=True)
    with open("results/context_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    eval_context_sweep()
