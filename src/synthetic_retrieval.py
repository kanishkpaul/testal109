import csv
import math
import os
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from src.resonator_lm import ResonatorLM
from src.transformer import TransformerLM


def half_life_decay_audit(
    half_lives: List[float] = [2.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 2048.0],
    distances: List[int] = [2048, 8192, 32768, 100000, 1000000],
) -> List[Dict]:
    """Quantifies amplitude and energy retention of linear resonant modes over long contexts."""
    results = []
    print("\n=== Theoretical Half-Life Decay Audit ===")
    header = f"{'Half-life':<12} | " + " | ".join([f"{d:>10} toks" for d in distances])
    print(header)
    print("-" * len(header))

    for hl in half_lives:
        row = {"half_life": hl}
        row_str = f"{hl:<12.1f} | "
        for d in distances:
            # Amplitude decay: A(d) = 2^(-d / hl)
            exponent = -d / hl
            if exponent < -100:
                amp = 0.0
                energy = 0.0
            else:
                amp = 2.0 ** exponent
                energy = amp ** 2
            row[f"amp_{d}"] = amp
            row[f"energy_{d}"] = energy

            if amp == 0.0 or amp < 1e-15:
                val_str = "  ~0.0    "
            elif amp < 1e-3:
                val_str = f"{amp:10.2e}"
            else:
                val_str = f"{amp*100:9.2f}%"
            row_str += val_str + " | "
        print(row_str)
        results.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/half_life_decay_audit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return results


def generate_passkey_dataset(
    n_samples: int,
    distance: int,
    passkey_len: int = 5,
    vocab_size: int = 64,
) -> List[Tuple[torch.Tensor, int]]:
    """Generates synthetic passkey prompt:
    'KEY: <passkey> DISTRACTORS ... QUERY: <first_digit> -> TARGET'
    """
    dataset = []
    for _ in range(n_samples):
        # Generate passkey digits (e.g. tokens 10 to 19)
        passkey = [random.randint(10, 19) for _ in range(passkey_len)]
        target_char = passkey[0]

        # Distractor tokens (e.g. tokens 20 to 50)
        distractors = [random.randint(20, 50) for _ in range(distance)]

        # Prompt format: [Prefix 1, 2] + passkey + distractors + [Query token 3]
        prefix = [1, 2]
        query = [3]
        tokens = prefix + passkey + distractors + query
        dataset.append((torch.tensor(tokens, dtype=torch.long), target_char))
    return dataset


def evaluate_passkey_retrieval(
    model: nn.Module,
    distances: List[int] = [64, 128, 256, 512, 1024, 2048],
    n_samples: int = 50,
    device: torch.device = torch.device("cpu"),
) -> List[Dict]:
    """Tests model retrieval accuracy as distractor distance increases."""
    model.eval()
    results = []

    print("\n=== Passkey Retrieval vs Distance ===")
    print(f"{'Distance':<12} | {'Accuracy (%)':<15} | {'Samples':<10}")
    print("-" * 45)

    for dist in distances:
        dataset = generate_passkey_dataset(n_samples=n_samples, distance=dist, vocab_size=model.vocab_size)
        correct = 0

        with torch.no_grad():
            for tokens, target in dataset:
                tokens = tokens.unsqueeze(0).to(device)
                logits = model(tokens)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred = logits[0, -1].argmax().item()
                if pred == target:
                    correct += 1

        acc = (correct / n_samples) * 100.0
        print(f"{dist:<12} | {acc:<15.1f} | {n_samples:<10}")
        results.append({"distance": dist, "accuracy": acc, "samples": n_samples})

    return results


if __name__ == "__main__":
    half_life_decay_audit()
