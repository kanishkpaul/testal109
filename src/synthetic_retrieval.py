import csv
import os
from typing import Dict, List


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
            # Only zero out true IEEE-754 underflow (2^-1074 is the smallest
            # subnormal double). Clamping earlier than this would discard real,
            # representable values such as the 2^-488.28 ~ 1.05e-147 amplitude
            # that a 2048-token half-life retains at 1M tokens.
            exponent = -d / hl
            amp = 2.0 ** exponent if exponent > -1074.0 else 0.0
            energy = amp ** 2  # may legitimately underflow to 0.0
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


if __name__ == "__main__":
    half_life_decay_audit()
