import json
import math
import os
from typing import Dict, List, Optional, Tuple
import torch


class CharacterTokenizer:
    """Character-level tokenizer with deterministic vocab mapping."""

    def __init__(self, char_to_id: Optional[Dict[str, int]] = None):
        self.char_to_id = char_to_id or {}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.unk_token = "<unk>"
        self.unk_id = self.char_to_id.get(self.unk_token, 0)

    @classmethod
    def build_from_text(cls, text: str) -> "CharacterTokenizer":
        # Frequency-sorted unique characters
        char_counts: Dict[str, int] = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        sorted_chars = sorted(char_counts.keys(), key=lambda c: char_counts[c], reverse=True)

        char_to_id = {"<unk>": 0}
        for c in sorted_chars:
            if c != "<unk>":
                char_to_id[c] = len(char_to_id)
        return cls(char_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, self.unk_token) for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            char_to_id = json.load(f)
        return cls(char_to_id)


def load_wikitext_split(split: str, data_dir: str = "data/wikitext-2-raw") -> str:
    filename = f"wiki.{split}.raw"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"WikiText split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataStream:
    """Pre-tokenized continuous tensor stream chunked into batches of (seq_len + 1)."""

    def __init__(self, token_ids: List[int], batch_size: int, seq_len: int, device: torch.device):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        # Total complete sequences per batch column
        n_tokens = len(token_ids)
        tokens_per_batch = n_tokens // batch_size
        trimmed = token_ids[: tokens_per_batch * batch_size]
        # Reshape to (batch_size, tokens_per_batch)
        self.data = torch.tensor(trimmed, dtype=torch.long).view(batch_size, tokens_per_batch)
        self.n_steps = (tokens_per_batch - 1) // seq_len

    def get_batch(self, step_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = (step_idx % self.n_steps) * self.seq_len
        x = self.data[:, start : start + self.seq_len].to(self.device, non_blocking=True)
        y = self.data[:, start + 1 : start + 1 + self.seq_len].to(self.device, non_blocking=True)
        return x, y

    def __len__(self) -> int:
        return self.n_steps


def compute_metrics(loss: float, correct: int, total_tokens: int) -> dict:
    """Computes test loss, perplexity, bits-per-character (BPC), and top-1 accuracy."""
    ppl = math.exp(min(loss, 100.0))
    bpc = loss / math.log(2.0)
    acc = (correct / total_tokens * 100.0) if total_tokens > 0 else 0.0
    return {
        "loss": loss,
        "ppl": ppl,
        "bpc": bpc,
        "acc": acc,
        "tokens": total_tokens,
    }
