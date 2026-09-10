import math
from typing import Dict
import torch
import torch.nn as nn
from src.data import CharDataStream, compute_metrics


@torch.no_grad()
def evaluate_stream(model: nn.Module, stream: CharDataStream) -> Dict[str, float]:
    """Evaluates cross-entropy loss, perplexity, BPC, and next-character accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for step_idx in range(len(stream)):
        x, y = stream.get_batch(step_idx)
        # Full sequence forward pass
        logits, _ = model(x) if hasattr(model, "pos_embed") else (model(x), None)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        preds = logits.argmax(dim=-1)

        total_loss += loss.item()
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    metrics = compute_metrics(avg_loss, total_correct, total_tokens)
    return metrics
