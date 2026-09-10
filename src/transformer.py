import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embedding (RoPE) to queries and keys."""
    # q, k: (B, H, T, D_h)
    _, _, t, d_h = q.shape
    d_h_even = d_h - (d_h % 2)
    dim = d_h_even // 2

    positions = torch.arange(start_pos, start_pos + t, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, device=q.device, dtype=q.dtype) / dim))
    freqs = torch.outer(positions, inv_freq)  # (T, dim)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., :dim]
        x2 = x[..., dim:d_h_even]
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q[..., :d_h_even]
    k_rot = k[..., :d_h_even]

    q_emb = torch.cat([q_rot[..., :dim] * cos - q_rot[..., dim:] * sin,
                       q_rot[..., :dim] * sin + q_rot[..., dim:] * cos], dim=-1)
    k_emb = torch.cat([k_rot[..., :dim] * cos - k_rot[..., dim:] * sin,
                       k_rot[..., :dim] * sin + k_rot[..., dim:] * cos], dim=-1)

    if d_h % 2 != 0:
        q_emb = torch.cat([q_emb, q[..., d_h_even:]], dim=-1)
        k_emb = torch.cat([k_emb, k[..., d_h_even:]], dim=-1)

    return q_emb, k_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch SDPA."""

    def __init__(self, d_model: int = 248, n_heads: int = 8, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 248 // 8 = 31
        self.dropout = dropout
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, start_pos=start_pos)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
            new_kv = (k, v)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=(t > 1), dropout_p=self.dropout if self.training else 0.0
            )
        else:
            new_kv = None
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.w_o(attn_out), new_kv


class TransformerBlock(nn.Module):
    """Transformer block with pre-RMSNorm, causal attention, and SwiGLU MLP."""

    def __init__(self, d_model: int = 248, n_heads: int = 8, d_ff: int = 992, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed, new_kv = self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + normed
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class TransformerLM(nn.Module):
    """Matched Transformer Language Model (~6.03M parameters)."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 248,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 992,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        pos_encoding: str = "learned",  # "learned", "rope", or "none"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_encoding = pos_encoding
        use_rope = (pos_encoding == "rope")

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "learned":
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
        else:
            self.pos_embed = None

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope=use_rope) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok_embed.weight, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        b, t = input_ids.shape
        x = self.tok_embed(input_ids)

        if self.pos_embed is not None:
            positions = torch.arange(start_pos, start_pos + t, device=input_ids.device)
            # When sequence exceeds max_seq_len, clamp positions to test out-of-context extrapolation
            positions = torch.clamp(positions, 0, self.max_seq_len - 1)
            x = x + self.pos_embed(positions).unsqueeze(0)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_c = layer(x, kv_cache=layer_cache, start_pos=start_pos)
            if new_c is not None:
                new_caches.append(new_c)

        logits = self.lm_head(self.final_norm(x))
        return logits, (new_caches if kv_caches is not None else None)

    def count_parameters(self) -> dict:
        attn_p = sum(p.numel() for l in self.layers for p in l.attn.parameters())
        mlp_p = sum(p.numel() for l in self.layers for p in l.mlp.parameters())
        norm_p = sum(p.numel() for l in self.layers for p in [l.norm1.weight, l.norm2.weight]) + self.final_norm.weight.numel()
        emb_p = self.tok_embed.weight.numel() + (self.pos_embed.weight.numel() if self.pos_embed else 0)
        head_p = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "embedding": emb_p,
            "mixer/attention": attn_p,
            "ffn": mlp_p,
            "norms": norm_p,
            "output_head": head_p,
        }
