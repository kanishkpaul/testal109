from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from src.resonator import ResonantFieldMixer
from src.transformer import RMSNorm, SwiGLU


class ResonatorBlock(nn.Module):
    """ResonatorLM Block combining ResonantFieldMixer and SwiGLU MLP."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        lambda_c: float = 1.0,
        local_kernel_size: int = 3,
        use_coupling: bool = True,
        use_local: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = ResonantFieldMixer(
            d_model=d_model,
            n_heads=n_heads,
            lambda_c=lambda_c,
            local_kernel_size=local_kernel_size,
            use_coupling=use_coupling,
            use_local=use_local,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(
        self,
        x_step: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        local_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        normed = self.norm1(x_step)
        mix_out, next_state, next_buf = self.mixer.step(normed, state, local_buffer)
        x_step = x_step + mix_out
        x_step = x_step + self.mlp(self.norm2(x_step))
        return x_step, next_state, next_buf


class ResonatorLM(nn.Module):
    """Matched ResonatorLM Language Model (~6.04M parameters)."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        lambda_c: float = 1.0,
        local_kernel_size: int = 3,
        use_coupling: bool = True,
        use_local: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ResonatorBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    lambda_c=lambda_c,
                    local_kernel_size=local_kernel_size,
                    use_coupling=use_coupling,
                    use_local=use_local,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass via causal FFT convolution."""
        x = self.tok_embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.final_norm(x))

    def init_states(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        states = [layer.mixer.init_state(batch_size, device, dtype) for layer in self.layers]
        buffers = [None for _ in self.layers]
        return states, buffers

    def step(
        self,
        token_ids: torch.Tensor,
        states: List[Tuple[torch.Tensor, torch.Tensor]],
        buffers: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Single-token recurrent generation step."""
        x = self.tok_embed(token_ids)
        next_states = []
        next_buffers = []
        for i, layer in enumerate(self.layers):
            x, next_s, next_b = layer.step(x, states[i], buffers[i])
            next_states.append(next_s)
            next_buffers.append(next_b)
        logits = self.lm_head(self.final_norm(x))
        return logits, next_states, next_buffers

    def count_parameters(self) -> dict:
        mixer_p = sum(p.numel() for l in self.layers for p in l.mixer.parameters())
        mlp_p = sum(p.numel() for l in self.layers for p in l.mlp.parameters())
        norm_p = sum(p.numel() for l in self.layers for p in [l.norm1.weight, l.norm2.weight]) + self.final_norm.weight.numel()
        emb_p = self.tok_embed.weight.numel()
        head_p = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "embedding": emb_p,
            "mixer/attention": mixer_p,
            "ffn": mlp_p,
            "norms": norm_p,
            "output_head": head_p,
        }
