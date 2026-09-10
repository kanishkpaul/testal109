import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def next_power_of_2(n: int) -> int:
    return 1 if n <= 0 else 1 << (n - 1).bit_length()


class ResonantFieldMixer(nn.Module):
    """Causal Resonant Field Mixer with dual execution modes:

    1. Full-sequence FFT convolution for training and prefill (O(T log T)).
    2. Recurrent state stepping for autoregressive decoding (O(1) state memory).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        min_half_life: float = 2.0,
        max_half_life: float = 2048.0,
        lambda_c: float = 1.0,
        local_kernel_size: int = 3,
        use_coupling: bool = True,
        use_local: bool = True,
        alpha_min: float = 1e-4,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.alpha_min = alpha_min
        self.lambda_c = lambda_c
        self.use_coupling = use_coupling
        self.use_local = use_local
        self.local_kernel_size = local_kernel_size

        # Grouped projection producing drive u and gate g
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        # Raw unconstrained parameters for alpha, omega, and phase per head
        self.raw_alpha = nn.Parameter(torch.empty(n_heads))
        self.raw_omega = nn.Parameter(torch.empty(n_heads))
        self.phase = nn.Parameter(torch.empty(n_heads))

        # Cross-head coupling parameter matrix: C = I + (lambda_c / sqrt(H)) * tanh(C_tilde)
        if self.use_coupling:
            self.coupling_raw = nn.Parameter(torch.zeros(n_heads, n_heads))
        else:
            self.register_parameter("coupling_raw", None)

        # Depthwise local lexical convolution with learned channel scale
        if self.use_local and local_kernel_size > 0:
            self.local_conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=local_kernel_size,
                groups=d_model,
                bias=False,
            )
            self.local_scale = nn.Parameter(torch.ones(d_model))
        else:
            self.local_conv = None
            self.local_scale = None

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._init_parameters(min_half_life, max_half_life)

    def _init_parameters(self, min_hl: float, max_hl: float) -> None:
        # Initialize half-lives logarithmically across [min_hl, max_hl]
        half_lives = torch.exp(
            torch.linspace(math.log(min_hl), math.log(max_hl), self.n_heads)
        )
        target_alphas = math.log(2.0) / half_lives

        # Invert softplus so softplus(raw_alpha) + alpha_min = target_alphas
        alphas_sub = torch.clamp(target_alphas - self.alpha_min, min=1e-6)
        with torch.no_grad():
            self.raw_alpha.copy_(torch.log(torch.exp(alphas_sub) - 1.0))

            # Frequencies spanning long-period (2pi / max_hl) to Nyquist (pi / 2)
            target_omegas = torch.exp(
                torch.linspace(
                    math.log(2.0 * math.pi / max_hl),
                    math.log(math.pi * 0.5),
                    self.n_heads,
                )
            )
            # Invert sigmoid: omega = pi * sigmoid(raw_omega)
            norm_omegas = torch.clamp(target_omegas / math.pi, min=1e-5, max=1.0 - 1e-5)
            self.raw_omega.copy_(torch.logit(norm_omegas))

            self.phase.zero_()
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.local_conv is not None:
                nn.init.normal_(self.local_conv.weight, std=0.02)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_min + F.softplus(self.raw_alpha)

    @property
    def omega(self) -> torch.Tensor:
        sig = torch.sigmoid(self.raw_omega).clamp(min=1e-6, max=1.0 - 1e-6)
        return math.pi * sig

    @property
    def half_lives(self) -> torch.Tensor:
        return math.log(2.0) / self.alpha

    def get_kernel(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Computes k_h[t] = exp(-alpha_h * t) * cos(omega_h * t + phi_h)."""
        t = torch.arange(length, device=device, dtype=dtype)
        decay = torch.exp(-self.alpha.unsqueeze(1) * t.unsqueeze(0))
        oscillation = torch.cos(self.omega.unsqueeze(1) * t.unsqueeze(0) + self.phase.unsqueeze(1))
        return decay * oscillation

    def _apply_coupling(self, y: torch.Tensor) -> torch.Tensor:
        """Applies cross-head coupling C = I + (lambda_c / sqrt(H)) * tanh(C_tilde)."""
        if not self.use_coupling or self.coupling_raw is None:
            return y
        c_matrix = torch.eye(self.n_heads, device=y.device, dtype=y.dtype) + (
            self.lambda_c / math.sqrt(self.n_heads)
        ) * torch.tanh(self.coupling_raw)
        return torch.einsum("gh, bthd -> btgd", c_matrix, y)

    def _apply_local_conv(self, u: torch.Tensor) -> torch.Tensor:
        """Applies causal depthwise convolution over sequence length."""
        if self.local_conv is None:
            return torch.zeros_like(u)
        # u: (B, T, D) -> conv1d input: (B, D, T)
        u_t = u.transpose(1, 2)
        # Pad on left by K - 1 to preserve strict causality
        u_pad = F.pad(u_t, (self.local_kernel_size - 1, 0))
        local_out = self.local_conv(u_pad).transpose(1, 2)
        return self.local_scale * local_out

    def forward_direct(self, x: torch.Tensor) -> torch.Tensor:
        """Slow O(T^2) reference causal convolution for numerical equivalence audits."""
        b, t_seq, d = x.shape
        proj = self.in_proj(x)
        u_total, g = proj.chunk(2, dim=-1)
        gamma = torch.sigmoid(g)

        u_heads = u_total.view(b, t_seq, self.n_heads, self.head_dim)
        k = self.get_kernel(t_seq, device=x.device, dtype=x.dtype)

        y = torch.zeros_like(u_heads)
        for i in range(t_seq):
            lags = torch.arange(i + 1, device=x.device)
            # k is (H, T); k[:, i - lags] is (H, i + 1); transpose to (i + 1, H)
            k_slice = k[:, i - lags].t().unsqueeze(0).unsqueeze(-1)
            u_slice = u_heads[:, : i + 1]
            y[:, i] = (k_slice * u_slice).sum(dim=1)

        y_coupled = self._apply_coupling(y).reshape(b, t_seq, d)
        z = gamma * y_coupled
        if self.use_local and self.local_conv is not None:
            z = z + self._apply_local_conv(u_total)
        return self.out_proj(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass via causal FFT convolution."""
        b, t_seq, d = x.shape
        proj = self.in_proj(x)
        u_total, g = proj.chunk(2, dim=-1)
        gamma = torch.sigmoid(g)

        u = u_total.view(b, t_seq, self.n_heads, self.head_dim)
        k = self.get_kernel(t_seq, device=x.device, dtype=x.dtype)

        # Pad to >= 2 * T to guarantee linear rather than circular convolution
        n_fft = next_power_of_2(2 * t_seq)

        u_perm = u.permute(0, 2, 3, 1)  # (B, H, D_h, T)
        k_expanded = k.unsqueeze(0).unsqueeze(2)  # (1, H, 1, T)

        u_f = torch.fft.rfft(u_perm, n=n_fft, dim=-1)
        k_f = torch.fft.rfft(k_expanded, n=n_fft, dim=-1)

        y_conv = torch.fft.irfft(u_f * k_f, n=n_fft, dim=-1)[..., :t_seq]
        y = y_conv.permute(0, 3, 1, 2)  # (B, T, H, D_h)

        y_coupled = self._apply_coupling(y).reshape(b, t_seq, d)
        z = gamma * y_coupled
        if self.use_local and self.local_conv is not None:
            z = z + self._apply_local_conv(u_total)
        return self.out_proj(z)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes complex recurrent state (real, imag) of shape (B, H, D_h)."""
        real = torch.zeros(batch_size, self.n_heads, self.head_dim, device=device, dtype=dtype)
        imag = torch.zeros(batch_size, self.n_heads, self.head_dim, device=device, dtype=dtype)
        return real, imag

    def step(
        self,
        x_step: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        local_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """Single-token autoregressive recurrent decoding step.

        Uses the mathematically verified phase convention: y = Re(e^{+i phi} * s).
        """
        b, d = x_step.shape
        proj = self.in_proj(x_step)
        u_step, g_step = proj.chunk(2, dim=-1)
        gamma = torch.sigmoid(g_step)

        u_h = u_step.view(b, self.n_heads, self.head_dim)
        s_real, s_imag = state

        # lambda = exp(-alpha + i * omega) = exp(-alpha) * (cos(omega) + i * sin(omega))
        decay = torch.exp(-self.alpha).view(1, self.n_heads, 1)
        cos_w = torch.cos(self.omega).view(1, self.n_heads, 1)
        sin_w = torch.sin(self.omega).view(1, self.n_heads, 1)

        lam_real = decay * cos_w
        lam_imag = decay * sin_w

        # s_{t+1} = lambda * s_t + u_t
        next_real = (lam_real * s_real - lam_imag * s_imag) + u_h
        next_imag = (lam_real * s_imag + lam_imag * s_real)

        # Output projection: Re(e^{+i * phi} * s_{t+1}) = cos(phi) * real - sin(phi) * imag
        cos_phi = torch.cos(self.phase).view(1, self.n_heads, 1)
        sin_phi = torch.sin(self.phase).view(1, self.n_heads, 1)
        y_h = cos_phi * next_real - sin_phi * next_imag

        y = y_h.unsqueeze(1)  # (B, 1, H, D_h)
        y_coupled = self._apply_coupling(y).reshape(b, d)
        z = gamma * y_coupled

        # Update local depthwise conv buffer if enabled
        if self.use_local and self.local_conv is not None:
            if local_buffer is None:
                local_buffer = torch.zeros(b, d, self.local_kernel_size, device=x_step.device, dtype=x_step.dtype)
            local_buffer = torch.cat([local_buffer[:, :, 1:], u_step.unsqueeze(-1)], dim=-1)
            conv_w = self.local_conv.weight.squeeze(1)  # (D, K)
            l_t = (local_buffer * conv_w.unsqueeze(0)).sum(dim=-1)
            z = z + self.local_scale * l_t

        out = self.out_proj(z)
        return out, (next_real, next_imag), local_buffer

    def compute_terminal_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes terminal recurrent state s_T analytically after full prompt prefill."""
        b, t_seq, _ = x.shape
        proj = self.in_proj(x)
        u_total, _ = proj.chunk(2, dim=-1)
        u = u_total.view(b, t_seq, self.n_heads, self.head_dim)

        # s_T = sum_{tau=0}^{T-1} lambda^{T - 1 - tau} * u_tau
        lags = torch.arange(t_seq, device=x.device, dtype=x.dtype)
        # time from tau to T - 1 is (T - 1 - tau)
        rev_lags = t_seq - 1 - lags

        decay = torch.exp(-self.alpha.unsqueeze(1) * rev_lags.unsqueeze(0))
        cos_term = torch.cos(self.omega.unsqueeze(1) * rev_lags.unsqueeze(0))
        sin_term = torch.sin(self.omega.unsqueeze(1) * rev_lags.unsqueeze(0))

        lam_power_real = (decay * cos_term).unsqueeze(0).unsqueeze(-1)  # (1, H, T, 1)
        lam_power_imag = (decay * sin_term).unsqueeze(0).unsqueeze(-1)

        u_perm = u.permute(0, 2, 1, 3)  # (B, H, T, D_h)
        s_real = (lam_power_real * u_perm).sum(dim=2)
        s_imag = (lam_power_imag * u_perm).sum(dim=2)
        return s_real, s_imag
