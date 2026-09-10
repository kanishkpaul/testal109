import math
from typing import Tuple
import torch
import torch.nn as nn


class DiagonalComplexSSM(nn.Module):
    """Generic 1D Complex / 2D Real Diagonal State-Space Model.

    State update:
        x_{t+1} = A x_t + B u_t
    Output:
        y_{t+1} = C x_{t+1} + D u_t

    For ResonatorLM equivalence:
        A = exp(-alpha) * [[cos(omega), -sin(omega)],
                           [sin(omega),  cos(omega)]]
        B = [1, 0]^T
        C = [cos(phi), -sin(phi)]
        D = 0
    """

    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward_mapped(
        self,
        u: torch.Tensor,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Runs 2D real linear dynamical system on input drive u of shape (B, T, H, D_h)."""
        b, t_seq, h, dh = u.shape
        device = u.device
        dtype = u.dtype

        # Decay and rotation components
        decay = torch.exp(-alpha).view(1, 1, h, 1)
        cos_w = torch.cos(omega).view(1, 1, h, 1)
        sin_w = torch.sin(omega).view(1, 1, h, 1)
        cos_p = torch.cos(phase).view(1, 1, h, 1)
        sin_p = torch.sin(phase).view(1, 1, h, 1)

        # Recurrent execution
        a_t = torch.zeros(b, 1, h, dh, device=device, dtype=dtype)
        b_t = torch.zeros(b, 1, h, dh, device=device, dtype=dtype)

        y_steps = []
        for t in range(t_seq):
            u_t = u[:, t : t + 1]  # (B, 1, H, D_h)
            # x_{t+1} = A * x_t + B * u_t
            next_a = decay * (cos_w * a_t - sin_w * b_t) + u_t
            next_b = decay * (sin_w * a_t + cos_w * b_t)
            a_t, b_t = next_a, next_b

            # y_{t+1} = C * x_{t+1}
            y_t = cos_p * a_t - sin_p * b_t
            y_steps.append(y_t)

        return torch.cat(y_steps, dim=1)


class GenericDiagonalSSMHead(nn.Module):
    """Unconstrained Diagonal State-Space Model baseline.

    Learns unconstrained complex pole lambda in polar form (r, theta)
    and unconstrained real/imaginary output projection weights.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        # Unconstrained log-magnitude and angle
        self.log_r = nn.Parameter(torch.empty(1))
        self.theta = nn.Parameter(torch.empty(1))
        # Unconstrained C projection weights
        self.c_real = nn.Parameter(torch.randn(1))
        self.c_imag = nn.Parameter(torch.randn(1))
        self._init_params()

    def _init_params(self):
        with torch.no_grad():
            self.log_r.fill_(-0.1)  # r ~ 0.90
            self.theta.fill_(0.5)

    @property
    def pole(self) -> complex:
        r = torch.sigmoid(self.log_r).item()
        th = self.theta.item()
        return complex(r * math.cos(th), r * math.sin(th))
