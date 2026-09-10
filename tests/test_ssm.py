import unittest
import torch
from src.resonator import ResonantFieldMixer
from src.ssm import DiagonalComplexSSM


class TestSSMEquivalence(unittest.TestCase):

    def test_ssm_numerical_identity(self):
        """DiagonalComplexSSM forward pass must be mathematically identical to ResonantFieldMixer."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 32
        d_model = 64
        n_heads = 4
        head_dim = d_model // n_heads

        mixer = ResonantFieldMixer(
            d_model=d_model,
            n_heads=n_heads,
            use_coupling=False,
            use_local=False,
        ).to(torch.float64)

        with torch.no_grad():
            mixer.phase.copy_(torch.randn(n_heads, dtype=torch.float64))

        ssm = DiagonalComplexSSM(n_heads=n_heads, head_dim=head_dim).to(torch.float64)

        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float64)

        # ResonantFieldMixer output (unprojected core field)
        proj = mixer.in_proj(x)
        u_total, _ = proj.chunk(2, dim=-1)
        u_heads = u_total.view(batch_size, seq_len, n_heads, head_dim)

        k = mixer.get_kernel(seq_len, device=x.device, dtype=torch.float64)
        # Direct causal convolution on heads
        y_res = torch.zeros_like(u_heads)
        for i in range(seq_len):
            lags = torch.arange(i + 1, device=x.device)
            k_slice = k[:, i - lags].t().unsqueeze(0).unsqueeze(-1)
            u_slice = u_heads[:, : i + 1]
            y_res[:, i] = (k_slice * u_slice).sum(dim=1)

        # DiagonalComplexSSM output
        y_ssm = ssm.forward_mapped(
            u=u_heads,
            alpha=mixer.alpha,
            omega=mixer.omega,
            phase=mixer.phase,
        )

        max_err = (y_res - y_ssm).abs().max().item()
        print(f"Max error between ResonantFieldMixer and DiagonalComplexSSM: {max_err:.2e}")
        self.assertLess(max_err, 1e-12, f"SSM and Resonator outputs diverge: {max_err}")


if __name__ == "__main__":
    unittest.main()
