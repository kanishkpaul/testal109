import math
import unittest
import torch
from src.resonator import ResonantFieldMixer


class TestResonantFieldMixer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 32
        self.batch_size = 2

    def test_direct_vs_fft_float64(self):
        """Direct convolution and FFT convolution must match to float64 machine precision."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            use_coupling=True,
            use_local=True,
        ).to(torch.float64)

        # Assign random non-zero phase to stress-test phase handling
        with torch.no_grad():
            mixer.phase.copy_(torch.randn(self.n_heads, dtype=torch.float64))

        x = torch.randn(self.batch_size, self.seq_len, self.d_model, dtype=torch.float64)
        out_direct = mixer.forward_direct(x)
        out_fft = mixer.forward(x)

        max_err = (out_direct - out_fft).abs().max().item()
        self.assertLess(max_err, 1e-12, f"Direct vs FFT float64 max error too high: {max_err}")

    def test_direct_vs_fft_float32(self):
        """Direct convolution and FFT convolution must match within float32 tolerance."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            use_coupling=True,
            use_local=True,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out_direct = mixer.forward_direct(x)
        out_fft = mixer.forward(x)

        max_err = (out_direct - out_fft).abs().max().item()
        self.assertLess(max_err, 1e-5, f"Direct vs FFT float32 max error too high: {max_err}")

    def test_fft_vs_recurrent_stepping(self):
        """Sequential recurrent stepping must equal full-sequence FFT convolution."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            use_coupling=True,
            use_local=True,
        ).to(torch.float64)

        with torch.no_grad():
            mixer.phase.copy_(torch.randn(self.n_heads, dtype=torch.float64))

        x = torch.randn(self.batch_size, self.seq_len, self.d_model, dtype=torch.float64)
        out_fft = mixer(x)

        state = mixer.init_state(self.batch_size, device=x.device, dtype=x.dtype)
        local_buf = None
        recurrent_outs = []

        for t in range(self.seq_len):
            x_t = x[:, t, :]
            out_t, state, local_buf = mixer.step(x_t, state, local_buf)
            recurrent_outs.append(out_t)

        out_recurrent = torch.stack(recurrent_outs, dim=1)
        max_err = (out_fft - out_recurrent).abs().max().item()
        self.assertLess(max_err, 1e-12, f"FFT vs recurrent max error too high: {max_err}")

    def test_phase_sign_audit(self):
        """Verify that e^{+i phi} matches the kernel cos(omega*t + phi), while e^{-i phi} fails."""
        T = 24
        alpha = 0.2
        omega = 0.5
        phi = 0.8

        t = torch.arange(T, dtype=torch.float64)
        k = torch.exp(-alpha * t) * torch.cos(omega * t + phi)
        u = torch.randn(T, dtype=torch.float64)

        # 1. Direct convolution
        y_conv = torch.zeros(T, dtype=torch.float64)
        for i in range(T):
            for j in range(i + 1):
                y_conv[i] += k[i - j] * u[j]

        # 2. Corrected recurrence: Re(e^{+i phi} * s)
        decay = math.exp(-alpha)
        lam_real = decay * math.cos(omega)
        lam_imag = decay * math.sin(omega)

        s_real, s_imag = 0.0, 0.0
        y_corrected = torch.zeros(T, dtype=torch.float64)
        y_paper_flawed = torch.zeros(T, dtype=torch.float64)

        for i in range(T):
            next_real = (lam_real * s_real - lam_imag * s_imag) + u[i].item()
            next_imag = (lam_real * s_imag + lam_imag * s_real)
            s_real, s_imag = next_real, next_imag

            # Correct: Re(e^{+i phi} s) = cos(phi)*real - sin(phi)*imag
            y_corrected[i] = math.cos(phi) * s_real - math.sin(phi) * s_imag
            # Paper Eq 4: Re(e^{-i phi} s) = cos(phi)*real + sin(phi)*imag
            y_paper_flawed[i] = math.cos(phi) * s_real + math.sin(phi) * s_imag

        err_corrected = (y_conv - y_corrected).abs().max().item()
        err_paper = (y_conv - y_paper_flawed).abs().max().item()

        self.assertLess(err_corrected, 1e-14, f"Corrected phase error: {err_corrected}")
        self.assertGreater(err_paper, 0.5, f"Paper equation should exhibit large phase discrepancy: {err_paper}")

    def test_prefill_and_decode_continuation(self):
        """Prompt prefill followed by recurrent decoding must match full FFT evaluation."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            use_coupling=True,
            use_local=False,  # Test core resonant terminal state
        ).to(torch.float64)

        prompt_len = 16
        decode_len = 16
        total_len = prompt_len + decode_len

        x = torch.randn(self.batch_size, total_len, self.d_model, dtype=torch.float64)
        out_full = mixer(x)

        # Run prefill on prompt
        x_prompt = x[:, :prompt_len]
        s_real, s_imag = mixer.compute_terminal_state(x_prompt)
        state = (s_real, s_imag)

        # Continue autoregressively for decode tokens
        decode_outs = []
        for t in range(prompt_len, total_len):
            x_t = x[:, t]
            out_t, state, _ = mixer.step(x_t, state)
            decode_outs.append(out_t)

        out_decode = torch.stack(decode_outs, dim=1)
        expected_decode = out_full[:, prompt_len:]
        max_err = (out_decode - expected_decode).abs().max().item()
        self.assertLess(max_err, 1e-12, f"Prefill + decode continuation error: {max_err}")

    def test_strict_causality(self):
        """Perturbing future suffix must produce zero change in the past prefix (< 1e-6)."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            use_coupling=True,
            use_local=True,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out1 = mixer(x)

        # Perturb only the second half of the sequence
        half = self.seq_len // 2
        x_perturbed = x.clone()
        x_perturbed[:, half:] += torch.randn_like(x_perturbed[:, half:])
        out2 = mixer(x_perturbed)

        prefix_err = (out1[:, :half] - out2[:, :half]).abs().max().item()
        self.assertLess(prefix_err, 1e-6, f"Causality violation: max prefix error {prefix_err} >= 1e-6")

    def test_parameter_constraints(self):
        """Alpha must remain strictly > alpha_min and omega in (0, pi)."""
        mixer = ResonantFieldMixer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            alpha_min=1e-4,
        )
        # Test extreme values
        mixer.raw_alpha.data.fill_(-100.0)
        self.assertGreaterEqual(mixer.alpha.min().item(), 1e-4 - 1e-7)

        mixer.raw_omega.data.fill_(-100.0)
        self.assertGreater(mixer.omega.min().item(), 0.0)
        mixer.raw_omega.data.fill_(100.0)
        self.assertLess(mixer.omega.max().item(), math.pi)

    def test_half_life_architectural_ceiling(self):
        """No head can exceed t_half = ln2/alpha_min, regardless of training.

        Guards the long-context argument. Reports must not say half-lives are
        "capped at 2048": 2048 is the initialisation ceiling, while the hard
        bound comes from alpha = alpha_min + softplus(raw) > alpha_min.
        """
        alpha_min = 1e-4
        mixer = ResonantFieldMixer(d_model=self.d_model, n_heads=self.n_heads, alpha_min=alpha_min)
        ceiling = math.log(2.0) / alpha_min

        # Drive raw_alpha to -inf: softplus -> 0, so alpha -> alpha_min from above.
        mixer.raw_alpha.data.fill_(-1e4)
        self.assertLess(mixer.half_lives.max().item(), ceiling)
        self.assertAlmostEqual(mixer.half_lives.max().item(), ceiling, delta=1.0)
        self.assertAlmostEqual(ceiling, 6931.47, places=1)

        # At initialisation the modes sit at or below the 2048 init ceiling.
        fresh = ResonantFieldMixer(d_model=self.d_model, n_heads=self.n_heads, max_half_life=2048.0)
        self.assertLessEqual(fresh.half_lives.max().item(), 2048.0 + 1e-3)

        # The 1M-token amplitude argument must hold at BOTH bounds.
        self.assertLess(2.0 ** (-1_000_000 / 2048.0), 1e-146)
        self.assertLess(2.0 ** (-1_000_000 / ceiling), 1e-43)

    def test_constant_recurrent_state_memory(self):
        """Recurrent state size must be independent of sequence length."""
        mixer = ResonantFieldMixer(d_model=self.d_model, n_heads=self.n_heads)
        state1 = mixer.init_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        num_elements = state1[0].numel() + state1[1].numel()
        expected_elements = 1 * self.n_heads * (self.d_model // self.n_heads) * 2
        self.assertEqual(num_elements, expected_elements)
        self.assertEqual(num_elements, self.d_model * 2)

    def test_gradient_direct_vs_fft(self):
        """Gradients of direct and FFT paths must match on input tensor."""
        mixer = ResonantFieldMixer(d_model=16, n_heads=2, use_coupling=False, use_local=False).to(torch.float64)
        x1 = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        y_direct = mixer.forward_direct(x1)
        loss_direct = y_direct.sum()
        loss_direct.backward()

        y_fft = mixer(x2)
        loss_fft = y_fft.sum()
        loss_fft.backward()

        grad_err = (x1.grad - x2.grad).abs().max().item()
        self.assertLess(grad_err, 1e-10, f"Gradient mismatch between direct and FFT: {grad_err}")


if __name__ == "__main__":
    unittest.main()
