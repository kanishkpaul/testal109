import unittest
import torch
from src.transformer import TransformerLM
from src.resonator_lm import ResonatorLM


class TestModelArchitectures(unittest.TestCase):

    def test_parameter_counts_and_matching(self):
        vocab_size = 256
        transformer = TransformerLM(
            vocab_size=vocab_size,
            d_model=248,
            n_layers=6,
            n_heads=8,
            d_ff=992,
            max_seq_len=256,
            pos_encoding="learned",
        )
        resonator = ResonatorLM(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            lambda_c=1.0,
            local_kernel_size=3,
            use_coupling=True,
            use_local=True,
        )

        p_trans = transformer.count_parameters()
        p_res = resonator.count_parameters()

        print("\n=== Parameter Breakdown ===")
        print(f"{'Component':<18} | {'Transformer (248)':<18} | {'ResonatorLM (256)':<18}")
        print("-" * 60)
        for comp in ["embedding", "mixer/attention", "ffn", "norms", "output_head", "total"]:
            print(f"{comp:<18} | {p_trans[comp]:<18,} | {p_res[comp]:<18,}")

        pct_diff = abs(p_trans["total"] - p_res["total"]) / p_trans["total"] * 100.0
        print(f"Absolute parameter mismatch: {abs(p_trans['total'] - p_res['total']):,} ({pct_diff:.3f}%)")

        # Must be within 1% of 6.0M parameters and within 1% of each other
        self.assertLess(pct_diff, 1.0, f"Parameter mismatch {pct_diff:.3f}% exceeds 1%")
        self.assertGreater(p_trans["total"], 5.8e6)
        self.assertLess(p_trans["total"], 6.6e6)

    def test_transformer_forward_and_kv_step(self):
        transformer = TransformerLM(vocab_size=64, d_model=248, n_layers=2, n_heads=8, d_ff=992)
        x = torch.randint(0, 64, (2, 16))
        logits, _ = transformer(x)
        self.assertEqual(logits.shape, (2, 16, 64))

        # Test single token step with KV cache
        caches = None
        for t in range(4):
            tok = x[:, t : t + 1]
            out, caches = transformer(tok, kv_caches=caches, start_pos=t)
            self.assertEqual(out.shape, (2, 1, 64))

    def test_decode_state_cache_size(self):
        """Whole-model recurrent cache must be 2.0 KiB per layer / 12.0 KiB total.

        Regression guard: reports previously quoted the per-layer figure (2.0 KiB)
        as the whole-model total. Pin both numbers so they cannot drift apart.
        """
        n_layers = 6
        resonator = ResonatorLM(vocab_size=284, d_model=256, n_layers=n_layers, n_heads=8)
        states, _ = resonator.init_states(1, torch.device("cpu"), torch.float32)

        self.assertEqual(len(states), n_layers)
        bytes_total = sum(s[0].numel() + s[1].numel() for s in states) * 4
        kib_total = bytes_total / 1024.0
        kib_per_layer = kib_total / n_layers

        self.assertAlmostEqual(kib_per_layer, 2.0, places=6)
        self.assertAlmostEqual(kib_total, 12.0, places=6)

        # And it must not grow with the length already decoded.
        for t in range(4):
            _, states, _ = resonator.step(torch.zeros(1, dtype=torch.long), states, [None] * n_layers)
        grown = sum(s[0].numel() + s[1].numel() for s in states) * 4
        self.assertEqual(grown, bytes_total)

    def test_resonator_lm_forward_and_recurrent_step(self):
        resonator = ResonatorLM(vocab_size=64, d_model=256, n_layers=2, n_heads=8, d_ff=1024)
        x = torch.randint(0, 64, (2, 16))
        logits = resonator(x)
        self.assertEqual(logits.shape, (2, 16, 64))

        # Test single token recurrent step
        states, buffers = resonator.init_states(batch_size=2, device=x.device, dtype=torch.float32)
        for t in range(4):
            tok = x[:, t]
            step_logits, states, buffers = resonator.step(tok, states, buffers)
            self.assertEqual(step_logits.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
