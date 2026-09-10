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
        states, buffers = resonator.init_states(1, torch.device("cpu"), torch.float32)
        self.assertEqual(len(states), n_layers)

        def cache_bytes(states, buffers):
            st = sum(s[0].numel() * s[0].element_size() + s[1].numel() * s[1].element_size()
                     for s in states)
            bf = sum(b.numel() * b.element_size() for b in buffers if b is not None)
            return st, bf

        # Buffers are lazily allocated inside step(), so decode once first.
        for _ in range(4):
            _, states, buffers = resonator.step(
                torch.zeros(1, dtype=torch.long), states, buffers
            )
        st_bytes, buf_bytes = cache_bytes(states, buffers)

        # Resonant state: the paper's B x H x d_h x 2 figure.
        self.assertAlmostEqual(st_bytes / 1024.0 / n_layers, 2.0, places=6)
        self.assertAlmostEqual(st_bytes / 1024.0, 12.0, places=6)
        # The K=3 depthwise local path needs a ring buffer to decode at all.
        self.assertAlmostEqual(buf_bytes / 1024.0 / n_layers, 3.0, places=6)
        # So the honest whole-model decode cache is 30.0 KiB, not 12.0.
        self.assertAlmostEqual((st_bytes + buf_bytes) / 1024.0, 30.0, places=6)

        # And none of it may grow with the length already decoded.
        for _ in range(8):
            _, states, buffers = resonator.step(
                torch.zeros(1, dtype=torch.long), states, buffers
            )
        self.assertEqual(cache_bytes(states, buffers), (st_bytes, buf_bytes))

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
