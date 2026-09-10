import unittest
import os
import tempfile
import torch
from src.data import CharacterTokenizer, CharDataStream, load_wikitext_split


class TestDataPipeline(unittest.TestCase):

    def test_tokenizer_and_wikitext_loading(self):
        train_text = load_wikitext_split("train")
        self.assertGreater(len(train_text), 10_000_000)

        tokenizer = CharacterTokenizer.build_from_text(train_text)
        print(f"Built character vocabulary size: {tokenizer.vocab_size}")
        self.assertGreaterEqual(tokenizer.vocab_size, 256)
        self.assertLessEqual(tokenizer.vocab_size, 300)

        # The committed vocabulary must match what the pipeline rebuilds from the
        # raw corpus. We write to a temp dir rather than the repo: a test must not
        # mutate a tracked artifact, or a failing run could silently rewrite the
        # vocab that every checkpoint in results/ was trained against.
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = os.path.join(tmp, "char_vocab.json")
            tokenizer.save(vocab_path)
            self.assertTrue(os.path.exists(vocab_path))
            roundtrip = CharacterTokenizer.load(vocab_path)
            self.assertEqual(roundtrip.char_to_id, tokenizer.char_to_id)

        committed = "data/char_vocab.json"
        if os.path.exists(committed):
            self.assertEqual(
                CharacterTokenizer.load(committed).char_to_id,
                tokenizer.char_to_id,
                "data/char_vocab.json is stale: it no longer matches the vocabulary "
                "rebuilt from data/wikitext-2-raw/wiki.train.raw",
            )

        # Test encoding and decoding
        sample = "ResonatorLM: Causal Resonant Field Mixing"
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(sample, decoded)

        # Test data stream chunking
        stream = CharDataStream(encoded * 100, batch_size=2, seq_len=16, device=torch.device("cpu"))
        x, y = stream.get_batch(0)
        self.assertEqual(x.shape, (2, 16))
        self.assertEqual(y.shape, (2, 16))
        # Verify next-character shift: y[:, 0] == x[:, 1]
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))


if __name__ == "__main__":
    unittest.main()
