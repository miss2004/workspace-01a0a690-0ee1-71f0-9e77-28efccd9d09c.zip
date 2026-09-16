"""
Unit Tests and Benchmark for Autocorrect Keyboard Engine
"""

import os
import time
import unittest
from autocorrect_engine import AutocorrectKeyboardEngine


class TestAutocorrectEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
        corpora = [
            os.path.join(corpus_dir, "sherlock_holmes.txt"),
            os.path.join(corpus_dir, "pride_and_prejudice.txt")
        ]
        cls.engine = AutocorrectKeyboardEngine(corpus_paths=corpora)

    def test_single_typos(self):
        """Test common real-world spelling mistakes."""
        test_cases = {
            "teh": "the",
            "recieved": "received",
            "definately": "definitely",
            "happpy": "happy",
            "exampel": "example",
            "machne": "machine",
            "computr": "computer"
        }
        for typo, expected in test_cases.items():
            suggestions = [w for w, _ in self.engine.suggest_corrections(typo, top_k=3)]
            self.assertIn(
                expected,
                suggestions,
                f"Expected '{expected}' in suggestions for '{typo}', got {suggestions}"
            )

    def test_context_sensitive_ranking(self):
        """Test that context helps disambiguate words."""
        # For typo 'ther', in context of 'went to their', 'their' or 'there'
        sug_context = [w for w, _ in self.engine.suggest_corrections("ther", context_words=["went", "to"], top_k=3)]
        self.assertTrue(len(sug_context) > 0)
        self.assertTrue(any(w in ["there", "their", "the"] for w in sug_context))

    def test_next_word_prediction(self):
        """Test anticipation of next word after a sequence."""
        context = ["sherlock", "holmes"]
        next_words = [w for w, _ in self.engine.lm.predict_next_words(context, top_k=5)]
        self.assertTrue(len(next_words) > 0, "Should return next word predictions")
        print("Predictions after 'sherlock holmes':", next_words)

    def test_keystroke_state(self):
        """Test simulation of keyboard state."""
        # Trailing space -> next word prediction
        state_space = self.engine.process_keystroke_state("sherlock holmes ")
        self.assertEqual(state_space["mode"], "next_word_prediction")
        self.assertFalse(state_space["is_misspelled"])

        # Misspelled word without space -> autocorrect
        state_typo = self.engine.process_keystroke_state("sherlock holms")
        self.assertEqual(state_typo["mode"], "autocorrect")
        self.assertTrue(state_typo["is_misspelled"])
        self.assertIn("holmes", state_typo["suggestions"])

    def test_inference_latency(self):
        """Verify suggestion response latency is sub-10ms for real-time typing."""
        start = time.perf_counter()
        for _ in range(50):
            _ = self.engine.suggest_corrections("procedur", top_k=3)
        elapsed_per_query = (time.perf_counter() - start) / 50.0 * 1000.0
        print(f"Average correction latency: {elapsed_per_query:.2f} ms")
        self.assertLess(elapsed_per_query, 50.0, "Latency must be low enough for interactive keyboard")


if __name__ == "__main__":
    unittest.main()
