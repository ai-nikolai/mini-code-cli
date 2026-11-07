"""Tests for the model context-window lookup and compaction budget.

The table is deliberately matched loosely (case-insensitively, longest family
name first). These tests pin the two properties that matter to the loop: a real
model id resolves at all, and the more specific family wins.
"""
import unittest

from mini_code_cli import config


class ContextWindowLookupTest(unittest.TestCase):

    def test_mixed_case_ids_resolve(self):
        # The Ornith entries are the ones that used to be unreachable: the
        # lookup lowercased the model id but not the table keys.
        self.assertEqual(
            config.context_window_for("ornith-ai/Ornith-1.5-35B-A3B"), 250_000)
        self.assertEqual(
            config.context_window_for("ORNITH-AI/ORNITH-1.5-122B-FP8"), 250_000)
        self.assertEqual(
            config.context_window_for("ornith-ai/Ornith-1.5-397B"), 250_000)

    def test_decorated_ids_resolve_to_the_family(self):
        self.assertEqual(
            config.context_window_for("Qwen/Qwen2.5-Coder-32B-Instruct"), 131_072)
        self.assertEqual(
            config.context_window_for("claude-3-5-sonnet-20241022"), 200_000)
        self.assertEqual(config.context_window_for("gpt-4o-2024-08-06"), 128_000)

    def test_the_most_specific_key_wins(self):
        # gpt-4o must beat gpt-4, and qwen2.5 must beat qwen2.
        self.assertEqual(config.context_window_for("gpt-4o"), 128_000)
        self.assertEqual(config.context_window_for("gpt-4-turbo"), 128_000)
        self.assertEqual(config.context_window_for("gpt-4"), 8_192)
        self.assertEqual(config.context_window_for("qwen2.5-72b"), 131_072)
        self.assertEqual(config.context_window_for("qwen2-72b"), 32_768)

    def test_unknown_and_empty_models_return_the_default(self):
        self.assertIsNone(config.context_window_for("no-such-model-xyz"))
        self.assertEqual(
            config.context_window_for("no-such-model-xyz", default=123), 123)
        self.assertEqual(config.context_window_for(None, default=123), 123)
        self.assertEqual(config.context_window_for("   ", default=123), 123)

    def test_every_key_is_lowercase_so_the_lookup_cannot_miss_it(self):
        for key in config.MODEL_CONTEXT_WINDOWS:
            self.assertEqual(key, key.lower(), msg=key)


class CompactionBudgetTest(unittest.TestCase):

    def test_explicit_limit_wins(self):
        budget = config.compaction_budget(
            model="ornith-ai/Ornith-1.5-35B-A3B", explicit_limit=50_000,
            max_output_tokens=2_000)
        self.assertEqual(budget.token_limit, 50_000)
        self.assertEqual(budget.reserved_output_tokens, 2_000)
        self.assertEqual(budget.source, "explicit")

    def test_known_model_derives_a_limit_below_its_window(self):
        budget = config.compaction_budget(
            model="ornith-ai/Ornith-1.5-35B-A3B", max_output_tokens=4_096)
        self.assertEqual(budget.source, "model")
        self.assertEqual(budget.context_window, 250_000)
        self.assertLess(budget.token_limit, budget.context_window)
        self.assertGreater(budget.token_limit, 200_000)

    def test_unknown_model_falls_back_conservatively(self):
        budget = config.compaction_budget(model="no-such-model-xyz")
        self.assertEqual(budget.source, "default")
        self.assertEqual(budget.context_window, config.DEFAULT_CONTEXT_TOKENS)
        self.assertGreaterEqual(budget.token_limit, config.MIN_COMPACTION_TOKEN_BUDGET)

    def test_reserve_is_clamped_to_half_the_window(self):
        budget = config.compaction_budget(
            model="gpt-4", explicit_limit=1, max_output_tokens=1_000_000)
        self.assertLessEqual(budget.reserved_output_tokens, 8_192 // 2)


if __name__ == "__main__":
    unittest.main()
