"""Tests for token-based compaction: the model-window table, the budget
resolution, the estimator, and the compaction trigger itself.

These are deliberately unit-level (no session, no disk, no loop) so a failure
points at exactly one decision. The agent-loop behaviour built on top of them is
covered in :mod:`tests.test_e2e_agent_loop` (``TokenBudgetCompactionTest``).

The thing being tested is a *policy*: compaction used to fire on
``len(messages) > max_messages``, which is a poor proxy for "will the next
request fit in the context window" -- ten large tool outputs can overflow a
window that two hundred short turns would not. The trigger is now the projected
input size plus the headroom kept for the reply, compared against a budget
derived from the model's own window -- and nothing else. The message count is
not consulted at all, so a long, cheap conversation is never compacted early.
"""

import unittest
from unittest import mock

from mini_code_cli import config, tokens, utils
from mini_code_cli.agent_loop import maybe_compact


USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def silent_console():
    from mini_code_cli import ui
    import io
    return ui.Console(stream=io.StringIO(), color=False, unicode=False, verbose=False)


class StubLLM:
    """Pop scripted ``(content, tool_calls, usage)`` replies, in order.

    The queue spells the fields a test cares about; the stub adds the reasoning
    slot, so what the loop receives has the real 4-field shape.
    """

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    def __call__(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if not self.queue:
            raise AssertionError("LLM stub called more times than scripted")
        content, tool_calls, usage = self.queue.pop(0)
        return content, tool_calls, usage, None


def long_messages(count, chars):
    """A transcript of ``count`` messages, each roughly ``chars/4`` tokens."""
    return [{"role": "user", "content": "x" * chars} for _ in range(count)]


# ---------------------------------------------------------------------------
# the model -> context window table
# ---------------------------------------------------------------------------

class ContextWindowLookupTest(unittest.TestCase):
    """``context_window_for`` maps a real model id to a plausible window."""

    def test_exact_ids(self):
        self.assertEqual(config.context_window_for("gpt-4o"), 128_000)
        self.assertEqual(config.context_window_for("gpt-4"), 8_192)

    def test_decorated_ids_match_their_family(self):
        # The ids real deployments use: versioned snapshots, org prefixes, tags.
        cases = {
            "gpt-4o-mini": 128_000,
            "gpt-4o-2024-08-06": 128_000,
            "gpt-4-turbo-2024-04-09": 128_000,
            "gpt-4.1-mini": 1_047_576,
            "gpt-3.5-turbo-0125": 16_385,
            "claude-3-5-sonnet-20241022": 200_000,
            "claude-sonnet-4-20250514": 200_000,
            "Qwen/Qwen2.5-Coder-32B-Instruct": 131_072,
            "qwen2.5-7b-instruct-awq": 131_072,
            "meta-llama/Llama-3.1-8B-Instruct": 131_072,
            "deepseek-r1:70b": 65_536,
            "deepseek-ai/DeepSeek-V3": 65_536,
            "mistralai/Mistral-7B-Instruct-v0.2": 32_768,
            "gemini-1.5-pro": 1_048_576,
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(config.context_window_for(model), expected)

    def test_the_most_specific_family_wins(self):
        # These are the pairs where a naive substring match picks the wrong one:
        # "gpt-4o" contains "gpt-4", and "qwen2.5" contains "qwen2".
        self.assertGreater(config.context_window_for("gpt-4o"),
                           config.context_window_for("gpt-4"))
        self.assertGreater(config.context_window_for("qwen2.5"),
                           config.context_window_for("qwen2"))

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(config.context_window_for("GPT-4O-MINI"), 128_000)

    def test_unknown_and_empty_models_return_the_default(self):
        self.assertIsNone(config.context_window_for("totally-unknown-model"))
        self.assertIsNone(config.context_window_for(""))
        self.assertIsNone(config.context_window_for(None))
        self.assertEqual(config.context_window_for("totally-unknown-model", 1_234), 1_234)


# ---------------------------------------------------------------------------
# resolving the compaction budget
# ---------------------------------------------------------------------------

class CompactionBudgetTest(unittest.TestCase):
    """The three-tier resolution: explicit -> model window -> fallback."""

    def test_explicit_limit_is_used_verbatim(self):
        budget = config.compaction_budget("gpt-4o", explicit_limit=5_000)
        self.assertEqual(budget.token_limit, 5_000)
        self.assertEqual(budget.source, "explicit")
        # The window is still reported, so the caller can explain the override.
        self.assertEqual(budget.context_window, 128_000)

    def test_explicit_limit_survives_a_model_that_is_unknown(self):
        budget = config.compaction_budget("mystery-model", explicit_limit=777)
        self.assertEqual(budget.token_limit, 777)
        self.assertEqual(budget.source, "explicit")

    def test_limit_is_derived_from_the_model_window(self):
        budget = config.compaction_budget("gpt-4o")
        # window - reserved output - 10% safety margin
        self.assertEqual(budget.context_window, 128_000)
        self.assertEqual(budget.reserved_output_tokens,
                         config.DEFAULT_RESERVED_OUTPUT_TOKENS)
        self.assertEqual(budget.token_limit, 128_000 - 8_192 - 12_800)
        self.assertLess(budget.token_limit, budget.context_window)
        self.assertEqual(budget.source, "model")

    def test_max_tokens_becomes_the_output_reserve(self):
        default = config.compaction_budget("gpt-4o")
        capped = config.compaction_budget("gpt-4o", max_output_tokens=2_048)
        self.assertEqual(capped.reserved_output_tokens, 2_048)
        # A smaller reply budget leaves more room for input.
        self.assertGreater(capped.token_limit, default.token_limit)
        self.assertEqual(capped.token_limit, 128_000 - 2_048 - 12_800)

    def test_an_unknown_model_falls_back_conservatively(self):
        budget = config.compaction_budget("some-local-checkpoint")
        self.assertEqual(budget.context_window, config.DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(budget.source, "default")
        self.assertGreater(budget.token_limit, 0)
        self.assertLess(budget.token_limit, config.DEFAULT_CONTEXT_TOKENS)

    def test_an_unnamed_model_also_falls_back(self):
        self.assertEqual(config.compaction_budget(None).source, "default")

    def test_a_small_window_still_yields_a_usable_budget(self):
        # gpt-4 has an 8k window: the 8k default reserve would swamp it.
        budget = config.compaction_budget("gpt-4")
        self.assertEqual(budget.context_window, 8_192)
        self.assertLessEqual(budget.reserved_output_tokens, 8_192 // 2)
        self.assertGreater(budget.token_limit, 1_000)
        self.assertLess(budget.token_limit, 8_192)

    def test_a_giant_window_does_not_get_a_giant_safety_margin(self):
        budget = config.compaction_budget("gemini-1.5-pro")
        margin = budget.context_window - budget.token_limit - budget.reserved_output_tokens
        self.assertEqual(margin, config.MAX_SAFETY_MARGIN_TOKENS)

    def test_an_absurd_max_tokens_cannot_produce_a_negative_budget(self):
        budget = config.compaction_budget("gpt-4", max_output_tokens=10_000_000)
        self.assertGreater(budget.token_limit, 0)

    def test_a_zero_or_negative_explicit_limit_is_ignored(self):
        budget = config.compaction_budget("gpt-4o", explicit_limit=0)
        self.assertEqual(budget.source, "model")
        self.assertEqual(budget.token_limit, 128_000 - 8_192 - 12_800)

    def test_the_convenience_resolver_agrees_with_the_budget(self):
        for model in ("gpt-4o", "gpt-4", "nope"):
            with self.subTest(model=model):
                self.assertEqual(
                    config.resolve_compaction_token_limit(model),
                    config.compaction_budget(model).token_limit)


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

class TokenEstimateTest(unittest.TestCase):
    """The cheap heuristic, and how it is combined with the server's count."""

    def test_empty_text_costs_nothing(self):
        self.assertEqual(tokens.estimate_text_tokens(""), 0)
        self.assertEqual(tokens.estimate_text_tokens(None), 0)

    def test_four_characters_per_token_rounding_up(self):
        self.assertEqual(tokens.estimate_text_tokens("abcd"), 1)
        self.assertEqual(tokens.estimate_text_tokens("abcde"), 2)
        self.assertEqual(tokens.estimate_text_tokens("x" * 400), 100)

    def test_a_message_costs_its_content_plus_boilerplate(self):
        bare = tokens.estimate_message_tokens({"role": "user", "content": "x" * 400})
        self.assertEqual(bare, 100 + tokens.MESSAGE_OVERHEAD_TOKENS)

    def test_tool_call_arguments_are_counted(self):
        without = tokens.estimate_message_tokens(
            {"role": "assistant", "content": "ok"})
        with_calls = tokens.estimate_message_tokens({
            "role": "assistant", "content": "ok",
            "tool_calls": [{"id": "c1", "function": {"name": "write_file",
                                                     "arguments": "x" * 400}}],
        })
        self.assertGreater(with_calls, without + 100)

    def test_a_malformed_message_never_raises(self):
        self.assertEqual(tokens.estimate_message_tokens("not a dict"), 0)
        self.assertEqual(tokens.estimate_message_tokens({"role": "user"}),
                         tokens.MESSAGE_OVERHEAD_TOKENS)

    def test_transcript_estimate_grows_with_the_transcript(self):
        self.assertEqual(tokens.estimate_messages_tokens([]), 0)
        self.assertEqual(tokens.estimate_messages_tokens(None), 0)
        small = tokens.estimate_messages_tokens([{"role": "user", "content": "hi"}])
        big = tokens.estimate_messages_tokens([{"role": "user", "content": "hi" * 400}])
        self.assertGreater(big, small)
        self.assertGreater(small, 0)

    def test_without_a_server_report_the_heuristic_is_used(self):
        messages = long_messages(3, 400)
        self.assertEqual(tokens.project_input_tokens(messages),
                         tokens.estimate_messages_tokens(messages))

    def test_a_report_covering_everything_is_used_as_is(self):
        messages = long_messages(3, 400)
        self.assertEqual(
            tokens.project_input_tokens(messages, 1_234, len(messages)), 1_234)

    def test_only_the_messages_appended_since_the_report_are_estimated(self):
        messages = long_messages(4, 400)
        projected = tokens.project_input_tokens(messages, 500, 2)
        tail = tokens.estimate_messages_tokens(messages[2:])
        self.assertEqual(projected, 500 + tail)
        self.assertGreater(projected, 500)          # the tail is not free

    def test_a_stale_report_is_ignored_after_the_transcript_shrank(self):
        # Compaction / /reset: the report describes a conversation that no longer
        # exists, so trusting it would keep triggering compaction forever.
        messages = long_messages(1, 400)
        self.assertEqual(tokens.project_input_tokens(messages, 99_999, 50),
                         tokens.estimate_messages_tokens(messages))

    def test_a_report_of_zero_tokens_is_ignored(self):
        # Many local servers report nothing at all.
        messages = long_messages(2, 400)
        self.assertEqual(tokens.project_input_tokens(messages, 0, 2),
                         tokens.estimate_messages_tokens(messages))


# ---------------------------------------------------------------------------
# the trigger itself
# ---------------------------------------------------------------------------

class CompactionTriggerTest(unittest.TestCase):
    """When ``maybe_compact`` fires, and which guard fired."""

    def setUp(self):
        self.stats = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "llm_calls": 0, "tool_calls": 0}

    def _compact(self, messages, replies=None, **kwargs):
        """Run maybe_compact with the LLM stubbed; returns its 3-tuple.

        The stub is left on ``self.stub`` so a test can assert how many calls the
        summariser actually made.
        """
        self.stub = StubLLM(replies or [("A short summary.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=self.stub):
            return maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30,
                self.stats, console=silent_console(), **kwargs)

    # -- token pressure ---------------------------------------------------

    def test_few_messages_over_the_token_budget_compact(self):
        messages = long_messages(2, 4_000)          # ~1k tokens each, 2 messages
        out, did_compact, index = self._compact(
            messages, token_limit=1_500, current_tokens=2_000)

        self.assertTrue(did_compact)
        self.assertEqual(index, 1)
        # No system prompt to preserve, so the old turns collapse to one summary
        # turn -- from two ~1k-token messages to one short one.
        self.assertEqual(len(out), 1)
        self.assertIn("A short summary.", out[0]["content"])
        self.assertEqual(self.stats["llm_calls"], 1)

    def test_a_large_transcript_under_the_budget_does_not_compact(self):
        messages = long_messages(200, 8)            # 200 turns, all tiny
        out, did_compact, index = self._compact(
            messages, token_limit=100_000)

        self.assertFalse(did_compact)
        self.assertIs(out, messages)
        self.assertEqual(index, 0)
        self.assertEqual(self.stats["llm_calls"], 0)   # no summariser call

    def test_the_output_reserve_is_what_pushes_it_over(self):
        # Input alone fits; input + room for the reply does not.
        messages = long_messages(1, 400)
        kwargs = dict(token_limit=1_000, reserved_output_tokens=200)

        # 900 + 200 = 1100 > 1000 -- over.
        # With the reserve removed: 900 > 1000 is false -- under.
        _, did_compact, _ = self._compact(
            messages, current_tokens=900, token_limit=1_000,
            reserved_output_tokens=0)
        self.assertFalse(did_compact)

        # 900 + 200 = 1100 > 1000 -- the reserve is what tips it over.
        _, did_compact, _ = self._compact(
            messages, current_tokens=900, **kwargs)
        self.assertTrue(did_compact)

        # 700 + 200 = 900 < 1000 -- comfortably under again.
        _, did_compact, _ = self._compact(
            messages, current_tokens=700, **kwargs)
        self.assertFalse(did_compact)

    def test_the_projection_is_computed_when_the_caller_has_none(self):
        messages = long_messages(4, 40_000)         # ~10k tokens each

        out, did_compact, _ = self._compact(messages, token_limit=5_000)
        self.assertTrue(did_compact)

    def test_an_unset_limit_uses_the_model_fallback(self):
        budget = config.compaction_budget("test-model")
        self.assertEqual(budget.source, "default")

        # A transcript comfortably over the fallback budget compacts ...
        over = long_messages(4, 200_000)            # ~200k tokens > the budget
        _, did_compact, _ = self._compact(
            over, token_limit=budget.token_limit,
            reserved_output_tokens=budget.reserved_output_tokens)
        self.assertTrue(did_compact)

        # ... while a short one in the same session does not.
        _, did_compact, _ = self._compact(
            long_messages(4, 40), token_limit=budget.token_limit,
            reserved_output_tokens=budget.reserved_output_tokens)
        self.assertFalse(did_compact)

    def test_no_limit_never_compacts(self):
        # token_limit=None: the "guard disabled" call shape, which must stay a
        # no-op however large the transcript is.
        _, did_compact, _ = self._compact(long_messages(50, 40_000))
        self.assertFalse(did_compact)

    def test_the_message_count_is_never_a_trigger(self):
        # Thousands of tiny turns, comfortably inside a huge token budget: the
        # old message cap would have compacted this; tokens alone must not.
        _, did_compact, index = self._compact(
            long_messages(5_000, 8), token_limit=10_000_000)

        self.assertFalse(did_compact)
        self.assertEqual(index, 0)
        self.assertEqual(self.stats["llm_calls"], 0)      # no summariser call

    # -- unchanged behaviour ----------------------------------------------

    def test_an_explicit_limit_is_honoured(self):
        messages = long_messages(1, 400)
        _, did_compact, _ = self._compact(
            messages, token_limit=1, current_tokens=2)
        self.assertTrue(did_compact)

    def test_compaction_keeps_the_system_prompt_and_bills_the_summariser(self):
        messages = [{"role": "system", "content": "SYSTEM"}] + long_messages(2, 8_000)
        out, did_compact, index = self._compact(
            messages, token_limit=1_000, current_tokens=4_000)

        self.assertTrue(did_compact)
        self.assertEqual(index, 1)
        self.assertEqual(out[0], {"role": "system", "content": "SYSTEM"})
        self.assertIn("A short summary.", out[1]["content"])
        # The summariser's spend is billed to the segment that is being closed.
        self.assertEqual(self.stats["llm_calls"], 1)
        self.assertEqual(self.stats["total_tokens"], USAGE["total_tokens"])

    def test_the_plan_hook_still_reinjects_the_goal_and_plan(self):
        out, did_compact, _ = self._compact(
            long_messages(2, 8_000), token_limit=1_000, current_tokens=4_000,
            plan_hook=lambda msgs: msgs.append(
                {"role": "user", "content": "GOAL + PLAN"}))

        self.assertTrue(did_compact)
        self.assertEqual(out[-1]["content"], "GOAL + PLAN")

    def test_an_empty_summary_keeps_the_transcript_and_the_segment(self):
        messages = long_messages(2, 8_000)
        out, did_compact, index = self._compact(
            messages, replies=[("", [], USAGE), ("", [], USAGE)],
            token_limit=1_000, current_tokens=4_000)

        self.assertFalse(did_compact)
        self.assertIs(out, messages)
        self.assertEqual(index, 0)


if __name__ == "__main__":
    unittest.main()
