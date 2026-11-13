"""Tests for the scratchpad, compaction-strategy and planning-strategy features.

The compaction/planning strategies are exercised at two levels:

* the pure registry/parsing helpers in :mod:`mini_code_cli.compaction` and
  :mod:`mini_code_cli.planning`;
* the real :class:`AgentSession` / :func:`maybe_compact` wiring, with the LLM
  stubbed exactly like the rest of the end-to-end suite.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from mini_code_cli import agent_files, agent_loop, compaction, planning, prompts, ui, utils
from mini_code_cli.agent_loop import AgentSession, maybe_compact

from test_e2e_agent_loop import (
    USAGE, StubLLM, make_args, run_loop, tool_call,
)
from test_interactive import _SessionCase


def _stats():
    return {"prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "llm_calls": 0, "tool_calls": 0}


class _AdvancedCase(_SessionCase):
    def new_session(self, start=True, **overrides):
        args = make_args(self.workdir, self.cache_dir,
                         plan=overrides.pop("plan", False))
        for key, value in overrides.items():
            setattr(args, key, value)
        console = self.console()
        session = AgentSession(args, console=console)
        if start:
            # Build a fresh session directly: start_session() would consult the
            # pointer this very session is about to write and ask to resume it.
            session.start_new_session()
        return session


class ScratchpadTest(_AdvancedCase):
    def test_scratchpad_creates_the_file_and_describes_it_in_the_prompt(self):
        session = self.new_session(scratchpad=True)

        path = agent_files.scratchpad_path(self.workdir)
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            self.assertIn("Scratchpad", handle.read())

        system = session.messages[0]["content"]
        self.assertIn("SCRATCHPAD MODE", system)
        self.assertIn(path, system)
        self.assertIn("additional scratch-pad files", system)
        self.assertIn("thoughts, plans, learnings", system)
        self.assertIn("history compaction", system)

    def test_scratchpad_is_off_by_default_and_creates_nothing(self):
        session = self.new_session()
        self.assertFalse(session.scratchpad)
        self.assertNotIn("SCRATCHPAD MODE", session.messages[0]["content"])
        self.assertFalse(os.path.exists(agent_files.scratchpad_path(self.workdir)))

    def test_scratchpad_is_inside_the_allowed_dir_so_file_tools_can_reach_it(self):
        self.new_session(scratchpad=True)
        path = agent_files.scratchpad_path(self.workdir)
        self.assertTrue(path.startswith(os.path.abspath(self.workdir) + os.sep))


class CompactionHelpersTest(unittest.TestCase):
    def test_builtin_strategies_are_registered(self):
        choices = compaction.strategy_choices()
        self.assertIn("summarise", choices)
        self.assertIn("predict-files", choices)
        self.assertIn("predict-messages", choices)
        self.assertEqual(compaction.DEFAULT_COMPACTION_STRATEGY, "summarise")

    def test_parse_keep_indexes_accepts_json_and_bare_lists(self):
        self.assertEqual(compaction.parse_keep_indexes("[2, 5, 9]"), [2, 5, 9])
        self.assertEqual(compaction.parse_keep_indexes('{"keep": [1, 3]}'), [1, 3])
        self.assertEqual(compaction.parse_keep_indexes("2, 5"), [2, 5])
        self.assertEqual(compaction.parse_keep_indexes("no indexes here"), [])
        self.assertEqual(compaction.parse_keep_indexes(""), [])

    def test_select_messages_keeps_order_and_excludes_system_and_tail(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "current"},
        ]
        kept = compaction.select_messages(
            messages, [3, 2, 1, 0, 99], tail=[messages[-1]])
        self.assertEqual([m["content"] for m in kept], ["q1", "a1"])

    def test_select_messages_keeps_tool_calls_only_with_their_results(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "body"},
            {"role": "user", "content": "current"},
        ]
        # Selecting the call pulls its result in automatically.
        kept = compaction.select_messages(messages, [2], tail=[messages[-1]])
        self.assertEqual([m["role"] for m in kept], ["assistant", "tool"])
        # A result without its call is an orphan and is dropped.
        self.assertEqual(compaction.select_messages(messages, [3], tail=[messages[-1]]), [])
        kept = compaction.select_messages(messages, [2, 3], tail=[messages[-1]])
        self.assertEqual([m["role"] for m in kept], ["assistant", "tool"])

    def test_unknown_strategy_is_an_error(self):
        with self.assertRaises(ValueError):
            compaction.build_extra_messages("nope", compaction.CompactionContext(
                summary="s", messages=[]))


class MaybeCompactStrategyTest(unittest.TestCase):
    def _messages(self):
        return [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "current question"},
        ]

    def _run(self, messages, strategy, hook):
        stats = _stats()
        stub = StubLLM([("A summary.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            return maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30, stats,
                token_limit=10, current_tokens=50, console=ui.Console(
                    stream=io.StringIO(), color=False, unicode=False),
                strategy=strategy, prediction_hook=hook)

    def test_predict_files_inserts_tools_between_summary_and_plan(self):
        messages = self._messages()

        def hook(strategy, context):
            self.assertEqual(strategy, "predict-files")
            self.assertEqual(context.summary, "A summary.")
            return [
                {"role": "assistant", "content": "", "tool_calls": [
                    tool_call("p1", "read_file", {"file_path": "x"})]},
                {"role": "tool", "tool_call_id": "p1", "content": "file body"},
            ]

        plan_hook = lambda msgs, tail: msgs.append(
            {"role": "user", "content": "PLAN"})
        stats = _stats()
        stub = StubLLM([("A summary.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            out, did, index = maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30, stats,
                token_limit=10, current_tokens=50,
                console=ui.Console(stream=io.StringIO(), color=False, unicode=False),
                plan_hook=plan_hook, strategy="predict-files",
                prediction_hook=hook)

        self.assertTrue(did)
        self.assertEqual(index, 1)
        self.assertEqual(out[0]["content"], "SYSTEM")
        self.assertIn("A summary.", out[1]["content"])
        self.assertEqual(out[2]["role"], "assistant")
        self.assertEqual(out[2]["tool_calls"][0]["id"], "p1")
        self.assertEqual(out[3]["role"], "tool")
        self.assertEqual(out[4]["content"], "PLAN")
        self.assertEqual(out[-1]["content"], "current question")

    def test_predict_messages_inserts_selected_originals(self):
        messages = self._messages()

        def hook(strategy, context):
            self.assertEqual(strategy, "predict-messages")
            return [dict(messages[1]), dict(messages[2])]

        stats = _stats()
        stub = StubLLM([("A summary.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            out, did, _ = maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30, stats,
                token_limit=10, current_tokens=50,
                console=ui.Console(stream=io.StringIO(), color=False, unicode=False),
                strategy="predict-messages", prediction_hook=hook)

        self.assertTrue(did)
        self.assertEqual([m["content"] for m in out[1:4]],
                         ["A summary." and out[1]["content"],
                          "question 1", "answer 1"])
        self.assertEqual(out[-1]["content"], "current question")

    def test_a_failing_strategy_degrades_to_the_summary(self):
        messages = self._messages()
        stats = _stats()
        stub = StubLLM([("A summary.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            out, did, _ = maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30, stats,
                token_limit=10, current_tokens=50,
                console=ui.Console(stream=io.StringIO(), color=False, unicode=False),
                strategy="predict-files",
                prediction_hook=lambda strategy, context: 1 / 0)

        self.assertTrue(did)
        # system + summary + the in-flight turn (the failed strategy adds none)
        self.assertEqual(len(out), 3)


class CompactionStrategyEndToEndTest(_AdvancedCase):
    def test_predict_messages_keeps_the_indexes_the_model_names(self):
        session = self.new_session(max_context_tokens=1,
                                   compaction_strategy="predict-messages")
        session.messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "current question"},
        ]
        stub = StubLLM([
            ("Summary of the work.", [], USAGE),   # summary
            ("[1, 2]", [], USAGE),                 # keep indexes
            ("Done.", [], USAGE),                  # the real step
        ])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            session._step()

        contents = [m.get("content") for m in session.messages]
        self.assertIn("Summary of the work.", contents[1])
        self.assertEqual(contents[2:4], ["question 1", "answer 1"])
        self.assertEqual(contents[4], "current question")
        self.assertEqual(contents[-1], "Done.")
        # The keep-prediction is an extra tool-free request, and it is billed.
        self.assertEqual(len(stub.calls), 3)
        self.assertEqual(session.total_usage["llm_calls"], 3)

    def test_predict_files_executes_predicted_reads_before_the_current_turn(self):
        target = os.path.join(self.workdir, "notes.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("important notes\n")

        session = self.new_session(max_context_tokens=1,
                                   compaction_strategy="predict-files")
        session.messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "current question"},
        ]
        stub = StubLLM([
            ("Summary of the work.", [], USAGE),                       # summary
            ("", [tool_call("p1", "read_file", {"file_path": target})], USAGE),
            ("Done.", [], USAGE),                                      # real step
        ])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            session._step()

        assistant = session.messages[2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "p1")
        self.assertEqual(session.messages[3]["role"], "tool")
        self.assertIn("important notes", session.messages[3]["content"])
        self.assertEqual(session.messages[4]["content"], "current question")
        # The predicted read is billed as a tool call (folded into the total
        # before the segment rollover) but not as part of the new segment.
        self.assertEqual(session.total_usage["tool_calls"], 1)
        self.assertEqual(session.token_stats["tool_calls"], 0)

    def test_unknown_strategy_falls_back_to_summarise(self):
        session = self.new_session(compaction_strategy="not-a-strategy")
        self.assertEqual(session.compaction_strategy, "summarise")
        self.assertIn("Unknown compaction strategy", self.output())


class PlanStrategyTest(_AdvancedCase):
    def test_open_plan_creates_a_document_and_skips_plan_requests(self):
        session = self.new_session(plan=True, plan_strategy="open-plan")
        path = agent_files.open_plan_path(self.workdir)
        self.assertTrue(os.path.isfile(path))
        system = session.messages[0]["content"]
        self.assertIn("OPEN PLAN MODE", system)
        self.assertIn(path, system)
        self.assertNotIn("CURRENT_PLAN.md -- an actual step-by-step plan", system)

        stub = StubLLM([("Done without a loop-authored plan.", [], USAGE)])
        # The session is already started (so its prompt/file could be
        # inspected); run() must not re-run resume detection and find it.
        with mock.patch.object(AgentSession, "start_session", lambda self: None):
            run_loop(session, "do the thing", stub)

        # No plan-generation and no plan-review request: one normal step only.
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(session.messages[-1]["content"],
                         "Done without a loop-authored plan.")

    def test_open_plan_command_shows_the_agent_document(self):
        session = self.new_session(plan=True, plan_strategy="open-plan")
        with open(agent_files.open_plan_path(self.workdir), "w", encoding="utf-8") as handle:
            handle.write("# Agent Plan\n\n1. read the data\n2. report\n")
        self.assertTrue(session._handle_command("/plan"))
        self.assertIn("read the data", self.output())

    def test_plan_strategy_implies_plan_mode(self):
        session = self.new_session(plan=False, plan_strategy="open-plan")
        self.assertTrue(session.plan_mode)

    def test_plan_verify_drops_the_final_no_tool_step(self):
        session = self.new_session(start=False, plan=True,
                                   plan_strategy="plan-verify")
        stub = StubLLM([
            ("Plan:\n1. do the thing", [], USAGE),      # plan generation
            ("Initial answer.", [], USAGE),             # settles -> review
            ("ALL TASKS DONE. Go back to user.", [], USAGE),  # review
            ("Final verification answer.", [], USAGE),   # post-review, no tools
        ])
        run_loop(session, "do the thing", stub)

        texts = [m.get("content") for m in session.messages]
        self.assertIn("Initial answer.", texts)
        self.assertNotIn("Final verification answer.", texts)
        # The user still saw the final answer.
        self.assertIn("Final verification answer.", self.output())

    def test_plan_verify_step_keeps_the_final_no_tool_step(self):
        session = self.new_session(start=False, plan=True,
                                   plan_strategy="plan-verify-step")
        stub = StubLLM([
            ("Plan:\n1. do the thing", [], USAGE),
            ("Initial answer.", [], USAGE),
            ("ALL TASKS DONE. Go back to user.", [], USAGE),
            ("Final step answer.", [], USAGE),
        ])
        run_loop(session, "do the thing", stub)

        texts = [m.get("content") for m in session.messages]
        self.assertIn("Final step answer.", texts)

    def test_plan_verify_keeps_a_post_review_tool_step(self):
        target = os.path.join(self.workdir, "data.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("alpha\n")

        session = self.new_session(start=False, plan=True,
                                   plan_strategy="plan-verify")
        stub = StubLLM([
            ("Plan:\n1. read the file", [], USAGE),
            ("Initial answer.", [], USAGE),
            ("TASKS still in progress.", [], USAGE),
            ("", [tool_call("c1", "read_file", {"file_path": target})], USAGE),
            ("After the tool.", [], USAGE),
            ("ALL TASKS DONE. Go back to user.", [], USAGE),
            ("Final dropped answer.", [], USAGE),
        ])
        run_loop(session, "read the file", stub)

        texts = [m.get("content") for m in session.messages]
        self.assertIn("After the tool.", texts)
        self.assertNotIn("Final dropped answer.", texts)
        tool_messages = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)


class PromptStrategyTest(unittest.TestCase):
    def test_system_prompt_includes_open_plan_and_scratchpad_blocks(self):
        text = prompts.system_prompt(
            {"read_file": {"description": "read"}},
            plan_mode=True,
            open_plan_path="/tmp/allowed/.mini/PLAN.md",
            scratchpad_path="/tmp/allowed/.mini/SCRATCHPAD.md",
            scratchpad_dir="/tmp/allowed/.mini",
        )
        self.assertIn("OPEN PLAN MODE", text)
        self.assertIn("SCRATCHPAD MODE", text)
        self.assertIn("explicit exceptions", text)
        self.assertNotIn("CURRENT_PLAN.md -- an actual step-by-step plan", text)

    def test_compaction_strategy_prompts_are_actionable(self):
        files = prompts.compaction_predict_files_prompt("SUMMARY", "do X")
        self.assertIn("SUMMARY", files)
        self.assertIn("do X", files)
        self.assertIn("read_file", files)
        keep = prompts.compaction_keep_prompt(
            [{"role": "user", "content": "hello"}], "SUMMARY")
        self.assertIn("[0] user: hello", keep)
        self.assertIn("JSON array", keep)


class SessionIdLineTest(_AdvancedCase):
    def _session_output(self):
        session = self.new_session()
        session._print_session_id()
        return self.output()

    def test_a_divider_precedes_the_first_assistant_style_line(self):
        lines = self._session_output().splitlines()
        assistant = next(i for i, line in enumerate(lines) if "Assistant" in line)
        divider = lines[assistant - 1].strip()
        self.assertTrue(divider)
        self.assertEqual(set(divider), {"-"})

    def test_the_context_bar_is_shown_for_a_new_session(self):
        # A fresh session starts from the system prompt, so the bar is already
        # meaningful (not missing until the first reply).
        out = self._session_output()
        self.assertRegex(out, r"\d+%")


class WelcomeBannerTest(_AdvancedCase):
    def _welcome(self, **overrides):
        args = make_args(self.workdir, self.cache_dir,
                         plan=overrides.pop("plan", False))
        for key, value in overrides.items():
            setattr(args, key, value)
        console = self.console()
        agent_loop.print_welcome(self.workdir, "http://localhost:1", args,
                                 "session-id", self.cache_dir, "test-model",
                                 console=console)
        return self.output()

    def test_banner_shows_the_compaction_and_planning_strategies(self):
        out = self._welcome(plan=True, compaction_strategy="predict-files",
                            plan_strategy="open-plan")
        self.assertIn("compaction  : predict-files", out)
        self.assertIn("planning    : open-plan", out)
        self.assertNotIn("plan mode off", out)

    def test_banner_marks_planning_off_when_plan_mode_is_disabled(self):
        out = self._welcome(compaction_strategy="summarise")
        self.assertIn("compaction  : summarise", out)
        self.assertIn("planning    : plan-verify-step (plan mode off)", out)

    def test_welcome_shows_the_ascii_mark(self):
        out = self._welcome()
        self.assertIn("a minimal coding agent CLI", out)
        self.assertIn("__  __", out)


class ParseArgsTest(unittest.TestCase):
    def test_new_flags_have_the_expected_defaults(self):
        with mock.patch.object(sys, "argv", ["mini-code"]):
            args = agent_loop.parse_args()
        self.assertFalse(args.scratchpad)
        self.assertEqual(args.compaction_strategy, "summarise")
        self.assertIsNone(args.plan_strategy)

    def test_new_flags_are_accepted(self):
        with mock.patch.object(
                sys, "argv",
                ["mini-code", "--scratchpad", "--compaction-strategy",
                 "predict-files", "--plan-strategy", "plan-verify"]):
            args = agent_loop.parse_args()
        self.assertTrue(args.scratchpad)
        self.assertEqual(args.compaction_strategy, "predict-files")
        self.assertEqual(args.plan_strategy, "plan-verify")
        # A strategy implies plan mode so the banner and the session agree.
        self.assertTrue(args.plan)
