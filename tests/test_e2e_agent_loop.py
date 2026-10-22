"""End-to-end tests for the agent loop.

These are the most important tests in the suite: they drive the *whole* loop the
way a real user would -- ``AgentSession(args).run(initial_prompt=...)`` -- which
covers user input -> LLM call -> real tool dispatch -> clean termination. The
one and only thing patched is ``mini_code_cli.utils.call_openai_server`` (the
LLM itself); everything else runs for real: argument handling, plan-mode wiring,
the genuine tools (a real ``grep`` / ``write_file`` actually executes on disk),
the ``--allowed-dir`` sandbox, sub-agent delegation, history compaction and the
on-disk persistence of the session.

Each scenario feeds the loop a scripted queue of LLM responses, so the tests are
deterministic and need no server or network.

Scenarios
---------
* ``PlainAutoModeTest``       -- plan mode OFF: one ``grep`` tool call, then a
                                  clean, self-terminating stop.
* ``PlanModeEndToEndTest``    -- plan mode ON: plan-gen + review are woven in, and
                                  the loop-authored ``CURRENT_GOAL.md`` /
                                  ``CURRENT_PLAN.md`` files land on disk.
* ``SubAgentEndToEndTest``    -- the main agent delegates to a sub-agent; both
                                  loops run through one shared LLM stub.
* ``ManualModeTest``          -- auto-mode OFF: one step, then the loop hands
                                  control back to the human instead of looping.
* ``ToolGatingTest``          -- --enable-shell gates the shell tool, and the
                                  shell-safety DENY tier refuses a destructive
                                  command even when shell is enabled.
* ``CompactionTest``          -- history compaction (the module-level helper):
                                  no-op under the token budget, summary + segment
                                  bump over it, and plan mode's goal/plan
                                  re-injection.
* ``RestoreSessionEndToEndTest`` -- a persisted session is restored *in place*
                                  (same id), a folder that does not satisfy the
                                  restore contract (no recorder) is refused and
                                  replaced by a fresh session, and
                                  ``--load-last-session`` continues whatever the
                                  ``.mini_tests`` pointer names without asking.
                                  (The tests record their pointer under
                                  ``.mini_tests``, never a real ``.mini``.)
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from mini_code_cli import persistence, tools, ui, utils
from mini_code_cli.agent_loop import AgentSession, maybe_compact
from mini_code_cli.tools import grep as grep_tool
from mini_code_cli.subagents import SubAgent, register, clear_registry


# The suite must never create a real ``.mini`` directory: that name belongs to
# the user (their skills folder + ``last_session.json`` resume pointer). Every
# session pointer the tests record therefore goes into ``.mini_tests`` instead,
# under the test's own temporary --allowed-dir. (The session histories
# themselves already live in per-test temporary cache dirs.)
persistence.SESSION_DIRNAME = ".mini_tests"


USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------

def make_args(allowed_dir, cache_dir, plan=False, enable_subagents=False):
    """Build the argparse-like namespace AgentSession.__init__ reads."""
    return argparse.Namespace(
        url="http://0.0.0.0:30000",
        max_tokens=None,
        temperature=1.0,
        model="test-model",
        api_key=None,
        cache_dir=cache_dir,
        restore_session=None,
        load_last_session=False,
        system_prompt=None,
        auto_mode=True,
        agent_md=None,
        prompt=None,
        ask_permission=False,
        sudo=False,
        allowed_dir=allowed_dir,
        enable_shell=False,
        enable_subagents=enable_subagents,
        skills=False,
        llm_timeout=30,
        tool_timeout=30,
        max_context_tokens=None,
        plan=plan,
        verbose=False,
    )


def tool_call(call_id, name, arguments):
    """An OpenAI-style tool call as the loop expects to receive it."""
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def silent_console():
    """A console whose output goes to a buffer instead of the screen."""
    return ui.Console(stream=io.StringIO(), color=False, unicode=False, verbose=False)


def count_from_messages(messages):
    """Derive a number of tool-call lines from a conversation (a fake LLM reply)."""
    tool_texts = [m.get("content", "") for m in messages if m.get("role") == "tool"]
    n = sum(1 for t in tool_texts for line in t.splitlines() if line.strip())
    return (f"Report: the tool produced {n} line(s).", [], USAGE)


class StubLLM:
    """A deterministic stand-in for ``utils.call_openai_server``.

    It pops responses from a queue in order and records every call's arguments
    in ``self.calls`` so tests can assert on what the loop actually sent the
    model. Each item in the queue may be either a callable (called with the
    live ``messages`` list) or a pre-built tuple; either way it spells the two
    fields a test cares about -- ``(content, tool_calls, usage)`` -- and the stub
    supplies the fourth (``reasoning_content=None``) so what the loop receives
    has the real 4-field shape.
    """

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    def __call__(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if not self.queue:
            raise AssertionError("LLM stub called more times than scripted")
        item = self.queue.pop(0)
        if callable(item):
            item = item(messages)
        content, tool_calls, usage = item
        return content, tool_calls, usage, None


def run_loop(session, prompt, stub):
    """Run one session with the LLM stubbed and stdin short-circuited.

    ``_read_user_line`` returns None (EOF) so the loop exits cleanly once the
    preset prompt's auto-mode work settles, and ``determine_vllm_model`` is
    stubbed so no real network probe happens at construction time.
    """
    patches = [
        mock.patch.object(utils, "call_openai_server", new=stub),
        mock.patch.object(utils, "determine_vllm_model", return_value=None),
        mock.patch.object(AgentSession, "_read_user_line", return_value=None),
    ]
    for p in patches:
        p.start()
    try:
        session.run(initial_prompt=prompt)
    finally:
        for p in patches:
            p.stop()
    return stub


# ---------------------------------------------------------------------------
# scenario 1 -- plain auto-mode, plan mode off
# ---------------------------------------------------------------------------

class PlainAutoModeTest(unittest.TestCase):
    """Plan mode OFF: one real grep tool call, then a clean stop."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_grep_runs_and_loop_terminates(self):
        target = os.path.join(self.workdir, "fruit.txt")
        lines = ["apple\n", "banana\n", "apple\n", "cherry\n", "apple\n"]
        with open(target, "w") as f:
            f.writelines(lines)
        expected = sum(1 for l in lines if "apple" in l)  # 3 -- independent ground truth
        real_grep = grep_tool("apple", target).strip()

        grep_call = tool_call("c1", "grep", {"pattern": "apple", "file_path": target})
        # The model's final answer is derived from the *real* tool output that is
        # now in the conversation, proving the tool result flowed through to the
        # model's context and back out as its answer.
        final_response = lambda messages: (f"Report: 'apple' appears {expected} times.", [], USAGE)

        stub = StubLLM([
            ("", [grep_call], USAGE),          # 1) ask to run grep
            final_response,                    # 2) report back, no tool calls -> stop
        ])

        prompt = "Use grep to count 'apple' in fruit.txt and report the number."
        args = make_args(self.workdir, self.cache_dir)
        session = AgentSession(args, console=silent_console())
        stub = run_loop(session, prompt, stub)

        # 1) Exactly the two scripted LLM calls happened.
        self.assertEqual(len(stub.calls), 2)
        # 2) The real grep tool ran and its stdout is recorded as a tool message.
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0].get("content", "").strip(), real_grep)
        # 3) The loop terminated with a clean assistant turn carrying the count.
        self.assertEqual(session._last_assistant_text(), f"Report: 'apple' appears {expected} times.")
        self.assertEqual(session.token_stats["llm_calls"], 2)
        self.assertEqual(session.token_stats["tool_calls"], 1)
        # 4) The session was persisted under a fresh session id.
        session_dir = os.path.join(self.cache_dir, session.unique_id)
        self.assertIn("history_0.json", os.listdir(session_dir))
        # Usage is recorded per segment, plus a whole-session total.
        self.assertIn("usage_0.json", os.listdir(session_dir))
        self.assertIn("total_usage.json", os.listdir(session_dir))
        self.assertNotIn("usage.json", os.listdir(session_dir))

    def test_sandbox_blocks_file_outside_allowed_dir(self):
        """An end-to-end write outside --allowed-dir is rejected by the real sandbox."""
        outside = tempfile.mkdtemp()
        write_call = tool_call("c1", "write_file",
                               {"file_path": os.path.join(outside, "bad.txt"), "content": "x"})
        stub = StubLLM([
            ("", [write_call], USAGE),                       # 1) try a sandbox-violating write
            ("Wrote nothing, the sandbox blocked it.", [], USAGE),  # 2) finish
        ])

        prompt = "Write a file outside the allowed directory."
        args = make_args(self.workdir, self.cache_dir)
        session = AgentSession(args, console=silent_console())
        stub = run_loop(session, prompt, stub)

        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertIn("not allowed outside allowed directory", tool_msgs[0].get("content", ""))
        self.assertFalse(os.path.exists(os.path.join(outside, "bad.txt")))  # nothing escaped
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# scenario 2 -- plan mode on
# ---------------------------------------------------------------------------

class PlanModeEndToEndTest(unittest.TestCase):
    """Plan mode ON: plan-gen + review woven in, tracking files written."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_plan_mode_end_to_end(self):
        target = os.path.join(self.workdir, "fruit.txt")
        with open(target, "w") as f:
            f.write("apple\napple\n")
        real_grep = grep_tool("apple", target).strip()

        grep_call = tool_call("c1", "grep", {"pattern": "apple", "file_path": target})

        stub = StubLLM([
            ("Plan:\n1. grep for apple\n2. report the count", [], USAGE),   # plan gen
            ("", [grep_call], USAGE),                                       # the grep step
            ("", [], USAGE),                                                # settle -> review due
            ("ALL TASKS DONE. Go back to user.", [], USAGE),                # plan review
            ("Done: grep reported " + str(len(real_grep.splitlines())) + " lines.", [], USAGE),  # final follow-up

        ])

        prompt = "Plan the work: grep for 'apple' in fruit.txt and report the count."
        args = make_args(self.workdir, self.cache_dir, plan=True)
        session = AgentSession(args, console=silent_console())
        stub = run_loop(session, prompt, stub)

        # 5 model calls in the plan-mode order (gen, step, settle, review, follow-up).
        self.assertEqual(len(stub.calls), 5)
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0].get("content", "").strip(), real_grep)
        # The loop-authored tracking files exist and are populated.
        session_dir = os.path.join(self.cache_dir, session.unique_id)
        goal_path = os.path.join(session_dir, "CURRENT_GOAL.md")
        plan_path = os.path.join(session_dir, "CURRENT_PLAN.md")
        self.assertTrue(os.path.isfile(goal_path))
        self.assertTrue(os.path.isfile(plan_path))
        with open(goal_path) as f:
            self.assertIn("apple", f.read())
        with open(plan_path) as f:
            self.assertIn("grep", f.read())
        # Clean termination with a final assistant turn.
        self.assertIn("2", session._last_assistant_text())
        self.assertEqual(session.token_stats["tool_calls"], 1)

        # --- billing: plan mode must not lose its planning calls ------------
        # Five model calls, so five billed LLM calls: the plan generation and the
        # completion review are ordinary (tool-free) requests and each costs one.
        # They used to be invisible here -- the usage line is only printed by
        # _step -- so this pins that they are actually charged.
        self.assertEqual(session.token_stats["llm_calls"], len(stub.calls))
        self.assertEqual(session.token_stats["llm_calls"], 5)
        self.assertEqual(session.token_stats["total_tokens"],
                         len(stub.calls) * USAGE["total_tokens"])
        # The session total agrees, and no compaction happened in this run so the
        # single live segment *is* the whole session's spend.
        self.assertEqual(session.total_usage["llm_calls"], 5)
        self.assertEqual(session.total_usage, session.token_stats)
        self.assertEqual(session.history_index, 0)
        # The two planning calls really are billed to the live segment on disk.
        session_dir = os.path.join(self.cache_dir, session.unique_id)
        seg0 = persistence.load_usage_segment(session_dir, 0)
        self.assertEqual(seg0["llm_calls"], 5)
        self.assertEqual(seg0, session.token_stats)
        total = persistence.load_total_usage_from_dir(session_dir)
        self.assertEqual(total["llm_calls"], 5)
        self.assertEqual(total["total_tokens"], 5 * USAGE["total_tokens"])
        # The last request's prompt size is remembered from the server's report,
        # which is what the token-based compaction trigger works from.
        self.assertEqual(session.last_prompt_tokens, USAGE["prompt_tokens"])


# ---------------------------------------------------------------------------
# scenario 3 -- sub-agent delegation end-to-end
# ---------------------------------------------------------------------------

class SubAgentEndToEndTest(unittest.TestCase):
    """The main agent delegates to a nested loop through one shared LLM stub."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_delegation_runs_nested_loop(self):
        class EchoSubAgent(SubAgent):
            name = "echo_test"
            description = "A test sub-agent that writes a file."
            tools = ["read_file", "write_file"]
            enable_shell = False
            max_steps = 5

        marker = os.path.join(self.workdir, "agent_done.txt")
        register(EchoSubAgent())
        try:
            parent_call = tool_call("pc1", "echo_test", {"task": f"Write {marker} and finish."})
            child_write = tool_call("cc1", "write_file", {"file_path": marker, "content": "done"})
            stub = StubLLM([
                ("", [parent_call], USAGE),                # parent delegates
                ("", [child_write], USAGE),                # child writes the file
                ("Final answer from sub-agent", [], USAGE),        # child finishes
                ("Sub-agent completed its task.", [], USAGE),       # parent finishes
            ])

            prompt = "Delegate a sub-agent to create marker.txt and finish."
            args = make_args(self.workdir, self.cache_dir, enable_subagents=True)
            session = AgentSession(args, console=silent_console())
            stub = run_loop(session, prompt, stub)

            self.assertEqual(len(stub.calls), 4)                 # 2 parent + 2 child calls
            self.assertTrue(os.path.isfile(marker))              # the sub-agent really wrote it
            tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
            self.assertTrue(any("Result from sub-agent 'echo_test'" in m.get("content", "")
                                for m in tool_msgs))
            self.assertEqual(session._last_assistant_text(), "Sub-agent completed its task.")
        finally:
            clear_registry()


# ---------------------------------------------------------------------------
# scenario 4 -- manual mode (auto-mode off)
# ---------------------------------------------------------------------------

class ManualModeTest(unittest.TestCase):
    """auto-mode OFF: one step runs, then the loop waits for the human."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_tool_call_then_waits_for_user(self):
        target = os.path.join(self.workdir, "notes.txt")
        with open(target, "w") as f:
            f.write("one\ntwo\n")
        real_grep = grep_tool("two", target).strip()

        grep_call = tool_call("c1", "grep", {"pattern": "two", "file_path": target})
        # Only ONE scripted response: if manual mode wrongly kept looping, the
        # stub would be asked for a second response and raise.
        stub = StubLLM([("", [grep_call], USAGE)])

        args = make_args(self.workdir, self.cache_dir)
        args.auto_mode = False
        session = AgentSession(args, console=silent_console())
        stub = run_loop(session, "grep notes.txt for 'two'.", stub)

        # Exactly one model call: the step ran the tool, then control returned
        # to the user (patched stdin then reports EOF and the session ends).
        self.assertEqual(len(stub.calls), 1)
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0].get("content", "").strip(), real_grep)
        self.assertEqual(session.token_stats["tool_calls"], 1)


# ---------------------------------------------------------------------------
# scenario 5 -- tool gating (shell disabled / shell-safety DENY)
# ---------------------------------------------------------------------------

class ToolGatingTest(unittest.TestCase):
    """The shell tool is opt-in, and DENY-tier commands are refused."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_shell_tool_withheld_without_enable_shell(self):
        shell_call = tool_call("c1", "run_shell_command", {"command": "echo hi"})
        stub = StubLLM([
            ("", [shell_call], USAGE),          # 1) model tries to use the shell
            ("Could not run the shell.", [], USAGE),  # 2) model reacts to the error
        ])

        args = make_args(self.workdir, self.cache_dir)   # enable_shell defaults False
        session = AgentSession(args, console=silent_console())
        run_loop(session, "Run echo hi.", stub)

        # The tool was never registered, so it is rejected as unknown -- and
        # nothing executed.
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Unknown tool: run_shell_command", tool_msgs[0].get("content", ""))
        self.assertEqual(session.token_stats["tool_calls"], 1)

    def test_deny_tier_command_refused_even_with_shell_enabled(self):
        shell_call = tool_call("c1", "run_shell_command", {"command": "rm -rf /"})
        stub = StubLLM([
            ("", [shell_call], USAGE),
            ("The command was refused.", [], USAGE),
        ])

        args = make_args(self.workdir, self.cache_dir)
        args.enable_shell = True
        session = AgentSession(args, console=silent_console())
        run_loop(session, "Delete everything.", stub)

        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("refused for safety reasons", tool_msgs[0].get("content", ""))
        # The session survived the refusal and let the model finish.
        self.assertEqual(session._last_assistant_text(), "The command was refused.")


# ---------------------------------------------------------------------------
# scenario 6 -- history compaction
# ---------------------------------------------------------------------------

class CompactionTest(unittest.TestCase):
    """The module-level compaction helper that keeps the context window bounded."""

    def setUp(self):
        self.stats = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "llm_calls": 0, "tool_calls": 0}

    def _messages(self, n=6):
        messages = [{"role": "system", "content": "SYSTEM"}]
        for i in range(n):
            messages.append({"role": "user", "content": f"question {i}"})
            messages.append({"role": "assistant", "content": f"answer {i}"})
        return messages

    def _compact(self, messages, **kwargs):
        stub = StubLLM([("A short summary of the session.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            return maybe_compact(
                messages, "http://localhost:1", "test-model", None, 30,
                self.stats, console=silent_console(), **kwargs)

    def test_under_the_budget_is_a_noop(self):
        messages = self._messages(1)
        out, did_compact, index = self._compact(messages, token_limit=100_000)

        self.assertFalse(did_compact)
        self.assertIs(out, messages)          # same list, untouched
        self.assertEqual(index, 0)
        self.assertEqual(self.stats["llm_calls"], 0)   # no summarization call

    def test_over_the_budget_summarises_and_starts_a_new_segment(self):
        messages = self._messages(4)          # 9 messages
        out, did_compact, index = self._compact(
            messages, token_limit=10, current_tokens=50)

        self.assertTrue(did_compact)
        self.assertEqual(index, 1)            # history_0 -> history_1
        # The system prompt survives; the old turns collapse to one summary turn.
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "SYSTEM")
        self.assertEqual(len(out), 2)
        self.assertIn("A short summary of the session.", out[1]["content"])
        # The summarization call is accounted for in the token totals.
        self.assertEqual(self.stats["llm_calls"], 1)
        self.assertEqual(self.stats["total_tokens"], USAGE["total_tokens"])

    def test_plan_hook_reinjects_goal_and_plan(self):
        messages = self._messages(4)
        out, did_compact, _ = self._compact(
            messages, token_limit=10, current_tokens=50,
            plan_hook=lambda msgs, tail: msgs.append({"role": "user", "content": "GOAL + PLAN"}))

        self.assertTrue(did_compact)
        self.assertEqual(out[-1]["content"], "GOAL + PLAN")


# ---------------------------------------------------------------------------
# scenario 6b -- per-segment usage (usage_{n}.json + total_usage.json)
# ---------------------------------------------------------------------------

class PerSegmentUsageTest(unittest.TestCase):
    """Usage is billed per segment so it survives a compaction.

    ``usage_{n}.json`` records what segment *n* cost, and ``total_usage.json``
    accumulates across every still-available segment, so a rollover (or a
    restore after one) still reports the whole session's spend.
    """

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_usage_survives_a_compaction(self):
        args = make_args(self.workdir, self.cache_dir)
        args.max_context_tokens = 1                 # any non-empty history is "over"
        session = AgentSession(args, console=silent_console())
        session.start_session()
        session.messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        stub = StubLLM([
            ("A summary of the session.", [], USAGE),   # the compaction call
            ("Hello.", [], USAGE),                      # this step's own call
        ])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            session._step()
        session.shutdown()

        session_dir = os.path.join(self.cache_dir, session.unique_id)
        self.assertEqual(session.history_index, 1)      # history_0 -> history_1
        self.assertIn("usage_0.json", os.listdir(session_dir))
        self.assertIn("usage_1.json", os.listdir(session_dir))
        self.assertIn("total_usage.json", os.listdir(session_dir))

        seg0 = persistence.load_usage_segment(session_dir, 0)
        seg1 = persistence.load_usage_segment(session_dir, 1)
        # The compaction call was billed to the segment it closed; the step's own
        # call was billed to the fresh one.
        self.assertEqual(seg0["llm_calls"], 1)
        self.assertEqual(seg1["llm_calls"], 1)
        self.assertEqual(seg1["total_tokens"], USAGE["total_tokens"])
        # The live segment counters hold only the current segment ...
        self.assertEqual(session.token_stats, seg1)
        # ... while the session total is the sum of both segments.
        total = persistence.load_total_usage_from_dir(session_dir)
        self.assertEqual(total["llm_calls"], 2)
        self.assertEqual(total["total_tokens"], 2 * USAGE["total_tokens"])
        self.assertEqual(total, session.total_usage)

    def test_total_is_rebuilt_from_the_segments_when_its_file_is_missing(self):
        session_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, session_dir, ignore_errors=True)
        persistence.save_usage_segment(
            {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
             "llm_calls": 1, "tool_calls": 0}, os.path.dirname(session_dir),
            os.path.basename(session_dir), 0)
        persistence.save_usage_segment(
            {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11,
             "llm_calls": 1, "tool_calls": 2}, os.path.dirname(session_dir),
            os.path.basename(session_dir), 1)

        total = persistence.load_total_usage_from_dir(session_dir)
        self.assertEqual(total["total_tokens"], 18)
        self.assertEqual(total["llm_calls"], 2)
        self.assertEqual(total["tool_calls"], 2)


# ---------------------------------------------------------------------------
# scenario 6c -- token-based compaction (the real trigger)
# ---------------------------------------------------------------------------

class TokenBudgetCompactionTest(unittest.TestCase):
    """The agent loop compacts on *context size*, not on the turn count.

    ``maybe_compact``'s policy is unit-tested in ``tests/test_tokens.py``; what
    matters here is the session-level wiring: the projected input size is handed
    to the trigger, a token-triggered rollover does the same bookkeeping a
    message-triggered one used to do (new history segment, usage files, totals),
    and the remembered server-reported prompt size is dropped so the rollover
    cannot immediately re-fire. The message count is never consulted.
    """

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _session(self, **overrides):
        args = make_args(self.workdir, self.cache_dir)
        for key, value in overrides.items():
            setattr(args, key, value)
        session = AgentSession(args, console=silent_console())
        # A second session in the same --allowed-dir would otherwise find the
        # .mini_tests pointer the first one wrote and *ask* whether to resume.
        # A test must never read stdin (that blocked the suite when run without
        # a TTY), so answer the offer explicitly.
        with mock.patch.object(AgentSession, "show_resume_choice", return_value="new"):
            session.start_session()
        return session

    def _run_step(self, session, replies):
        stub = StubLLM(replies)
        with mock.patch.object(utils, "call_openai_server", new=stub):
            session._step()
        return stub

    def test_the_budget_is_resolved_from_the_model(self):
        session = self._session(model="gpt-4o")
        self.assertEqual(session.compaction_budget.source, "model")
        self.assertEqual(session.compaction_budget.context_window, 128_000)
        self.assertLess(session.compaction_budget.token_limit, 128_000)

        # --max-context-tokens overrides it outright.
        explicit = self._session(max_context_tokens=4_242)
        self.assertEqual(explicit.compaction_budget.token_limit, 4_242)
        self.assertEqual(explicit.compaction_budget.source, "explicit")

    def test_a_token_triggered_compaction_rolls_the_segment(self):
        # A budget of 1 token means the very first step is over it -- which is
        # how a real overflow behaves, only sooner.
        session = self._session(max_context_tokens=1)
        stub = self._run_step(session, [
            ("A summary of the work so far.", [], USAGE),   # the compaction call
            ("Done.", [], USAGE),                           # the step's own call
        ])
        session.shutdown()

        self.assertEqual(len(stub.calls), 2)
        self.assertEqual(session.history_index, 1)          # history_0 -> history_1
        # The transcript really was replaced by the summary turn.
        self.assertIn("A summary of the work so far.", session.messages[1]["content"])

        session_dir = os.path.join(self.cache_dir, session.unique_id)
        for name in ("history_0.json", "history_1.json",
                     "usage_0.json", "usage_1.json", "total_usage.json"):
            with self.subTest(file=name):
                self.assertIn(name, os.listdir(session_dir))

        # The summariser is billed to the segment it closes; the step's own call
        # to the fresh one. The session total spans both.
        self.assertEqual(persistence.load_usage_segment(session_dir, 0)["llm_calls"], 1)
        self.assertEqual(persistence.load_usage_segment(session_dir, 1)["llm_calls"], 1)
        total = persistence.load_total_usage_from_dir(session_dir)
        self.assertEqual(total["llm_calls"], 2)
        self.assertEqual(total["total_tokens"], 2 * USAGE["total_tokens"])
        self.assertEqual(total, session.total_usage)

    def test_a_rollover_flushes_the_closing_segment(self):
        """``history_0.json`` must hold every message its recorded turns point at.

        The interactive loop only persists on quit, so the compaction is the last
        chance to write the segment it is closing. Without that flush the file
        stayed short by every turn recorded in it, and a ``/reset`` (or a restore)
        rewound to a state the user never saw.
        """
        session = self._session(max_context_tokens=1)
        session.messages = [{"role": "system", "content": "S"},
                            {"role": "user", "content": "the question"},
                            {"role": "assistant", "content": "the answer"}]
        session.summary = [{"turn": 1, "what": "assistant", "history_file": 0, "index": 2}]

        self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])
        session.shutdown()

        self.assertEqual(session.history_index, 1)
        session_dir = os.path.join(self.cache_dir, session.unique_id)
        with open(os.path.join(session_dir, "history_0.json"), "r", encoding="utf-8") as f:
            closed = json.load(f)
        self.assertEqual([m.get("content") for m in closed],
                         ["S", "the question", "the answer"])
        # Atomic writes: nothing half-written is left lying around.
        self.assertEqual([n for n in os.listdir(session_dir) if ".tmp." in n], [])

    def test_the_typed_turn_survives_the_rollover(self):
        """The line typed just before a compaction is still the turn answered.

        The bug: the rollover replaced the transcript with (system, summary), so
        the user message this step existed to answer was summarised away. The
        model answered the summary ("Understood. What would you like me to
        do?"), and the typed line was missing from the live transcript, from the
        new segment, from the recorder's target and therefore from a restore --
        from the user's side, they typed an instruction and nothing happened.
        """
        session = self._session(max_context_tokens=1)
        typed = "please fix the parser bug in main.py"
        session.messages.append({"role": "user", "content": typed})

        self._run_step(session, [
            ("A summary of the work so far.", [], USAGE),   # the compaction call
            ("Understood. Fixing the parser.", [], USAGE),  # the step's own call
        ])
        session.shutdown()

        # 1) the model was asked to answer the typed line, not the summary.
        self.assertEqual(session.messages[-2], {"role": "user", "content": typed})
        self.assertIn("A summary of the work so far.", session.messages[-3]["content"])

        # 2) the recorder's last record reaches it, so /history and /reset can.
        last = session.summary[-1]
        self.assertEqual(last["history_file"], session.history_index)
        self.assertIn(typed, [m.get("content") for m in session._segment_upto(last)])

        # 3) it is in the segment the recorder points at, so a restore has it.
        session_dir = os.path.join(self.cache_dir, session.unique_id)
        with open(os.path.join(session_dir, f"history_{last['history_file']}.json"),
                  "r", encoding="utf-8") as f:
            segment = json.load(f)
        self.assertIn(typed, [m.get("content") for m in segment])

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = session_dir
        restored = AgentSession(args, console=silent_console())
        restored.start_session()
        self.assertEqual([m.get("content") for m in restored.messages],
                         [m.get("content") for m in session.messages])
        self.assertIn(typed, [m.get("content") for m in restored.messages])

    def test_the_stale_prompt_report_is_dropped_after_a_rollover(self):
        session = self._session(max_context_tokens=1)
        session.last_prompt_tokens = 12_345
        session.last_prompt_message_count = len(session.messages)

        self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])

        # Keeping the pre-compaction report would describe a transcript that no
        # longer exists, and would immediately re-trigger compaction next step.
        self.assertEqual(session.last_prompt_tokens, USAGE["prompt_tokens"])
        self.assertEqual(session.last_prompt_message_count, 2)

    def test_many_short_messages_never_compact_on_their_own(self):
        # 500 tiny turns sit far below any token budget. The number of messages
        # is not a trigger any more, so nothing may happen here -- previously a
        # message-count cap would have compacted a perfectly healthy history.
        session = self._session(model="gpt-4o")
        session.messages = ([{"role": "system", "content": "S"}]
                            + [{"role": "user", "content": "hi"}] * 500)

        stub = self._run_step(session, [("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 1)        # no summariser call
        self.assertEqual(session.history_index, 0)
        self.assertEqual(len(session.messages), 502)   # transcript untouched

    def test_compaction_ignores_a_max_messages_style_attribute(self):
        # A ``max_messages`` attribute on the args namespace must have no
        # effect: the wiring reads tokens only, so an over-large turn count is
        # not compacted even when something sets that attribute.
        session = self._session(model="gpt-4o")
        session.args.max_messages = 2
        session.messages = ([{"role": "system", "content": "S"}]
                            + [{"role": "user", "content": "hi"}] * 20)

        stub = self._run_step(session, [("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(session.history_index, 0)

    def test_a_large_transcript_compacts_on_tokens_with_few_messages(self):
        # Four messages, each ~50k tokens: well over the fallback budget, and
        # only four messages in total -- so this can only be a token trigger.
        session = self._session()                    # unknown model -> fallback
        session.messages = [{"role": "system", "content": "S"}] + [
            {"role": "user", "content": "x" * 200_000} for _ in range(4)]

        stub = self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 2)
        self.assertEqual(session.history_index, 1)
        self.assertLess(sum(len(m.get("content") or "")
                            for m in session.messages), 4 * 200_000)

    def test_the_server_reported_prompt_size_drives_the_trigger(self):
        # The transcript is tiny, but the server says the last prompt was huge:
        # that count is authoritative for the prefix it covered, so it compacts.
        session = self._session(max_context_tokens=1_000)
        session.last_prompt_tokens = 5_000
        session.last_prompt_message_count = len(session.messages)

        stub = self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 2)
        self.assertEqual(session.history_index, 1)

    def test_the_output_reserve_counts_towards_the_limit(self):
        # 1_000-token budget with a 900-token reply reserve: a request that leaves
        # almost no room for the answer is already too big.
        session = self._session(max_context_tokens=1_000, max_tokens=900)
        self.assertEqual(session.compaction_budget.reserved_output_tokens, 900)
        session.last_prompt_tokens = 200
        session.last_prompt_message_count = len(session.messages)

        stub = self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 2)        # 200 + 900 > 1_000
        self.assertEqual(session.history_index, 1)

    def test_the_same_request_fits_when_the_reserve_is_smaller(self):
        session = self._session(max_context_tokens=1_000, max_tokens=100)
        session.last_prompt_tokens = 200
        session.last_prompt_message_count = len(session.messages)

        stub = self._run_step(session, [("Done.", [], USAGE)])

        self.assertEqual(len(stub.calls), 1)        # 200 + 100 < 1_000
        self.assertEqual(session.history_index, 0)

    def test_token_compaction_still_reinjects_the_plan_and_goal_in_plan_mode(self):
        session = self._session(plan=True, max_context_tokens=1)
        session.plan_manager.write_goal("do the thing")
        session.plan_manager.write_plan("1. do it")

        self._run_step(session, [("Summary.", [], USAGE), ("Done.", [], USAGE)])

        self.assertEqual(session.history_index, 1)
        joined = "\n".join(str(m.get("content") or "") for m in session.messages)
        self.assertIn("do the thing", joined)        # CURRENT_GOAL.md re-injected
        self.assertIn("1. do it", joined)            # CURRENT_PLAN.md re-injected

    def test_the_restored_session_starts_from_an_estimate(self):
        original = self._session()
        original.messages = [{"role": "system", "content": "S"},
                             {"role": "user", "content": "hello"}]
        original.shutdown()

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = os.path.join(self.cache_dir, original.unique_id)
        restored = AgentSession(args, console=silent_console())
        restored.start_session()

        # No prompt has been sent in this process, so there is no server report
        # to trust: the projection must fall back to the heuristic estimate.
        self.assertEqual(restored.last_prompt_tokens, 0)
        self.assertEqual(restored.last_prompt_message_count, 0)
        self.assertGreater(restored._projected_input_tokens(), 0)


# ---------------------------------------------------------------------------
# scenario 7 -- restoring a persisted session
# ---------------------------------------------------------------------------

class RestoreSessionEndToEndTest(unittest.TestCase):
    """A persisted session is restored in place, not replaced by a new one."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _run_one_turn(self):
        stub = StubLLM([
            ("First answer.", [], USAGE),
        ])
        args = make_args(self.workdir, self.cache_dir)
        session = AgentSession(args, console=silent_console())
        run_loop(session, "Remember this.", stub)
        return session

    def test_round_trip_restores_messages_usage_and_id(self):
        original = self._run_one_turn()
        session_dir = os.path.join(self.cache_dir, original.unique_id)

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = session_dir
        restored = AgentSession(args, console=silent_console())
        restored.start_session()

        self.assertEqual(restored.unique_id, original.unique_id)     # in place
        self.assertEqual(restored.session_dir, session_dir)
        self.assertEqual([m.get("content") for m in restored.messages],
                         [m.get("content") for m in original.messages])
        self.assertEqual(restored.token_stats["total_tokens"],
                         original.token_stats["total_tokens"])
        self.assertEqual(restored.total_usage["total_tokens"],
                         original.total_usage["total_tokens"])

    def test_load_last_session_continues_the_previous_session(self):
        """--load-last-session: the whole point of the session pointer.

        Session A runs one real turn, which records A in
        ``--allowed-dir/.mini_tests``.
        Session B is then built with --load-last-session (and no
        --restore-session): it must continue A *in place* -- same id, same
        messages -- rather than start a fresh session, and must not ask.
        """
        original = self._run_one_turn()

        args = make_args(self.workdir, self.cache_dir)
        args.load_last_session = True
        resumed = AgentSession(args, console=silent_console())
        with mock.patch.object(AgentSession, "show_resume_choice",
                               side_effect=AssertionError("asked despite --load-last-session")):
            resumed.start_session()

        self.assertEqual(resumed.unique_id, original.unique_id)
        self.assertEqual(resumed.session_dir, original.session_dir)
        self.assertEqual([m.get("content") for m in resumed.messages],
                         [m.get("content") for m in original.messages])
        self.assertEqual(resumed.token_stats["total_tokens"],
                         original.token_stats["total_tokens"])

    def test_load_last_session_without_a_pointer_starts_fresh(self):
        """No pointer recorded yet: nothing to continue, so start clean."""
        args = make_args(self.workdir, self.cache_dir)
        args.load_last_session = True
        session = AgentSession(args, console=silent_console())
        with mock.patch.object(AgentSession, "show_resume_choice",
                               side_effect=AssertionError("offered a resume for a missing session")):
            session.start_session()

        self.assertIsNotNone(session.unique_id)
        self.assertEqual(len(session.messages), 1)          # just the system prompt
        self.assertEqual(session.messages[0]["role"], "system")

    def test_load_last_session_pointer_survives_a_second_run(self):
        """The pointer keeps naming the live session across runs."""
        first = self._run_one_turn()

        args = make_args(self.workdir, self.cache_dir)
        args.load_last_session = True
        second = AgentSession(args, console=silent_console())
        second.start_session()

        from mini_code_cli import persistence
        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(first.session_dir))
        self.assertEqual(second.unique_id, first.unique_id)

    def test_restore_session_takes_precedence_over_the_flag(self):
        """--restore-session wins when both are given."""
        original = self._run_one_turn()

        args = make_args(self.workdir, self.cache_dir)
        args.load_last_session = True
        args.restore_session = original.session_dir
        session = AgentSession(args, console=silent_console())
        session.start_session()

        self.assertEqual(session.unique_id, original.unique_id)

    def test_a_folder_without_a_recorder_is_refused(self):
        """No summary_history.json means unrestorable: a fresh session, plus a note.

        There is no rebuild-the-recorder-from-the-transcript path any more: it
        had to invent turn boundaries, and then a restored session behaved
        differently under /reset than a recorded one.
        """
        session_dir = os.path.join(self.cache_dir, "bare_session")
        os.makedirs(session_dir)
        with open(os.path.join(session_dir, "history_0.json"), "w") as f:
            json.dump([
                {"role": "system", "content": "S"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ], f)

        console = silent_console()
        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = session_dir
        session = AgentSession(args, console=console)
        session.start_session()

        # A brand-new session, not the refused folder ...
        self.assertNotEqual(session.unique_id, "bare_session")
        self.assertNotEqual(session.session_dir, session_dir)
        self.assertEqual(session.summary, [])
        self.assertEqual([m["role"] for m in session.messages], ["system"])
        # ... and the folder itself was not written into.
        self.assertEqual(os.listdir(session_dir), ["history_0.json"])


# ---------------------------------------------------------------------------
# scenario 8 -- the tool set itself (edit + the richer common parameters)
# ---------------------------------------------------------------------------

class ToolBehaviourTest(unittest.TestCase):
    """The new ``edit`` tool, the extended parameters, and their schemas."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.workdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_read_file_offset_and_limit(self):
        path = self._write("lines.txt", "one\ntwo\nthree\nfour\n")

        self.assertEqual(tools.read_file(path), "one\ntwo\nthree\nfour\n")
        partial = tools.read_file(path, offset=2, limit=2)
        self.assertIn("two", partial)
        self.assertIn("three", partial)
        self.assertNotIn("one\n", partial)
        self.assertNotIn("four", partial)
        self.assertIn("[lines 2-3 of 4]", partial)
        self.assertIn("Error", tools.read_file(path, offset=99))

    def test_read_file_reports_a_missing_file(self):
        self.assertTrue(tools.read_file(os.path.join(self.workdir, "nope")).startswith("Error"))

    def test_edit_replaces_a_unique_match(self):
        path = self._write("code.py", "def f():\n    return 1\n")

        result = tools.edit(path, "return 1", "return 2")

        self.assertIn("replaced 1 occurrence", result)
        with open(path) as f:
            self.assertEqual(f.read(), "def f():\n    return 2\n")

    def test_edit_refuses_an_ambiguous_match(self):
        path = self._write("dup.txt", "x = 1\nx = 1\n")

        result = tools.edit(path, "x = 1", "x = 2")

        self.assertIn("matches 2 times", result)
        with open(path) as f:                     # nothing was written
            self.assertEqual(f.read(), "x = 1\nx = 1\n")

    def test_edit_replace_all_is_explicit(self):
        path = self._write("dup.txt", "x = 1\nx = 1\n")

        self.assertIn("replaced 2 occurrence", tools.edit(path, "x = 1", "x = 2",
                                                          replace_all=True))
        with open(path) as f:
            self.assertEqual(f.read(), "x = 2\nx = 2\n")

    def test_edit_refuses_a_missing_or_empty_old_string(self):
        path = self._write("f.txt", "hello\n")

        self.assertIn("was not found", tools.edit(path, "goodbye", "hi"))
        self.assertIn("must not be empty", tools.edit(path, "", "hi"))

    def test_edit_works_end_to_end_and_is_sandboxed(self):
        path = self._write("target.txt", "before\n")
        edit_call = tool_call("c1", "edit",
                              {"file_path": path, "old_string": "before",
                               "new_string": "after"})
        stub = StubLLM([
            ("", [edit_call], USAGE),
            ("Done.", [], USAGE),
        ])
        session = AgentSession(make_args(self.workdir, tempfile.mkdtemp()),
                               console=silent_console())
        run_loop(session, "Edit target.txt.", stub)

        with open(path) as f:
            self.assertEqual(f.read(), "after\n")

        # Outside --allowed-dir it is refused, exactly like the other file tools.
        from mini_code_cli.sandbox import tool_paths_are_allowed
        outside = os.path.join(tempfile.gettempdir(), "elsewhere.txt")
        self.assertFalse(tool_paths_are_allowed(
            "edit", {"file_path": outside}, self.workdir))
        self.assertTrue(tool_paths_are_allowed(
            "edit", {"file_path": path}, self.workdir))

    def test_the_schema_exposes_the_new_tool_and_common_parameters(self):
        schema = {t["function"]["name"]: t["function"]
                  for t in tools.util_get_tools()}

        self.assertIn("edit", schema)
        edit_params = schema["edit"]["parameters"]
        self.assertEqual(set(edit_params["required"]),
                         {"file_path", "old_string", "new_string"})
        self.assertEqual(edit_params["properties"]["replace_all"]["type"], "boolean")

        # Common, non-breaking knobs are offered ...
        self.assertIn("offset", schema["read_file"]["parameters"]["properties"])
        self.assertIn("limit", schema["read_file"]["parameters"]["properties"])
        for name in ("recursive", "include", "ignore_case", "line_numbers"):
            self.assertIn(name, schema["grep"]["parameters"]["properties"])
        self.assertIn("root", schema["glob"]["parameters"]["properties"])
        self.assertIn("max_chars", schema["read_website"]["parameters"]["properties"])
        self.assertIn("cwd", schema["run_shell_command"]["parameters"]["properties"])

    def test_a_malformed_tool_call_is_reported_not_fatal(self):
        # The model sent a JSON list instead of an object. That used to raise
        # AttributeError (tool_params.get) and end the session; now it is fed
        # back as an ordinary tool error and the loop keeps going.
        bad_call = {"id": "c1", "function": {"name": "read_file", "arguments": "[1, 2]"}}
        cache_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, cache_dir, ignore_errors=True)
        stub = StubLLM([
            ("", [bad_call], USAGE),
            ("Recovered.", [], USAGE),
        ])
        session = AgentSession(make_args(self.workdir, cache_dir),
                               console=silent_console())
        run_loop(session, "Read something.", stub)

        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("must be a JSON object", tool_msgs[0]["content"])
        self.assertEqual(session._last_assistant_text(), "Recovered.")

    def test_grep_is_recursive_with_an_include_filter(self):
        os.makedirs(os.path.join(self.workdir, "sub"))
        self._write("a.py", "needle\n")
        self._write(os.path.join("sub", "b.py"), "needle\n")
        self._write(os.path.join("sub", "c.txt"), "needle\n")

        flat = tools.grep("needle", self.workdir)
        self.assertIn("a.py", flat)
        self.assertNotIn("b.py", flat)              # not recursive by default

        found = tools.grep("needle", self.workdir, recursive=True, include="*.py")
        self.assertIn("b.py", found)
        self.assertNotIn("c.txt", found)            # filtered out
        self.assertIn("No matches", tools.grep("absent-needle", self.workdir))

    def test_glob_resolves_against_a_root(self):
        self._write("one.py", "")
        self._write("two.txt", "")

        found = tools.glob("*.py", root=self.workdir)
        self.assertIn("one.py", found)
        self.assertNotIn("two.txt", found)




if __name__ == "__main__":
    unittest.main()
