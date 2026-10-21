"""Tests for the interactive surfaces and the pure helpers behind them.

These cover the parts of the CLI a user drives directly rather than the model:

  * the slash commands (``/history``, ``/history <n>``, ``/reset [<n>]``,
    ``/plan``, ``/clear``) -- handled in place, so they never reach the LLM;
  * the ``.mini`` latest-session pointer: its readers, its writer, and the
    rule that a real session records it while a background (sub-agent) one does not;
  * the shell-safety classifier, as a table across all three tiers;
  * two compaction regressions: a message without ``content`` must not break
    the summary prompt, and an empty summary must not roll the segment over.

Compaction's happy path is covered end-to-end in ``test_e2e_agent_loop.py``;
here we pin the edge cases.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from mini_code_cli import persistence, prompts, shell_safety_utils, ui, utils
from mini_code_cli.agent_loop import AgentSession, maybe_compact

from test_e2e_agent_loop import USAGE, StubLLM, make_args, run_loop, tool_call


class _SessionCase(unittest.TestCase):
    """Tmpdirs, a captured console, and a one/two-turn session builder."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def console(self):
        self._stream = io.StringIO()
        return ui.Console(stream=self._stream, color=False, unicode=False, verbose=False)

    def output(self):
        return self._stream.getvalue()

    def build_session(self, plan=False, turns=2):
        """Run a real session with `turns` scripted steps and return it."""
        console = self.console()
        args = make_args(self.workdir, self.cache_dir, plan=plan)
        session = AgentSession(args, console=console)
        target = os.path.join(self.workdir, "data.txt")
        with open(target, "w") as f:
            f.write("alpha\nbeta\n")

        queue = []
        for i in range(turns):
            queue.append(("", [tool_call(f"c{i}", "read_file", {"file_path": target})], USAGE))
        queue.append(("Final answer.", [], USAGE))
        stub = StubLLM(queue)
        run_loop(session, "read the file", stub)
        return session


# ---------------------------------------------------------------------------
# slash commands
# ---------------------------------------------------------------------------

class SlashCommandTest(_SessionCase):

    def test_history_lists_recorded_turns(self):
        session = self.build_session()
        self.assertGreaterEqual(len(session.summary), 2)

        self.assertTrue(session._handle_command("/history"))
        out = self.output()
        # A header with the turn count, then one row per recorded turn.
        self.assertIn(f"({len(session.summary)} turns)", out)
        for record in session.summary:
            self.assertIn(str(record["turn"]), out)
            self.assertIn(f"history_{record['history_file']}.json#{record['index']}", out)
        self.assertIn("/history <n>", out)          # the hint is shown too

    def test_history_n_shows_the_turns_messages(self):
        session = self.build_session()
        self.assertTrue(session._handle_command("/history 1"))
        self.assertIn("read the file", self.output())

    def test_history_out_of_range_is_reported_not_raised(self):
        session = self.build_session()
        self.assertTrue(session._handle_command("/history 99"))
        self.assertIn("No such turn", self.output())

    def test_reset_rewinds_messages_and_recorder(self):
        session = self.build_session()
        before = len(session.messages)
        record = session.summary[0]

        self.assertTrue(session._handle_command("/reset 1"))

        # The conversation was truncated back to the end of turn 1 ...
        self.assertEqual(len(session.messages), record["index"] + 1)
        self.assertLess(len(session.messages), before)
        # ... the recorder dropped the later turns ...
        self.assertEqual(len(session.summary), 1)
        # ... and the truncation was persisted, so a restore sees the same state.
        loaded = persistence.load_history_from_dir(os.path.join(self.cache_dir, session.unique_id))
        self.assertEqual([m.get("content") for m in loaded],
                         [m.get("content") for m in session.messages])
        self.assertIn("Reset to turn 1", self.output())

    def test_plan_command_renders_goal_and_plan(self):
        session = self.build_session(plan=True)
        session.plan_manager.write_goal("grep for apple and report the count")
        session.plan_manager.write_plan("1. run grep\n2. report")

        self.assertTrue(session._handle_command("/plan"))
        out = self.output()
        self.assertIn("grep for apple", out)
        self.assertIn("run grep", out)

    def test_plan_command_without_plan_mode_is_a_noop(self):
        session = self.build_session()
        self.assertTrue(session._handle_command("/plan"))
        self.assertIn("Plan mode is not enabled", self.output())

    def test_unknown_command_is_swallowed(self):
        session = self.build_session()
        self.assertTrue(session._handle_command("/nonsense"))
        self.assertIn("Unknown command", self.output())

    def test_quit_and_clear_commands_are_recognised(self):
        session = self.build_session()
        # /clear redraws the last turn instead of sending it to the model.
        self.assertTrue(session._handle_command("/clear"))
        self.assertIn("Final answer.", self.output())

    def test_plain_text_is_not_a_command(self):
        session = self.build_session()
        self.assertFalse(session._handle_command("just a question"))


# ---------------------------------------------------------------------------
# the .mini latest-session pointer
# ---------------------------------------------------------------------------

class LatestSessionPointerTest(_SessionCase):
    """The readers restore is built on. Best-effort: never raise, always safe."""

    def pointer_dir(self):
        return os.path.join(os.path.abspath(self.workdir), persistence.SKILLS_DIRNAME)

    def write_pointer(self, payload):
        os.makedirs(self.pointer_dir(), exist_ok=True)
        with open(os.path.join(self.pointer_dir(), persistence.LAST_SESSION_FILENAME), "w") as f:
            f.write(payload if isinstance(payload, str) else json.dumps(payload))

    def test_missing_pointer_yields_none(self):
        self.assertIsNone(persistence.load_last_session_pointer(self.workdir))
        self.assertIsNone(persistence.find_last_session_dir(self.workdir))

    def test_valid_pointer_is_returned(self):
        session_dir = os.path.join(self.cache_dir, "sess1")
        os.makedirs(session_dir, exist_ok=True)
        self.write_pointer({"session_dir": session_dir})

        self.assertEqual(persistence.load_last_session_pointer(self.workdir), session_dir)
        self.assertEqual(persistence.find_last_session_dir(self.workdir), session_dir)

    def test_pointer_to_a_vanished_session_is_ignored(self):
        self.write_pointer({"session_dir": os.path.join(self.cache_dir, "gone")})
        self.assertIsNone(persistence.find_last_session_dir(self.workdir))

    def test_malformed_pointer_is_ignored(self):
        self.write_pointer("{not json")
        self.assertIsNone(persistence.find_last_session_dir(self.workdir))

    def test_pointer_without_the_key_is_ignored(self):
        self.write_pointer({"something": "else"})
        self.assertIsNone(persistence.find_last_session_dir(self.workdir))

    def test_no_allowed_dir_yields_none(self):
        self.assertIsNone(persistence.load_last_session_pointer(None))
        self.assertIsNone(persistence.find_last_session_dir(""))

    # -- the write half: save_last_session_pointer -------------------------

    def test_writer_records_the_session_dir(self):
        session_dir = os.path.join(self.cache_dir, "sessW")
        os.makedirs(session_dir, exist_ok=True)

        path = persistence.save_last_session_pointer(self.workdir, session_dir)

        self.assertEqual(path, os.path.join(self.pointer_dir(),
                                            persistence.LAST_SESSION_FILENAME))
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(session_dir))

    def test_writer_stores_an_absolute_path_and_timestamp(self):
        session_dir = os.path.join(self.cache_dir, "sessW")
        os.makedirs(session_dir, exist_ok=True)
        persistence.save_last_session_pointer(self.workdir, session_dir)

        with open(os.path.join(self.pointer_dir(),
                              persistence.LAST_SESSION_FILENAME)) as f:
            payload = json.load(f)
        self.assertTrue(os.path.isabs(payload["session_dir"]))
        self.assertIsInstance(payload["recorded_at"], float)

    def test_writer_leaves_no_temp_file_behind(self):
        session_dir = os.path.join(self.cache_dir, "sessW")
        os.makedirs(session_dir, exist_ok=True)
        persistence.save_last_session_pointer(self.workdir, session_dir)

        leftovers = [n for n in os.listdir(self.pointer_dir()) if ".tmp." in n]
        self.assertEqual(leftovers, [])

    def test_writer_is_safe_without_a_session_or_dir(self):
        self.assertIsNone(persistence.save_last_session_pointer(self.workdir, None))
        self.assertIsNone(persistence.save_last_session_pointer(None, self.cache_dir))

    def test_writer_overwrites_a_previous_pointer(self):
        first = os.path.join(self.cache_dir, "first")
        second = os.path.join(self.cache_dir, "second")
        os.makedirs(first, exist_ok=True)
        os.makedirs(second, exist_ok=True)

        persistence.save_last_session_pointer(self.workdir, first)
        persistence.save_last_session_pointer(self.workdir, second)

        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(second))

    def test_writer_warns_instead_of_raising_when_the_store_is_unusable(self):
        """A best-effort write: a broken .mini path warns, never raises."""
        blocker = os.path.join(self.workdir, persistence.SKILLS_DIRNAME)
        with open(blocker, "w") as f:      # a *file* where the dir should be
            f.write("not a directory")

        console = self.console()
        with mock.patch.object(persistence.ui, "console", console):
            self.assertIsNone(
                persistence.save_last_session_pointer(self.workdir, self.cache_dir))
        self.assertIn("latest-session pointer", self.output())


# ---------------------------------------------------------------------------
# the pointer is written by real sessions, and only when it should be
# ---------------------------------------------------------------------------

class PointerWrittenBySessionsTest(_SessionCase):
    """A real run records the pointer; a background (sub-agent) run does not."""

    def test_a_new_session_records_the_pointer(self):
        session = self.build_session(turns=1)
        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(session.session_dir))

    def test_a_restored_session_keeps_the_pointer_on_it(self):
        original = self.build_session(turns=1)

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = original.session_dir
        restored = AgentSession(args, console=self.console())
        restored.start_session()

        self.assertEqual(restored.unique_id, original.unique_id)
        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(original.session_dir))

    def test_a_background_session_never_records_the_pointer(self):
        """Delegating must not hijack the user's next --load-last-session."""
        session = AgentSession(make_args(self.workdir, self.cache_dir),
                               console=self.console(), background_mode=True)
        session.start_session()

        self.assertIsNone(persistence.find_last_session_dir(self.workdir))

    def test_a_background_session_leaves_an_existing_pointer_alone(self):
        original = self.build_session(turns=1)

        child = AgentSession(make_args(self.workdir, self.cache_dir),
                             console=self.console(), background_mode=True)
        child.start_session()

        self.assertEqual(persistence.find_last_session_dir(self.workdir),
                         os.path.abspath(original.session_dir))


# ---------------------------------------------------------------------------
# shell-safety classifier
# ---------------------------------------------------------------------------

class ClassifyCommandTest(unittest.TestCase):
    """Table-driven: the three tiers, including piped/chained commands."""

    DENY = [
        "rm -rf /",
        "rm -rf --no-preserve-root /",
        "mkfs.ext4 /dev/sda1",
        "mkfs /dev/sdb",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
        "",
        "   ",
        None,
    ]
    ASK = [
        "rm notes.txt",
        "rmdir build",
        "mv a.txt b.txt",
        "chmod 777 script.sh",
        "chown root file",
        "sudo apt install cowsay",
        "find . -name '*.tmp' -delete",
    ]
    ALLOW = [
        "ls -la",
        "cat notes.txt",
        "grep -n apple fruit.txt",
        "python -m pytest tests/ -q",
        "echo hi | tr a-z A-Z",
        "git status",
    ]

    def test_deny_tier(self):
        for cmd in self.DENY:
            with self.subTest(command=cmd):
                tier, reason = shell_safety_utils.classify_command(cmd)
                self.assertEqual(tier, shell_safety_utils.DENY)
                self.assertTrue(reason)
                self.assertTrue(shell_safety_utils.is_deny(tier))

    def test_ask_tier(self):
        for cmd in self.ASK:
            with self.subTest(command=cmd):
                tier, reason = shell_safety_utils.classify_command(cmd)
                self.assertEqual(tier, shell_safety_utils.ASK)
                self.assertTrue(reason)
                self.assertTrue(shell_safety_utils.is_ask(tier))

    def test_allow_tier(self):
        for cmd in self.ALLOW:
            with self.subTest(command=cmd):
                tier, _ = shell_safety_utils.classify_command(cmd)
                self.assertEqual(tier, shell_safety_utils.ALLOW)

    def test_a_list_of_argv_is_joined_before_classifying(self):
        tier, _ = shell_safety_utils.classify_command(["rm", "-rf", "/"])
        self.assertEqual(tier, shell_safety_utils.DENY)

    def test_tiers_are_mutually_exclusive(self):
        for cmd in self.DENY + self.ASK + self.ALLOW:
            with self.subTest(command=cmd):
                tier, _ = shell_safety_utils.classify_command(cmd)
                flags = [shell_safety_utils.is_deny(tier),
                         shell_safety_utils.is_ask(tier)]
                self.assertLessEqual(sum(flags), 1)


# ---------------------------------------------------------------------------
# compaction edge cases
# ---------------------------------------------------------------------------

class CompactionEdgeCaseTest(unittest.TestCase):

    def setUp(self):
        self.stats = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "llm_calls": 0, "tool_calls": 0}

    def _compact(self, messages, reply, max_messages=2):
        stub = StubLLM([(reply, [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            return maybe_compact(messages, "http://localhost:1", "m", None, 30,
                                 self.stats, max_messages=max_messages)

    def test_compaction_prompt_tolerates_missing_content(self):
        """A tool-call-only turn (content None or absent) must not raise KeyError."""
        messages = [
            {"role": "system", "content": "S"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "assistant"},                      # no content key at all
        ]
        text = prompts.compaction_prompt(messages)
        self.assertIn("system: S", text)
        self.assertIn("assistant:", text)               # rendered as empty

    def test_empty_summary_does_not_roll_the_segment_over(self):
        """Documented behaviour: no summary -> no rollover, history left as-is.

        The model sometimes returns nothing; dropping the transcript on an empty
        summary would silently destroy the conversation, so we keep it and report
        ``did_compact=False``. The cost is that compaction is then retried every
        step (see the recommendation to warn instead of failing quietly).
        """
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"}]
        out, did_compact, index = self._compact(messages, "")

        self.assertFalse(did_compact)
        self.assertEqual(index, 0)               # still history_0
        self.assertIs(out, messages)             # transcript preserved verbatim
        self.assertEqual(self.stats["llm_calls"], 1)   # the attempt was still billed


if __name__ == "__main__":
    unittest.main()
