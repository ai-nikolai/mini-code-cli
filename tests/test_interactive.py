"""Tests for the interactive surfaces and the pure helpers behind them.

These cover the parts of the CLI a user drives directly rather than the model:

  * the slash commands (``/history``, ``/history <n>``, ``/history last``,
    ``/reset [<n>|last]``, ``/plan``, ``/clear``) -- handled in place, so they
    never reach the LLM;
  * history navigation and rewinding as an *on-disk* contract: turn-selector
    validation, the destructive-by-explicit-command rule, deletion of the
    segments a rewind moved past, rollback when a write fails, and a restore
    that must not resurrect a rewound-away tail;
  * robustness of a damaged session folder: corrupt or empty segments, and a
    restore that refuses (rather than repairs) a recorder it cannot trust;
  * the ``.mini`` latest-session pointer: its readers, its writer, and the
    rule that a real session records it while a background (sub-agent) one does not;
  * the shared ``oneline`` helper (one implementation, three callers);
  * ``resolve_session_dir``, the one rule for what ``--restore-session`` means;
  * model resolution (the lazy, exception-safe ``/v1/models`` probe);
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

from mini_code_cli import agent_loop, persistence, prompts, shell_safety_utils, ui, utils
from mini_code_cli.agent_loop import AgentSession, maybe_compact
from mini_code_cli.config import compaction_budget

from test_e2e_agent_loop import USAGE, StubLLM, make_args, run_loop, tool_call


# The pointer this suite exercises must never land in a real ``.mini``
# directory (that is the user's skills folder + resume pointer), so it is
# recorded under ``.mini_tests`` instead. See mini_code_cli/persistence.py.
persistence.SESSION_DIRNAME = ".mini_tests"


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

    # -- history helpers (shared by the rewind / robustness tests) ---------

    def session_dir(self, session):
        """The on-disk folder backing ``session``."""
        return os.path.join(self.cache_dir, session.unique_id)

    def segment_names(self, session):
        """The history/usage files actually present for ``session``, sorted."""
        return sorted(n for n in os.listdir(self.session_dir(session))
                      if n.startswith("history_") or n.startswith("usage_"))

    def snapshot(self, session):
        """Every file in the session folder, name -> raw text, for "unchanged" checks."""
        directory = self.session_dir(session)
        texts = {}
        for name in sorted(os.listdir(directory)):
            with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
                texts[name] = f.read()
        return texts

    def write_segment(self, session, index, messages):
        """Write one ``history_{index}.json`` for ``session``."""
        return persistence.save_history(messages, self.cache_dir,
                                        session.unique_id, index)

    def step(self, session, queue):
        """Drive one LLM step against a scripted queue (no user input involved)."""
        stub = StubLLM(queue)
        with mock.patch.object(utils, "call_openai_server", new=stub):
            session._step()
        return stub

    def make_two_segment_session(self):
        """A real session whose recorder spans ``history_0.json`` and ``history_1.json``.

        One ordinary turn is recorded in segment 0; the compaction budget is then
        tightened to a single token so the next step rolls over (a *token*
        triggered compaction is the only way a new segment ever starts) and
        records its turns in segment 1. The result is the state a cross-segment
        ``/reset`` has to handle: turns recorded in both segments, segment 1 live.
        """
        session = self.build_session(turns=1)
        self.assertEqual(persistence.history_segment_indices(self.session_dir(session)), [0])

        session.compaction_budget = compaction_budget(
            model="test-model", explicit_limit=1, max_output_tokens=None)
        self.step(session, [("A summary of the start.", [], USAGE),
                            ("After the compaction.", [], USAGE)])

        self.assertEqual(persistence.history_segment_indices(self.session_dir(session)),
                         [0, 1])
        self.assertEqual(session.history_index, 1)
        self.assertEqual([r["history_file"] for r in session.summary], [0, 0, 1, 1])
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
        loaded = persistence.load_history_from_dir(
            os.path.join(self.cache_dir, session.unique_id), session.history_index)
        self.assertEqual([m.get("content") for m in loaded],
                         [m.get("content") for m in session.messages])
        self.assertIn("Reset to turn 1", self.output())

    # -- turn selectors: one rule for /history and /reset ------------------

    def test_history_rejects_zero_negative_and_out_of_range_turns(self):
        """``summary[int(arg) - 1]`` had no lower bound: ``/history 0`` showed
        the *last* turn and ``/history -2`` the third-to-last, silently."""
        session = self.build_session(turns=1)
        for bad in ("0", "-1", "-2", "99"):
            with self.subTest(turn=bad):
                self.assertTrue(session._handle_command("/history " + bad))
        self.assertEqual(self.output().count("No such turn"), 4)

    def test_history_rejects_a_non_numeric_turn(self):
        session = self.build_session(turns=1)
        self.assertTrue(session._handle_command("/history junk"))
        self.assertIn("No such turn 'junk'", self.output())

    def test_history_last_shows_the_most_recent_turn(self):
        session = self.build_session(turns=1)
        self.assertTrue(session._handle_command("/history last"))
        self.assertNotIn("No such turn", self.output())
        self.assertIn("Final answer.", self.output())

    def test_reset_rejects_an_invalid_turn_without_touching_anything(self):
        session = self.build_session(turns=1)
        before = self.snapshot(session)
        messages = [m.get("content") for m in session.messages]

        for bad in ("0", "-2", "junk"):
            with self.subTest(turn=bad):
                self.assertTrue(session._handle_command("/reset " + bad))

        self.assertEqual(self.output().count("No such turn"), 3)
        self.assertEqual([m.get("content") for m in session.messages], messages)
        self.assertEqual(self.snapshot(session), before)

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
# rewinding: the on-disk contract
# ---------------------------------------------------------------------------

class HistoryRewindRegressionTest(_SessionCase):
    """``/reset`` must stick: across a restart, across segments, on any disk.

    The bugs these pin down all shared one shape -- the in-memory session was
    rewound while the *disk* kept (or was missing) something else, so a later
    restore silently undid the rewind or shortened the transcript.
    """

    def test_cross_segment_reset_deletes_higher_segments(self):
        """A rewind past a compaction removes the segments it moved past.

        Leaving ``history_1.json`` behind was the actual bug: a restore then
        picked the newest segment, so the orphan resurrected exactly the turns
        the user had just rewound away.
        """
        session = self.make_two_segment_session()
        session._persist()                     # as a quit would: flush segment 1 + usage_1
        self.assertEqual(self.segment_names(session),
                         ["history_0.json", "history_1.json",
                          "usage_0.json", "usage_1.json"])

        self.assertTrue(session._handle_command("/reset 1"))

        self.assertEqual(self.segment_names(session), ["history_0.json", "usage_0.json"])
        self.assertEqual(session.history_index, 0)
        self.assertIn("Removed 2 later history/usage segment file(s).", self.output())

    def test_cross_segment_reset_survives_a_restart(self):
        session = self.make_two_segment_session()
        session._persist()
        self.assertIn("After the compaction.",
                      [m.get("content") for m in session.messages])

        self.assertTrue(session._handle_command("/reset 1"))
        kept = [m.get("content") for m in session.messages]

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = self.session_dir(session)
        restored = AgentSession(args, console=self.console())
        with mock.patch.object(AgentSession, "show_resume_choice", return_value="new"):
            restored.start_session()

        self.assertEqual([m.get("content") for m in restored.messages], kept)
        self.assertNotIn("After the compaction.",
                         [m.get("content") for m in restored.messages])
        self.assertEqual(len(restored.summary), 1)          # the dropped turns are gone
        self.assertEqual(restored.history_index, 0)         # and we resume on segment 0

    def test_reset_reloads_segment_usage_and_preserves_the_session_total(self):
        """A rewind does not un-spend tokens, and must not cross-wire segments."""
        session = self.make_two_segment_session()
        session._persist()
        directory = self.session_dir(session)
        segment_0 = persistence.load_usage_segment(directory, 0)
        thread_total = dict(session.total_usage)
        self.assertNotEqual(segment_0, persistence.load_usage_segment(directory, 1))

        self.assertTrue(session._handle_command("/reset 1"))

        # The segment counters are the *target* segment's, so the next save cannot
        # write the later segment's numbers over history_0's usage file.
        self.assertEqual(session.token_stats, segment_0)
        self.assertEqual(session.total_usage, thread_total)
        self.assertEqual(persistence.load_total_usage_from_dir(directory), thread_total)

    def test_reset_rolls_back_when_the_write_fails(self):
        """A failed persist leaves the session exactly as it was (no half-rewind)."""
        session = self.make_two_segment_session()
        session._persist()
        before = self.snapshot(session)
        messages = [m.get("content") for m in session.messages]
        summary = [dict(r) for r in session.summary]
        index = session.history_index

        with mock.patch.object(agent_loop, "save_history",
                               side_effect=OSError("No space left on device")):
            self.assertTrue(session._handle_command("/reset 1"))

        self.assertIn("Could not persist the rewind", self.output())
        self.assertEqual([m.get("content") for m in session.messages], messages)
        self.assertEqual([dict(r) for r in session.summary], summary)
        self.assertEqual(session.history_index, index)
        self.assertEqual(self.snapshot(session), before)

    def test_reset_last_is_accepted(self):
        session = self.build_session(turns=1)
        before = [m.get("content") for m in session.messages]

        self.assertTrue(session._handle_command("/reset last"))

        self.assertNotIn("No such turn", self.output())
        self.assertEqual([m.get("content") for m in session.messages], before)

    def test_bare_reset_on_a_settled_transcript_reports_already_at_turn(self):
        """No argument, nothing to drop: say so and write nothing at all."""
        session = self.build_session(turns=1)
        before = self.snapshot(session)

        self.assertTrue(session._handle_command("/reset"))

        self.assertIn("Already at turn 2", self.output())
        self.assertIn("nothing to rewind", self.output())
        self.assertEqual(self.snapshot(session), before)

    def test_bare_reset_with_an_unrecorded_tail_warns_and_does_not_truncate(self):
        """Messages that no turn records must never be destroyed by a bare /reset."""
        session = self.build_session(turns=1)
        session.messages.append({"role": "user", "content": "never recorded"})
        before = [m.get("content") for m in session.messages]

        self.assertTrue(session._handle_command("/reset"))

        self.assertIn("were never recorded", self.output())
        self.assertIn("refusing to discard", self.output())
        self.assertEqual([m.get("content") for m in session.messages], before)

    def test_explicit_reset_forces_the_rewind_past_an_unrecorded_tail(self):
        """...but an explicit turn is an explicit instruction, so it does rewind."""
        session = self.build_session(turns=1)
        session.messages.append({"role": "user", "content": "never recorded"})

        self.assertTrue(session._handle_command("/reset 1"))

        self.assertIn("Reset to turn 1", self.output())
        self.assertEqual(len(session.messages), session.summary[0]["index"] + 1)
        self.assertNotIn("never recorded", [m.get("content") for m in session.messages])

    def test_every_completed_step_records_a_turn(self):
        """Even an empty reply is recorded, so a bare /reset has no phantom tail."""
        session = self.build_session(turns=1)
        count = len(session.summary)

        self.step(session, [("", [], USAGE)])          # no text, no tool calls

        self.assertEqual(len(session.summary), count + 1)
        self.assertEqual(session.summary[-1]["what"], "assistant (no response)")
        self.assertEqual(session.summary[-1]["index"], len(session.messages) - 1)

    def test_shutdown_names_the_files_that_really_exist(self):
        session = self.make_two_segment_session()
        session._persist()
        session._handle_command("/reset 1")

        session.shutdown()

        self.assertIn("history_0.json + summary_history.json", self.output())
        self.assertNotIn("history_0..", self.output())
        self.assertEqual([n for n in os.listdir(self.session_dir(session))
                          if n.startswith("history_")], ["history_0.json"])


# ---------------------------------------------------------------------------
# robustness: a damaged session folder must degrade, never crash
# ---------------------------------------------------------------------------

class HistoryRobustnessTest(_SessionCase):
    """Corrupt segments, untrustworthy recorders, unrestorable folders.

    A folder this version cannot read is *refused* -- one note saying why, then a
    clean fresh session. Nothing is repaired, renumbered or rebuilt, because a
    repaired recorder silently disagrees with the transcripts on disk.
    """

    def test_history_and_reset_survive_a_corrupt_segment(self):
        session = self.make_two_segment_session()
        session._persist()
        # Corrupt the *older* segment: the live one is served from memory.
        with open(os.path.join(self.session_dir(session), "history_0.json"), "w") as f:
            f.write("{not json")

        self.assertTrue(session._handle_command("/history 1"))
        self.assertIn("Could not load the history segment", self.output())

        messages = [m.get("content") for m in session.messages]
        before = self.snapshot(session)
        self.assertTrue(session._handle_command("/reset 1"))
        self.assertIn("session is unchanged", self.output())
        self.assertEqual([m.get("content") for m in session.messages], messages)
        self.assertEqual(self.snapshot(session), before)

    def test_an_empty_or_absurd_segment_index_is_unusable(self):
        session = self.build_session(turns=1)
        self.write_segment(session, 5, [])

        self.assertIsNone(session._load_segment(5))       # empty list: not a segment
        self.assertIsNone(session._load_segment(-1))
        self.assertIsNone(session._load_segment("junk"))
        self.assertIsNone(session._load_segment(404))     # missing file

    def _restore(self, session):
        """Restore ``session``'s folder and return (the new session, its folder)."""
        folder = self.session_dir(session)
        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = folder
        restored = AgentSession(args, console=self.console())
        restored.start_session()
        return restored, folder

    def test_a_junk_recorder_is_refused_not_repaired(self):
        """Dropping/renumbering records used to shift every later turn number."""
        session = self.build_session(turns=1)
        persistence.save_summary(
            [None, "nope", {"turn": "x", "index": -5},
             {"what": "assistant", "history_file": 0, "index": 1}],
            self.cache_dir, session.unique_id)

        restored, _ = self._restore(session)

        self.assertNotEqual(restored.unique_id, session.unique_id)   # a fresh session
        self.assertEqual(restored.summary, [])
        self.assertEqual([m["role"] for m in restored.messages], ["system"])
        self.assertIn("Cannot restore", self.output())
        self.assertIn("record 1 is not a JSON object", self.output())

    def test_a_folder_without_a_recorder_is_not_restored(self):
        """No summary_history.json: refused, not rebuilt from the transcript."""
        folder = os.path.join(self.cache_dir, "bare-session")
        os.makedirs(folder)
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"}]
        persistence.save_history(messages, self.cache_dir, "bare-session", 1)

        args = make_args(self.workdir, self.cache_dir)
        args.restore_session = folder
        restored = AgentSession(args, console=self.console())
        restored.start_session()

        self.assertNotEqual(restored.unique_id, "bare-session")
        self.assertEqual(restored.summary, [])
        self.assertIn("no readable summary_history.json", self.output())
        # The refused folder is left untouched: nothing was written into it.
        self.assertEqual(sorted(os.listdir(folder)), ["history_1.json"])

    def test_an_unreadable_recorder_is_refused(self):
        session = self.build_session(turns=1)
        with open(os.path.join(self.session_dir(session), "summary_history.json"), "w") as f:
            f.write("{not json")

        restored, _ = self._restore(session)

        self.assertNotEqual(restored.unique_id, session.unique_id)
        self.assertIn("no readable summary_history.json", self.output())

    def test_a_recorder_pointing_at_a_missing_segment_is_refused(self):
        """The last record names where the session is: no guessing, no scan."""
        session = self.build_session(turns=1)
        self.assertGreaterEqual(len(session.summary), 1)
        os.unlink(os.path.join(self.session_dir(session), "history_0.json"))

        restored, _ = self._restore(session)

        self.assertNotEqual(restored.unique_id, session.unique_id)
        self.assertIn("history_0.json is missing, empty or unreadable", self.output())
        self.assertEqual([m["role"] for m in restored.messages], ["system"])

    def test_a_valid_session_still_round_trips(self):
        """The strict contract must not break the normal case."""
        session = self.build_session(turns=2)
        session._persist()
        original = [m.get("content") for m in session.messages]

        restored, folder = self._restore(session)

        self.assertEqual(restored.unique_id, session.unique_id)
        self.assertEqual(restored.session_dir, folder)
        self.assertEqual([m.get("content") for m in restored.messages], original)
        self.assertEqual(restored.summary, session.summary)
        self.assertEqual(restored.history_index, session.history_index)
        self.assertEqual(restored.token_stats, persistence.load_usage_segment(folder, 0))
        self.assertNotIn("Cannot restore", self.output())


# ---------------------------------------------------------------------------
# the .mini latest-session pointer
# ---------------------------------------------------------------------------

class LatestSessionPointerTest(_SessionCase):
    """The readers restore is built on. Best-effort: never raise, always safe."""

    def pointer_dir(self):
        return persistence.pointer_dir(self.workdir)

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
        """A best-effort write: a broken .mini_tests path warns, never raises."""
        blocker = os.path.join(self.workdir, persistence.SESSION_DIRNAME)
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
# model resolution (the /v1/models probe)
# ---------------------------------------------------------------------------

class ResolveModelTest(_SessionCase):
    """The vLLM model probe is lazy and exception-safe."""

    def build(self, model=None, console=None):
        args = make_args(self.workdir, self.cache_dir)
        args.model = model
        return AgentSession(args, console=console or self.console())

    def test_an_explicit_model_skips_the_probe_entirely(self):
        with mock.patch.object(utils, "determine_vllm_model",
                               side_effect=AssertionError("probed despite --model")):
            session = self.build(model="my-model")
        self.assertEqual(session.model, "my-model")

    def test_the_probe_supplies_a_model_when_none_was_named(self):
        with mock.patch.object(utils, "determine_vllm_model", return_value="served-model"):
            session = self.build(model=None)
        self.assertEqual(session.model, "served-model")

    def test_a_failed_probe_falls_back_and_warns(self):
        args = make_args(self.workdir, self.cache_dir)
        args.model = None
        console = self.console()
        with mock.patch.object(utils, "determine_vllm_model",
                               side_effect=OSError("connection refused")):
            session = AgentSession(args, console=console)

        self.assertIsNone(session.model)                     # degraded, not fatal
        self.assertIn("Could not detect a model", self.output())

    def test_a_failed_probe_does_not_swallow_interrupts(self):
        """A bare `except:` used to hide KeyboardInterrupt during construction."""
        args = make_args(self.workdir, self.cache_dir)
        args.model = None
        with mock.patch.object(utils, "determine_vllm_model",
                               side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                AgentSession(args, console=self.console())

    def test_the_probe_passes_the_api_key_and_a_bounded_timeout(self):
        args = make_args(self.workdir, self.cache_dir)
        args.model = None
        args.api_key = "secret"
        with mock.patch.object(utils, "determine_vllm_model",
                               return_value=None) as probe:
            AgentSession(args, console=self.console())
        _, kwargs = probe.call_args
        self.assertEqual(kwargs["api_key"], "secret")
        self.assertLessEqual(kwargs["timeout"], 500)


# ---------------------------------------------------------------------------
# resolving what --restore-session refers to
# ---------------------------------------------------------------------------

class ResolveSessionDirTest(unittest.TestCase):
    """One canonical resolver: an existing session dir, a bare id, or None."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cache = os.path.join(self.root, "cache")
        self.session = os.path.join(self.cache, "1792-abc")
        os.makedirs(self.session)
        for name in ("history_0.json", "history_1.json", "summary_history.json"):
            with open(os.path.join(self.session, name), "w") as f:
                f.write("[]")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_none_and_empty(self):
        for value in (None, ""):
            self.assertIsNone(persistence.resolve_session_dir(value))
            self.assertIsNone(persistence.resolve_session_dir(value, self.cache))

    def test_a_session_directory_is_returned(self):
        self.assertEqual(persistence.resolve_session_dir(self.session), self.session)

    def test_a_file_is_not_a_session(self):
        """A file path used to resolve to its parent folder -- an unnamed session."""
        path = os.path.join(self.session, "history_0.json")
        self.assertEqual(persistence.resolve_session_dir(path), path)
        self.assertEqual(persistence.resolve_session_dir(path, self.cache), path)

    def test_a_bare_session_id_resolves_under_the_cache_dir(self):
        self.assertEqual(
            persistence.resolve_session_dir("1792-abc", self.cache), self.session)

    def test_a_bare_id_without_a_cache_dir_is_left_alone(self):
        self.assertEqual(persistence.resolve_session_dir("1792-abc"), "1792-abc")

    def test_a_nonexistent_path_is_passed_through_unchanged(self):
        """A typo must stay a typo: the caller reports it, we do not invent a dir."""
        missing = os.path.join(self.root, "nope")
        self.assertEqual(persistence.resolve_session_dir(missing, self.cache), missing)
        self.assertEqual(persistence.resolve_session_dir("not-a-session", self.cache),
                         "not-a-session")

    def test_a_tilde_is_expanded(self):
        with mock.patch.dict(os.environ, {"HOME": self.root}):
            self.assertEqual(
                persistence.resolve_session_dir("~/cache/1792-abc"), self.session)

    def test_the_private_duplicates_are_gone(self):
        self.assertFalse(hasattr(persistence, "_restore_dir"))
        self.assertFalse(hasattr(persistence, "_resolve_history_path"))

    def test_the_single_file_loaders_are_gone(self):
        """``history.json`` is not a format this version reads, at all."""
        self.assertFalse(hasattr(persistence, "load_history"))
        messages = [{"role": "user", "content": "segment"}]
        with open(os.path.join(self.session, "history_0.json"), "w") as f:
            json.dump(messages, f)
        with open(os.path.join(self.session, "history.json"), "w") as f:
            json.dump([{"role": "user", "content": "single file"}], f)

        # A caller must name the segment it wants ...
        self.assertIsNone(persistence.load_history_from_dir(self.session, None))
        # ... and when it does, the segment wins over any single-file history.
        self.assertEqual(persistence.load_history_from_dir(self.session, 0), messages)


# ---------------------------------------------------------------------------
# the shared one-line helper
# ---------------------------------------------------------------------------

class OnelineTest(unittest.TestCase):
    """``utils.oneline`` is the single implementation behind three call sites."""

    def test_flattens_newlines_and_whitespace(self):
        self.assertEqual(utils.oneline("alpha\nbeta\r\n  gamma  "), "alpha beta gamma")
        self.assertEqual(utils.oneline("a\t\tb"), "a b")
        self.assertEqual(utils.oneline("   "), "")

    def test_no_limit_keeps_the_whole_line(self):
        text = "x" * 500
        self.assertEqual(utils.oneline(text), text)

    def test_limit_truncates_with_an_ellipsis(self):
        self.assertEqual(utils.oneline("abcdefghij", 4), "abcd...")
        self.assertEqual(utils.oneline("abc", 4), "abc")     # fits: untouched

    def test_accepts_non_string_input(self):
        self.assertEqual(utils.oneline(None), "None")
        self.assertEqual(utils.oneline(42), "42")

    def test_the_duplicate_definitions_are_gone(self):
        """No module re-implements it: agent_loop / subagents import, ui delegates."""
        from mini_code_cli import agent_loop
        from mini_code_cli.subagents import base as subagent_base

        self.assertFalse(hasattr(agent_loop, "_oneline"))
        self.assertFalse(hasattr(subagent_base, "_oneline"))
        self.assertIs(subagent_base.oneline, utils.oneline)

    def test_the_console_renderer_delegates(self):
        console = ui.Console(stream=io.StringIO(), color=False, unicode=False)
        self.assertEqual(console._oneline("a\nb", 3), utils.oneline("a\nb", 3))
        self.assertEqual(console._oneline("a\nb"), "a b")


class PluralTest(unittest.TestCase):
    """One ``plural`` helper, shared by the agent loop and the console.

    The compaction notes, the good-bye box and the usage line all build counts
    for a human to read; before this was factored out there were three inline
    ``'' if n == 1 else 's'`` expressions, and a "1 messages" would have been
    invisible until a release note said it.
    """

    def test_singular_and_plural(self):
        self.assertEqual(utils.plural(1, "message"), "1 message")
        self.assertEqual(utils.plural(0, "message"), "0 messages")
        self.assertEqual(utils.plural(2, "message"), "2 messages")

    def test_a_custom_suffix_is_honoured(self):
        # The helper concatenates a suffix rather than inflecting English, so a
        # noun that takes "-es" works and anything irregular stays the caller's
        # problem (none of the call sites need it).
        self.assertEqual(utils.plural(1, "box", "es"), "1 box")
        self.assertEqual(utils.plural(3, "box", "es"), "3 boxes")

    def test_the_console_usage_line_uses_it(self):
        console = ui.Console(stream=io.StringIO(), color=False, unicode=False)
        self.assertIn("| 1 LLM call", console.usage_inline({}, llm_calls=1))
        self.assertIn("| 2 LLM calls", console.usage_inline({}, llm_calls=2))


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

    def console(self):
        self._stream = io.StringIO()
        return ui.Console(stream=self._stream, color=False, unicode=False, verbose=False)

    def output(self):
        return self._stream.getvalue()

    def _compact(self, messages, replies, token_limit=1):
        # A 1-token budget means any non-empty transcript is "over budget", which
        # is how a real context overflow behaves -- only sooner. Compaction is
        # token-triggered only; the message count is not a criterion.
        if isinstance(replies, str):
            replies = [replies]
        stub = StubLLM([(r, [], USAGE) for r in replies])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            result = maybe_compact(messages, "http://localhost:1", "m", None, 30,
                                   self.stats, token_limit=token_limit,
                                   console=self.console())
        return result, stub

    def test_success_reports_what_was_compacted_where_and_what_was_kept(self):
        """The default (non-verbose) console has to answer the user's questions.

        A rollover leaves no trace in the transcript, so the notes are the only
        thing the user has: they must say *what* was folded away (a summary
        preview), *how much* (before -> after), *where it went* (which segment
        file was closed and which was started) and that the instruction being
        answered was not lost. All of it is ``warn``, i.e. visible without
        ``--verbose``.
        """
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "the new instruction"}]
        (out, did_compact, index), _ = self._compact(
            messages, ["The user asked a question and got an answer."])
        text = self.output()

        self.assertTrue(did_compact)
        self.assertEqual(len(out), 3)            # system + summary + turn in flight
        # the summary content itself, not just a message count
        self.assertIn("Summary preview: The user asked a question", text)
        # how much was folded away, and which segment file it went into
        self.assertIn("Compacted history_0.json (4 messages) into a summary", text)
        self.assertIn("history_1.json now starts with 3 messages", text)
        self.assertIn("summary + 1 kept message", text)
        # and that the instruction this step is answering survived the rollover
        self.assertIn("The current turn (1 message) is kept verbatim", text)

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

    def test_empty_summary_is_retried_then_abandoned_without_rollover(self):
        """No summary twice -> no rollover, transcript kept, both calls billed.

        An empty summary must never be mistaken for a successful compaction: the
        caller would think the history had shrunk while the over-long transcript
        was kept and the segment index silently stayed behind. We retry once,
        then give up -- keeping the messages and the index -- and say so.
        """
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"}]
        (out, did_compact, index), stub = self._compact(messages, ["", ""])

        self.assertFalse(did_compact)
        self.assertEqual(index, 0)               # still history_0
        self.assertIs(out, messages)             # transcript preserved verbatim
        self.assertEqual(len(stub.calls), 2)           # exactly one retry
        self.assertEqual(self.stats["llm_calls"], 2)   # both attempts were billed
        self.assertIn("Could not compact history", self.output())
        # ... and it must say that nothing was lost, not just that it failed.
        self.assertIn("nothing was lost", self.output())
        self.assertIn("Keeping all 3 messages and history_0.json", self.output())

    def test_empty_summary_then_a_real_one_still_compacts(self):
        """A single empty reply is recovered from by the retry."""
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"}]
        (out, did_compact, index), stub = self._compact(
            messages, ["", "A short summary."])

        self.assertTrue(did_compact)
        self.assertEqual(index, 1)               # history_0 -> history_1
        self.assertEqual(out[0], {"role": "system", "content": "S"})
        self.assertIn("A short summary.", out[1]["content"])
        self.assertEqual(len(stub.calls), 2)

    def test_a_failing_summariser_call_keeps_the_transcript(self):
        """A raised error must not lose the history or the segment index."""
        messages = [{"role": "system", "content": "S"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"}]
        boom = mock.Mock(side_effect=RuntimeError("connection refused"))
        with mock.patch.object(utils, "call_openai_server", new=boom):
            out, did_compact, index = maybe_compact(
                messages, "http://localhost:1", "m", None, 30, self.stats,
                token_limit=1, console=self.console())

        self.assertFalse(did_compact)
        self.assertEqual(index, 0)
        self.assertIs(out, messages)
        self.assertIn("Failed to compact history", self.output())
        self.assertIn("Keeping all 3 messages unchanged", self.output())



# ---------------------------------------------------------------------------
# the good-bye box, the version invariant and the -m entry point
# ---------------------------------------------------------------------------

class FarewellBoxTest(_SessionCase):
    """The exit banner: what it says, when it says it, and how often.

    It is the last thing a user sees, and the only place the quiet (non-verbose)
    view states what the session left behind, so the content and the "exactly
    once, normal exits only" rule are both pinned here.
    """

    def make(self):
        """A started session whose console writes into ``self.output()``."""
        session = AgentSession(make_args(self.workdir, self.cache_dir),
                               console=self.console())
        with mock.patch.object(AgentSession, "show_resume_choice", return_value="new"):
            session.start_session()
        return session

    def test_the_box_shows_the_version_the_sign_off_and_the_state(self):
        session = self.make()
        session.console.emit("")           # separator, keeps assertions simple

        session.shutdown()

        from mini_code_cli import __version__

        text = self.output()
        self.assertIn(f"MiniCodeCLI v{__version__} - goodbye", text)
        self.assertIn("Goodbye, and thanks for using MiniCodeCLI!", text)
        self.assertIn("session id", text)
        self.assertIn(session.unique_id, text)
        self.assertIn("saved under", text)
        self.assertIn("history", text)
        self.assertIn("conversation", text)
        self.assertIn("tokens", text)
        self.assertIn("calls", text)

    def test_it_is_not_the_startup_banner(self):
        # The welcome box lists how the session *will* run; the farewell box
        # must not repeat any of that once it is over.
        session = self.make()
        session.console.emit("")

        session.shutdown()

        text = self.output()
        for absent in ("allowed dir", "server url", "permissions :", "model       :"):
            self.assertNotIn(absent, text)

    def test_it_prints_exactly_once_across_repeated_shutdowns(self):
        session = self.make()
        session.console.emit("")

        session.shutdown()
        session.shutdown()

        self.assertEqual(self.output().count("Goodbye, and thanks"), 1)

    def test_no_box_on_the_interrupt_and_fatal_paths(self):
        session = self.make()
        session.console.emit("")

        session.shutdown(farewell=False)

        self.assertNotIn("Goodbye, and thanks", self.output())
        # ... but the session was still persisted ("Saved session to ...").
        self.assertIn("Saved session to", self.output())

    def test_the_interrupt_path_in_run_prints_no_box(self):
        session = self.make()
        session.console.emit("")
        with mock.patch.object(AgentSession, "print_welcome"), \
                mock.patch.object(AgentSession, "print_announcements"), \
                mock.patch.object(AgentSession, "_warn_compaction_budget"), \
                mock.patch.object(AgentSession, "start_session"), \
                mock.patch.object(AgentSession, "_print_session_id"), \
                mock.patch.object(AgentSession, "_emit_tip"), \
                mock.patch.object(AgentSession, "_read_user_line",
                                  side_effect=KeyboardInterrupt):
            session.run()

        text = self.output()
        self.assertIn("Interrupting: Mini Code Exiting...", text)
        self.assertNotIn("Goodbye, and thanks", text)


def scan_project_version(path):
    """Read ``[project] version`` by scanning lines -- the 3.10 fallback.

    ``tomllib`` exists from 3.11 only, and CI runs 3.10 as well, so this is the
    code path that leg takes; it is factored out (rather than inlined) so the
    test below exercises it on *every* interpreter, not just the oldest one.
    Section tracking is the point: a ``version`` key in some other table must not
    be mistaken for the project's.
    """
    section = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("["):
                section = stripped
            elif section == "[project]" and stripped.startswith("version"):
                return stripped.split("=", 1)[1].strip().strip('"')
    raise AssertionError(f"no version in [project] of {path}")


def declared_version(path):
    """The version in ``pyproject.toml``, read the way the package reads it."""
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # Python 3.10
        return scan_project_version(path)
    with open(path, "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


class VersionTest(unittest.TestCase):

    def test_the_package_version_matches_pyproject(self):
        # ``pyproject.toml`` is the single source of truth (the welcome banner
        # reads it through ``__init__``), and the release workflow refuses a tag
        # that disagrees with it -- so a forgotten bump should fail here first.
        from mini_code_cli import __version__

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(__version__, declared_version(os.path.join(root, "pyproject.toml")))

    def test_the_310_fallback_agrees_with_the_real_file(self):
        # Exercised on every interpreter, so a broken fallback cannot hide until
        # the 3.10 leg runs -- and a stray ``version`` in another table must not
        # win.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "pyproject.toml")
        self.assertEqual(scan_project_version(path), declared_version(path))

    def test_the_fallback_ignores_a_version_in_another_table(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write('[tool.other]\nversion = "9.9.9"\n\n'
                         '[project]\nname = "demo"\nversion = "1.2.3"\n')
            scratch = handle.name
        try:
            self.assertEqual(scan_project_version(scratch), "1.2.3")
        finally:
            os.unlink(scratch)


class ModuleEntryPointTest(unittest.TestCase):

    def test_python_dash_m_starts_the_cli(self):
        # ``mini_code_cli/__main__.py`` exists so the package runs without the
        # ``mini-code`` script on PATH; ``--help`` proves the wiring without
        # needing a server or a session.
        import subprocess
        import sys

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run([sys.executable, "-m", "mini_code_cli", "--help"],
                                cwd=root, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: mini-code", result.stdout)


if __name__ == "__main__":
    unittest.main()
