"""Tests for the `.mini` latest-session pointer used by --load-last-session.

A tiny pointer is recorded in `--allowed-dir/.mini/last_session.json` at the
end of every interactive session. `--load-last-session` consults that pointer
*first* (an explicit record of the session the user last worked in) and only
falls back to the timestamp heuristic over the cache dir when there is none.
"""
import io
import os
import sys
import types
import tempfile
import json
import shutil
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import agent_loop, ui, persistence, skills


def _make_args(**kw):
    a = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None, api_key=None,
        system_prompt=None, agent_md=None, prompt=None, enable_shell=False,
        ask_permission=False, auto_mode=False, restore_session=None, llm_timeout=30,
        tool_timeout=30, load_last_session=False, verbose=False, max_tokens=None,
        temperature=1.0, plan=False, enable_subagents=False, enable_skills=False,
        sudo=False,
    )
    a.__dict__.update(kw)
    return a


class TestPointerBasics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name

    def _session_dir(self, name="sess"):
        d = os.path.join(self.tmp.name, "cache", name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "history.json"), "w").write("[]")
        return d

    def test_save_then_load_roundtrip(self):
        sdir = self._session_dir("one")
        path = persistence.save_last_session_pointer(self.allowed, sdir)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))
        payload = json.load(open(path))
        self.assertEqual(payload["session_dir"], os.path.abspath(sdir))
        self.assertIn("recorded_at", payload)
        self.assertEqual(persistence.load_last_session_pointer(self.allowed),
                         os.path.abspath(sdir))

    def test_load_missing_returns_none(self):
        self.assertIsNone(persistence.load_last_session_pointer(self.allowed))

    def test_load_malformed_returns_none(self):
        persistence.save_last_session_pointer(self.allowed, self._session_dir("x"))
        p = os.path.join(self.allowed, ".mini", persistence.LAST_SESSION_FILENAME)
        open(p, "w").write("not json {[")
        self.assertIsNone(persistence.load_last_session_pointer(self.allowed))

    def test_load_pointing_to_missing_dir_returns_none(self):
        path = persistence.save_last_session_pointer(self.allowed, self._session_dir("x"))
        shutil.rmtree(os.path.join(self.tmp.name, "cache", "x"))
        self.assertIsNone(persistence.load_last_session_pointer(self.allowed))

    def test_save_none_is_a_noop(self):
        self.assertIsNone(persistence.save_last_session_pointer(self.allowed, None))
        self.assertIsNone(
            persistence.load_last_session_pointer(self.allowed))


class TestFindLastSessionResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name
        self.cache = os.path.join(self.tmp.name, "cache")

    def _session_dir(self, name):
        d = os.path.join(self.cache, name)
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, "history.json")
        open(f, "w").write("[]")
        return d, f

    def test_pointer_wins_over_newer_mtime(self):
        s1, _ = self._session_dir("older")
        s2, f2 = self._session_dir("newer")
        os.utime(f2, (1e12, 1e12))  # far in the future -> newest by mtime
        # Record a pointer to s1 (the older one).
        persistence.save_last_session_pointer(self.allowed, s1)
        got = persistence.find_last_session_dir(self.cache, allowed_dir=self.allowed,
                                                use_pointer=True)
        self.assertEqual(got, os.path.abspath(s1))

    def test_falls_back_to_mtime_when_no_pointer(self):
        s1, _ = self._session_dir("older")
        s2, f2 = self._session_dir("newer")
        os.utime(f2, (1e12, 1e12))
        got = persistence.find_last_session_dir(self.cache, allowed_dir=self.allowed,
                                                use_pointer=True)
        self.assertEqual(got, os.path.abspath(s2))

    def test_use_pointer_false_ignores_pointer(self):
        s1, _ = self._session_dir("older")
        s2, f2 = self._session_dir("newer")
        os.utime(f2, (1e12, 1e12))
        persistence.save_last_session_pointer(self.allowed, s1)
        got = persistence.find_last_session_dir(self.cache, allowed_dir=self.allowed,
                                                use_pointer=False)
        self.assertEqual(got, os.path.abspath(s2))

    def test_none_when_everything_empty(self):
        os.makedirs(self.cache, exist_ok=True)
        got = persistence.find_last_session_dir(self.cache, allowed_dir=self.allowed,
                                                use_pointer=True)
        self.assertIsNone(got)


class TestPointerAndSkillsCoexist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name

    def test_pointer_file_does_not_appear_as_a_skill(self):
        # Use skills so .mini + SKILLS.md exist.
        skills.list_skills(self.allowed)
        # A session records its pointer in the same .mini.
        cache = os.path.join(self.tmp.name, "cache")
        sdir = os.path.join(cache, "abc")
        os.makedirs(sdir); open(os.path.join(sdir, "history.json"), "w").write("[]")
        persistence.save_last_session_pointer(self.allowed, sdir)
        # Skills still work, and the pointer file never shows up as a skill.
        listing = skills.list_skills(allowed_dir=self.allowed)
        self.assertNotIn(persistence.LAST_SESSION_FILENAME, listing)
        self.assertIn("Skills", open(skills.summary_path(self.allowed)).read())

    def test_create_skill_after_pointer_written(self):
        cache = os.path.join(self.tmp.name, "cache")
        sdir = os.path.join(cache, "abc")
        os.makedirs(sdir); open(os.path.join(sdir, "history.json"), "w").write("[]")
        persistence.save_last_session_pointer(self.allowed, sdir)
        out = skills.create_skill(self.allowed, "greet", "greets")
        self.assertIn("Created", out)
        self.assertIn("greet", skills.list_skills(self.allowed))


class TestPrintSessionIdAndDivider(unittest.TestCase):
    """The interactive run() should show the id as an Assistant turn in brackets,
    and draw a short divider before the user's turn."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _fake_llm(self, session):
        def fake(messages, **kw):
            return "hi there", [], {"prompt_tokens": 1, "completion_tokens": 1,
                                    "total_tokens": 2}
        return fake

    def test_session_id_renderedin_assistant_turn_and_pointer_written(self):
        allowed = self.tmp.name
        args = _make_args(cache_dir=os.path.join(self.tmp.name, "cache"),
                          allowed_dir=allowed, prompt="hello")
        sess = agent_loop.AgentSession(args, console=self.console)
        uid = sess.unique_id

        with patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             patch.object(agent_loop.utils, "call_openai_server", self._fake_llm(sess)), \
             patch.object(self.console, "read_line", side_effect=iter(["/quit"])):
            sess.run(initial_prompt="hello")

        out = self.buf.getvalue()
        # (1) id rendered as an Assistant turn carrying only the id in brackets.
        self.assertIn(f"[{uid}]", out)
        # It sits on the same line as the 'Assistant' role marker.
        self.assertRegex(out, r"(?m)^> Assistant\n   \[%s\]$" % uid)
        # No noisy "saved under" line in the session-id turn.
        self.assertNotIn("saved under", out)

        # (3) a pointer was written under allowed_dir/.mini.
        ptr = os.path.join(allowed, ".mini", persistence.LAST_SESSION_FILENAME)
        self.assertTrue(os.path.isfile(ptr))
        payload = json.load(open(ptr))
        self.assertEqual(payload["session_dir"],
                         os.path.abspath(os.path.join(self.tmp.name, "cache", uid)))

    def test_short_divider_before_preset_you_turn(self):
        args = _make_args(cache_dir=os.path.join(self.tmp.name, "cache"),
                          allowed_dir=self.tmp.name, prompt="hello")
        sess = agent_loop.AgentSession(args, console=self.console)
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             patch.object(agent_loop.utils, "call_openai_server", self._fake_llm(sess)), \
             patch.object(self.console, "read_line", side_effect=iter(["/quit"])):
            sess.run(initial_prompt="hello")
        out = self.buf.getvalue()
        # The preset "You" turn is preceded by a short divider line.
        lines = out.splitlines()
        idx = next(i for i, l in enumerate(lines) if l.strip() == "> You")
        divider = lines[idx - 1].strip()
        self.assertTrue(set(divider) <= {"-", " "})
        self.assertIn("-", divider)
        self.assertLess(len(divider.strip("-")), 12)


if __name__ == "__main__":
    unittest.main()
