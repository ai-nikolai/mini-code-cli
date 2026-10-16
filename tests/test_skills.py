"""Tests for the skills feature (skills.py + its wiring into agent_loop.py).

Covers:
  * name-safety / path-traversal guards,
  * the four skill tools (list / read / create / run) in isolation,
  * the high-level SKILLS.md summary,
  * running a skill's Python scripts (with args, and with invalid names),
  * the loop wiring: --skills exposes the four tools and the skill prompt, and
    skills stay disabled without the flag.

All tests are fully offline -- no server is needed.
"""
import io
import os
import sys
import types
import tempfile
import unittest
import unittest.mock as mock
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import agent_loop, tools, skills, ui
from mini_code_cli import prompts


def _make_args(**kw):
    a = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None, api_key=None,
        system_prompt=None, agent_md=None, prompt=None, enable_shell=False,
        ask_permission=False, auto_mode=False, restore_session=None, llm_timeout=30,
        tool_timeout=30, load_last_session=False, verbose=False, max_tokens=None,
        temperature=1.0, plan=False, enable_subagents=False, enable_skills=False,
    )
    a.__dict__.update(kw)
    return a


class TestSafeName(unittest.TestCase):
    def test_rejects_traversal(self):
        self.assertIsNone(skills._safe_name(".."))
        self.assertIsNone(skills._safe_name("."))
        self.assertIsNone(skills._safe_name("a/b"))
        self.assertIsNone(skills._safe_name("a\\b"))
        self.assertIsNone(skills._safe_name("../etc/passwd"))
        self.assertIsNone(skills._safe_name(""))
        self.assertIsNone(skills._safe_name("   "))
        self.assertIsNone(skills._safe_name(None))

    def test_allows_plain_segment(self):
        self.assertEqual(skills._safe_name("sum"), "sum")
        self.assertEqual(skills._safe_name("  my-skill_1 "), "my-skill_1")


class TestSkillsTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name

    def test_list_skills_empty_creates_mini(self):
        out = skills.list_skills(self.allowed)
        self.assertIn("No skills yet", out)
        self.assertTrue(os.path.isdir(skills.skills_dir(self.allowed)))
        self.assertTrue(os.path.isfile(skills.summary_path(self.allowed)))

    def test_create_then_list(self):
        skills.create_skill(self.allowed, "greet", "Greets a person.", title="Greeter")
        listing = skills.list_skills(self.allowed)
        self.assertIn("greet", listing)
        self.assertIn("Greets a person.", listing)
        # SKILLS.md reflects the new skill.
        summary = open(skills.summary_path(self.allowed)).read()
        self.assertIn("## greet", summary)
        self.assertIn("Greets a person.", summary)

    def test_create_requires_description(self):
        out = skills.create_skill(self.allowed, "x", description=None)
        self.assertIn("Error", out)

    def test_create_rejects_traversal_names(self):
        self.assertIn("invalid", skills.create_skill(self.allowed, "../evil", "x"))
        self.assertIn("invalid", skills.create_skill(self.allowed, "a/b", "x"))

    def test_create_no_overwrite_then_overwrite(self):
        first = skills.create_skill(self.allowed, "s", "one")
        self.assertIn("Created", first)
        second = skills.create_skill(self.allowed, "s", "two")
        self.assertIn("already exists", second)
        third = skills.create_skill(self.allowed, "s", "two", overwrite=True)
        self.assertIn("Created", third)

    def test_read_skill_unknown(self):
        out = skills.read_skill(self.allowed, "ghost")
        self.assertIn("Error", out)
        self.assertIn("no skill named", out)

    def test_read_skill_contents(self):
        skills.create_skill(self.allowed, "greet", "Greets a person.", title="Greeter")
        out = skills.read_skill(self.allowed, "greet")
        self.assertIn("# Greeter", out)

    def test_read_skill_rejects_traversal(self):
        out = skills.read_skill(self.allowed, "../../etc/passwd")
        self.assertIn("invalid", out)

    def test_run_skill_unknown_script(self):
        skills.create_skill(self.allowed, "s", "d")
        out = skills.run_skill(self.allowed, "s", "missing")
        self.assertIn("Error", out)
        self.assertIn("no script", out)

    def test_run_skill_executes_script_with_args(self):
        skills.create_skill(self.allowed, "calc", "d")
        script_path = os.path.join(skills.skill_dir(self.allowed, "calc"), "add.py")
        with open(script_path, "w") as f:
            f.write("import sys, argparse\n"
                    "p = argparse.ArgumentParser()\n"
                    "p.add_argument('--a', type=int, default=0)\n"
                    "p.add_argument('--b', type=int, default=0)\n"
                    "a = p.parse_args()\n"
                    "print('result', a.a + a.b)\n")
        out = skills.run_skill(self.allowed, "calc", "add", args="--a 3 --b 4")
        self.assertIn("result 7", out)

    def test_run_skill_rejects_traversal(self):
        skills.create_skill(self.allowed, "s", "d")
        out = skills.run_skill(self.allowed, "s", "../tools")
        self.assertIn("invalid", out)

    def test_scripts_are_listed_in_summary(self):
        skills.create_skill(self.allowed, "calc", "d")
        script_path = os.path.join(skills.skill_dir(self.allowed, "calc"), "add.py")
        open(script_path, "w").write("print('hi')\n")
        # list_skills refreshes SKILLS.md to match what is on disk (including
        # scripts dropped into a skill after it was created), so the summary
        # now reflects the new script.
        skills.list_skills(self.allowed)
        summary = open(skills.summary_path(self.allowed)).read()
        self.assertIn("Scripts: add", summary)


class TestSkillToolSchemas(unittest.TestCase):
    def test_partial_hides_bound_arg(self):
        # A partial bound on allowed_dir must not expose allowed_dir in the schema.
        bound = tools.util_get_tools(functions=skills._skill_functions("/tmp/x"))
        names = [t["function"]["name"] for t in bound]
        self.assertEqual(sorted(names), sorted(skills.SKILL_TOOL_NAMES))
        for t in bound:
            props = t["function"]["parameters"]["properties"]
            self.assertNotIn("allowed_dir", props)
        # run_skill requires skill_name/script_name but not allowed_dir.
        run_schema = next(t for t in bound if t["function"]["name"] == "run_skill")
        self.assertIn("script_name", run_schema["function"]["parameters"]["properties"])
        self.assertIn("args", run_schema["function"]["parameters"]["properties"])
        self.assertNotIn("allowed_dir", run_schema["function"]["parameters"]["properties"])

    def test_regular_tools_still_work(self):
        # Sanity: the default tools module still builds schemas without functions=.
        schema = tools.util_get_tools()
        names = [t["function"]["name"] for t in schema]
        self.assertIn("read_file", names)
        self.assertIn("grep", names)
        # util_ helpers are excluded.
        self.assertNotIn("util_get_tools", names)


class TestSkillsPrompt(unittest.TestCase):
    def test_skills_instructions_present_with_dir(self):
        text = prompts.skills_instructions("/tmp/whatever")
        self.assertIn("SKILLS MODE", text)
        self.assertIn("list_skills", text)
        self.assertIn("run_skill", text)

    def test_system_prompt_has_skills_block_when_dir_given(self):
        with_prompt = prompts.system_prompt(tools_dict=None, allowed_dir="/tmp", skills_dir="/tmp")
        without = prompts.system_prompt(tools_dict=None, allowed_dir="/tmp", skills_dir=None)
        self.assertIn("SKILLS MODE", with_prompt)
        self.assertNotIn("SKILLS MODE", without)


class TestSkillsWiringInLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name

    def _make(self, **kw):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            return agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name,
                                                      allowed_dir=self.allowed, **kw))

    def test_disabled_by_default(self):
        sess = self._make()
        self.assertFalse(sess.enable_skills)
        self.assertNotIn("list_skills", sess.tools_dict)
        self.assertNotIn("list_skills", [t["function"]["name"] for t in sess.llm_tools])

    def test_enabled_with_flag(self):
        sess = self._make(enable_skills=True)
        self.assertTrue(sess.enable_skills)
        for name in skills.SKILL_TOOL_NAMES:
            self.assertIn(name, sess.tools_dict)
            self.assertIn(name, [t["function"]["name"] for t in sess.llm_tools])

    def test_system_prompt_gets_skills_block(self):
        sess = self._make(enable_skills=True)
        self.assertIn("SKILLS MODE", sess.messages[0]["content"])

    def test_system_prompt_no_skills_block_without_flag(self):
        sess = self._make(enable_skills=False)
        self.assertNotIn("SKILLS MODE", sess.messages[0]["content"])

    def test_skill_tool_execution_end_to_end(self):
        # A mocked LLM that calls create_skill then list_skills, verifying the
        # tools actually run through the normal loop.
        seen = {"n": 0}

        def fake_server(messages, **kw):
            n = seen["n"]
            seen["n"] += 1
            if n == 0:
                tc = [{"id": "1", "function": {
                    "name": "create_skill",
                    "arguments": '{"skill_name": "ping", "description": "says pong", "title": "Pong"}'}}]
                return "creating", tc, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if n == 1:
                tc = [{"id": "2", "function": {
                    "name": "list_skills", "arguments": "{}"}}]
                return "listing", tc, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            return "done", None, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"), \
             mock.patch.object(agent_loop.utils, "call_openai_server", side_effect=fake_server), \
             mock.patch.object(ui.Console, "read_line", side_effect=iter(["/quit"])):
            sess = agent_loop.AgentSession(
                _make_args(cache_dir=self.tmp.name, allowed_dir=self.allowed,
                           auto_mode=True, prompt="create a skill then list", enable_skills=True),
                console=ui.Console(stream=io.StringIO(), color=False, unicode=False))
            sess.run(initial_prompt="create a skill then list")

        # The skill was really created on disk inside the allowed dir.
        self.assertTrue(os.path.isfile(skills.skill_md_path(self.allowed, "ping")))
        self.assertTrue(os.path.isfile(skills.summary_path(self.allowed)))
