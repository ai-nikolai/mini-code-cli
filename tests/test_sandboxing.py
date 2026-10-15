"""Unit tests for the --allowed-dir sandbox enforcement (change #5)."""
import os
import sys
import tempfile
import unittest

# Allow running as `python tests/test_sandboxing.py` without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli.agent_loop import (
    is_within_allowed_dir,
    tool_paths_are_allowed,
    _FILE_TOOL_PATH_ARGS,
)


class TestIsWithinAllowedDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.child = os.path.join(self.root, "sub", "file.txt")

    def test_direct_child_allowed(self):
        self.assertTrue(is_within_allowed_dir(self.child, self.root))

    def test_traversal_denied(self):
        # ../ escapes the root even if the path text starts inside it.
        sneaky = os.path.join(self.root, "sub", "..", "..", "secret.txt")
        self.assertFalse(is_within_allowed_dir(sneaky, self.root))

    def test_sibling_denied(self):
        outside = os.path.join(os.path.dirname(self.root), "elsewhere.txt")
        self.assertFalse(is_within_allowed_dir(outside, self.root))

    def test_relative_root(self):
        # A relative allowed_dir should be resolved and still contain its children.
        cwd = os.getcwd()
        # Build a path inside cwd to check containment works with a relative root.
        target = os.path.join(cwd, "somefile.txt")
        self.assertTrue(is_within_allowed_dir(target, "."))


class TestToolSandboxGuard(unittest.TestCase):
    """The per-tool allowed-dir guard (tool_paths_are_allowed).

    Regression test for the glob sandbox bypass: `glob` uses a `pattern`
    argument rather than `file_path`, so a naive check of `file_path` let it
    read anywhere on the system. This guard maps each tool to its path arg(s).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.child = os.path.join(self.root, "sub", "f.txt")

    def test_glob_pattern_is_sandboxed(self):
        # Inside the allowed dir -> allowed.
        self.assertTrue(tool_paths_are_allowed("glob", {"pattern": self.child}, self.root))

    def test_glob_pattern_escape_denied(self):
        # A glob pattern that reaches outside must be denied (the old bypass).
        sneaky = os.path.join(self.root, "..", "..", "etc")
        self.assertFalse(tool_paths_are_allowed("glob", {"pattern": sneaky}, self.root))
        self.assertFalse(tool_paths_are_allowed("glob", {"pattern": "/etc/passwd"}, self.root))

    def test_file_tools_still_sandboxed(self):
        self.assertTrue(tool_paths_are_allowed("read_file", {"file_path": self.child}, self.root))
        self.assertTrue(tool_paths_are_allowed("write_file", {"file_path": self.child}, self.root))
        self.assertTrue(tool_paths_are_allowed("grep", {"pattern": "x", "file_path": self.child}, self.root))
        self.assertFalse(tool_paths_are_allowed("read_file", {"file_path": "/etc/passwd"}, self.root))

    def test_non_filesystem_tool_passes_through(self):
        # Tools with no path argument (e.g. the shell tool) are not filesystem
        # operations and should pass the guard unchanged.
        self.assertTrue(tool_paths_are_allowed("run_shell_command", {"command": "ls -la"}, self.root))

    def test_empty_path_is_allowed(self):
        # An empty/absent path argument means "nothing to check" -> allowed.
        self.assertTrue(tool_paths_are_allowed("glob", {"pattern": ""}, self.root))
        self.assertTrue(tool_paths_are_allowed("read_file", {}, self.root))

    def test_every_fs_tool_has_a_path_mapping(self):
        # Documentation/defence: the four filesystem tools must all be covered.
        for name in ("read_file", "write_file", "grep", "glob"):
            self.assertIn(name, _FILE_TOOL_PATH_ARGS)


if __name__ == "__main__":
    unittest.main()
