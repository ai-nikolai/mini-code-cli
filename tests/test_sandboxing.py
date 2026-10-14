"""Unit tests for the --allowed-dir sandbox enforcement (change #5)."""
import os
import sys
import tempfile
import unittest

# Allow running as `python tests/test_sandboxing.py` without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli.agent_loop import is_within_allowed_dir


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


if __name__ == "__main__":
    unittest.main()
