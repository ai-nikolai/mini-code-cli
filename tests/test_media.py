"""Tests for image staging, content blocks and request materialisation."""
import os
import tempfile
import unittest
from unittest import mock

from mini_code_cli import media


class MediaTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "board.png")
        with open(self.path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\nfake-bytes")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_stage_is_file_backed_and_materialises_on_request(self):
        ref = media.stage_image(self.path, self.directory)
        self.assertEqual(ref["type"], "image_ref")
        self.assertEqual(ref["mime"], "image/png")
        self.assertTrue(os.path.isfile(ref["path"]))
        self.assertNotIn("base64", str(ref))

        content = media.user_content("look", [ref])
        self.assertEqual(media.content_text(content), "look\n[image: board.png]")
        resolved = media.materialize_messages(
            [{"role": "user", "content": content}])[0]["content"]
        self.assertEqual(resolved[1]["type"], "image_url")
        self.assertTrue(resolved[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"))

    def test_missing_attachment_becomes_a_text_marker(self):
        ref = {"type": "image_ref", "path": "/does/not/exist.png",
               "mime": "image/png", "name": "gone.png"}
        content = media.user_content("look", [ref])
        resolved = media.materialize_messages(
            [{"role": "user", "content": content}])[0]["content"]
        self.assertEqual(resolved[1]["type"], "text")
        self.assertIn("image unavailable", resolved[1]["text"])

    def test_unsupported_types_and_oversized_images_are_rejected(self):
        text = os.path.join(self.directory, "notes.txt")
        with open(text, "w", encoding="utf-8") as handle:
            handle.write("x")
        with self.assertRaises(ValueError):
            media.stage_image(text, self.directory)
        with mock.patch.object(media, "MAX_IMAGE_BYTES", 1):
            with self.assertRaises(ValueError):
                media.stage_image(self.path, self.directory)

    def test_string_content_is_unchanged(self):
        self.assertEqual(media.content_text("hello"), "hello")
        self.assertEqual(media.user_content("hello", []), "hello")
        self.assertEqual(media.materialize_messages(
            [{"role": "user", "content": "hello"}]),
            [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
