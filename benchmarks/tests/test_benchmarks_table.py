"""Tests for the scorecard table reader (``benchmarks/table.py``)."""
import json
import os
import tempfile
import unittest

from benchmarks import table


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)


def _outcome(instance, outcome, prompt=None, completion=None, seconds=10.0):
    payload = {
        "suite": "terminal-bench",
        "instance_id": instance,
        "outcome": outcome,
        "resolved": outcome == "resolved",
        "duration_seconds": seconds,
    }
    if prompt is not None:
        payload["extra"] = {"metrics": {
            "prompt_tokens": prompt,
            "completion_tokens": completion or 0,
            "total_tokens": prompt + (completion or 0),
        }}
    return json.dumps(payload) + "\n"


class TableTest(unittest.TestCase):
    def _campaign(self, root):
        agent_dir = os.path.join(root, "campaign", "mini-code")
        trial = "terminal-bench-mini-code-swe-ornith"
        _write(os.path.join(agent_dir, "manifest.json"), {
            "trials": [{
                "run_id": trial,
                "agent": "mini-code",
                "setting": "swe",
                "model": "ornith",
            }],
        })
        _write(os.path.join(agent_dir, "trials", trial, "results.jsonl"),
               _outcome("ok", "resolved", 100, 10, 8.0)
               + _outcome("no", "unresolved", 200, 20, 12.0)
               + _outcome("err", "error"))
        return agent_dir

    def test_reads_manifest_labels_and_per_task_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            agent_dir = self._campaign(root)
            rows = table.collect([agent_dir])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["agent"], row["setting"], row["model"]),
                         ("mini-code", "swe", "ornith"))
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["resolved"], 1)
        self.assertEqual(row["errors"], 1)
        self.assertAlmostEqual(row["raw"], 1 / 3)
        self.assertAlmostEqual(row["clean"], 0.5)
        self.assertEqual(row["token_trials"], 2)
        self.assertEqual(row["prompt_tokens"], 300)
        self.assertEqual(row["completion_tokens"], 30)
        self.assertEqual(row["total_tokens"], 330)
        self.assertEqual(row["mean_total_tokens"], 165)
        self.assertEqual(row["mean_seconds"], 10.0)

    def test_campaign_directory_finds_each_agent_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            self._campaign(root)
            rows = table.collect([root])
        self.assertEqual([row["setting"] for row in rows], ["swe"])

    def test_markdown_notes_incomplete_token_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            agent_dir = self._campaign(root)
            rows = table.collect([agent_dir])
        text = table.render_markdown(rows)
        self.assertIn("| mini-code | swe | 1/3 | 33.3% | 1 | 50.0% |", text)
        self.assertIn("Token means use trials with metrics", text)
        self.assertIn("`mini-code/swe` 2/3", text)

    def test_orphan_results_file_is_labelled_by_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "trials", "old-run", "results.jsonl")
            _write(path, _outcome("ok", "resolved", 10, 1))
            rows = table.collect([path])
        self.assertEqual(rows[0]["setting"], "old-run")
        self.assertEqual(rows[0]["model"], "?")

    def test_missing_path_is_an_error(self):
        with self.assertRaises(SystemExit):
            table.collect(["/definitely/not/a/results/path"])


if __name__ == "__main__":
    unittest.main()
