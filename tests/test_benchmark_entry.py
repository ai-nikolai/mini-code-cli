"""The ``./benchmark`` entry point, the shared preamble, and the summary CSV.

These are the three things a user meets first, so they are pinned by tests: the
command a person types, the file that tells every agent the same thing, and the
CSV that is the output of a run.
"""
import csv
import io
import os
import tempfile
import unittest

from benchmarks import campaign
from benchmarks import summary as summary_module
from benchmarks.agents import AGENT_MD, load_agent_preamble
from benchmarks.config import load_config
from benchmarks.agents import get_adapter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PreambleTests(unittest.TestCase):
    def test_there_is_one_shared_file(self):
        path, text = load_agent_preamble()
        self.assertEqual(path, AGENT_MD)
        self.assertTrue(os.path.isfile(path))
        self.assertIn("/testbed", text)
        # The three rules the benchmark relies on.
        self.assertIn("git commit", text)
        self.assertIn("Reproduce", text)

    def test_every_agent_receives_that_preamble(self):
        """All three adapters, one file -- checked through the argv they build."""
        config = load_config(os.path.join(
            ROOT, "benchmarks/configs/ornith_30b.json"))
        models = {m.name: m for m in config.models}
        seen = {}
        for spec in config.agents:
            adapter = get_adapter(spec)
            model = models[spec.models[0]]
            seen[spec.name] = adapter.argv("fix the bug", model,
                                           workspace="/testbed")
        _, text = load_agent_preamble()
        head = text.strip().splitlines()[0]

        self.assertIn(AGENT_MD, seen["mini-code"])
        self.assertEqual(seen["claude-code"][seen["claude-code"].index(
            "--append-system-prompt") + 1], text)
        self.assertTrue(any(part.startswith(head)
                            for part in seen["pi"]))

    def test_inline_preamble_is_gone_from_the_suite(self):
        """One source of truth: the suite must not re-add its own text."""
        with open(os.path.join(ROOT, "benchmarks/suites/swebench.py"),
                  encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("SWE_PREAMBLE", source)


class CampaignTests(unittest.TestCase):
    def test_both_datasets_are_runnable(self):
        self.assertIn("swe-bench", campaign.DATASETS)
        self.assertIn("terminalbench-2.1", campaign.DATASETS)
        self.assertIn("short_eval", campaign.MODES)
        self.assertIn("short", campaign.MODES)

    def test_mode_narrows_the_matrix(self):
        config, suite_name, label = campaign.build_config(
            "swe-bench", "mini-code", "short_eval")
        self.assertEqual(suite_name, "swebench")
        self.assertEqual(label, "swe-bench")
        self.assertEqual([a.name for a in config.agents], ["mini-code"])
        self.assertEqual([s.name for s in config.suites], ["swebench"])
        self.assertEqual(config.suites[0].limit, 3)
        self.assertFalse(config.suites[0].options["skip_evaluation"])

    def test_smoke_mode_leaves_the_graders_off(self):
        config, _, _ = campaign.build_config("swe-bench", "pi", "short")
        self.assertTrue(config.suites[0].options["skip_evaluation"])

    def test_terminal_bench_21_points_at_the_local_checkout(self):
        config, suite_name, label = campaign.build_config(
            "terminalbench-2.1", "claude-code", "short")
        self.assertEqual(suite_name, "terminal-bench")
        self.assertEqual(label, "terminalbench-2.1")
        options = config.suites[0].options
        self.assertTrue(str(options.get("dataset_dir")).endswith("terminal-bench-2-1"))

    def test_dry_run_writes_a_summary_but_starts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buffer):
                code = campaign.main([
                    "--dataset", "swe-bench", "--agent", "mini-code",
                    "--mode", "short", "--dry-run",
                    "--results-dir", tmp, "--run-id", "dry"])
            self.assertEqual(code, 0)
            printed = buffer.getvalue()
            self.assertIn("miniconda", printed + "miniconda"
                          if False else printed)
            self.assertIn("summary:", printed)
            path = os.path.join(tmp, "_dry_run", "dry",
                                summary_module.SUMMARY_FILENAME)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "dry-run")
            self.assertEqual(rows[0]["agent"], "mini-code")


class SummaryTests(unittest.TestCase):
    def _trial_outcome(self, resolved, outcome="ok", seconds=10.0,
                       extra=None):
        from benchmarks.suites.base import TrialOutcome

        status = TrialOutcome.RESOLVED if resolved else outcome
        if resolved:
            status = TrialOutcome.RESOLVED
        return TrialOutcome(suite="swebench", instance_id="i",
                            outcome=status, duration_seconds=seconds,
                            extra=extra or {})

    def test_row_has_the_documented_columns(self):
        row = summary_module.row_for("swe-bench", "mini-code", "short_eval",
                                     "run", [self._trial_outcome(True),
                                             self._trial_outcome(False)])
        self.assertEqual(set(row), set(summary_module.SUMMARY_COLUMNS))
        self.assertEqual(row["trials"], 2)
        self.assertEqual(row["resolved"], 1)
        self.assertEqual(row["pass_rate"], 0.5)
        self.assertEqual(row["mean_reward"], 0.5)
        self.assertEqual(row["mean_seconds"], 10.0)

    def test_rewards_come_from_the_suite_when_it_reports_them(self):
        row = summary_module.row_for(
            "terminalbench-2.1", "pi", "short", "run",
            [self._trial_outcome(False, extra={"rewards": {"tests": 1.0}})])
        self.assertEqual(row["mean_reward"], 1.0)

    def test_csv_round_trips_with_empty_cells_for_missing_numbers(self):
        row = summary_module.row_for("swe-bench", "pi", "full", "run", [])
        with tempfile.TemporaryDirectory() as tmp:
            path = summary_module.write_summary(tmp, [row])
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        self.assertTrue(text.startswith(",".join(summary_module.SUMMARY_COLUMNS)))
        self.assertIn("swe-bench,pi,full,0,0,,", text)


if __name__ == "__main__":
    unittest.main()
