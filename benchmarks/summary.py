"""``summary_results.csv``: one row per (dataset, agent, mode).

The suites already write a per-run CSV (:mod:`benchmarks.results`); this module
turns a *campaign* -- several runs, one per agent, all under one results
directory -- into the single file a person opens first:

    results/<dataset>-<agent>-<mode>-<timestamp>/summary_results.csv

The columns are fixed, because this file is the deliverable of a benchmark run and
scripts read it (``dataset,agent,mode,trials,resolved,pass_rate,mean_reward,
mean_seconds,mean_tokens,run_id,status``). Anything the harness cannot measure is
left empty rather than guessed: an empty ``mean_tokens`` means "this suite does not
report tokens", not zero.
"""
import csv
import json
import os

#: Column order of ``summary_results.csv`` -- part of the interface, do not reorder.
SUMMARY_COLUMNS = (
    "dataset", "agent", "mode", "trials", "resolved", "pass_rate",
    "mean_reward", "mean_seconds", "mean_tokens", "run_id", "status",
)

#: Filename written into the campaign directory.
SUMMARY_FILENAME = "summary_results.csv"


def _mean(values):
    """Mean of the numbers present, or ``None`` so the cell stays empty."""
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _rewards(outcomes):
    """Per-trial reward, from whichever shape the suite recorded.

    Terminal-Bench reports ``{"verifier_result": {"rewards": {...}}}`` and the
    suite keeps that mapping under ``extra["rewards"]``; a task resolved only when
    the best metric reaches 1.0. SWE-bench is binary, so its resolved flag is the
    reward -- using the same column for both keeps one CSV readable.
    """
    values = []
    for outcome in outcomes:
        extra = getattr(outcome, "extra", None) or {}
        rewards = extra.get("rewards") if isinstance(extra, dict) else None
        if rewards:
            values.append(max(float(v) for v in rewards.values()))
        else:
            values.append(1.0 if outcome.resolved else 0.0)
    return values


def _tokens(outcomes):
    """Per-trial token totals, when the agent reported them."""
    values = []
    for outcome in outcomes:
        extra = getattr(outcome, "extra", None) or {}
        if not isinstance(extra, dict):
            continue
        tokens = extra.get("tokens") or extra.get("total_tokens")
        if isinstance(tokens, (int, float)):
            values.append(tokens)
    return values


def row_for(dataset, agent, mode, run_id, outcomes, status="ok"):
    """Build one summary row from a run's outcomes."""
    trials = len(outcomes)
    resolved = sum(1 for o in outcomes if o.resolved)
    if status == "ok":
        errors = [o for o in outcomes if o.outcome in ("error", "timeout")]
        if errors and not resolved and len(errors) == trials:
            status = "all_failed"
        elif errors:
            status = "partial_failures"
    return {
        "dataset": dataset,
        "agent": agent,
        "mode": mode,
        "trials": trials,
        "resolved": resolved,
        "pass_rate": (round(resolved / trials, 4) if trials else None),
        "mean_reward": _mean(_rewards(outcomes)),
        "mean_seconds": _mean([o.duration_seconds for o in outcomes]),
        "mean_tokens": _mean(_tokens(outcomes)),
        "run_id": run_id,
        "status": status,
    }


def write_summary(results_dir, rows, filename=SUMMARY_FILENAME):
    """Write ``summary_results.csv`` into ``results_dir``; return its path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key))
                             for key in SUMMARY_COLUMNS})
    return path


def read_run_outcomes(run_dir):
    """Outcomes of one ``python -m benchmarks run`` directory.

    ``results.jsonl`` is the machine-readable record; a run that died before
    writing it still counts (as a row with zero trials), because a missing agent in
    a comparison is exactly what the summary must not hide.
    """
    from .results import read_results

    for name in ("results.jsonl", os.path.join("results", "results.jsonl")):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return read_results(path)
    return []


def run_metadata(run_dir):
    """``(run_id, manifest)`` for a run directory, tolerating a missing manifest."""
    manifest_path = os.path.join(run_dir, "manifest.json")
    run_id = os.path.basename(os.path.abspath(run_dir))
    if not os.path.isfile(manifest_path):
        return run_id, None
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return run_id, None
    return manifest.get("run_id") or run_id, manifest


__all__ = ["SUMMARY_COLUMNS", "SUMMARY_FILENAME", "read_run_outcomes",
           "row_for", "run_metadata", "write_summary"]
