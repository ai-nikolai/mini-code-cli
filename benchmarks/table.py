"""Scorecard tables from stored benchmark results.

The suite already writes everything needed to rebuild a comparison: each
campaign keeps a ``manifest.json`` per agent run (which trial is which setting)
and each trial keeps a ``results.jsonl`` (one outcome per task, including the
agent-reported token metrics). This module joins the two and prints the table a
README quotes:

    python -m benchmarks.table results/tb21-clean1-* results/tb21-combined1_*

It reads only files on disk -- no container, dataset or scheduler -- so it runs
on a login node against results copied off the GPU host. Unlike
:mod:`benchmarks.analyze`, which reports the vLLM server counters, the token
columns here come from each task's ``extra.metrics`` block. Those are complete
per task; a server counter can be partial when the server restarted mid-run.
"""
import argparse
import json
import os
import sys

from .metrics import outcome_metrics
from .results import read_results

#: Directories that hold per-trial payloads, not run manifests. Pruning them
#: keeps a walk over a results tree fast even when it contains thousands of
#: harness logs.
_PRUNE = {"_prepare", "trials", "harbor", "logs", ".cache", "__pycache__"}

_HEADER = ("agent", "setting", "resolved", "raw", "err", "clean", "mean s",
           "tokens", "tok/task", "tok n")


def _manifest_trials(manifest_path):
    """``(agent, setting, model, run_id)`` for every trial in a manifest."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return []
    trials = []
    for trial in manifest.get("trials") or []:
        run_id = trial.get("run_id")
        if not run_id:
            continue
        trials.append((
            str(trial.get("agent") or manifest.get("agent") or "?"),
            str(trial.get("setting") or "default"),
            str(trial.get("model") or "?"),
            run_id,
        ))
    return trials


def _fallback_label(results_path):
    """Derive a label when an old run has no manifest beside it."""
    name = os.path.basename(os.path.dirname(results_path))
    return "?", name, "?"


def _collect(root):
    """``(key, outcomes)`` pairs found under *root*.

    A manifest is authoritative: it maps the trial directory to an agent,
    setting and model. If a directory has no manifest, its ``results.jsonl``
    files are still read and labelled by directory name, so an interrupted run
    does not silently disappear from the table.
    """
    found = []
    manifests = []
    orphan_results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _PRUNE]
        if "manifest.json" in filenames:
            manifests.append(os.path.join(dirpath, "manifest.json"))
        if "results.jsonl" in filenames:
            orphan_results.append(os.path.join(dirpath, "results.jsonl"))

    claimed = set()
    for manifest_path in manifests:
        run_dir = os.path.dirname(manifest_path)
        for agent, setting, model, run_id in _manifest_trials(manifest_path):
            results_path = os.path.join(run_dir, "trials", run_id,
                                        "results.jsonl")
            claimed.add(os.path.abspath(results_path))
            outcomes = read_results(results_path)
            found.append(((agent, setting, model), outcomes))

    for results_path in orphan_results:
        if os.path.abspath(results_path) in claimed:
            continue
        found.append((_fallback_label(results_path), read_results(results_path)))
    return found


def collect(paths):
    """Aggregate outcomes by ``(agent, setting, model)`` across *paths*."""
    cells = {}
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"table: no such file or directory: {path!r}")
        if os.path.isfile(path):
            if not path.endswith("results.jsonl"):
                raise SystemExit(
                    f"table: expected a results directory or results.jsonl: {path!r}")
            key = _fallback_label(path)
            pairs = [(key, read_results(path))]
        else:
            pairs = _collect(path)
        for key, outcomes in pairs:
            cells.setdefault(key, []).extend(outcomes)

    rows = []
    for (agent, setting, model), outcomes in cells.items():
        rows.append(_row(agent, setting, model, outcomes))
    return rows


def _row(agent, setting, model, outcomes):
    n = len(outcomes)
    resolved = sum(1 for outcome in outcomes if outcome.resolved)
    errors = sum(1 for outcome in outcomes
                 if outcome.outcome == outcome.ERROR)
    timeouts = sum(1 for outcome in outcomes
                   if outcome.outcome == outcome.TIMEOUT)
    comparable = n - errors - timeouts

    durations = [outcome.duration_seconds for outcome in outcomes
                 if outcome.duration_seconds is not None]
    prompt = completion = total = 0
    token_trials = 0
    for outcome in outcomes:
        metrics = outcome_metrics(outcome)
        if not any(key in metrics for key in
                   ("prompt_tokens", "completion_tokens", "total_tokens")):
            continue
        token_trials += 1
        prompt += int(metrics.get("prompt_tokens") or 0)
        completion += int(metrics.get("completion_tokens") or 0)
        total += int(metrics.get("total_tokens")
                     or (int(metrics.get("prompt_tokens") or 0)
                         + int(metrics.get("completion_tokens") or 0)))

    return {
        "agent": agent,
        "setting": setting,
        "model": model,
        "n": n,
        "resolved": resolved,
        "raw": (resolved / n) if n else None,
        "errors": errors,
        "clean": (resolved / comparable) if comparable else None,
        "mean_seconds": (sum(durations) / len(durations)) if durations else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "token_trials": token_trials,
        "mean_total_tokens": (total / token_trials) if token_trials else None,
    }


def _fmt_rate(value):
    return "-" if value is None else f"{value:.1%}"


def _fmt_mean(value):
    return "-" if value is None else f"{value:,.0f}"


def _cells(row):
    n = row["n"]
    return (
        row["agent"],
        row["setting"],
        f"{row['resolved']}/{n}",
        _fmt_rate(row["raw"]),
        str(row["errors"]),
        _fmt_rate(row["clean"]),
        _fmt_mean(row["mean_seconds"]),
        f"{row['total_tokens']:,}",
        _fmt_mean(row["mean_total_tokens"]),
        f"{row['token_trials']}/{n}",
    )


def _sort(rows, key):
    if key == "raw":
        return sorted(rows, key=lambda row: (
            -(row["raw"] if row["raw"] is not None else -1),
            row["agent"], row["setting"]))
    if key == "tokens":
        return sorted(rows, key=lambda row: (
            -(row["mean_total_tokens"] or 0), row["agent"], row["setting"]))
    return sorted(rows, key=lambda row: (
        row["agent"], row["setting"], row["model"]))


def render_table(rows, sort="raw"):
    """A fixed-width terminal table."""
    rows = _sort(rows, sort)
    body = [_cells(row) for row in rows]
    widths = [max(len(_HEADER[i]), *(len(cells[i]) for cells in body))
              if body else len(_HEADER[i]) for i in range(len(_HEADER))]
    lines = ["  ".join(header.ljust(widths[i])
                       for i, header in enumerate(_HEADER)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(widths[i])
                           for i, cell in enumerate(cells)).rstrip()
                 for cells in body)
    incomplete = [row for row in rows if row["token_trials"] < row["n"]]
    if incomplete:
        lines.append("")
        lines.append("token means use trials with metrics: " + ", ".join(
            f"{row['agent']}/{row['setting']} {row['token_trials']}/{row['n']}"
            for row in incomplete))
    return "\n".join(lines)


def render_markdown(rows, sort="raw", title="Benchmark scorecard"):
    """The same table as markdown, for a README or a PR comment."""
    rows = _sort(rows, sort)
    lines = [f"# {title}", "",
             "| " + " | ".join(_HEADER) + " |",
             "|" + "|".join(["---"] * len(_HEADER)) + "|"]
    lines.extend("| " + " | ".join(_cells(row)) + " |" for row in rows)
    incomplete = [row for row in rows if row["token_trials"] < row["n"]]
    if incomplete:
        lines += ["", "Token means use trials with metrics: " + ", ".join(
            f"`{row['agent']}/{row['setting']}` {row['token_trials']}/{row['n']}"
            for row in incomplete) + "."]
    lines += ["", "`raw` = resolved / n; `clean` = resolved / (n - errors - "
              "timeouts); tokens are the per-task sums from `results.jsonl`."]
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.table",
        description="Scorecard table from stored benchmark results.")
    parser.add_argument("paths", nargs="+",
                        help="campaign directory, agent run directory, or results.jsonl")
    parser.add_argument("--sort", choices=("raw", "agent", "tokens"),
                        default="raw", help="row order (default: raw score)")
    parser.add_argument("--md", default=None, metavar="PATH",
                        help="also write the table as markdown")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="also write the rows as JSON")
    return parser


def _write(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = collect(args.paths)
    print(render_table(rows, sort=args.sort))
    if not rows:
        print("(no results found; point at a campaign or agent run directory)",
              file=sys.stderr)
    if args.md:
        _write(args.md, render_markdown(rows, sort=args.sort))
        print(f"\nmarkdown: {args.md}")
    if args.json:
        _write(args.json, json.dumps({"rows": _sort(rows, args.sort)},
                                     indent=2, sort_keys=True) + "\n")
        print(f"json:     {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
