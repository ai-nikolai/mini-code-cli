"""Summarise benchmark results: ``python -m benchmarks.analyze <csv|run-dir> ...``

``benchmarks run --csv`` (and ``benchmarks report --csv``) write one row per
trial; this module turns any number of those files -- or the run directories they
came from -- into a single table. It is deliberately a *reader*: nothing here
touches a container, a dataset or an agent, so it runs anywhere (a laptop, a login
node without a scheduler) and can be pointed at results copied off the GPU host.

    python -m benchmarks.analyze benchmarks/results/ornith-30b-smoke.csv
    python -m benchmarks.analyze benchmarks/runs/20260101T... --md summary.md
    python -m benchmarks.analyze results/*.csv --csv combined.csv --json out.json

Rows are merged on ``(suite, agent, setting, model)``: two runs of the same
configuration are the same cell, and adding them is what makes "3 samples" and
"3 samples again" comparable. ``n = 0`` rows are kept (a trial that produced no
outcome is a finding, not a rounding error) and are printed with ``-`` rates.
"""
import argparse
import json
import os
import sys

from .results import (CSV_COLUMNS, read_csv, read_results, summarise, to_csv)

KEY = ("suite", "agent", "setting", "model")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def rows_from_path(path):
    """Rows from a CSV, a ``summary.json``, or a run directory.

    A run directory is read in this order: ``summary.json`` (already aggregated by
    the runner), then ``manifest.json`` + ``trials/*/results.jsonl`` (which is
    what a run whose summary step was interrupted still has). Anything else is an
    error -- silently analysing nothing is the one failure mode that matters here.
    """
    if os.path.isdir(path):
        summary = os.path.join(path, "summary.json")
        if os.path.isfile(summary):
            with open(summary, encoding="utf-8") as handle:
                return list(json.load(handle).get("rows") or [])
        return _rows_from_run_dir(path)
    if str(path).endswith((".csv", ".tsv")):
        return read_csv(path)
    if str(path).endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "rows" in payload:
            return list(payload.get("rows") or [])
        if isinstance(payload, list):
            return payload
    raise SystemExit(f"analyze: cannot read results from {path!r} "
                     f"(expected a .csv, a summary.json, or a run directory)")


def _rows_from_run_dir(run_dir):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"analyze: {run_dir!r} is not a run directory "
                         f"(no manifest.json and no summary.json)")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    outcomes = {}
    trials = manifest.get("trials") or []
    for trial in trials:
        path = os.path.join(run_dir, "trials", trial["run_id"], "results.jsonl")
        outcomes[trial["run_id"]] = read_results(path)
    return summarise(trials, outcomes)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(rows):
    """Merge rows with the same (suite, agent, setting, model); sort for output.

    Durations are merged as a *pooled mean* weighted by the number of outcomes
    that had a duration, which is the only merge that does not invent data: a
    plain mean of means would over-weight a 1-instance run next to a 300-instance
    one.
    """
    merged = {}
    order = []
    for row in rows:
        key = tuple(row.get(part) for part in KEY)
        if key not in merged:
            merged[key] = {
                "suite": row.get("suite"),
                "agent": row.get("agent"),
                "setting": row.get("setting"),
                "model": row.get("model"),
                "run_id": row.get("run_id"),
                "n": 0, "resolved": 0, "unresolved": 0, "errors": 0,
                "timeouts": 0, "command_errors": 0,
                "_duration_total": 0.0, "_duration_n": 0, "_runs": 0,
            }
            order.append(key)
        cell = merged[key]
        cell["_runs"] += 1
        for field in ("n", "resolved", "unresolved", "errors", "timeouts",
                      "command_errors"):
            cell[field] += int(row.get(field) or 0)
        mean = row.get("mean_seconds")
        count = int(row.get("n") or 0)
        if mean is not None and count:
            cell["_duration_total"] += float(mean) * count
            cell["_duration_n"] += count
        if not cell.get("run_id") or cell["_runs"] == 1:
            cell["run_id"] = row.get("run_id") or cell.get("run_id")

    out = []
    for key in order:
        cell = merged[key]
        n = cell.pop("_duration_n", 0) or 0
        total = cell.pop("_duration_total", 0.0)
        cell["rate"] = round(cell["resolved"] / cell["n"], 4) if cell["n"] else None
        cell["mean_seconds"] = round(total / n, 2) if n else None
        out.append(cell)
    out.sort(key=lambda r: (str(r.get("suite")), str(r.get("agent")),
                            str(r.get("setting")), str(r.get("model"))))
    return out


def totals(rows):
    """One ``ALL`` row across every cell (the number a README quotes)."""
    n = sum(int(r.get("n") or 0) for r in rows)
    resolved = sum(int(r.get("resolved") or 0) for r in rows)
    duration_total = sum(float(r["mean_seconds"]) * int(r.get("n") or 0)
                         for r in rows if r.get("mean_seconds") is not None)
    return {
        "suite": "ALL", "agent": "", "setting": "", "model": "", "run_id": "",
        "n": n, "resolved": resolved,
        "unresolved": sum(int(r.get("unresolved") or 0) for r in rows),
        "errors": sum(int(r.get("errors") or 0) for r in rows),
        "timeouts": sum(int(r.get("timeouts") or 0) for r in rows),
        "command_errors": sum(int(r.get("command_errors") or 0) for r in rows),
        "rate": round(resolved / n, 4) if n else None,
        "mean_seconds": round(duration_total / n, 2) if n else None,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

HEADER = ("suite", "agent", "setting", "model", "n", "ok", "rate", "err",
          "to", "cmdfail", "mean s")


def _cells(row):
    rate = "-" if row.get("rate") is None else f"{float(row['rate']):.0%}"
    mean = ("-" if row.get("mean_seconds") is None
            else f"{float(row['mean_seconds']):.0f}")
    return (str(row.get("suite") or ""), str(row.get("agent") or ""),
            str(row.get("setting") or ""), str(row.get("model") or ""),
            str(row.get("n", 0)), str(row.get("resolved", 0)), rate,
            str(row.get("errors", 0)), str(row.get("timeouts", 0)),
            str(row.get("command_errors", 0)), mean)


def render_table(rows, include_total=True):
    """A fixed-width table for a terminal (no dependency, no colour)."""
    body = [_cells(row) for row in rows]
    if include_total and rows:
        body.append(_cells(totals(rows)))
    widths = [max(len(HEADER[i]), *(len(r[i]) for r in body)) if body
              else len(HEADER[i]) for i in range(len(HEADER))]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(HEADER)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for cells in body:
        lines.append("  ".join(c.ljust(widths[i])
                               for i, c in enumerate(cells)).rstrip())
    return "\n".join(lines)


def render_markdown(rows, title="Benchmark results"):
    """The same table as markdown, for a README or a PR comment."""
    lines = [f"# {title}", "",
             "| " + " | ".join(HEADER) + " |",
             "|" + "|".join(["---"] * len(HEADER)) + "|"]
    for row in list(rows) + ([totals(rows)] if rows else []):
        lines.append("| " + " | ".join(_cells(row)) + " |")
    lines += ["", "`ok` = resolved instances, `to` = timeouts, `cmdfail` = commands "
              "that did not finish (a missing harness lands here, not in the score)."]
    return "\n".join(lines) + "\n"


def _clean(rows):
    """Rows trimmed to the CSV columns (drop the bookkeeping keys)."""
    return [{key: row.get(key) for key in CSV_COLUMNS} for row in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.analyze",
        description="Summarise one or more benchmark result files.")
    parser.add_argument("paths", nargs="+",
                        help="summary CSV, summary.json, or a run directory")
    parser.add_argument("--md", default=None, metavar="PATH",
                        help="also write the table as markdown")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="also write the aggregated rows as JSON")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="also write the aggregated rows as CSV")
    parser.add_argument("--no-total", action="store_true",
                        help="omit the ALL row")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = []
    for path in args.paths:
        if not os.path.exists(path):
            print(f"error: no such file or directory: {path}", file=sys.stderr)
            return 2
        rows.extend(rows_from_path(path))
    merged = aggregate(rows)

    print(render_table(merged, include_total=not args.no_total))
    if not merged:
        print("(no rows: the inputs produced no outcomes)", file=sys.stderr)

    if args.md:
        _write(args.md, render_markdown(merged))
        print(f"\nmarkdown: {args.md}")
    if args.json:
        _write(args.json, json.dumps({
            "rows": _clean(merged),
            "total": totals(merged),
            "sources": args.paths,
        }, indent=2, sort_keys=True) + "\n")
        print(f"json:     {args.json}")
    if args.csv:
        to_csv(_clean(merged), args.csv)
        print(f"csv:      {args.csv}")
    return 0


def _write(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
