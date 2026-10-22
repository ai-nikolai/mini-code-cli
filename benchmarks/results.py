"""Results: collect outcomes, aggregate them, and write a comparable report.

Two rules shape this module:

* **A report must never contain a secret.** Agent logs and command lines are full
  of ``-e API_KEY=...``; everything that is written out goes through
  :func:`redact`, which replaces the *value* of any environment-like variable
  whose name looks sensitive and keeps the name, so the report stays debuggable.
* **Numbers must be comparable.** Rates are computed per (suite, agent, model)
  from :class:`~benchmarks.suites.base.TrialOutcome` only -- the suites resolve
  their own scoring rules (a TB task is resolved by its tests, a SWE-bench
  instance by the harness), and this module just counts.
"""
import json
import os
import re

#: An environment assignment whose *value* is replaced: the name is kept because
#: "which key was used" is useful, the value never is.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"
    r"[A-Z0-9_]*)(\s*[=:]\s*)(['\"]?)([^\s'\",&]+)")

#: Bare keys that agents print in headers or JSON bodies.
_BARE_SECRET = re.compile(
    r"(?i)\b(sk-[A-Za-z0-9_\-]{8,}|ghp_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,})")


def redact(text):
    """Replace credential-looking substrings with ``***``.

    Applied to everything the harness writes: ``results.jsonl``, ``summary.md``
    and every per-command log. Raw agent output is *not* kept anywhere on disk --
    a log is the artifact most likely to be pasted into an issue, and an agent
    that echoes its own environment is normal. The patterns are narrow (secret
    *values*, not whole lines), so a failure is still readable.
    """
    if text is None:
        return ""
    text = str(text)
    text = _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}"
                                           f"{m.group(3)}***", text)
    return _BARE_SECRET.sub("***", text)


def outcome_from_dict(payload):
    """Rebuild a :class:`TrialOutcome` from its ``to_dict`` form."""
    from .suites.base import TrialOutcome

    return TrialOutcome(
        suite=payload.get("suite", ""),
        instance_id=payload.get("instance_id", ""),
        outcome=payload.get("outcome", TrialOutcome.ERROR),
        duration_seconds=payload.get("duration_seconds"),
        detail=payload.get("detail", ""),
        artifacts=payload.get("artifacts", ()),
        extra=payload.get("extra"),
    )


def read_results(path):
    """Read ``results.jsonl`` (one outcome per line, JSON)."""
    outcomes = []
    if not os.path.isfile(path):
        return outcomes
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            outcomes.append(outcome_from_dict(json.loads(line)))
    return outcomes


def write_results(path, outcomes):
    """Write outcomes as JSONL, redacted, one per line, sorted for stable diffs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for outcome in sorted(outcomes, key=lambda o: (o.suite, o.instance_id)):
            payload = outcome.to_dict()
            if payload.get("detail"):
                payload["detail"] = redact(payload["detail"])
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def summarise(trials, outcomes_by_trial):
    """Aggregate outcomes per trial.

    ``trials`` is the manifest's list of trial descriptions and
    ``outcomes_by_trial`` maps ``run_id -> [TrialOutcome]``. Keying by run id (not
    by suite) is what keeps "mini-code:plan" and "mini-code:default" apart -- they
    share a suite but must never share a row. A trial with no outcomes still gets
    a row with ``n = 0``, because a silently missing agent is exactly what a
    benchmark report must not hide.
    """
    return [_row(trial, outcomes_by_trial.get(trial.get("run_id"), []))
            for trial in trials]


def _row(trial, outcomes):
    """One report row: counts plus a rate, with the denominator spelled out."""
    total = len(outcomes)
    resolved = sum(1 for o in outcomes if o.resolved)
    errors = sum(1 for o in outcomes if o.outcome == "error")
    timeouts = sum(1 for o in outcomes if o.outcome == "timeout")
    durations = [o.duration_seconds for o in outcomes
                 if o.duration_seconds is not None]
    return {
        "suite": trial.get("suite"),
        "agent": trial.get("agent"),
        "setting": trial.get("setting"),
        "model": trial.get("model"),
        "run_id": trial.get("run_id"),
        "n": total,
        "resolved": resolved,
        "unresolved": total - resolved - errors - timeouts,
        "errors": errors,
        "timeouts": timeouts,
        "command_errors": len(trial.get("failed_commands") or []),
        "rate": (round(resolved / total, 4) if total else None),
        "mean_seconds": (round(sum(durations) / len(durations), 2)
                         if durations else None),
    }


def write_summary(run_dir, rows, manifest=None):
    """Write ``summary.json`` and ``summary.md`` into ``run_dir``; return both paths."""
    json_path = os.path.join(run_dir, "summary.json")
    md_path = os.path.join(run_dir, "summary.md")
    payload = {"run_id": (manifest or {}).get("run_id"),
               "git_sha": (manifest or {}).get("git_sha"),
               "rows": rows}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(rows, manifest))
    return json_path, md_path


def render_markdown(rows, manifest=None):
    """A table a human can paste into a report.

    Columns are fixed and the rate is shown with its denominator, because
    "0.67" from 3 instances and "0.67" from 300 are not the same claim.
    """
    lines = ["# Benchmark summary", ""]
    if manifest:
        lines += [
            f"- run: `{manifest.get('run_id')}`",
            f"- git: `{manifest.get('git_sha')}`"
            + (" (dirty)" if manifest.get("git_dirty") else ""),
            f"- started: {manifest.get('started_at')}",
            f"- finished: {manifest.get('finished_at')}",
            "",
        ]
    header = ("| suite | agent | setting | model | resolved | n | rate | "
              "errors | timeouts | cmd fail | mean s |")
    divider = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines += [header, divider]
    for row in rows:
        rate = "-" if row["rate"] is None else f"{row['rate']:.2%}"
        mean = "-" if row["mean_seconds"] is None else f"{row['mean_seconds']:.1f}"
        lines.append(
            f"| {row['suite']} | {row['agent']} | {row['setting']} | "
            f"{row['model']} | {row['resolved']} | {row['n']} | {rate} | "
            f"{row['errors']} | {row['timeouts']} | {row.get('command_errors', 0)} | "
            f"{mean} |")
    lines.append("")
    lines.append("Rates are resolved/n per trial; `n = 0` means the trial produced "
                 "no result, and `cmd fail` counts commands that did not finish "
                 "cleanly (a missing harness shows up there, not as a score).")
    return "\n".join(lines) + "\n"
