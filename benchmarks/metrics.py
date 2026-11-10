"""Extract per-instance agent metrics from the logs a trial already writes.

The harness does not need to change any agent: mini-code prints a token/call
summary, Claude Code emits a JSON result, and the runner logs command durations.
This module reads those artifacts and puts a small, uniform ``metrics`` block on
each :class:`~benchmarks.suites.base.TrialOutcome`:

    duration_seconds, prompt_tokens, completion_tokens, total_tokens,
    llm_calls, tool_calls, turns, tool_names, cost_usd

Missing values stay absent -- a local vLLM server may not report tokens, and a
"0" would be a lie. The results CSV then leaves those cells empty.
"""
import datetime
import glob
import json
import os
import re

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_USAGE = re.compile(
    r"tokens\s*:\s*input=(\d+)\s+output=(\d+)\s+toolcalls=(\d+)\s+"
    r"total=(\d+)(?:\s*\|\s*(\d+)\s+LLM calls)?")
_GOODBYE_TOKENS = re.compile(
    r"tokens\s*:\s*([\d,]+)\s+in\s*/\s*([\d,]+)\s+out\s*\(([\d,]+)\s+total\)")
_GOODBYE_CALLS = re.compile(r"calls\s*:\s*(\d+)\s+LLM,\s*(\d+)\s+tool")
_EXIT = re.compile(r"\[exit\s+\d+\s+in\s+([0-9.]+)s\]")
_TOOL_NAME = re.compile(r"\n\s*tool\s*\n\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _int(value):
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _clean(text):
    return _ANSI.sub("", text or "")


def _json_objects(text):
    """Yield parsed JSON objects from lines that look like a JSON record."""
    for line in _clean(text).splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            yield payload


def parse_text(text):
    """Best-effort metrics from one agent log. Never raises."""
    text = _clean(text)
    metrics = {}
    if not text:
        return metrics

    usage = _USAGE.findall(text)
    if usage:
        prompt, completion, tools, total, calls = usage[-1]
        metrics.update({
            "prompt_tokens": _int(prompt),
            "completion_tokens": _int(completion),
            "total_tokens": _int(total),
            "tool_calls": _int(tools),
        })
        if calls:
            metrics["llm_calls"] = _int(calls)
            metrics["turns"] = _int(calls)

    # The goodbye box is the final, authoritative per-session total.
    goodbye = _GOODBYE_TOKENS.search(text)
    if goodbye:
        metrics.update({
            "prompt_tokens": _int(goodbye.group(1)),
            "completion_tokens": _int(goodbye.group(2)),
            "total_tokens": _int(goodbye.group(3)),
        })
    calls = _GOODBYE_CALLS.search(text)
    if calls:
        metrics["llm_calls"] = _int(calls.group(1))
        metrics["turns"] = _int(calls.group(1))
        metrics["tool_calls"] = _int(calls.group(2))

    names = sorted(set(_TOOL_NAME.findall(text)))
    if names:
        metrics["tool_names"] = names

    exits = _EXIT.findall(text)
    if exits:
        metrics["duration_seconds"] = _float(exits[-1])

    # Claude Code --print --output-format json / Pi event records.
    for payload in _json_objects(text):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            input_tokens = (usage.get("input_tokens")
                            or usage.get("prompt_tokens"))
            output_tokens = (usage.get("output_tokens")
                             or usage.get("completion_tokens"))
            if input_tokens is not None:
                metrics["prompt_tokens"] = _int(input_tokens)
            if output_tokens is not None:
                metrics["completion_tokens"] = _int(output_tokens)
            if input_tokens is not None or output_tokens is not None:
                metrics["total_tokens"] = (
                    (_int(input_tokens) or 0) + (_int(output_tokens) or 0))
        if payload.get("num_turns") is not None:
            metrics["turns"] = _int(payload["num_turns"])
            metrics.setdefault("llm_calls", metrics["turns"])
        if payload.get("duration_ms") is not None:
            metrics["duration_seconds"] = _float(
                float(payload["duration_ms"]) / 1000.0)
        if payload.get("total_cost_usd") is not None:
            metrics["cost_usd"] = _float(payload["total_cost_usd"])
        tool_uses = text.count('"type":"tool_use"') + text.count('"type": "tool_use"')
        if tool_uses:
            metrics["tool_calls"] = tool_uses

    return {key: value for key, value in metrics.items()
            if value is not None and value != ""}


def _iso_seconds(start, end):
    try:
        first = datetime.datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        last = datetime.datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return round((last - first).total_seconds(), 3)
    except (TypeError, ValueError):
        return None


def _harbor_trial_metrics(result_path):
    """Duration/tokens from one harbor task's ``result.json``."""
    try:
        with open(result_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    metrics = {}
    execution = payload.get("agent_execution") or {}
    seconds = _iso_seconds(execution.get("started_at"),
                           execution.get("finished_at"))
    if seconds is not None:
        metrics["duration_seconds"] = seconds
    result = payload.get("agent_result") or {}
    prompt = result.get("n_input_tokens")
    completion = result.get("n_output_tokens")
    if prompt is not None:
        metrics["prompt_tokens"] = _int(prompt)
    if completion is not None:
        metrics["completion_tokens"] = _int(completion)
    if prompt is not None or completion is not None:
        metrics["total_tokens"] = (_int(prompt) or 0) + (_int(completion) or 0)
    steps = payload.get("step_results")
    if isinstance(steps, list) and steps:
        metrics["turns"] = len(steps)
    return metrics


def _log_candidates(trial_dir, instance_id, suite):
    """Agent logs that may describe one instance/task, best first."""
    candidates = []
    safe = str(instance_id)
    if suite == "swebench":
        candidates.append(os.path.join(trial_dir, "instances", safe, "agent.log"))
        candidates.extend(sorted(glob.glob(os.path.join(
            trial_dir, "logs", f"*agent*{safe}*.log"))))
    else:
        for root in sorted(glob.glob(os.path.join(trial_dir, "harbor", "*",
                                                   f"{safe}*"))):
            candidates.append(os.path.join(root, "trial.log"))
            candidates.extend(sorted(glob.glob(os.path.join(root, "agent", "*"))))
    return [path for path in candidates if os.path.isfile(path)]


def enrich_outcomes(trial_dir, outcomes):
    """Attach ``extra["metrics"]`` to each outcome from its own artifacts."""
    for outcome in outcomes:
        suite = getattr(outcome, "suite", "")
        metrics = {}
        for path in _log_candidates(trial_dir, outcome.instance_id, suite):
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            metrics.update(parse_text(text))
        if suite == "terminal-bench":
            for result_path in glob.glob(os.path.join(
                    trial_dir, "harbor", "*", f"{outcome.instance_id}*",
                    "result.json")):
                metrics.update(_harbor_trial_metrics(result_path))
        if metrics.get("duration_seconds") is not None:
            outcome.duration_seconds = metrics["duration_seconds"]
        if metrics:
            outcome.extra["metrics"] = metrics
    return outcomes


def outcome_metrics(outcome):
    """The metrics block for an outcome, with top-level duration as fallback."""
    extra = getattr(outcome, "extra", None) or {}
    metrics = dict(extra.get("metrics") or {}) if isinstance(extra, dict) else {}
    if outcome.duration_seconds is not None:
        metrics.setdefault("duration_seconds", outcome.duration_seconds)
    return metrics


METRIC_COLUMNS = (
    "duration_seconds", "prompt_tokens", "completion_tokens",
    "total_tokens", "llm_calls", "tool_calls", "turns", "cost_usd",
)


METRICS_CSV_COLUMNS = (
    "dataset", "suite", "agent", "setting", "model", "run_id",
    "instance_id", "outcome", "resolved", *METRIC_COLUMNS,
    "tool_names", "detail",
)


def metric_row(trial, adapter, outcome):
    """One flat row for ``metrics.csv`` (one row per task/instance)."""
    values = outcome_metrics(outcome)
    dataset = str(trial.suite.dataset or "")
    version = getattr(trial.suite, "version", None)
    if version:
        dataset = f"{dataset}@{version}"
    row = {
        "dataset": dataset,
        "suite": trial.suite.name,
        "agent": trial.agent.name,
        "setting": trial.agent.setting,
        "model": trial.model.name,
        "run_id": trial.run_id,
        "instance_id": outcome.instance_id,
        "outcome": outcome.outcome,
        "resolved": outcome.resolved,
        "detail": outcome.detail,
    }
    for key in METRIC_COLUMNS:
        row[key] = values.get(key)
    names = values.get("tool_names") or []
    row["tool_names"] = ";".join(names) if isinstance(names, list) else str(names)
    return row


def write_metrics(path, rows):
    """Write the detailed per-instance metrics CSV (header always present)."""
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_CSV_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key))
                             for key in METRICS_CSV_COLUMNS})
    return path


def merge_metrics(paths, out_path):
    """Concatenate per-run ``metrics.csv`` files into one campaign metrics file."""
    import csv

    rows = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return write_metrics(out_path, rows)
