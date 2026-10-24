"""The one command a person runs: ``benchmarks/run_benchmarks.py``.

    python benchmarks/run_benchmarks.py --dataset swe-bench --agent mini-code --mode short_eval
    python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent all --mode short

It exists because the harness underneath has three moving parts -- a config, a
suite and an agent -- and "which config, which limit, which timeouts" is exactly
the knowledge that should not have to be retyped (or mis-remembered) per run.

What it does:

* picks the config for the dataset and applies the mode as an override (limit,
  concurrency, evaluation on/off, timeouts);
* expands ``--agent all`` into one run per agent, all under one campaign directory;
* runs each agent's matrix, then writes ``results/<campaign>/summary_results.csv``
  with one row per agent (:mod:`benchmarks.summary`);
* ``--dry-run`` prints the commands that would run and the summary path, without
  touching a container or the model server.

Modes are deliberately few. ``short`` is a smoke run whose purpose is "does this
agent work end to end", so graders are off and it is cheap to repeat;
``short_eval`` is the same size with the graders on, so the number it prints is a
real score; ``full`` drops the limit and gives agents an hour.
"""
import argparse
import datetime
import os
import sys

from . import summary as summary_module
from .cli import _cmd_prepare, _reuse_prepared_inputs
from .config import ConfigError, load_config, parse_config, validate_config
from .runner import Runner

EXIT_OK = 0
EXIT_PROBLEM = 2

#: ``--dataset`` -> (config file, suite name, label used in the CSV).
DATASETS = {
    "swe-bench": ("benchmarks/configs/ornith_30b.json", "swebench", "swe-bench"),
    "swebench": ("benchmarks/configs/ornith_30b.json", "swebench", "swe-bench"),
    "terminalbench-2.1": ("benchmarks/configs/terminal_bench_21.json",
                          "terminal-bench", "terminalbench-2.1"),
    "terminal-bench-2.1": ("benchmarks/configs/terminal_bench_21.json",
                           "terminal-bench", "terminalbench-2.1"),
    "terminalbench-2.0": ("benchmarks/configs/ornith_30b.json",
                          "terminal-bench", "terminalbench-2.0"),
    "terminal-bench": ("benchmarks/configs/ornith_30b.json", "terminal-bench",
                       "terminalbench-2.0"),
}

#: ``--mode`` -> (limit, evaluation on?, option overrides).
MODES = {
    "short": (3, False, {}),
    "short_eval": (3, True, {}),
    "full": (None, True, {"agent_timeout_sec": 3600, "test_timeout_sec": 3600,
                          "instance_seconds": 7200, "agent_seconds": 3600}),
}

AGENT_CHOICES = ("mini-code", "pi", "claude-code", "all")
ALL_AGENTS = ("mini-code", "pi", "claude-code")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="benchmarks/run_benchmarks.py",
        description="Run one benchmark: pick a dataset, an agent and a mode.")
    parser.add_argument("--dataset", "-d", required=True, choices=sorted(DATASETS),
                        help="swe-bench | terminalbench-2.1 | terminalbench-2.0")
    parser.add_argument("--agent", "-a", default="all", choices=AGENT_CHOICES,
                        help="mini-code | pi | claude-code | all (default: all)")
    # Both spellings are accepted on purpose: `--mode` in documentation, `-mode`
    # is what people type after reading a single-dash example.
    parser.add_argument("--mode", "-mode", "-m", dest="mode",
                        default="short_eval", choices=sorted(MODES),
                        help="short | short_eval | full (default: short_eval)")
    parser.add_argument("--limit", type=int, default=None,
                        help="override the mode's task limit")
    parser.add_argument("--run-id", default=None,
                        help="campaign name (default: <dataset>-<agent>-<mode>-<time>)")
    parser.add_argument("--results-dir", default="results",
                        help="where the campaign directory and CSV go (default: results)")
    parser.add_argument("--n-concurrent", type=int, default=None,
                        help="override concurrency")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and the summary path, run nothing")
    parser.add_argument("--scratch-dir", default=None,
                        help="where container images are unpacked (default: "
                             "$BENCHMARK_SCRATCH, else <results-dir>/.scratch)")
    return parser


def configure_scratch(directory, scratch_dir=None):
    """Point Apptainer's temporary and cache directories at real disk.

    A container image is unpacked into ``APPTAINER_TMPDIR``, and ``/tmp`` is
    routinely a 1 GB partition: pulling a SWE-bench image there fails with
    ``no space left on device`` while unpacking ``miniconda.sh``, and the failure
    looks like a broken dataset rather than a full disk. The same pull on a real
    filesystem takes a minute.
    """
    root = scratch_dir or os.environ.get("BENCHMARK_SCRATCH") or \
        os.path.join(directory, ".scratch")
    for name, sub in (("APPTAINER_TMPDIR", "tmp"),
                      ("APPTAINER_CACHEDIR", "apptainer")):
        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
        os.environ.setdefault(name, path)
    os.environ.setdefault("SINGULARITY_TMPDIR", os.environ["APPTAINER_TMPDIR"])
    return root


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def build_config(dataset, agent, mode, limit=None, n_concurrent=None,
                 config_path=None):
    """The config for one agent's run, with the mode applied on top.

    The config file stays the source of truth for models and endpoints; this only
    narrows the matrix to one suite and one agent, and sets the size and the
    evaluation switches that the mode names.
    """
    path, suite_name, label = DATASETS[dataset]
    mode_limit, evaluate, overrides = MODES[mode]
    limit = mode_limit if limit is None else limit

    config = load_config(config_path or path)
    documented = config.to_dict()
    documented.pop("_comment", None)

    agents = documented.get("agents") or []
    if agent != "all":
        agents = [a for a in agents if a.get("name") == agent]
        if not agents:
            raise ConfigError(
                f"agent {agent!r} is not configured in {config_path or path}")
    documented["agents"] = agents

    suites = []
    for suite in documented.get("suites") or []:
        if suite.get("name") != suite_name:
            continue
        suite = dict(suite)
        if limit is not None:
            suite["limit"] = limit
        options = dict(suite.get("options") or {})
        if n_concurrent is not None:
            options["n_concurrent"] = n_concurrent
            options["max_workers"] = n_concurrent
        if suite_name == "swebench":
            options["skip_evaluation"] = not evaluate
            if "instance_seconds" in overrides:
                options["instance_seconds"] = overrides["instance_seconds"]
        else:
            if "agent_timeout_sec" in overrides:
                options["agent_timeout_sec"] = overrides["agent_timeout_sec"]
        if evaluate and "test_timeout_sec" in overrides:
            options["test_timeout_sec"] = overrides["test_timeout_sec"]
        suite["options"] = options
        suites.append(suite)
    if not suites:
        raise ConfigError(f"suite {suite_name!r} is not in {config_path or path}")
    documented["suites"] = suites
    return parse_config(documented), suite_name, label


def main(argv=None):
    args = build_parser().parse_args(argv)
    label = DATASETS[args.dataset][2]
    agents = list(ALL_AGENTS) if args.agent == "all" else [args.agent]
    run_id = args.run_id or f"{label.replace('.', '_')}-{args.agent}-{args.mode}-{_timestamp()}"
    directory = os.path.join(args.results_dir, run_id)
    summary_path = os.path.join(directory, summary_module.SUMMARY_FILENAME)

    prepared = []
    try:
        for agent in agents:
            config, suite_name, _ = build_config(
                args.dataset, agent, args.mode, limit=args.limit,
                n_concurrent=args.n_concurrent)
            problems = validate_config(config, check_env=not args.dry_run)
            if problems:
                raise ConfigError("; ".join(problems))
            prepared.append((agent, config, suite_name))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PROBLEM

    scratch = configure_scratch(directory, args.scratch_dir)
    print(f"dataset: {label}   mode: {args.mode}   agents: {', '.join(agents)}")
    print(f"scratch:  {scratch}")
    print(f"campaign: {directory}")
    print(f"summary:  {summary_path}")

    rows = []
    for agent, config, suite_name in prepared:
        agent_dir = os.path.join(directory, agent)
        trials = config.trials(suites=[suite_name])
        if not trials:
            print(f"error: agent {agent!r} produced no trials for {suite_name}",
                  file=sys.stderr)
            return EXIT_PROBLEM
        # The campaign directory *is* this agent's output dir, so the prepared
        # inputs and the run directory line up (`prepare` writes under
        # <output_dir>/_prepare/<trial>, a run reads <output_dir>/<run>/<trial>`).
        config.output_dir = agent_dir
        # Materialising the dataset is a separate, cacheable step: doing it once
        # per agent is what keeps a run from dying with "no instance list" after
        # the containers were already set up.
        if not args.dry_run and _cmd_prepare(config, trials) != EXIT_OK:
            print(f"error: could not prepare inputs for {agent}", file=sys.stderr)
            return EXIT_PROBLEM
        _reuse_prepared_inputs(config, trials)
        runner = Runner(config, dry_run=args.dry_run, output=agent_dir)
        if args.dry_run:
            for message in runner.plan(trials):
                print(message)
            rows.append(summary_module.row_for(
                label, agent, args.mode, f"{run_id}/{agent}", [], status="dry-run"))
            continue
        manifest = runner.run(trials, run_id=f"{run_id}/{agent}")
        outcomes = summary_module.read_run_outcomes(
            runner.run_dir(f"{run_id}/{agent}"))
        rows.append(summary_module.row_for(
            label, agent, args.mode,
            (manifest or {}).get("run_id") or f"{run_id}/{agent}", outcomes))

    if args.dry_run:
        dry_dir = os.path.join(args.results_dir, "_dry_run", run_id)
        summary_module.write_summary(dry_dir, rows)
        print(f"(dry run) would write {summary_path}")
        return EXIT_OK

    path = summary_module.write_summary(directory, rows)
    with open(path, encoding="utf-8") as handle:
        print()
        print(handle.read().rstrip())
    print(f"\nsummary written to {path}")
    return EXIT_OK


__all__ = ["ALL_AGENTS", "DATASETS", "MODES", "build_config", "build_parser",
           "main"]
