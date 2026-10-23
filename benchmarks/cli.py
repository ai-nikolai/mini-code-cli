"""Command line: ``python -m benchmarks <command>``.

    list       what is installed / configured (suites, agents, settings, models)
    validate   check a config and the preconditions it implies
    prepare    materialise what a run needs (e.g. the SWE-bench instance list)
    plan       dry run: print the exact commands, execute nothing (the default)
    run        execute the matrix
    report     rebuild a run's summary from its results
    models     print the vLLM / LiteLLM commands for a configured model

``plan`` is the command to reach for first, and ``check``-style failures are
reported as messages plus exit code 2 rather than tracebacks: this is a tool a
human drives against a GPU host, and a stack trace is not an error message.
"""
import argparse
import json
import os
import shutil
import sys

from . import __version__, docker as docker_module, models as models_module
from .agents import ADAPTERS, AgentError, get_adapter
from .config import ConfigError, load_config, parse_config, validate_config
from .results import (default_csv_path, read_results, summarise, to_csv,
                      write_summary)
from .runner import Runner, _execute, _write_json
from .suites import ALIASES, SUITES, SuiteError, get_suite

EXIT_OK = 0
EXIT_PROBLEM = 2


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Benchmark harness for coding agents (Terminal-Bench, SWE-bench).")
    parser.add_argument("--version", action="version",
                        version=f"benchmarks {__version__}")
    parser.add_argument("-c", "--config", default=None,
                        help="run config (JSON, or TOML on Python 3.11+)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list suites, agents, settings and models")

    validate = sub.add_parser("validate", help="check a config and preconditions")
    validate.add_argument("--allow-host", action="store_true",
                          help="do not require a container")

    prepare = sub.add_parser("prepare", help="materialise dataset/instance lists")
    plan = sub.add_parser("plan", help="dry run: print the commands (default)")

    run = sub.add_parser("run", help="execute the matrix")
    run.add_argument("--run-id", default=None,
                     help="id for the run directory (default: timestamp + sha)")
    run.add_argument("--allow-host", action="store_true",
                     help="run the agents on this host instead of a container")
    run.add_argument("--no-resume", action="store_true",
                     help="redo trials that already finished")
    run.add_argument("--dry-run", action="store_true",
                     help="print the commands instead of executing them")
    run.add_argument("--csv", nargs="?", const="auto", default=None,
                     metavar="PATH",
                     help="also write the summary as CSV "
                          "(default: benchmarks/results/<run-id>.csv)")

    analyze = sub.add_parser(
        "analyze", help="summarise result CSVs (see also: python -m benchmarks.analyze)",
        add_help=False)
    analyze.add_argument("analyze_args", nargs=argparse.REMAINDER,
                         help="paths and flags, forwarded to benchmarks.analyze")

    report = sub.add_parser("report", help="rebuild a run's summary")
    report.add_argument("--run", required=True, help="run directory")
    report.add_argument("--csv", nargs="?", const="auto", default=None,
                        metavar="PATH",
                        help="also write the summary as CSV "
                             "(default: benchmarks/results/<run-id>.csv)")

    show_models = sub.add_parser("models", help="print serving commands")
    show_models.add_argument("--name", default=None,
                             help="model name from the config (default: all)")

    # `-c/--config` is accepted before *or* after the subcommand: both spellings
    # are common, and failing on the one a user typed is pure friction.
    for target in (validate, prepare, plan, run, show_models):
        target.add_argument("-c", "--config", dest="subconfig", default=None,
                            help="run config (may also be given before the command)")

    for target in (prepare, plan, run):
        target.add_argument("--suite", action="append", default=None,
                            help="only this suite (repeatable)")
        target.add_argument("--agent", action="append", default=None,
                            help="only this agent name or 'agent:setting'")
        target.add_argument("--model", action="append", default=None,
                            help="only this model name")
        target.add_argument("--limit", type=int, default=None,
                            help="cap instances/tasks per trial (smoke runs)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "subconfig", None):
        args.config = args.subconfig
    try:
        return _dispatch(args, parser)
    except (ConfigError, AgentError, SuiteError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_PROBLEM


def _dispatch(args, parser):
    if args.command == "list":
        return _cmd_list()
    if args.command == "report":
        return _cmd_report(args.run, args.csv)
    if args.command == "analyze":
        from .analyze import main as analyze_main

        return analyze_main(args.analyze_args)

    config = _require_config(args)
    if args.command == "models":
        return _cmd_models(config, args.name)
    if args.command == "validate":
        return _cmd_validate(config, args.allow_host)

    trials = _select_trials(config, args)
    if args.command == "prepare":
        return _cmd_prepare(config, trials)
    if args.command == "run":
        _reuse_prepared_inputs(config, trials)
        if args.dry_run:
            Runner(config, dry_run=True).plan(trials)
            return EXIT_OK
        runner = Runner(config, dry_run=False, allow_host=args.allow_host,
                        resume=not args.no_resume)
        csv_path = _csv_target(args.csv, args.run_id)
        runner.run(trials, run_id=args.run_id, csv_path=csv_path)
        return EXIT_OK
    runner = Runner(config, dry_run=True)
    runner.plan(trials)
    return EXIT_OK


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _cmd_list():
    print(f"benchmarks {__version__}")
    print("\nsuites:")
    for name, suite_class in sorted(SUITES.items()):
        print(f"  {name:<16} {getattr(suite_class, 'install_hint', '')}")
    if ALIASES:
        print("  aliases: " + ", ".join(sorted(ALIASES)))
    print("\nagents:")
    for name, adapter_class in sorted(ADAPTERS.items()):
        settings = ", ".join(sorted(adapter_class.settings))
        print(f"  {name:<16} speaks={adapter_class.speaks:<9} settings: {settings}")
        if adapter_class.install_hint:
            print(f"  {'':<16} {adapter_class.install_hint}")
    print("\nrun a config with: python -m benchmarks plan --config <file>")
    return EXIT_OK


def _cmd_validate(config, allow_host):
    problems = validate_config(config, check_env=True)
    for suite_spec in config.suites:
        suite = get_suite(suite_spec)
        for model in _models_for(config, None):
            problems += suite.preflight(model, require_binary=True)
    for agent_spec in config.agents:
        adapter = get_adapter(agent_spec)
        for model in _models_for(config, agent_spec):
            problems += adapter.preflight(model, require_binary=True)
    if not allow_host and docker_module.docker_version() is None:
        problems.append("docker is not available on this host (pass --allow-host "
                        "only if the agents really should run here)")

    for problem in dict.fromkeys(problems):
        print(f"problem: {problem}")
    print(f"validate: {len(config.suites)} suite(s), {len(config.agents)} agent "
          f"setting(s), {len(config.models)} model(s), "
          f"{len(dict.fromkeys(problems))} problem(s)")
    return EXIT_OK if not problems else EXIT_PROBLEM


def _reuse_prepared_inputs(config, trials):
    """Copy inputs from an earlier ``prepare`` into the run's own directory.

    ``prepare`` materialises per-trial datasets under ``_prepare/<trial>/`` while
    a run reads them from ``<run-dir>/`` -- so the documented
    ``prepare`` then ``run`` sequence used to fail with "no instance list".
    """
    for trial in trials:
        prepared = os.path.join(config.output_dir, "_prepare", trial.run_id,
                                "instances.jsonl")
        target = os.path.join(config.output_dir, trial.run_id, "instances.jsonl")
        if os.path.isfile(prepared) and not os.path.isfile(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(prepared, target)
            print(f"reusing prepared instances: {prepared} -> {target}")


def _cmd_prepare(config, trials):
    """Materialise per-trial inputs (currently: the SWE-bench instance list)."""
    prepared = 0
    for trial in trials:
        suite = get_suite(trial.suite)
        prepare = getattr(suite, "prepare_command", None)
        if prepare is None:
            print(f"{trial.run_id}: nothing to prepare for suite "
                  f"{trial.suite.name!r}")
            continue
        run_dir = os.path.join(config.output_dir, "_prepare", trial.run_id)
        os.makedirs(run_dir, exist_ok=True)
        command = prepare(trial, run_dir)
        log_path = os.path.join(run_dir, "prepare.log")
        status = _execute(command, log_path, print)
        print(f"{trial.run_id}: {status} (log: {log_path})")
        if status != "ok":
            return EXIT_PROBLEM
        prepared += 1
    print(f"prepare: {prepared} trial(s) ready")
    return EXIT_OK


def _cmd_models(config, name=None):
    for model in config.models:
        if name and model.name != name:
            continue
        print(f"# {model.name}: {model.model} ({model.api}) at {model.base_url}")
        if model.api == "openai":
            print("$ " + " ".join(models_module.vllm_argv(model.model)))
            print("  # then, for Claude Code only, the Anthropic-compatible proxy:")
            print("$ " + " ".join(models_module.litellm_argv("<config>.yaml")))
        else:
            print("  # Claude Code talks to this endpoint directly; it must "
                  "expose /v1/messages")
        print()
    return EXIT_OK


def _cmd_report(run_dir, csv_target=None):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"error: no manifest.json in {run_dir}", file=sys.stderr)
        return EXIT_PROBLEM
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    outcomes = {}
    trials = manifest.get("trials") or []
    for trial in trials:
        path = os.path.join(run_dir, "trials", trial["run_id"], "results.jsonl")
        outcomes[trial["run_id"]] = read_results(path)
    rows = summarise(trials, outcomes)
    csv_path = _csv_target(csv_target, manifest.get("run_id")
                           or os.path.basename(os.path.abspath(run_dir)))
    json_path, md_path = write_summary(run_dir, rows, manifest,
                                       csv_path=csv_path)
    print(f"report: {json_path}")
    print(f"report: {md_path}")
    if csv_path:
        to_csv(rows, csv_path)
        print(f"csv:    {csv_path}")
        print(f"analyse: python -m benchmarks.analyze {csv_path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _csv_target(value, run_id):
    """Resolve ``--csv``: ``None`` -> no CSV, ``"auto"`` -> the default path."""
    if value is None:
        return None
    if value == "auto":
        return default_csv_path(run_id)
    return value


def _models_for(config, agent_spec):
    """The models a (suite, agent) pair will actually be combined with."""
    if agent_spec is None or not agent_spec.models:
        return list(config.models)
    return [m for m in config.models if m.name in agent_spec.models]


def _require_config(args):
    if not args.config:
        raise ConfigError("this command needs --config (see "
                          "benchmarks/configs/default.json)")
    return load_config(args.config)


def _select_trials(config, args):
    trials = config.trials(suites=args.suite, agents=args.agent,
                           models=args.model, limit=args.limit)
    if not trials:
        raise ConfigError("the filters selected no trials; check --suite/--agent/"
                          "--model against `python -m benchmarks list`")
    return trials


def load_or_default(path=None, data=None):
    """Convenience for tests and notebooks: a config from a path or a mapping."""
    if data is not None:
        return parse_config(data, source="<memory>")
    return load_config(path)


def _write_run_config(run_dir, config):
    return _write_json(os.path.join(run_dir, "config.resolved.json"),
                       config.to_dict())


if __name__ == "__main__":  # pragma: no cover - exercised via __main__
    sys.exit(main())
