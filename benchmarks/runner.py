"""Run orchestration: expand the matrix, execute the commands, record everything.

A run is a directory. Everything needed to answer "what did this number come
from?" is inside it:

``manifest.json``
    git SHA (and whether the tree was dirty), harness/agent versions, the resolved
    config, one entry per trial, Docker image digests, host, timestamps.
``trials/<trial>/trial.json``
    the exact commands of that trial, and the agent's metadata (setting/options).
``trials/<trial>/logs/*.log``
    stdout/stderr of every command, verbatim.
``trials/<trial>/results.jsonl``
    the trial's outcomes, redacted.
``summary.json`` / ``summary.md``
    the cross-agent table.

Two behaviours are deliberate:

* **Dry run is the default in the CLI.** :meth:`Runner.plan` executes nothing --
  it prints the exact argv (no shell), the environment variable *names* it would
  set, and the Docker actions, so a config can be reviewed before it costs a day
  of GPU time.
* **Agent execution is refused outside a container** unless ``allow_host`` is
  passed. Benchmark agents run with ``--sudo``/``bypassPermissions`` and write to
  whatever they can reach; the guard is what keeps that inside a disposable
  environment (see :meth:`Runner.container_guard`).
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

from . import config as config_module
from . import docker as docker_module
from . import runtime as runtime_module
from .agents import get_adapter
from .results import read_results, summarise, write_results, write_summary
from .suites import get_suite


class Runner:
    """Executes (or plans) a :class:`~benchmarks.config.RunConfig`."""

    def __init__(self, run_config, dry_run=True, allow_host=False, resume=True,
                 concurrency=None, output=None):
        self.config = run_config
        self.dry_run = dry_run
        self.allow_host = allow_host
        self.resume = resume
        self.concurrency = concurrency or run_config.concurrency
        self.output = output or run_config.output_dir
        self.messages = []

    # -- logging ---------------------------------------------------------

    def say(self, message=""):
        """Print and remember a line (``plan`` output is also returned to callers)."""
        print(message)
        self.messages.append(message)

    # -- run identity ----------------------------------------------------

    def new_run_id(self, suites, agents, models):
        """``<utc timestamp>-<sha7>``: sortable, and it names the code that ran."""
        sha = git_sha() or "nogit"
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = uuid.uuid5(uuid.NAMESPACE_URL, "|".join(
            ["/".join(suites), "/".join(agents), "/".join(models)])).hex[:6]
        return f"{stamp}-{sha[:7]}-{digest}"

    def run_dir(self, run_id):
        return os.path.join(self.output, run_id)

    # -- safety ----------------------------------------------------------

    def container_guard(self, adapter):
        """Raise unless it is safe to launch ``adapter`` here.

        The check is deliberately conservative: either we are inside a container,
        or the caller passed ``--allow-host``. A benchmark agent granted shell
        access must not run in a directory someone cares about.
        """
        # Apptainer containers are containers: the agents run inside a task image
        # just as they do under Docker, so the guard is satisfied by the runtime
        # being apptainer even though there is no daemon to ask.
        if self.allow_host or inside_container() \
                or self.config.resolved_runtime() == "apptainer":
            return
        raise RuntimeError(
            f"refusing to launch agent {adapter.label!r} outside a container: "
            f"benchmark settings include unattended shell access "
            f"(`--sudo`, `bypassPermissions`). Run inside Docker, or pass "
            f"--allow-host if you really mean it")

    # -- planning --------------------------------------------------------

    def plan(self, trials):
        """Print the commands ``run`` would execute. Touches nothing."""
        run_id = "plan-" + self.new_run_id(
            sorted({t.suite.name for t in trials}),
            sorted({t.agent.label for t in trials}),
            sorted({t.model.name for t in trials}))
        self.say(f"# plan ({len(trials)} trial(s), dry run: nothing is executed)")
        self.say(f"# run directory would be: {self.run_dir(run_id)}")
        for trial in trials:
            adapter = get_adapter(trial.agent)
            suite = get_suite(trial.suite)
            trial_dir = self.run_dir(os.path.join(run_id, "trials", trial.run_id))
            self.say("")
            self.say(f"## {trial.run_id}")
            self.say(f"   suite={trial.suite.name} "
                     f"dataset={suite.dataset_label(trial.suite)}")
            self.say(f"   agent={adapter.label} model={trial.model.name} "
                     f"({trial.model.base_url}, {trial.model.api})")
            self.say(f"   runtime={trial.container_runtime or 'none installed'}"
                     f" (config: {self.config.runtime})")
            for note in suite.notes(trial.model) + adapter.notes(trial.model):
                self.say(f"   note: {note}")
            try:
                commands = self._with_config_env(
                    suite.commands(trial, os.getcwd(), trial_dir, adapter))
            except Exception as e:  # noqa: BLE001 - a plan must never crash
                self.say(f"   ! cannot expand yet: {e}")
                for line in suite.describe(trial, adapter, trial_dir):
                    self.say(f"   {line}")
                continue
            for command in commands:
                self.say(f"   [{command.label}]")
                self.say(f"   $ {command.rendered()}")
                if command.env:
                    names = ", ".join(sorted(command.env))
                    self.say(f"     env: {names}")
                for note in command.notes:
                    self.say(f"     note: {note}")
        self.say("")
        self.say("# nothing was executed; run the same command without --dry-run")
        return self.messages

    # -- execution -------------------------------------------------------

    def run(self, trials, run_id=None, csv_path=None):
        """Execute every trial, collecting outcomes; returns the manifest.

        ``csv_path`` also writes the summary table as CSV (by convention
        ``benchmarks/results/<run-id>.csv``, which is git-ignored).
        """
        run_id = run_id or self.new_run_id(
            sorted({t.suite.name for t in trials}),
            sorted({t.agent.label for t in trials}),
            sorted({t.model.name for t in trials}))
        run_dir = self.run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)
        manifest = self._manifest(run_id, trials)
        self._write_manifest(run_dir, manifest)
        self.say(f"run: {run_id} -> {run_dir}")

        outcomes_by_trial = {}
        failures_by_trial = {}
        for index, trial in enumerate(trials, 1):
            self.say(f"[{index}/{len(trials)}] {trial.run_id}")
            trial_dir = os.path.join(run_dir, "trials", trial.run_id)
            outcomes, failures = self._run_trial(trial, trial_dir)
            outcomes_by_trial[trial.run_id] = outcomes
            failures_by_trial[trial.run_id] = failures
            if not self.dry_run:
                write_results(os.path.join(trial_dir, "results.jsonl"), outcomes)

        manifest["finished_at"] = _now()
        manifest["trials"] = [
            _trial_record(t, self._trial_status(outcomes_by_trial, t),
                          failures_by_trial.get(t.run_id, []))
            for t in trials]
        self._write_manifest(run_dir, manifest)
        rows = summarise(manifest["trials"], outcomes_by_trial)
        json_path, md_path = write_summary(run_dir, rows, manifest,
                                          csv_path=csv_path)
        self.say("")
        self.say(f"summary: {json_path}")
        self.say(f"report:  {md_path}")
        if csv_path:
            self.say(f"csv:     {csv_path}")
            self.say(f"analyse: python -m benchmarks.analyze {csv_path}")
        return manifest

    def _run_trial(self, trial, trial_dir):
        """Run one trial's commands in order; collect its outcomes."""
        adapter = get_adapter(trial.agent)
        suite = get_suite(trial.suite)
        os.makedirs(os.path.join(trial_dir, "logs"), exist_ok=True)
        done_marker = os.path.join(trial_dir, "DONE")

        if self.resume and os.path.isfile(done_marker):
            self.say("   (already done; keeping recorded results)")
            recorded = _read_json(os.path.join(trial_dir, "trial.json"), {})
            return (read_results(os.path.join(trial_dir, "results.jsonl")),
                    recorded.get("failed_commands", []))

        if not self.dry_run:
            self.container_guard(adapter)

        try:
            commands = self._with_config_env(
                suite.commands(trial, os.getcwd(), trial_dir, adapter))
        except Exception as e:  # noqa: BLE001 - one broken suite must not hide the rest
            # A trial that cannot even expand (missing instance list, no runtime,
            # unresolvable agent) is a *result*: it is recorded as a failed command
            # so the report shows "cmd fail: 1, n: 0" instead of silently omitting
            # the cell, and the message is kept in the trial directory.
            if not self.dry_run:
                _write_json(os.path.join(trial_dir, "trial.json"), {
                    "run_id": trial.run_id,
                    "suite": trial.suite.to_dict(),
                    "agent": adapter.metadata(),
                    "model": trial.model.to_dict(),
                    "failed_commands": [{"label": "expand trial",
                                         "status": "error", "log": None}],
                    "error": str(e),
                })
                with open(os.path.join(trial_dir, "DONE"), "w",
                          encoding="utf-8") as handle:
                    handle.write(_now() + "\n")
            self.say(f"   ! cannot expand this trial: {e}")
            return [], [{"label": f"expand trial: {e}", "status": "error",
                         "log": None}]

        _write_json(os.path.join(trial_dir, "trial.json"), {
            "run_id": trial.run_id,
            "suite": trial.suite.to_dict(),
            "dataset": suite.dataset_label(trial.suite),
            "agent": adapter.metadata(),
            "model": trial.model.to_dict(),
            "limit": trial.limit,
            "timeouts": trial.timeouts,
            "commands": [_command_record(c) for c in commands],
        })

        failures = []
        for number, command in enumerate(commands, 1):
            log_path = os.path.join(trial_dir, "logs",
                                    f"{number:02d}-{_safe(command.label)}.log")
            if self.dry_run:
                self.say(f"   would run: {command.rendered()}")
                continue
            started = time.monotonic()
            status = _execute(command, log_path, self.say)
            self.say(f"   {command.label}: {status} "
                     f"({time.monotonic() - started:.1f}s)")
            if status != "ok":
                # A command that did not finish cleanly is the difference between
                # "the agent scored 0" and "the harness never ran"; the report
                # must be able to tell those apart.
                failures.append({"label": command.label, "status": status,
                                 "log": os.path.relpath(log_path, trial_dir)})

        outcomes = []
        if not self.dry_run:
            try:
                outcomes = suite.collect(trial_dir, trial.run_id)
            except Exception as e:  # noqa: BLE001 - a missing report is a result
                self.say(f"   ! could not collect results: {e}")
                outcomes = []
            record = _read_json(os.path.join(trial_dir, "trial.json"), {})
            record["failed_commands"] = failures
            _write_json(os.path.join(trial_dir, "trial.json"), record)
            with open(done_marker, "w", encoding="utf-8") as handle:
                handle.write(_now() + "\n")
        return outcomes, failures

    def _with_config_env(self, commands):
        """Merge the config's ``env`` block into every command's environment.

        The block is how a run points all agents at one endpoint
        (``ANTHROPIC_BASE_URL`` and friends) without repeating it per agent; a
        command's own value wins, because the adapter knows its agent's needs
        better than the config does.
        """
        if not self.config.env:
            return commands
        for command in commands:
            merged = dict(self.config.env)
            merged.update(command.env or {})
            command.env = merged
        return commands

    # -- manifest --------------------------------------------------------

    def _manifest(self, run_id, trials):
        """Everything needed to reproduce the run, minus secrets."""
        digests = {}
        for image in sorted({t.suite.options.get("image") for t in trials
                             if t.suite.options and t.suite.options.get("image")}):
            digests[image] = docker_module.image_digest(image)
        return {
            "run_id": run_id,
            "started_at": _now(),
            "finished_at": None,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "benchmarks_version": _package_version(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "docker": docker_module.docker_version(),
            "runtime": (self.config.resolved_runtime()
                        or self.config.runtime),
            "runtime_version": runtime_module.version(
                self.config.resolved_runtime() or ""),
            "image_digests": digests,
            "config": self.config.to_dict(),
            "config_source": self.config.source,
            "suites": sorted({t.suite.name for t in trials}),
            "datasets": {t.suite.name: _suite_dataset(t.suite) for t in trials},
            "trials": [_trial_record(t, "pending") for t in trials],
        }

    def _write_manifest(self, run_dir, manifest):
        _write_json(os.path.join(run_dir, "manifest.json"), manifest)

    def _trial_status(self, outcomes_by_trial, trial):
        if trial.run_id not in outcomes_by_trial:
            return "pending"
        return "done" if outcomes_by_trial[trial.run_id] else "no-results"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _suite_dataset(suite_spec):
    return {"dataset": suite_spec.dataset, "version": suite_spec.version,
            "split": suite_spec.split}


def _trial_record(trial, status, failures=None):
    return {
        "run_id": trial.run_id,
        "suite": trial.suite.name,
        "dataset": trial.suite.dataset,
        "agent": trial.agent.name,
        "setting": trial.agent.setting,
        "model": trial.model.name,
        "model_id": trial.model.model,
        "limit": trial.limit,
        "status": status,
        "failed_commands": [f["label"] for f in (failures or [])],
    }


def _command_record(command):
    return {
        "label": command.label,
        "argv": list(command.argv),
        "env_names": sorted(command.env),
        "cwd": command.cwd,
        "timeout": command.timeout,
        "artifacts": list(command.artifacts),
        "notes": list(command.notes),
    }


def _execute(command, log_path, say):
    """Run one planned command, teeing its output into ``log_path``.

    The environment is the parent's plus ``command.env`` with ``{env:NAME}``
    placeholders resolved from the parent environment, so a key is passed by name
    and never written into a command line or a log.
    """
    from .results import redact

    env = dict(os.environ)
    for key, value in command.env.items():
        env[key] = expand_env(str(value), os.environ)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {redact(command.rendered())}\n")
        log.flush()
        try:
            result = subprocess.run(
                command.argv, cwd=command.cwd, env=env, timeout=command.timeout,
                input=command.stdin, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            log.write(f"\n[timeout after {command.timeout}s]\n")
            return "timeout"
        except (OSError, ValueError) as e:
            log.write(f"\n[could not start: {e}]\n")
            return "error"
        # Redacted on the way in: agent output routinely echoes its own
        # environment, and a log is the artifact most likely to be shared.
        log.write(redact(result.stdout or ""))
        log.write(redact(result.stderr or ""))
        log.write(f"\n[exit {result.returncode} in "
                  f"{time.monotonic() - started:.1f}s]\n")
    return "ok" if result.returncode == 0 else f"exit {result.returncode}"


def expand_env(value, environ=None):
    """Resolve ``{env:NAME}`` placeholders; an unset variable expands to ``''``.

    Expanding to ``''`` rather than raising keeps the failure where it belongs: the
    agent reports "no API key", which is a result, instead of the runner dying
    with a ``KeyError`` halfway through a matrix.
    """
    environ = environ if environ is not None else os.environ
    text = str(value)
    while "{env:" in text:
        start = text.index("{env:")
        end = text.find("}", start)
        if end == -1:
            break
        name = text[start + 5:end]
        text = text[:start] + environ.get(name, "") + text[end + 1:]
    return text


def inside_container():
    """Best-effort "am I containerised?" check (Docker, Podman, Kubernetes)."""
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8", errors="replace") as handle:
            cgroup = handle.read()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd"))


def git_sha():
    """The repository's HEAD SHA, or ``None`` outside a git checkout."""
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty():
    """Whether the working tree has uncommitted changes (``None`` if unknown)."""
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _package_version():
    try:
        from . import __version__

        return __version__
    except ImportError:  # pragma: no cover - never in practice
        return "unknown"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _safe(name):
    return config_module._slug(name)[:60]


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
