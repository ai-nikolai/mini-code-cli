"""Terminal-Bench suite: wraps the official harness, either generation of it.

Terminal-Bench ships **two** harnesses, and which one is correct depends on the
dataset version:

``harbor`` (Terminal-Bench 2.x, the current one)
    ``harbor run -d terminal-bench@2.1 -a claude-code -m anthropic/...``
    Harbor is the official framework for Terminal-Bench 2.0/2.1 (it is by the
    Terminal-Bench authors) and it also ships first-class agents for Claude Code
    and Pi, which is what makes those two comparable on the same task set.

``tb`` (Terminal-Bench 1.x / ``terminal-bench-core``)
    ``tb run --dataset terminal-bench-core==0.1.1 --agent-import-path ...``
    Pinned against ``terminal-bench`` 0.2.18.

Both are *wrapped*, never reimplemented: the harness builds the task container,
drives the agent inside it and runs the task's own tests. This adapter decides
what to pass and reads the result back.

Result files differ too, so :meth:`collect` understands both:

* harbor: ``<jobs-dir>/<job-name>/result.json`` ->
  ``{"trial_results": [{"task_name", "verifier_result": {"rewards": {...}},
  "exception_info"}]}``
* tb: ``<output-path>/<run-id>/results.json`` ->
  ``{"results": [{"task_id", "is_resolved", "failure_mode"}]}``
"""
import json
import os
import re

from .. import forks as forks_module
from .. import runtime as runtime_module
from .base import PlannedCommand, Suite, SuiteError, TrialOutcome


class TerminalBenchSuite(Suite):
    name = "terminal-bench"
    install_hint = ("pip install harbor   (Terminal-Bench 2.x)  |  "
                    "uv tool install terminal-bench   (1.x, the 'tb' CLI)")
    results_filename = "result.json"

    # -- which harness ---------------------------------------------------

    def harness(self, spec=None):
        """``"harbor"`` (default, 2.x) or ``"tb"`` (1.x/core).

        An explicitly passed spec wins; otherwise the spec this suite was built
        for decides. That matters for :meth:`dataset_label`, which is also called
        with a bare spec from the report side.
        """
        for candidate in (spec, getattr(self, "spec", None)):
            options = (candidate.options if candidate is not None else None) or {}
            if "harness" in options:
                return str(options["harness"])
        return "harbor"

    def binary(self, spec=None):
        """The harness executable.

        ``options.harbor_bin`` pins it explicitly, which is what a cluster needs:
        harbor usually lives in its own environment (it is heavy, and its
        dependencies should not leak into the benchmark's), so it is often not on
        ``PATH``. The fork check asks *this* script's interpreter what it imports,
        so the answer has to be the script that will actually run.
        """
        from ..agents.base import which

        spec = spec if spec is not None else getattr(self, "spec", None)
        configured = ((getattr(spec, "options", None) or {})
                      .get("harbor_bin"))
        if configured:
            return os.path.expanduser(str(configured))
        if self.harness(spec) == "harbor":
            return which("harbor") or which("hr")
        return which("tb") or which("terminal-bench")

    def local_dataset_dir(self, spec):
        """A local task collection given as ``options.dataset_dir``, or ``None``.

        Terminal-Bench 2.1 exists as a checkout of tasks (each with its own
        ``Dockerfile``) before the registry publishes ``terminal-bench@2.1``.
        Harbor runs *any* directory of tasks with ``-p``, so pointing at the
        checkout is the whole story: no registry copy, no `-d`, no staged
        dataset -- harbor builds each task's ``.sif`` from its Dockerfile.
        """
        raw = (getattr(spec, "options", None) or {}).get("dataset_dir")
        if not raw:
            return None
        return os.path.abspath(os.path.expanduser(str(raw)))

    def dataset_label(self, spec):
        """``dataset@version`` for harbor, ``dataset==version`` for tb."""
        local = self.local_dataset_dir(spec)
        if local:
            return f"{os.path.basename(local)} (local)"
        if not spec.version:
            return str(spec.dataset or "")
        joiner = "@" if self.harness(spec) == "harbor" else "=="
        return f"{spec.dataset}{joiner}{spec.version}"

    # -- preflight -------------------------------------------------------

    def preflight(self, model, require_binary=True):
        spec = getattr(self, "spec", None)
        problems = []
        if require_binary and not self.binary(spec):
            name = "harbor" if self.harness(spec) == "harbor" else "tb"
            problems.append(f"suite 'terminal-bench': the {name!r} CLI is not on "
                            f"PATH ({self.install_hint})")
        chosen = runtime_module.resolve(self._runtime_hint())
        if require_binary and chosen is None:
            problems.append(
                "suite 'terminal-bench': no container runtime is usable "
                f"({' or '.join(runtime_module.RUNTIMES)} not on PATH); every task "
                "runs in a container")
        elif require_binary and chosen == "apptainer" and \
                self.harness(spec) == "harbor":
            problems.append(
                "suite 'terminal-bench': harbor drives task containers through "
                "Docker; its Apptainer backend (`--env singularity`) converts the "
                "images but then needs /usr/bin/python3 inside them to bootstrap "
                "its RPC server, which Terminal-Bench images do not ship, so this "
                "suite needs a Docker host (see benchmarks/README.md)")
        return problems

    def _runtime_hint(self):
        return getattr(self, "runtime", None)

    def notes(self, model):
        spec = getattr(self, "spec", None)
        notes = [
            "tasks run inside their own containers: the agent must be able to "
            "reach the model endpoint from there (bind vLLM to 0.0.0.0 and use "
            "a host address the container can route to)",
        ]
        binary = self.binary(spec)
        if not binary:
            name = "harbor" if self.harness(spec) == "harbor" else "tb"
            notes.append(f"{name!r} is not installed here; `run` will fail, "
                         f"`plan` works")
        chosen = runtime_module.resolve(self._runtime_hint())
        if chosen == "apptainer":
            notes.append("container runtime: apptainer; harbor reaches it through "
                         "`--env singularity`, which only works for tasks that "
                         "declare docker_image in task.toml")
        elif chosen == "docker":
            notes.append("container runtime: docker")
        else:
            notes.append("no container runtime found (docker/apptainer)")
        if self.harness(spec) == "harbor":
            notes.append("Terminal-Bench 2.x is executed by harbor; `harbor "
                         "datasets list` shows the dataset@version strings")
            info = forks_module.state("harbor")
            if forks_module.cli_uses_fork(binary):
                # The CLI runs in the environment that owns the script, which is
                # not necessarily this interpreter -- so ask that one.
                notes.append(
                    f"using our harbor checkout ({info['branch']} "
                    f"{info['sha']}): a task that ships only a Dockerfile has "
                    f"its .sif built on demand, so no staging is needed")
            elif info["present"] and binary:
                notes.append(
                    f"{binary} imports harbor from "
                    f"{forks_module.cli_origin(binary) or 'an unknown location'}, "
                    f"not from {info['path']}: the unpatched release has no "
                    f"Dockerfile-only support, so the staging step runs. "
                    f"`pip install -e harbor --no-deps` in that environment to "
                    f"switch")
        return notes

    @staticmethod
    def _docker_available():
        import shutil

        return shutil.which("docker") is not None

    # -- preparation -----------------------------------------------------

    def apptainer_dataset_dir(self, run_dir):
        """The staged, SIF-pointing dataset: ``<run>/apptainer/<dataset-slug>``."""
        return os.path.join(run_dir, "apptainer",
                            _dataset_slug(self._spec_or_none()))

    def prepare_command(self, trial, run_dir):
        """Build the derived Apptainer images a Terminal-Bench run needs.

        Only for ``runtime=apptainer``: on Docker, harbor converts/builds images
        itself and there is nothing to prepare. ``None`` means "nothing to do",
        which is what ``benchmarks prepare`` reports.
        """
        if (trial.container_runtime or "docker") != "apptainer":
            return None
        if self.local_dataset_dir(trial.suite):
            # The tasks are a local checkout and harbor builds each .sif from the
            # task's own Dockerfile, so there is nothing to stage or download.
            return None
        spec = trial.suite
        options = spec.options or {}
        argv = [
            _python(), "-m", "benchmarks.suites.tb_apptainer", "prepare",
            "--out", self.apptainer_dataset_dir(run_dir),
            "--sif-cache", str(options.get("singularity_image_cache_dir")
                               or _default_sif_cache()),
            "--jobs", str(int(options.get("sif_build_jobs", 2))),
        ]
        if options.get("exported_tasks_dir"):
            argv += ["--tasks-dir", str(options["exported_tasks_dir"])]
        else:
            # `harbor download <dataset> --export` is the only way to obtain the
            # task files; the export is cached under a git-ignored directory.
            argv += ["--dataset", self.dataset_label(spec),
                     "--download-dir", os.path.join(_cache_root(), "tb")]
        if trial.limit:
            argv += ["--limit", str(int(trial.limit))]
        for task_id in spec.task_ids:
            argv += ["--task-id", str(task_id)]
        if options.get("bake_mini_code", True):
            argv += ["--mini-code", _mini_code_dir()]
        return PlannedCommand(label="terminal-bench: derive Apptainer images",
                              argv=argv,
                              timeout=options.get("sif_build_timeout_sec"))

    def _spec_or_none(self):
        return getattr(self, "spec", None)

    # -- commands --------------------------------------------------------

    def commands(self, trial, workspace, run_dir, adapter):
        if self.harness(trial.suite) == "harbor":
            commands = []
            if self._needs_apptainer_prepare(trial, run_dir):
                # Deriving the images is part of the run, not a separate chore:
                # `run` on a Docker-free machine has to work in one command.
                commands.append(self.prepare_command(trial, run_dir))
            commands.append(self._harbor_command(trial, workspace, run_dir,
                                                 adapter))
            return commands
        return [self._tb_command(trial, workspace, run_dir, adapter)]

    def _harbor_has_dockerfile_support(self, spec):
        """Whether the harbor that will run supports Dockerfile-only tasks.

        Asked of the script's own interpreter: harbor is usually in another
        environment, so "can *this* process import harbor?" is the wrong question.
        """
        return forks_module.cli_uses_fork(self.binary(spec))

    def _needs_apptainer_prepare(self, trial, run_dir):
        """True when this trial runs under Apptainer and has no staged dataset."""
        if (trial.container_runtime or "docker") != "apptainer":
            return False
        if self.local_dataset_dir(trial.suite):
            # The tasks are already a local directory; nothing to stage.
            return False
        if self._harbor_has_dockerfile_support(trial.suite):
            # The patched harbor builds a task's .sif from its Dockerfile on
            # demand, so there is nothing to stage and no container alias to
            # maintain: `harbor run -d terminal-bench@2.0` is the whole story.
            return False
        staged = self.apptainer_dataset_dir(run_dir)
        if not os.path.isdir(staged):
            return True
        return not any(os.path.isfile(os.path.join(staged, name, "task.toml"))
                       for name in os.listdir(staged))

    def _harbor_dataset_args(self, trial, run_dir):
        """``-d dataset@version`` or, under Apptainer, ``-p <staged dataset>``.

        Harbor's registry copy is what it would download on Docker. Under
        Apptainer each task needs its derived SIF, and ``task.toml`` in the staged
        copy points at it -- so the run targets the staged directory instead.
        """
        spec = trial.suite
        local = self.local_dataset_dir(spec)
        if local:
            # A local collection of tasks: harbor reads <dir>/<task>/task.toml
            # and builds each image from the task's own Dockerfile.
            return ["-p", local]
        if (trial.container_runtime or "docker") != "apptainer":
            return ["-d", self.dataset_label(spec)]
        if self._harbor_has_dockerfile_support(spec):
            return ["-d", self.dataset_label(spec)]
        return ["-p", self.apptainer_dataset_dir(run_dir)]

    def _harbor_command(self, trial, workspace, run_dir, adapter):
        """``harbor run ...`` -- one job per trial; harbor parallelises inside."""
        spec = trial.suite
        model = trial.model
        options = spec.options or {}
        jobs_dir = self.ensure_dir(os.path.join(run_dir, "harbor"))
        job_name = trial.run_id

        argv = [self.binary(spec) or "harbor", "run",
                *self._harbor_dataset_args(trial, run_dir),
                "--jobs-dir", str(jobs_dir),
                "--job-name", job_name]

        selection = adapter.harbor_selection(model)
        if selection.get("agent"):
            argv += ["-a", str(selection["agent"])]
        else:
            # harbor resolves the import path *in its own interpreter*, so a
            # module from this repository only imports if the repository is on
            # PYTHONPATH (see the env below). `-a` is the non-deprecated spelling
            # and accepts an import path just as well.
            argv += ["-a", str(selection["import_path"])]
        kwargs = dict(selection.get("kwargs") or {})
        # An agent whose code lives in this repository is not in the task image
        # (harbor's TB path mounts nothing from the host), so hand its package
        # over for upload -- otherwise every task dies with
        # "No module named mini_code_cli", which reads like a model failure.
        python_path = adapter.harbor_python_path()
        if python_path and not selection.get("agent"):
            kwargs.setdefault("python_path", python_path)
        for key, value in kwargs.items():
            rendered = _kwarg(value)
            if rendered == "":
                continue
            argv += ["--agent-kwarg", f"{key}={rendered}"]

        # The provider prefix must match the wire protocol the *agent* speaks:
        # `anthropic/...` makes harbor point Claude Code at an Anthropic endpoint,
        # `openai/...` is what every other agent here consumes.
        provider = options.get("provider") or (
            "anthropic" if model.api == "anthropic" else "openai")
        argv += ["-m", f"{provider}/{model.model}"]
        argv += ["--n-concurrent", str(int(options.get(
            "n_concurrent", trial.limit or 1)))]
        if trial.limit:
            argv += ["--n-tasks", str(int(trial.limit))]
        if options.get("max_retries"):
            argv += ["--max-retries", str(int(options["max_retries"]))]
        if options.get("cpus"):
            argv += ["--cpus", str(options["cpus"])]
        if options.get("memory"):
            argv += ["--memory", str(options["memory"])]
        container = trial.container_runtime
        env_backend = options.get("env_backend")
        if env_backend is None and container == "apptainer":
            # harbor's Apptainer backend is spelled 'singularity'.
            env_backend = "singularity"
        if env_backend:
            argv += ["--env", str(env_backend)]
        for kwarg in options.get("environment_kwargs") or ():
            argv += ["--environment-kwarg", str(kwarg)]
        # A 2.1 task budgets 600 s for its environment, which is the image build
        # on Docker but the *start* on Apptainer (SIF extract + the agent runtime
        # going in). That is not enough on a network filesystem: the trial dies
        # with "Environment start timed out after 600.0 seconds" even though the
        # server came up. The multiplier raises the budget harbor derives from the
        # task, rather than overriding the task's own numbers.
        if options.get("environment_build_timeout_multiplier"):
            argv += ["--environment-build-timeout-multiplier",
                     str(float(options["environment_build_timeout_multiplier"]))]
        if options.get("agent_setup_timeout_multiplier"):
            argv += ["--agent-setup-timeout-multiplier",
                     str(float(options["agent_setup_timeout_multiplier"]))]
        if env_backend == "singularity" and not options.get("environment_kwargs"):
            cache = options.get("singularity_image_cache_dir") or _default_sif_cache()
            argv += ["--environment-kwarg",
                     f"singularity_image_cache_dir={cache}"]
            # Harbor defaults to no_mount=home,tmp,bind-paths, and 'bind-paths'
            # also suppresses the /etc/resolv.conf bind -> the container has no
            # DNS, so apt/curl (the verifier!) fail with "could not resolve".
            argv += ["--environment-kwarg",
                     f"singularity_no_mount="
                     f"{options.get('singularity_no_mount', 'home,tmp')}"]
        # Never wait for a human: a benchmark job starts unattended.
        argv += ["--yes"]

        env = dict(adapter.env(model))
        if not selection.get("agent"):
            env.setdefault("PYTHONPATH", _repo_root())
        return PlannedCommand(
            label=f"harbor run {job_name}",
            argv=argv,
            env=env,
            cwd=workspace,
            timeout=options.get("harness_timeout_sec"),
            artifacts=(os.path.join(jobs_dir, job_name, "result.json"),
                       os.path.join(jobs_dir, job_name, "config.json")),
            notes=[f"dataset: {self.dataset_label(spec)}",
                   f"agent: {trial.agent.label}",
                   "harbor writes <jobs-dir>/<job-name>/result.json"],
        )

    def _tb_command(self, trial, workspace, run_dir, adapter):
        """``tb run ...`` (Terminal-Bench 1.x / core datasets)."""
        spec = trial.suite
        model = trial.model
        runs_dir = self.ensure_dir(os.path.join(run_dir, "tb"))
        run_id = trial.run_id
        options = spec.options or {}

        argv = [self.binary(spec) or "tb", "run", "--run-id", run_id,
                "--output-path", str(runs_dir)]

        if options.get("use_legacy_dataset_flags"):
            if spec.dataset:
                argv += ["--dataset-name", spec.dataset]
            if spec.version:
                argv += ["--dataset-version", spec.version]
        else:
            argv += ["--dataset", self.dataset_label(spec)]
        if options.get("dataset_path"):
            argv += ["--dataset-path", str(options["dataset_path"])]

        selection = adapter.harness_selection(model)
        if selection.get("agent"):
            argv += ["--agent", str(selection["agent"])]
        else:
            argv += ["--agent-import-path", str(selection["import_path"])]
        for key, value in (selection.get("kwargs") or {}).items():
            rendered = _kwarg(value)
            if rendered == "":
                # An empty kwarg would only be noise in the printed command, and
                # tb would pass an empty string where the agent expects a default.
                continue
            argv += ["--agent-kwarg", f"{key}={rendered}"]

        provider = options.get("provider") or (
            "anthropic" if model.api == "anthropic" else "openai")
        argv += ["--model", f"{provider}/{model.model}"]
        argv += ["--n-concurrent-trials", str(int(options.get(
            "n_concurrent_trials", trial.limit or 1)))]
        argv += ["--n-attempts", str(int(options.get("n_attempts", 1)))]
        argv += ["--global-agent-timeout-sec", str(int(
            options.get("agent_timeout_sec", _timeout(trial, "agent_seconds"))))]
        argv += ["--global-test-timeout-sec", str(int(
            options.get("test_timeout_sec", _timeout(trial, "test_seconds"))))]
        if trial.limit:
            argv += ["--n-tasks", str(int(trial.limit))]
        for task_id in spec.task_ids:
            argv += ["--task-id", task_id]
        for excluded in options.get("exclude_task_ids", ()):
            argv += ["--exclude-task-id", str(excluded)]
        if options.get("no_rebuild"):
            argv += ["--no-rebuild"]
        if options.get("cleanup"):
            argv += ["--cleanup"]

        return PlannedCommand(
            label=f"tb run {run_id}",
            argv=argv,
            env=dict(adapter.env(model)),
            cwd=workspace,
            timeout=options.get("harness_timeout_sec"),
            artifacts=(os.path.join(runs_dir, run_id, "results.json"),
                       os.path.join(runs_dir, run_id, "run_metadata.json")),
            notes=[f"dataset: {self.dataset_label(spec)}",
                   f"agent: {trial.agent.label}"],
        )

    def describe(self, trial, adapter, run_dir):
        """What a plan shows when the trial cannot be expanded yet.

        Under Apptainer the missing piece is usually the staged SIF dataset, so the
        plan prints the two commands that create it and then use it, rather than
        the bare "would run" line.
        """
        if (trial.container_runtime or "docker") != "apptainer":
            return [f"would run {self.name} with {adapter.label}"]
        lines = []
        try:
            prepare = self.prepare_command(trial, run_dir)
        except SuiteError as e:
            return [f"! {e}"]
        if prepare is not None:
            lines.append("1. " + prepare.rendered())
        stage = len(lines) + 1
        lines += [
            f"{stage}. harbor run -p {self.apptainer_dataset_dir(run_dir)} "
            f"--env singularity "
            f"--environment-kwarg singularity_image_cache_dir=<cache> "
            f"--environment-kwarg singularity_no_mount=home,tmp "
            f"-a <agent> -m <provider>/<model> --n-tasks "
            f"{trial.limit or '<n>'} --yes",
            "   (each staged task.toml points docker_image at its derived .sif)",
        ]
        return lines

    # -- results ---------------------------------------------------------

    def collect(self, run_dir, run_id=None):
        """Read whichever harness's result file exists under ``run_dir``."""
        harbor_path = self._harbor_results_path(run_dir, run_id)
        if harbor_path:
            return self._collect_harbor(harbor_path)
        tb_path = self._tb_results_path(run_dir, run_id)
        if tb_path:
            return self._collect_tb(tb_path)
        raise SuiteError(
            f"terminal-bench: no results under {run_dir} (looked for "
            f"harbor/<job>/result.json and tb/<run>/results.json); did the run "
            f"finish?")

    def _collect_harbor(self, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("trial_results") or []
        if not entries:
            # A job whose trials never started (no environment, missing image)
            # carries no trial_results -- only per-eval exception stats. Those
            # trials are outcomes too: dropping them would turn "every task
            # errored" into "nothing ran", which is a different claim.
            return self._collect_harbor_stats(payload)
        outcomes = []
        for entry in entries:
            rewards = ((entry.get("verifier_result") or {}).get("rewards") or {})
            failed = bool(entry.get("exception_info"))
            if failed:
                outcome = TrialOutcome.ERROR
            elif rewards and max(float(v) for v in rewards.values()) >= 1.0:
                outcome = TrialOutcome.RESOLVED
            else:
                outcome = TrialOutcome.UNRESOLVED
            outcomes.append(TrialOutcome(
                suite=self.name,
                instance_id=str(entry.get("task_name") or entry.get("trial_name") or ""),
                outcome=outcome,
                detail=_exception_detail(entry),
                extra={"rewards": rewards,
                       "trial_name": entry.get("trial_name")},
            ))
        return outcomes

    def _collect_harbor_stats(self, payload):
        """Outcomes from harbor's per-eval summary (no per-trial records).

        Harbor writes ``trial_results`` only when the trials ran far enough to
        produce them; a job that finished has its rewards and exceptions in
        ``stats.evals`` instead. Both are results: a 0.0 reward is a real
        (unresolved) outcome, and an exception is an error.
        """
        outcomes = []
        stats = payload.get("stats") or {}
        for eval_name, entry in (stats.get("evals") or {}).items():
            # reward_stats is <metric> -> <reward value> -> [task names].
            for metric, values in (entry.get("reward_stats") or {}).items():
                for reward, task_names in (values or {}).items():
                    try:
                        value = float(reward)
                    except (TypeError, ValueError):
                        continue
                    for task_name in task_names or []:
                        outcomes.append(TrialOutcome(
                            suite=self.name, instance_id=str(task_name),
                            outcome=(TrialOutcome.RESOLVED if value >= 1.0
                                     else TrialOutcome.UNRESOLVED),
                            detail=f"{metric}={value}",
                            extra={"eval": eval_name, metric: value}))
            for exception_type, task_names in \
                    (entry.get("exception_stats") or {}).items():
                for task_name in task_names or []:
                    outcomes.append(TrialOutcome(
                        suite=self.name, instance_id=str(task_name),
                        outcome=TrialOutcome.ERROR,
                        detail=f"{exception_type}: trials never started "
                               f"(eval {eval_name})",
                        extra={"eval": eval_name,
                               "exception_type": exception_type}))

        if not outcomes and stats.get("n_errored_trials"):
            outcomes.append(TrialOutcome(
                suite=self.name, instance_id="<unknown>",
                outcome=TrialOutcome.ERROR,
                detail=f"{stats['n_errored_trials']} trial(s) errored"))
        return _dedupe(outcomes)

    def _collect_tb(self, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        outcomes = []
        for entry in payload.get("results", []) or []:
            resolved = bool(entry.get("is_resolved"))
            outcomes.append(TrialOutcome(
                suite=self.name,
                instance_id=str(entry.get("task_id") or entry.get("trial_name") or ""),
                outcome=(TrialOutcome.RESOLVED if resolved
                         else TrialOutcome.UNRESOLVED),
                detail=str(entry.get("failure_mode") or ""),
                artifacts=[entry["recording_path"]] if entry.get("recording_path") else [],
                extra={"trial_name": entry.get("trial_name"),
                       "input_tokens": entry.get("total_input_tokens"),
                       "output_tokens": entry.get("total_output_tokens")},
            ))
        return outcomes

    def _harbor_results_path(self, run_dir, job_name=None):
        base = os.path.join(run_dir, "harbor")
        if job_name:
            path = os.path.join(base, job_name, "result.json")
            return path if os.path.isfile(path) else None
        return _newest(os.path.join(base, "*", "result.json"))

    def _tb_results_path(self, run_dir, run_id=None):
        base = os.path.join(run_dir, "tb")
        if run_id:
            path = os.path.join(base, run_id, "results.json")
            return path if os.path.isfile(path) else None
        return _newest(os.path.join(base, "*", "results.json"))


def _python():
    """The interpreter running the harness (same one harbor is installed into)."""
    import sys

    return sys.executable


def _dataset_slug(spec):
    """A filesystem-safe name for a dataset (``terminal-bench-2.0``)."""
    if spec is None:
        return "dataset"
    name = str(getattr(spec, "dataset", "") or "dataset").replace("/", "_")
    version = getattr(spec, "version", None)
    clean = f"{name}-{version}" if version else name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", clean)


def _cache_root():
    """Git-ignored cache shared by datasets and derived images."""
    return os.environ.get("BENCH_CACHE") or os.path.join("benchmarks", ".cache")


def _default_sif_cache():
    """A cache directory that survives between runs (git-ignored)."""
    return os.environ.get("BENCH_SIF_CACHE") or os.path.join(_cache_root(), "sif")


def _mini_code_dir():
    """The ``mini_code_cli`` package to bake into the derived SIF."""
    return os.path.join(_repo_root(), "mini_code_cli")


def _repo_root():
    """The directory that contains the ``benchmarks`` package.

    Passed to harbor as ``PYTHONPATH`` so ``-a benchmarks.harbor_agents:...``
    resolves there; harbor runs in whatever interpreter it was installed into.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(here)


def _newest(pattern):
    """The most recently modified file matching a one-level glob, or ``None``."""
    import glob

    candidates = [(os.path.getmtime(p), p) for p in glob.glob(pattern)]
    return max(candidates)[1] if candidates else None


def _dedupe(outcomes):
    """One outcome per task: harbor retries report the same task several times.

    The *best* attempt wins (resolved > unresolved > error), because a task that
    succeeded on a retry was solved; an un-deduplicated count would silently
    inflate ``n`` and turn a solved task into a half-error.
    """
    rank = {TrialOutcome.RESOLVED: 3, TrialOutcome.UNRESOLVED: 2,
            TrialOutcome.ERROR: 0, TrialOutcome.TIMEOUT: 1}
    best = {}
    for outcome in outcomes:
        current = best.get(outcome.instance_id)
        if current is None or rank.get(outcome.outcome, 0) > rank.get(
                current.outcome, 0):
            best[outcome.instance_id] = outcome
    return [best[key] for key in sorted(best)]


def _exception_detail(entry):
    info = entry.get("exception_info") or {}
    if not info:
        return ""
    return str(info.get("exception_type") or info.get("message") or "")


def _kwarg(value):
    """Serialise an ``--agent-kwarg`` value.

    Lists (the argv template) go over as JSON: the harness parses JSON, strings
    and booleans from this value, and JSON is the only lossless spelling for a
    list of arguments that may themselves contain ``=`` or spaces.
    """
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True) if value else ""
    if value is None:
        return ""
    return str(value)


def _timeout(trial, key):
    """The trial's budget for ``key``, with the config's default as a floor."""
    return (trial.timeouts or {}).get(key, 3_600 if key == "agent_seconds" else 1_800)
