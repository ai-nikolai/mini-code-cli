"""SWE-bench suite: agent-side patch generation + the official evaluation harness.

SWE-bench splits cleanly in two, and this module keeps that split:

1. **Produce a patch.** The agent runs against the instance's prepared environment
   (the image SWE-bench builds for its base commit) and its ``git diff`` is the
   prediction. Nothing here grades anything.
2. **Grade it.** ``python -m swebench.harness.run_evaluation`` is invoked with the
   predictions file; it applies each patch, runs the instance's FAIL_TO_PASS /
   PASS_TO_PASS tests in the official container and writes a report.

So this adapter reimplements *no* part of SWE-bench: it writes the mandated
``predictions.jsonl`` (``instance_id`` / ``model_name_or_path`` / ``model_patch``),
drives the agent, and reads the harness's report back.

Pinned against ``swebench`` 5.0.2:

* ``run_evaluation -d <dataset|json> -s <split> -p <preds> -id <run_id>
  --max_workers N -t <test timeout> --report_dir <dir> [-i ID ...]``;
* report file ``<report_dir>/<model_name_or_path>.<run_id>.json`` with
  ``resolved_ids`` / ``unresolved_ids`` / ``error_ids`` / ``completed_instances``.

Datasets: ``SWE-bench/SWE-bench_Verified`` (split ``test``) is the academic
baseline; ``SWE-bench/SWE-bench_Lite`` works the same way. ``revision`` pins the
dataset commit, and the resolved revision is recorded in the run manifest.
"""
import json
import shlex
import os

from .. import forks as forks_module
from . import swebench_apptainer
from .. import runtime as runtime_module
from ..agents import container_setup
from .base import PlannedCommand, Suite, SuiteError, TrialOutcome

#: The repository root, so the shared `.sif` cache can live under `benchmarks/runs`
#: (ignored by git) no matter which directory a run was started from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class SweBenchSuite(Suite):
    name = "swebench"
    install_hint = "pip install 'swebench>=5' (brings datasets + docker)"
    results_filename = "predictions.jsonl"

    #: The default base image name template for an instance environment. This is
    #: the long-standing upstream naming; override with ``options.image_template``
    #: if a harness release changes it.
    DEFAULT_IMAGE_TEMPLATE = "swebench/sweb.eval.x86_64.{instance_id}"

    # -- preflight -------------------------------------------------------

    def preflight(self, model, require_binary=True):
        problems = []
        if not require_binary:
            return problems
        if not _module_available("swebench"):
            problems.append(f"suite 'swebench': the swebench package is not "
                            f"importable ({self.install_hint})")
        problems += forks_module.preflight_problems()
        chosen = runtime_module.resolve(self._runtime_hint())
        if chosen is None or not runtime_module.available(chosen):
            problems.append(
                "suite 'swebench': no container runtime is usable "
                f"({' or '.join(runtime_module.RUNTIMES)} not on PATH); the agent "
                "runs inside the instance image")
        return problems

    def _runtime_hint(self):
        """The configured runtime (``auto``/``docker``/``apptainer``).

        The suite used to read ``self.runtime``, which never exists -- runtime is
        a run-level setting -- so every lookup silently fell through to ``auto``
        and picked apptainer on a machine without a Docker daemon even when the
        config pinned docker.
        """
        return (getattr(self, "runtime", None)
                or getattr(self.spec, "runtime", None))

    def notes(self, model):
        chosen = runtime_module.resolve(self._runtime_hint())
        notes = [
            "the agent runs inside the instance image, so the endpoint must be "
            "reachable from a container and the agent CLI must be installed there "
            "(options.install_script, or a prebuilt image)",
        ]
        if chosen == "apptainer":
            if forks_module.swebench_fork_active():
                notes.append(
                    "grading runs through our SWE-bench checkout "
                    "(mini-code/apptainer): "
                    "`python -m swebench.harness.run_evaluation --backend "
                    "apptainer`, i.e. the upstream harness with a container "
                    "backend that needs no Docker daemon")
            else:
                notes.append(
                    "grading is swebench's own `grading.get_eval_report`, driven by "
                    "benchmarks.suites.swebench_apptainer: the instance image runs "
                    "under `apptainer exec`, so no Docker daemon is involved")
        else:
            notes.append("grading is the upstream harness's: patches are applied "
                         "and the instance tests run in the official container")
        if chosen:
            notes.append(f"container runtime: {chosen} "
                         f"({runtime_module.image_ref(chosen, 'swebench/...')})")
            notes += runtime_module.translation_notes(
                chosen, limits=getattr(self, "docker", None))
        else:
            notes.append("no container runtime found (docker/apptainer); `plan` "
                         "still shows the command shape, `run` will fail")
        return notes

    # -- instances -------------------------------------------------------

    def instances_path(self, run_dir):
        return os.path.abspath(os.path.join(run_dir, "instances.jsonl"))

    def load_instances(self, run_dir):
        path = self.instances_path(run_dir)
        if not os.path.isfile(path):
            raise SuiteError(
                f"swebench: no instance list at {path}. Run "
                f"`python -m benchmarks prepare --suite swebench ...` first "
                f"(it materialises the pinned dataset), or point "
                f"options.instances_file at a local JSONL of instances.")
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def prepare_command(self, trial, run_dir):
        """A ``PlannedCommand`` that materialises the pinned instance list."""
        spec = trial.suite
        argv = [
            _python(), "-m", "benchmarks.suites.swebench_env", "instances",
            "--dataset", spec.dataset or "SWE-bench/SWE-bench_Verified",
            "--split", spec.split or "test",
            "--out", self.instances_path(run_dir),
        ]
        if spec.version:
            argv += ["--revision", spec.version]
        if trial.limit:
            argv += ["--limit", str(trial.limit)]
        if trial.suite.options.get("instances_file"):
            argv += ["--from-file", str(trial.suite.options["instances_file"])]
        return PlannedCommand(label="swebench: materialise instances", argv=argv,
                              timeout=1_800)

    # -- commands --------------------------------------------------------

    def commands(self, trial, workspace, run_dir, adapter, timeouts=None):
        """Expand one trial into build/agent/diff commands plus the evaluation.

        The per-instance commands are generated from the prepared instance list, so
        ``plan`` on a machine without the dataset shows the shape of the run
        (see :meth:`describe`) and ``run`` shows the exact commands.
        """
        # Every command below runs with cwd=run_dir, but the upstream harness
        # re-enters the repository, so it has to be given absolute paths.
        run_dir = os.path.abspath(run_dir)
        instances = self.load_instances(run_dir)
        commands = []
        # Apptainer has no daemon, so an instance image has to exist as a .sif
        # *before* the agent runs inside it -- pulling after the agent was a
        # 1.8s exit 255 ("failed to get checksum for docker://...").
        if not (trial.suite.options or {}).get("skip_evaluation") \
                and self.apptainer_backend(trial):
            commands.append(self.images_command(trial, run_dir))
        commands.extend(self._instance_commands(trial, instances, run_dir,
                                                adapter, timeouts))
        commands.append(self.predictions_command(trial, run_dir))
        if not (trial.suite.options or {}).get("skip_evaluation"):
            commands.append(self.evaluate_command(trial, run_dir))
        return commands

    def apptainer_backend(self, trial):
        """True when this trial's grading has to avoid the Docker daemon."""
        return (trial.container_runtime or "docker") == "apptainer"

    def sif_cache(self):
        """Where instance images are cached as `.sif` (shared across runs)."""
        configured = (getattr(self.spec, "options", None) or {}) \
            .get("sif_cache")
        if configured:
            return str(configured)
        return os.path.join(_REPO_ROOT, "benchmarks", "runs", "_cache", "swe-sif")

    def images_command(self, trial, run_dir):
        """Prefetch the instance images as `.sif` (Apptainer has no daemon)."""
        argv = [
            _python(), "-m", "benchmarks.suites.swebench_apptainer", "pull",
            "--instances", self.instances_path(run_dir),
            "--sif-cache", self.sif_cache(),
            "--log-dir", self.ensure_dir(os.path.join(run_dir, "pulls")),
        ]
        if trial.limit:
            argv += ["--limit", str(trial.limit)]
        return PlannedCommand(label="swebench: cache instance images as .sif",
                              argv=argv,
                              timeout=(getattr(self.spec, "options", None)
                                       or {}).get("pull_timeout_sec"))

    def _instance_commands(self, trial, instances, run_dir, adapter, timeouts=None):
        """The per-instance commands: install (optional), agent, capture patch."""
        options = trial.suite.options or {}
        runs_dir = self.ensure_dir(os.path.join(run_dir, "instances"))
        image_template = options.get("image_template", self.DEFAULT_IMAGE_TEMPLATE)
        agent_timeout = int((timeouts or {}).get("instance_seconds", 5_400))
        # `plan` on a machine with no runtime should still show *a* command line;
        # docker is the spelling the upstream harnesses assume, and preflight is
        # what reports the missing binary.
        container = trial.container_runtime or "docker"
        commands = []

        for instance in instances:
            instance_id = instance["instance_id"]
            image = self._image(instance, container, image_template)
            if container == "apptainer":
                # No daemon: the runnable artefact is the .sif the pull step
                # cached, not the docker:// reference (which apptainer would try
                # to resolve against the registry on every exec).
                cached = swebench_apptainer.sif_path(self.sif_cache(), instance_id)
                if os.path.isfile(cached):
                    image = cached
            workdir = self.ensure_dir(os.path.join(runs_dir, instance_id))
            patch_path = os.path.join(workdir, "model.patch")
            log_path = os.path.join(workdir, "agent.log")

            # The agent itself may not exist in the instance image (claude-code,
            # pi): install it once per instance, inside the image, before the run.
            tools_dir = self.ensure_dir(os.path.join(workdir, "agent-tools"))
            for target, contents in adapter.container_setup_files(trial.model).items():
                relative = target[len(container_setup.PREFIX):].lstrip("/")
                destination = os.path.join(tools_dir, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(destination, "w", encoding="utf-8") as handle:
                    handle.write(contents)
            install_commands = adapter.install_commands()
            if install_commands:
                commands.append(PlannedCommand(
                    label=f"swebench: install agent tools for {instance_id}",
                    argv=runtime_module.container_argv(
                        container, image,
                        ["bash", "-lc", " && ".join(install_commands)],
                        limits=dict(trial.docker, network="none"),
                        # npm needs ~300 MiB; the image's writable overlay is
                        # 64 MiB, so the tools live on the host instead.
                        mounts=[(tools_dir, container_setup.PREFIX)],
                        workdir="/testbed"),
                    timeout=int(options.get("install_timeout_sec", 1_800)),
                    notes=["installs the agent's own runtime (node) in the image"]))

            if options.get("install_script"):
                # Preparation happens *inside* the instance image, once per
                # instance, before the agent runs.
                commands.append(PlannedCommand(
                    label=f"swebench: install agent for {instance_id}",
                    argv=runtime_module.container_argv(
                        container, image, ["bash", "/tmp/install.sh"],
                        env=options.get("install_env"),
                        limits=dict(trial.docker, network="none"),
                        mounts=[(options["install_script"], "/tmp/install.sh:ro")]),
                    timeout=int(options.get("install_timeout_sec", 1_800))))

            agent_argv = adapter.argv(_problem_statement(instance), trial.model,
                                      workspace="/testbed", container=True)
            if install_commands:
                # The tools live on the bind mount, and the image's writable
                # overlay is thrown away between execs, so /usr/local/bin
                # symlinks do not survive: put them on PATH for this command.
                agent_argv = [
                    "bash", "-lc",
                    f"export PATH={container_setup.PREFIX}/bin:$PATH; "
                    f"exec {shlex.join(agent_argv)}",
                ]
            commands.append(PlannedCommand(
                label=f"swebench: agent {instance_id}",
                argv=runtime_module.container_argv(
                    container, image,
                    agent_argv,
                    container_name=f"bench-{trial.run_id}-{_short(instance_id)}",
                    env=adapter.container_env(trial.model),
                    mounts=list(adapter.container_mounts())
                    + [(tools_dir, container_setup.PREFIX)],
                    limits=trial.docker,
                    network=trial.docker.get("network"),
                    workdir="/testbed"),
                cwd=run_dir,
                timeout=agent_timeout,
                artifacts=(log_path,),
                notes=["the agent's stdout/stderr are captured by the runner"]))

            # The patch is the prediction: capture everything the agent changed,
            # including new files (hence `git add -A`), from the instance itself.
            commands.append(PlannedCommand(
                label=f"swebench: capture patch {instance_id}",
                argv=runtime_module.container_argv(
                    container, image,
                    ["bash", "-lc", "git add -A && git diff --cached > /out/model.patch"],
                    limits=trial.docker,
                    mounts=[(workdir, "/out")],
                    workdir="/testbed"),
                cwd=run_dir,
                timeout=int(options.get("patch_timeout_sec", 600)),
                artifacts=(patch_path,),
                notes=["runs git inside the container; no user input is involved"]))
        return commands

    @staticmethod
    def _image(instance, container, image_template):
        """The instance's image name.

        The dataset ships an authoritative ``image`` (its tags escape ``__`` to
        ``_1776_``, which the ``image_template`` cannot reproduce), so it wins
        whenever it is present -- and for Apptainer it is also what gets pulled
        into the ``.sif`` cache, so both sides must agree on one name.
        """
        image = str(instance.get("image") or "").strip()
        if image and (container == "apptainer" or "{instance_id}" not in
                      str(image_template)):
            return image
        return str(image_template).format(instance_id=instance["instance_id"])

    def predictions_command(self, trial, run_dir):
        """Turn the per-instance patches into the harness's ``predictions.jsonl``."""
        argv = [
            _python(), "-m", "benchmarks.suites.swebench_env", "predictions",
            "--instances", self.instances_path(run_dir),
            "--patches-dir", os.path.join(run_dir, "instances"),
            "--model-name", trial.model.model,
            "--out", os.path.join(run_dir, self.results_filename),
        ]
        return PlannedCommand(label="swebench: write predictions", argv=argv,
                              timeout=600,
                              artifacts=(os.path.join(run_dir,
                                                      self.results_filename),))

    def evaluate_command(self, trial, run_dir):
        """The official evaluation harness, on the predictions written above."""
        spec = trial.suite
        options = spec.options or {}
        if self.apptainer_backend(trial):
            return self.apptainer_evaluate_command(trial, run_dir)
        argv = [
            _python(), "-m", "swebench.harness.run_evaluation",
            "-d", spec.dataset or "SWE-bench/SWE-bench_Verified",
            "-s", spec.split or "test",
            "-p", os.path.join(run_dir, self.results_filename),
            "-id", trial.run_id,
            "--max_workers", str(int(options.get("max_workers", 4))),
            "-t", str(int(options.get("test_timeout_sec", 1_800))),
            "--report_dir", run_dir,
        ]
        if trial.limit:
            # The harness rejects unknown ids, so only pass the ones this run
            # actually materialised. Absent list (a `plan`) means "no -i": the
            # predictions file already scopes the run.
            try:
                argv += ["-i", *[i["instance_id"]
                                  for i in self.load_instances(run_dir)]]
            except SuiteError:
                pass
        return PlannedCommand(
            label=f"swebench: evaluate {trial.run_id}", argv=argv, cwd=run_dir,
            timeout=options.get("evaluate_timeout_sec"),
            artifacts=(os.path.join(run_dir, f"{trial.run_id}.json"),))

    def apptainer_evaluate_command(self, trial, run_dir):
        """Grade without a Docker daemon.

        Two ways to do that, and the patched upstream harness wins when it is
        installed: it is the real ``run_evaluation`` (so a bug there is *our* bug
        to fix, not a difference we paper over) plus a container backend. The
        in-repo ``swebench_apptainer`` is the fallback for an unpatched install.
        """
        if forks_module.swebench_fork_active():
            return self.fork_evaluate_command(trial, run_dir)
        options = trial.suite.options or {}
        report = os.path.join(run_dir, f"{trial.run_id}.json")
        argv = [
            _python(), "-m", "benchmarks.suites.swebench_apptainer", "evaluate",
            "--instances", self.instances_path(run_dir),
            "--patches-dir", os.path.join(run_dir, "instances"),
            "--report", report,
            "--work-dir", self.ensure_dir(os.path.join(run_dir, "eval")),
            "--sif-cache", self.sif_cache(),
            "--timeout", str(int(options.get("test_timeout_sec", 1_800))),
        ]
        if trial.limit:
            argv += ["--limit", str(trial.limit)]
        return PlannedCommand(
            label=f"swebench: evaluate {trial.run_id} (apptainer)", argv=argv,
            cwd=run_dir, timeout=options.get("evaluate_timeout_sec"),
            artifacts=(report,))

    def fork_evaluate_command(self, trial, run_dir):
        """``python -m swebench.harness.run_evaluation --backend apptainer``."""
        spec = trial.suite
        options = spec.options or {}
        argv = [
            _python(), "-m", "swebench.harness.run_evaluation",
            "-d", spec.dataset or "SWE-bench/SWE-bench_Verified",
            "-s", spec.split or "test",
            "-p", os.path.join(run_dir, self.results_filename),
            "-id", trial.run_id,
            "--backend", "apptainer",
            "--sif-cache", self.sif_cache(),
            "-t", str(int(options.get("test_timeout_sec", 1_800))),
            "--max_workers", str(int(options.get("max_workers", 1))),
        ]
        if trial.limit:
            try:
                argv += ["-i", *[i["instance_id"]
                                  for i in self.load_instances(run_dir)]]
            except SuiteError:
                pass
        return PlannedCommand(
            label=f"swebench: evaluate {trial.run_id} (fork, apptainer)",
            argv=argv, cwd=run_dir,
            timeout=options.get("evaluate_timeout_sec"),
            artifacts=(os.path.join(run_dir, f"{trial.run_id}.json"),))

    def describe(self, trial, adapter, run_dir):
        """Show the shape of a SWE-bench trial before its dataset is downloaded.

        ``plan`` calls this when the instance list does not exist yet: the prepare
        step, one per-instance command with placeholders, and the evaluation are
        printed so the run can still be reviewed (and the cost understood).
        """
        stub = [{"instance_id": "<instance_id>",
                 "problem_statement": "<problem statement from the dataset>"}]
        shape = [self.prepare_command(trial, run_dir)]
        # Same order as `commands()`: the .sif has to exist before anything runs
        # inside it, or the plan would describe a sequence that cannot work.
        if self.apptainer_backend(trial):
            shape.append(self.images_command(trial, run_dir))
        shape += self._instance_commands(trial, stub, run_dir, adapter)
        shape.append(self.predictions_command(trial, run_dir))
        shape.append(self.evaluate_command(trial, run_dir))
        lines = [f"{number}. {command.rendered()}"
                 for number, command in enumerate(shape, 1)]
        return lines

    # -- results ---------------------------------------------------------

    def collect(self, run_dir, run_id=None):
        """Read the harness report; fall back to the predictions file."""
        report = self._find_report(run_dir, run_id)
        if report is None:
            raise SuiteError(
                f"swebench: no report file in {run_dir} (expected "
                f"<report_dir>/<model>.<run_id>.json from run_evaluation)")
        with open(report, encoding="utf-8") as handle:
            payload = json.load(handle)
        resolved = set(payload.get("resolved_ids") or [])
        unresolved = set(payload.get("unresolved_ids") or [])
        errors = set(payload.get("error_ids") or [])
        outcomes = []
        for instance_id in sorted(resolved | unresolved | errors):
            if instance_id in resolved:
                outcome = TrialOutcome.RESOLVED
            elif instance_id in errors:
                outcome = TrialOutcome.ERROR
            else:
                outcome = TrialOutcome.UNRESOLVED
            outcomes.append(TrialOutcome(suite=self.name, instance_id=instance_id,
                                         outcome=outcome))
        return outcomes

    @staticmethod
    def _find_report(run_dir, run_id=None):
        if not os.path.isdir(run_dir):
            return None
        candidates = []
        # The harness puts its report where *it* likes -- the apptainer backend
        # writes logs/evaluation/<run_id>/results.json -- so look at every depth
        # and identify the report by its contents, not by its name or directory.
        for root, _dirs, names in os.walk(run_dir):
            for name in names:
                if not name.endswith(".json"):
                    continue
                if run_id and run_id not in name and run_id not in root:
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, encoding="utf-8") as handle:
                        payload = json.load(handle)
                except (OSError, ValueError):
                    continue
                if isinstance(payload, dict) and ("resolved_ids" in payload
                                                  or "completed_instances" in payload):
                    candidates.append((os.path.getmtime(path), path))
        return max(candidates)[1] if candidates else None

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _module_available(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _python():
    import sys

    return sys.executable


def _short(instance_id):
    """A docker-safe short name (instance ids contain ``/`` and ``__``)."""
    return str(instance_id).replace("/", "-").replace("__", "-")[-24:]


def _problem_statement(instance):
    return str(instance.get("problem_statement") or instance.get("text") or "")


