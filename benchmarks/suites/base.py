"""Suite interface: what a benchmark must provide to be runnable here.

A suite owns *everything benchmark-specific*: how to get its task list, the exact
command that runs it, where it writes results, and how to read those results back
into a common shape. The runner never learns what a "task" or an "instance" is
beyond :class:`TrialOutcome`.

The contract is deliberately narrow, because each upstream harness does its own
execution and evaluation:

* :meth:`Suite.preflight` -- binary/config problems that block a real run;
* :meth:`Suite.notes`      -- things worth printing in ``plan`` but not blocking;
* :meth:`Suite.commands`   -- the exact argv(s) to execute for one trial;
* :meth:`Suite.collect`    -- parse the harness's own artifacts into outcomes.
"""
import os


class SuiteError(RuntimeError):
    """A suite cannot run as configured."""


class Trial:
    """One cell of the matrix: a suite, an agent setting, a model, a limit."""

    def __init__(self, suite, agent, model, limit=None, timeouts=None,
                 docker=None):
        self.suite = suite
        self.agent = agent
        self.model = model
        self.limit = limit
        # The run-level limits are copied in, not looked up: a suite must be able
        # to build its command line from the trial alone (which is also what
        # makes `plan` a pure function of the config).
        self.timeouts = dict(timeouts or {})
        self.docker = dict(docker or {})

    @property
    def run_id(self):
        """Deterministic id: same config in, same id out (this is what resumability
        keys on, alongside the git SHA recorded in the manifest)."""
        from ..config import _slug

        return _slug(f"{self.suite.name}-{self.agent.label}-{self.model.name}")

    def __repr__(self):
        return f"Trial({self.run_id!r})"


class PlannedCommand:
    """A command the runner may execute, with everything needed to record it.

    ``argv`` is a list -- never a shell string -- except for the few places where
    the upstream harness genuinely takes a shell command (Terminal-Bench's
    installed agents run inside a tmux session); those set ``shell=True`` and say
    so in ``notes``.
    """

    def __init__(self, label, argv, env=None, cwd=None, timeout=None,
                 artifacts=(), notes=(), stdin=None):
        self.label = label
        self.argv = list(argv)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.artifacts = tuple(artifacts)
        self.notes = list(notes)
        self.stdin = stdin

    def rendered(self):
        """A copy-pasteable rendering, with secret placeholders left as-is."""
        import shlex

        return shlex.join(self.argv)

    def __repr__(self):
        return f"PlannedCommand({self.label!r})"


class Suite:
    """Base class for the benchmark suites."""

    #: Registry name (``--suite`` on the CLI).
    name = ""

    #: Where the upstream harness comes from, for the preflight message.
    install_hint = ""

    #: The artifact this suite writes that :meth:`collect` reads.
    results_filename = ""

    def __init__(self, spec=None):
        # The spec is kept so preflight/notes can answer questions that depend on
        # the config (which harness, which dataset) before a trial exists.
        self.spec = spec

    def preflight(self, model, require_binary=True):
        """Return blocking problems (missing harness, missing dataset, ...)."""
        return []

    def notes(self, model):
        """Return non-blocking remarks for ``plan``."""
        return []

    def commands(self, trial, workspace, run_dir, adapter):
        """Return ``[PlannedCommand, ...]`` that execute this trial."""
        raise NotImplementedError

    def describe(self, trial, adapter, run_dir):
        """Human-readable lines for ``plan`` when the run cannot be expanded yet.

        Terminal-Bench can always be expanded (one ``tb run``), but SWE-bench
        needs its instance list first; ``describe`` is how a dry run stays useful
        without downloading a dataset.
        """
        return [f"would run {self.name} with {adapter.label}"]

    def collect(self, run_dir):
        """Return ``list[TrialOutcome]`` parsed from ``run_dir``'s artifacts."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------

    def dataset_label(self, spec):
        """``dataset==version`` or ``dataset``: one string for manifests/logs."""
        if spec.version:
            return f"{spec.dataset}=={spec.version}"
        return str(spec.dataset or "")

    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
        return path


class TrialOutcome:
    """The common result shape every suite reduces to.

    ``outcome`` is deliberately coarse (``resolved`` / ``unresolved`` /
    ``error`` / ``timeout``) so a report can compare agents across benchmarks
    without pretending the two scoring rules are the same.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ERROR = "error"
    TIMEOUT = "timeout"

    def __init__(self, suite, instance_id, outcome, duration_seconds=None,
                 detail=None, artifacts=(), extra=None):
        self.suite = suite
        self.instance_id = instance_id
        self.outcome = outcome
        self.duration_seconds = duration_seconds
        self.detail = detail or ""
        self.artifacts = tuple(artifacts)
        self.extra = dict(extra or {})

    @property
    def resolved(self):
        return self.outcome == self.RESOLVED

    def to_dict(self):
        payload = {
            "suite": self.suite,
            "instance_id": self.instance_id,
            "outcome": self.outcome,
            "resolved": self.resolved,
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = round(float(self.duration_seconds), 3)
        if self.detail:
            payload["detail"] = self.detail
        if self.artifacts:
            payload["artifacts"] = list(self.artifacts)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload
