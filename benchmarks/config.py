"""Configuration model for the benchmark harness.

One JSON (or TOML, on 3.11+) file describes *what* to run: which suites, which
agents and settings, which models, and the limits around them. Everything the
runner needs comes from here, so a run is reproducible from the file plus the git
SHA and dataset revision recorded in its manifest.

The file is deliberately a plain data structure rather than a Python module: the
agent/setting/model matrix is expanded mechanically, which is what keeps the
comparison honest (every agent sees the same suites, models and limits).
"""
import json
import os
import re

# Keys are validated by the suites/agents themselves; these are the ones this
# module needs to exist and be sane.
DEFAULT_TIMEOUTS = {
    "instance_seconds": 5_400,   # one task/instance end to end
    "agent_seconds": 3_600,      # the agent's own turn budget
    "test_seconds": 1_800,       # the suite's tests/validation
}
DEFAULT_DOCKER = {
    "cpus": 4,
    "memory": "16g",
    "pids_limit": 2_048,
    "network": "bridge",
}
DEFAULT_OUTPUT_DIR = os.path.join("benchmarks", "runs")

#: Which container runtime to use by default: ``"auto"`` means "docker if it is
#: installed, otherwise apptainer" (see ``benchmarks/runtime.py``). HPC machines
#: frequently have no Docker daemon at all, and pinning the runtime in the config
#: is what keeps a comparison reproducible across two such machines.
DEFAULT_RUNTIME = "auto"


class ConfigError(ValueError):
    """The configuration file is unusable; the message names what to fix."""


def _slug(text):
    """A filesystem- and shell-safe identifier built from ``text``.

    Run ids end up in paths and in ``docker`` names, so anything that is not
    ``[A-Za-z0-9._-]`` collapses to a single dash.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned or "unnamed"


class ModelSpec:
    """One model endpoint, as the agents see it.

    ``api`` is the wire protocol the *client* must speak: ``"openai"`` for
    anything OpenAI-compatible (vLLM, SGLang, LiteLLM) and ``"anthropic"`` for
    Claude Code, which speaks the Anthropic Messages API and therefore needs
    either api.anthropic.com or an Anthropic-mode proxy (see ``models.py``).
    """

    def __init__(self, name, model, base_url, api="openai", api_key_env=None,
                 notes=None):
        self.name = str(name)
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api = str(api)
        self.api_key_env = str(api_key_env) if api_key_env else None
        self.notes = str(notes) if notes else ""

    def __repr__(self):
        return f"ModelSpec({self.name!r}, {self.model!r}, {self.api})"

    def to_dict(self):
        """A JSON-safe description (never the key itself -- only its env name)."""
        payload = {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api": self.api,
        }
        if self.api_key_env:
            payload["api_key_env"] = self.api_key_env
        if self.notes:
            payload["notes"] = self.notes
        return payload


class AgentSpec:
    """One agent plus the setting preset it should be run with.

    ``models`` restricts the matrix: an agent that speaks the Anthropic API is
    only meaningful against an Anthropic endpoint, so listing models per agent is
    how the config expresses that instead of producing invalid cross-product
    cells (and a wall of preflight noise).
    """

    def __init__(self, name, setting="default", options=None, models=None):
        self.name = str(name)
        self.setting = str(setting or "default")
        self.options = dict(options or {})
        self.models = tuple(str(m) for m in (models or ()))

    @property
    def label(self):
        """``mini-code:plan`` -- stable, used in run ids and report rows."""
        return f"{self.name}:{self.setting}"

    def __repr__(self):
        return f"AgentSpec({self.label!r})"

    def to_dict(self):
        payload = {"name": self.name, "setting": self.setting}
        if self.options:
            payload["options"] = dict(self.options)
        if self.models:
            payload["models"] = list(self.models)
        return payload


class SuiteSpec:
    """One benchmark, pinned by dataset name/revision."""

    def __init__(self, name, dataset=None, version=None, split=None, limit=None,
                 task_ids=None, options=None):
        self.name = str(name)
        self.dataset = str(dataset) if dataset else None
        self.version = str(version) if version else None
        self.split = str(split) if split else None
        self.limit = int(limit) if limit else None
        self.task_ids = tuple(str(t) for t in (task_ids or ()))
        self.options = dict(options or {})

    def __repr__(self):
        return f"SuiteSpec({self.name!r}, {self.dataset!r}, {self.version!r})"

    def to_dict(self):
        payload = {"name": self.name}
        for key in ("dataset", "version", "split", "limit"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.task_ids:
            payload["task_ids"] = list(self.task_ids)
        if self.options:
            payload["options"] = dict(self.options)
        return payload


class RunConfig:
    """The whole expansion: suites x agents x models, plus the limits."""

    def __init__(self, suites, agents, models, output_dir=DEFAULT_OUTPUT_DIR,
                 concurrency=1, timeouts=None, docker=None, env=None,
                 runtime=None, source=None):
        self.suites = list(suites)
        self.agents = list(agents)
        self.models = list(models)
        self.output_dir = str(output_dir)
        self.concurrency = max(int(concurrency), 1)
        self.timeouts = dict(DEFAULT_TIMEOUTS)
        self.timeouts.update(timeouts or {})
        self.docker = dict(DEFAULT_DOCKER)
        self.docker.update(docker or {})
        self.env = dict(env or {})
        # Resolved lazily so a config that says "auto" is still a pure data
        # structure: `plan` on a Docker host and `plan` on an Apptainer host show
        # their own commands, and the manifest records what was actually used.
        self.runtime = str(runtime).strip().lower() if runtime else DEFAULT_RUNTIME
        self.source = source

    def resolved_runtime(self):
        """The runtime this machine will use, or ``None`` when none is installed."""
        from .runtime import resolve

        return resolve(self.runtime)

    # -- expansion -------------------------------------------------------

    def trials(self, suites=None, agents=None, models=None, limit=None):
        """Expand the matrix into :class:`~benchmarks.suites.base.Trial` objects.

        The filters accept names as given in the config (``--suite`` on the CLI),
        and ``limit`` caps the *per-trial* instance count, which is what a smoke
        run needs. Order is stable: suite, then agent, then model -- so two runs
        with the same config list the same trials in the same order.
        """
        from .suites.base import Trial

        def wanted(items, spec, attr="name"):
            if not items:
                return True
            values = {str(i) for i in items}
            return getattr(spec, attr) in values

        out = []
        for suite in self.suites:
            if not wanted(suites, suite):
                continue
            for agent in self.agents:
                if not wanted(agents, agent, attr="name") and \
                        agent.label not in {str(a) for a in agents}:
                    continue
                for model in self.models:
                    if not wanted(models, model):
                        continue
                    if agent.models and model.name not in agent.models:
                        continue
                    out.append(Trial(
                        suite=suite, agent=agent, model=model,
                        # An explicit --limit wins; otherwise the suite's own
                        # limit (if any) applies, so a config is self-contained.
                        limit=limit if limit is not None else suite.limit,
                        timeouts=self.timeouts, docker=self.docker,
                        runtime=self.runtime))
        return out

    def to_dict(self):
        return {
            "output_dir": self.output_dir,
            "concurrency": self.concurrency,
            "timeouts": dict(self.timeouts),
            "docker": dict(self.docker),
            "env": dict(self.env),
            "runtime": self.runtime,
            "suites": [s.to_dict() for s in self.suites],
            "agents": [a.to_dict() for a in self.agents],
            "models": [m.to_dict() for m in self.models],
        }


# ---------------------------------------------------------------------------
# loading / validation
# ---------------------------------------------------------------------------

def _read_source(path):
    """Parse ``path`` (JSON, or TOML where ``tomllib`` exists) into a mapping."""
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    if path.endswith(".toml"):
        try:
            import tomllib  # Python >= 3.11
        except ImportError as e:  # pragma: no cover - 3.10 with a .toml file
            raise ConfigError(
                "TOML configs need Python 3.11+ (tomllib); use a .json config on "
                "this interpreter") from e
        try:
            return tomllib.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ConfigError(f"{path}: invalid TOML: {e}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from e


def _require(mapping, key, where):
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def parse_config(data, source=None):
    """Build a :class:`RunConfig` from an already-parsed mapping.

    Unknown top-level keys are rejected rather than ignored: a typo in a config
    that silently runs the wrong matrix is worse than a hard error. Keys *inside*
    ``options`` are passed through untouched, because they belong to the suite or
    agent that consumes them.
    """
    known = {"output_dir", "concurrency", "timeouts", "docker", "env", "runtime",
             "suites", "agents", "models"}
    # Keys starting with "_" are comments (the configs here are hand-written and
    # JSON has no comment syntax); anything else unknown is a typo.
    unknown = sorted(k for k in set(data) - known if not str(k).startswith("_"))
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(unknown)}")

    suites = []
    for entry in _require(data, "suites", "config"):
        if isinstance(entry, str):
            suites.append(SuiteSpec(name=entry))
            continue
        suites.append(SuiteSpec(
            name=_require(entry, "name", "suite"),
            dataset=entry.get("dataset"),
            version=entry.get("version"),
            split=entry.get("split"),
            limit=entry.get("limit"),
            task_ids=entry.get("task_ids"),
            options=entry.get("options"),
        ))

    models = []
    for entry in _require(data, "models", "config"):
        models.append(ModelSpec(
            name=_require(entry, "name", "model"),
            model=_require(entry, "model", "model"),
            base_url=_require(entry, "base_url", "model"),
            api=entry.get("api", "openai"),
            api_key_env=entry.get("api_key_env"),
            notes=entry.get("notes"),
        ))

    agents = []
    for entry in _require(data, "agents", "config"):
        if isinstance(entry, str):
            agents.append(AgentSpec(name=entry))
            continue
        settings = entry.get("settings") or [entry.get("setting") or "default"]
        for setting in settings:
            agents.append(AgentSpec(name=_require(entry, "name", "agent"),
                                    setting=setting,
                                    options=entry.get("options"),
                                    models=entry.get("models")))

    config = RunConfig(
        suites=suites,
        agents=agents,
        models=models,
        output_dir=data.get("output_dir", DEFAULT_OUTPUT_DIR),
        concurrency=data.get("concurrency", 1),
        timeouts=data.get("timeouts"),
        docker=data.get("docker"),
        env=data.get("env"),
        runtime=data.get("runtime"),
        source=source,
    )
    problems = validate_config(config)
    if problems:
        raise ConfigError("; ".join(problems))
    return config


def load_config(path):
    """Read, parse and validate a config file."""
    return parse_config(_read_source(path), source=path)


def validate_config(config, check_env=False):
    """Return a list of human-readable problems (empty when the config is usable).

    Structural problems -- empty matrix, duplicate names, bad numbers -- are
    always reported: a config that cannot describe a run is a bug in the config.

    ``check_env`` additionally requires every ``api_key_env`` to be *set*. It is
    off by default so ``plan`` works on a laptop with no credentials, and on for
    ``validate`` and ``run``, where a missing key would otherwise surface as an
    authentication error an hour into a run.
    """
    problems = []
    if not config.suites:
        problems.append("config: no suites specified")
    if not config.agents:
        problems.append("config: no agents specified")
    if not config.models:
        problems.append("config: no models specified")

    for kind, items in (("suite", config.suites), ("model", config.models)):
        seen = set()
        for item in items:
            if item.name in seen:
                problems.append(f"{kind} names must be unique: {item.name!r}")
            seen.add(item.name)

    labels = [a.label for a in config.agents]
    duplicates = {label for label in labels if labels.count(label) > 1}
    for label in sorted(duplicates):
        problems.append(f"agent setting listed twice: {label!r}")

    for model in config.models:
        if model.api not in ("openai", "anthropic"):
            problems.append(
                f"model {model.name!r}: api must be 'openai' or 'anthropic', "
                f"not {model.api!r}")
        if not model.base_url.startswith(("http://", "https://")):
            problems.append(
                f"model {model.name!r}: base_url must be an http(s) URL, "
                f"not {model.base_url!r}")
        if check_env and model.api_key_env and \
                not (os.environ.get(model.api_key_env)
                     or config.env.get(model.api_key_env)):
            # A key may come from the process environment or from the config's own
            # `env` block (which the harness injects into every command); either
            # satisfies "the agent will have it".
            problems.append(
                f"model {model.name!r}: {model.api_key_env} is not set in the "
                f"environment")

    for name, value in config.timeouts.items():
        if not isinstance(value, int) or value <= 0:
            problems.append(f"timeout {name!r} must be a positive integer")
    if config.concurrency < 1:
        problems.append("concurrency must be at least 1")

    from .runtime import AUTO as _AUTO, RUNTIMES as _RUNTIMES

    if config.runtime not in _RUNTIMES + (_AUTO,):
        problems.append(
            f"runtime must be one of {', '.join(_RUNTIMES + (_AUTO,))}, "
            f"not {config.runtime!r}")
    return problems
