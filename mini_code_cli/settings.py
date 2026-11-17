"""Session-local settings: load, merge, validate and persist.

A session owns one small JSON file next to its history::

    <cache-dir>/<session-id>/settings.json

There is deliberately no global profile.  A bare ``mini-code`` imports the
settings of the most recent session (the existing ``.mini/last_session.json``
pointer), re-checks the saved model/key, and asks the provider/model questions
before a new session.  The file is mode 0600 because it may contain API keys
for one or more endpoint/model pairs.

Precedence is: explicit CLI arguments, environment, the previous session's
settings, then built-in defaults.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

from . import compaction as compaction_mod
from . import planning
from . import ui
from . import utils
from .config import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MODEL_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_TIMEOUT,
)
from .persistence import find_last_session_dir, resolve_session_dir

SETTINGS_FILENAME = "settings.json"
SETTINGS_VERSION = 1

#: Model/key validation is a best-effort startup probe; keep it short so an
#: unreachable server does not delay an interactive session for long.
MODEL_CHECK_TIMEOUT = 5


def key_id(url, model):
    """Stable id for a key scoped to one endpoint + model pair."""
    return f"{str(url or '').rstrip('/')}|{model or ''}"


@dataclass(frozen=True)
class SettingSpec:
    """One user-editable setting.

    ``cache`` is shown by ``/settings`` and decides whether applying the value
    invalidates the LLM prompt cache: ``none``, ``prompt``, ``tools``, or
    ``model``.  A setting that affects tools also affects the generated prompt.
    """

    name: str                         # CLI-ish name used by /settings
    attr: str                         # argparse dest
    kind: str                         # str | int | float | bool | choice
    default: Any
    choices: tuple[str, ...] = ()
    mutable: bool = True
    cache: str = "none"


def _specs():
    return (
        SettingSpec("model", "model", "str", None, cache="model"),
        SettingSpec("url", "url", "str", DEFAULT_MODEL_URL, cache="model"),
        SettingSpec("api-key", "api_key", "str", "", mutable=True,
                    cache="model"),
        SettingSpec("temperature", "temperature", "float",
                    DEFAULT_TEMPERATURE),
        SettingSpec("max-tokens", "max_tokens", "int", None),
        SettingSpec("max-context-tokens", "max_context_tokens", "int", None),
        SettingSpec("compaction-strategy", "compaction_strategy", "choice",
                    compaction_mod.DEFAULT_COMPACTION_STRATEGY,
                    choices=compaction_mod.strategy_choices()),
        SettingSpec("plan-strategy", "plan_strategy", "choice",
                    planning.DEFAULT_PLAN_STRATEGY,
                    choices=planning.PLAN_STRATEGIES, cache="prompt"),
        SettingSpec("allow-nudge", "allow_nudge", "bool", False),
        SettingSpec("nudge-limit", "nudge_limit", "int", 1),
        SettingSpec("tool-timeout", "tool_timeout", "int",
                    DEFAULT_TOOL_TIMEOUT),
        SettingSpec("llm-timeout", "llm_timeout", "int", DEFAULT_HTTP_TIMEOUT),
        SettingSpec("verbose", "verbose", "bool", False),
        SettingSpec("plan", "plan", "bool", False, cache="prompt"),
        SettingSpec("enable-shell", "enable_shell", "bool", False,
                    cache="tools"),
        SettingSpec("enable-subagents", "enable_subagents", "bool", False,
                    cache="tools"),
        SettingSpec("enable-skills", "enable_skills", "bool", False,
                    cache="tools"),
        SettingSpec("sudo", "sudo", "bool", False),
        SettingSpec("ask-permission", "ask_permission", "bool", False),
        SettingSpec("background", "background", "bool", False, cache="tools"),
        SettingSpec("scratchpad", "scratchpad", "bool", False, cache="prompt"),
        SettingSpec("auto-mode", "auto_mode", "bool", False),
    )


SPECS = _specs()
BY_NAME = {spec.name: spec for spec in SPECS}
ALIASES = {spec.attr: spec.name for spec in SPECS}
ALIASES.update({"skills": "enable-skills", "subagents": "enable-subagents",
                "ask_permission": "ask-permission"})

#: Settings that require a new process/session and are shown by /settings.
IMMUTABLE = (
    ("tui", "fixed-screen UI"),
    ("allowed-dir", "file-tool sandbox"),
    ("cache-dir", "session storage"),
    ("restore-session", "restore target"),
    ("load-last-session", "resume mode"),
    ("prompt", "initial prompt"),
    ("system-prompt", "custom system prompt"),
    ("agent-md", "extra agent instructions"),
)


def _bool_value(raw):
    text = str(raw).strip().lower()
    if text in ("on", "true", "yes", "y", "1"):
        return True
    if text in ("off", "false", "no", "n", "0"):
        return False
    raise ValueError("expected on/off, true/false or 1/0")


def parse_value(spec: SettingSpec, raw):
    """Parse a user-supplied value for ``spec``; raises ValueError."""
    if spec.kind == "bool":
        return _bool_value(raw)
    if spec.kind == "int":
        value = int(str(raw).strip())
        if value < 0:
            raise ValueError("must be >= 0")
        return value
    if spec.kind == "float":
        value = float(str(raw).strip())
        if value < 0:
            raise ValueError("must be >= 0")
        return value
    if spec.kind == "choice":
        text = str(raw).strip()
        if text not in spec.choices:
            raise ValueError("choose one of: " + ", ".join(spec.choices))
        return text
    return str(raw).strip()


def format_value(spec: SettingSpec, value):
    if spec.name == "api-key":
        if not value:
            return "(not set)"
        return value[:3] + "…" + value[-4:] if len(value) > 8 else "***"
    if value is None:
        return "(derived)"
    if spec.kind == "bool":
        return "on" if value else "off"
    return str(value)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, mode=0o700, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class Settings:
    """The values for one session, plus the resolved API-key reference."""

    values: dict[str, Any] = field(default_factory=dict)
    api_key: str = ""
    api_key_env: str = ""
    persist_api_key: bool = False
    #: Stored keys, keyed by ``url|model``. Values may be in memory only when
    #: the user declined to persist them; ``persisted_key_ids`` decides what
    #: ``save`` writes. Environment references are not secrets and are always
    #: persisted as a reference.
    api_keys: dict[str, str] = field(default_factory=dict)
    api_key_envs: dict[str, str] = field(default_factory=dict)
    persisted_key_ids: set[str] = field(default_factory=set)
    #: True when the active key was supplied explicitly (CLI/typed) in this
    #: process, so selecting a model must keep it rather than replace it with
    #: an older stored key for that model.
    key_entered: bool = False
    #: True when the wizard already displayed the key prompt in this run (so a
    #: skip is not immediately followed by the same question again).
    key_asked: bool = False
    enabled: bool = True
    #: Set when the startup wizard already asked this process, so the
    #: new-session path does not ask the same questions a second time.
    startup_asked: bool = False
    #: Whether the model came from an explicit ``--model`` (not a saved value).
    model_explicit: bool = False

    def __post_init__(self):
        for spec in SPECS:
            self.values.setdefault(spec.attr, spec.default)

    # -- API keys by endpoint/model ---------------------------------------
    def key_for(self, url=None, model=None):
        """Return ``(key, env_name, persisted)`` for one endpoint/model."""
        url = self.values.get("url") if url is None else url
        model = self.values.get("model") if model is None else model
        kid = key_id(url, model)
        env = self.api_key_envs.get(kid, "")
        if env:
            return os.environ.get(env, ""), env, False
        key = self.api_keys.get(kid, "")
        return key, "", kid in self.persisted_key_ids

    def remember_key(self, url=None, model=None):
        """Associate the currently active key with one endpoint/model."""
        url = self.values.get("url") if url is None else url
        model = self.values.get("model") if model is None else model
        kid = key_id(url, model)
        if self.api_key_env:
            self.api_key_envs[kid] = self.api_key_env
            self.api_keys.pop(kid, None)
            self.persisted_key_ids.discard(kid)
        elif self.api_key:
            self.api_keys[kid] = self.api_key
            self.api_key_envs.pop(kid, None)
            if self.persist_api_key:
                self.persisted_key_ids.add(kid)
            else:
                self.persisted_key_ids.discard(kid)

    def use_key(self, key, env_name="", persist=False):
        """Make ``key`` active and remember it for the current model."""
        self.api_key = str(key or "")
        self.api_key_env = str(env_name or "")
        self.persist_api_key = bool(persist and self.api_key
                                    and not self.api_key_env)
        self.key_entered = bool(self.api_key)
        self.remember_key()

    def load_key_for(self, url=None, model=None):
        """Load a stored key for one endpoint/model; False if none exists."""
        key, env, persisted = self.key_for(url, model)
        if not key and not env:
            return False
        self.api_key = key
        self.api_key_env = env
        self.persist_api_key = persisted
        self.key_entered = False
        return True

    def forget_key(self, url=None, model=None):
        """Drop the stored key for one endpoint/model."""
        kid = key_id(
            self.values.get("url") if url is None else url,
            self.values.get("model") if model is None else model)
        self.api_keys.pop(kid, None)
        self.api_key_envs.pop(kid, None)
        self.persisted_key_ids.discard(kid)

    def key_is_valid(self, timeout=MODEL_CHECK_TIMEOUT):
        """Probe the active key: True/False, or None when unknown."""
        if not self.api_key:
            return None
        url = self.values.get("url") or DEFAULT_MODEL_URL
        return utils.probe_api_key(url, self.api_key, timeout=timeout)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_args(cls, args):
        values = {}
        for spec in SPECS:
            if spec.attr == "api_key":
                continue
            value = getattr(args, spec.attr, spec.default)
            # ``None`` is the runtime "plan mode off" marker; the setting
            # itself keeps the concrete strategy that plan mode would use.
            if spec.attr == "plan_strategy" and value is None:
                value = spec.default
            values[spec.attr] = value
        if getattr(args, "plan_strategy", None):
            values["plan"] = True
        key = getattr(args, "api_key", None) or ""
        env_name = ""
        if not key:
            key = os.environ.get("MINI_CODE_API_KEY", "")
            if key:
                env_name = "MINI_CODE_API_KEY"
        result = cls(values=values)
        if key:
            result.use_key(key, env_name=env_name)
        result.model_explicit = bool(getattr(args, "model", None))
        return result

    @classmethod
    def load(cls, session_dir):
        """Load a session's settings; a missing/corrupt file yields defaults."""
        path = os.path.join(session_dir, SETTINGS_FILENAME)
        payload = _read_json(path)
        if not payload:
            return cls()
        values = payload.get("values")
        if not isinstance(values, dict):
            return cls()
        result = cls()
        for spec in SPECS:
            if spec.attr == "api_key":
                continue
            if spec.attr in values:
                try:
                    result.values[spec.attr] = parse_value(spec, values[spec.attr])
                except (TypeError, ValueError):
                    pass

        api_keys = payload.get("api_keys")
        if isinstance(api_keys, dict):
            result.api_keys = {
                str(kid): str(key) for kid, key in api_keys.items()
                if key and not str(kid).startswith("_")}
            result.persisted_key_ids = set(result.api_keys)
        api_key_envs = payload.get("api_key_envs")
        if isinstance(api_key_envs, dict):
            result.api_key_envs = {
                str(kid): str(env) for kid, env in api_key_envs.items()
                if env and not str(kid).startswith("_")}
        for kid in list(result.api_key_envs):
            if kid in result.api_keys:
                # A concrete key is stronger than an environment reference.
                result.api_key_envs.pop(kid)

        # Older files stored one active key/environment reference at the top
        # level. Migrate it to the endpoint/model map unless the map already
        # has an entry for the saved model.
        key, env, _persisted = result.key_for()
        if not key and not env:
            if payload.get("api_key"):
                result.use_key(str(payload["api_key"]), persist=True)
            elif payload.get("api_key_env"):
                result.use_key(
                    os.environ.get(str(payload["api_key_env"]), ""),
                    env_name=str(payload["api_key_env"]))
        result.load_key_for()
        result.key_entered = False
        return result

    def save(self, session_dir):
        """Persist this session's settings with mode 0600."""
        if not self.enabled:
            return None
        # Keys are remembered when supplied/loaded; save only persists the
        # entries the user opted to write. Re-associating the active key here
        # would copy an explicit CLI key onto whatever model is current.
        payload = {"version": SETTINGS_VERSION, "values": dict(self.values)}
        literal = {kid: key for kid, key in self.api_keys.items()
                   if kid in self.persisted_key_ids and key}
        envs = {kid: env for kid, env in self.api_key_envs.items() if env}
        if literal:
            payload["api_keys"] = literal
        if envs:
            payload["api_key_envs"] = envs
        path = os.path.join(session_dir, SETTINGS_FILENAME)
        _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    # -- merge ------------------------------------------------------------
    @classmethod
    def resolve(cls, args, saved=None):
        """Merge CLI, environment and the previous session's settings."""
        saved = saved or cls()
        result = cls()
        result.api_keys = dict(saved.api_keys)
        result.api_key_envs = dict(saved.api_key_envs)
        result.persisted_key_ids = set(saved.persisted_key_ids)
        result.model_explicit = bool(getattr(args, "model", None))
        explicit = {
            "model": result.model_explicit,
            "url": getattr(args, "url", DEFAULT_MODEL_URL) != DEFAULT_MODEL_URL,
            "temperature": getattr(args, "temperature", DEFAULT_TEMPERATURE)
            != DEFAULT_TEMPERATURE,
            "max_tokens": getattr(args, "max_tokens", None) is not None,
            "max_context_tokens": getattr(args, "max_context_tokens", None)
            is not None,
            "compaction_strategy": getattr(
                args, "compaction_strategy",
                compaction_mod.DEFAULT_COMPACTION_STRATEGY)
            != compaction_mod.DEFAULT_COMPACTION_STRATEGY,
            "plan_strategy": getattr(args, "plan_strategy", None) is not None,
            "nudge_limit": getattr(args, "nudge_limit", 1) != 1,
            "tool_timeout": getattr(args, "tool_timeout", DEFAULT_TOOL_TIMEOUT)
            != DEFAULT_TOOL_TIMEOUT,
            "llm_timeout": getattr(args, "llm_timeout", DEFAULT_HTTP_TIMEOUT)
            != DEFAULT_HTTP_TIMEOUT,
        }
        for spec in SPECS:
            if spec.attr == "api_key":
                continue
            if spec.kind == "bool":
                cli = getattr(args, spec.attr, spec.default)
                if cli:
                    result.values[spec.attr] = True
                elif spec.attr in saved.values:
                    result.values[spec.attr] = saved.values[spec.attr]
                else:
                    result.values[spec.attr] = spec.default
            elif explicit.get(spec.attr):
                result.values[spec.attr] = getattr(args, spec.attr)
            elif spec.attr in saved.values:
                value = saved.values[spec.attr]
                if spec.attr == "plan_strategy" and value is None:
                    value = spec.default
                result.values[spec.attr] = value
            else:
                result.values[spec.attr] = spec.default

        # An explicit --plan-strategy implies plan mode, matching the CLI
        # parser. A *saved* strategy does not: plan mode is its own on/off
        # setting, and the strategy stays stored while plan mode is off.
        if explicit.get("plan_strategy"):
            result.values["plan"] = True

        # API key: CLI > environment > stored key for this url/model > legacy
        # previous-session value > empty.
        url = result.values.get("url")
        model = result.values.get("model")
        if getattr(args, "api_key", None):
            result.use_key(str(args.api_key))
        else:
            env_name = "MINI_CODE_API_KEY"
            env_key = os.environ.get(env_name, "")
            if env_key:
                result.use_key(env_key, env_name=env_name)
            else:
                saved_key, saved_env, saved_persist = saved.key_for(url, model)
                if saved_key or saved_env:
                    result.use_key(saved_key, saved_env, saved_persist)
                    result.key_entered = False
                elif (not saved.api_keys and not saved.api_key_envs
                      and (saved.api_key or saved.api_key_env)):
                    # Legacy Settings objects that predate the per-model map.
                    result.use_key(saved.api_key, saved.api_key_env,
                                   saved.persist_api_key)
                    result.key_entered = False
        return result

    def apply_to_args(self, args):
        for spec in SPECS:
            if spec.attr == "api_key":
                continue
            # max-tokens is derived from the model when the user did not set it.
            # A settings snapshot stores None in that case; do not overwrite an
            # already resolved session value with None.
            if (spec.attr == "max_tokens"
                    and self.values.get(spec.attr) is None
                    and getattr(args, "max_tokens", None) is not None):
                continue
            if spec.attr == "plan_strategy" and not self.values.get("plan"):
                # None is the "plan mode off" marker the parser/loop already
                # understand; a stored default must not silently turn plan mode
                # on for every session.
                setattr(args, "plan_strategy", None)
                continue
            setattr(args, spec.attr, self.values.get(spec.attr, spec.default))
        args.api_key = self.api_key or None
        return args

    def capture_from_args(self, args):
        """Refresh values after the session resolved derived values.

        ``plan_strategy`` is deliberately not recaptured: the runtime uses
        ``None`` there to mean "plan mode is off", while the setting itself
        keeps the strategy that will be used when plan mode is enabled. Without
        this the UI showed the strategy as ``(derived)`` even though it has a
        concrete value.
        """
        for spec in SPECS:
            if spec.attr in ("api_key", "max_tokens", "plan_strategy"):
                continue
            setattr_value = getattr(args, spec.attr, spec.default)
            self.values[spec.attr] = setattr_value

    # -- UI ---------------------------------------------------------------
    def rows(self):
        rows = []
        for index, spec in enumerate(SPECS, 1):
            value = self.api_key if spec.name == "api-key" else self.values.get(
                spec.attr, spec.default)
            formatted = format_value(spec, value)
            is_default = value == spec.default
            display = formatted
            # "(not set)" and "(derived)" already say what the value is; a
            # trailing "(default)" there would just be noise.
            if is_default and formatted not in ("(not set)", "(derived)"):
                display += " (default)"
            rows.append({
                "number": index,
                "name": spec.name,
                "value": formatted,
                "display": display,
                "is_default": is_default,
                "raw": value,
                "mutable": spec.mutable,
                "cache": spec.cache,
                "spec": spec,
            })
        return rows

    def by_number(self, number):
        for row in self.rows():
            if row["number"] == int(number):
                return row
        raise ValueError(f"no setting number {number}")


def interactive_startup(args):
    """Whether startup may ask the user questions.

    A one-shot ``--prompt`` run, a non-TTY (pipes, CI) and ``--no-settings``
    must never block on input.
    """
    return (args.prompt is None and sys.stdin.isatty()
            and not getattr(args, "no_settings", False))


def _source_session(args):
    if getattr(args, "restore_session", None):
        return resolve_session_dir(args.restore_session, args.cache_dir)
    return find_last_session_dir(args.allowed_dir)


def _ask(console, prompt_text):
    answer = console.read_line(prompt_text)
    return "" if answer is None else answer.strip()


DEEPSEEK_URL = "https://api.deepseek.com"

#: Stable DeepSeek model aliases used when ``/v1/models`` cannot be queried
#: (for example before an API key is entered).
DEEPSEEK_MODELS = ("deepseek-chat", "deepseek-reasoner")

def _provider_index(settings):
    """Map the configured URL to the wizard's 1-based provider choice."""
    url = str(settings.values.get("url") or DEFAULT_MODEL_URL).rstrip("/")
    if url == str(DEFAULT_MODEL_URL).rstrip("/"):
        return 1
    if url == DEEPSEEK_URL.rstrip("/"):
        return 2
    return 3


def model_is_valid(settings, timeout=MODEL_CHECK_TIMEOUT):
    """Whether the configured model is still advertised by its server.

    ``True`` also means "could not check": if the probe fails (server down,
    no key, network error) the saved model is kept rather than nagging the
    user every startup. A reachable server that does not list the model is
    the case that re-opens the provider/model question.
    """
    model = settings.values.get("model")
    if not model:
        return False
    url = str(settings.values.get("url") or DEFAULT_MODEL_URL)
    served = utils.list_served_models(
        url, api_key=settings.api_key or None, timeout=timeout)
    if served is None:
        return True
    return model in served


def _ask_key(console, settings, provider):
    """Prompt for a key and ask whether to persist it (default: no).

    Returns True when a key is available afterwards.
    """
    if provider == "localhost" and not settings.api_key:
        # Local vLLM normally ignores the key; an env/CLI key is optional.
        return False
    if settings.api_key:
        return True
    if provider == "deepseek":
        console.note(
            "DeepSeek API key: create one at "
            "https://platform.deepseek.com/api_keys or set "
            "MINI_CODE_API_KEY.", "info")
        prompt = "DeepSeek API key (hidden; Enter to skip): "
    else:
        console.note(
            "API key for this server (leave blank if it does not need one).",
            "info")
        prompt = "API key (hidden; Enter to skip): "
    # ``read_secret`` masks the typed value in the fixed TUI and uses getpass
    # on the ordinary console.
    settings.key_asked = True
    key = console.read_secret(prompt)
    if key is None:
        return False
    key = key.strip()
    if not key:
        return False
    answer = _ask(
        console,
        "Store this key? Type 'y' to write it to settings.json (mode 0600), "
        "or Enter to keep it in memory for this session only: ").lower()
    settings.use_key(key, persist=answer in ("y", "yes"))
    return True


def _deepseek_model_choices(settings):
    """Live DeepSeek model ids, falling back to the stable aliases."""
    served = utils.list_served_models(
        DEEPSEEK_URL, api_key=settings.api_key or None,
        timeout=MODEL_CHECK_TIMEOUT)
    models = []
    for name in list(served or ()) + list(DEEPSEEK_MODELS):
        if name and name not in models:
            models.append(name)
    return models


def _ask_deepseek_model(settings, console):
    """Ask which DeepSeek model to use, preferring the live served list."""
    models = _deepseek_model_choices(settings)
    current = settings.values.get("model")
    default = models.index(current) + 1 if current in models else 1
    choice = console.choose("DeepSeek model to use", models, default=default)
    if choice is None:
        settings.values["model"] = current or models[0]
    else:
        settings.values["model"] = models[choice - 1]


def provider_name(settings):
    return {1: "localhost", 2: "deepseek", 3: "custom"}[
        _provider_index(settings)]


def _finish_key_for_model(settings, console, provider, ask_if_missing=True):
    """Load/remember the key for the selected model and check it.

    A key entered in this wizard (or supplied by the CLI/environment) belongs
    to the selected model. Otherwise a key stored for that exact endpoint/model
    is loaded. When no key exists yet, ``ask_if_missing`` controls whether the
    user is prompted here (DeepSeek already asked before the model list).
    """
    if settings.key_entered:
        # A key supplied in this run belongs to the model just selected.
        settings.remember_key()
    elif settings.load_key_for():
        # A stored key exists for this exact endpoint/model.
        pass
    else:
        # Do not silently reuse another model's key.
        settings.api_key = ""
        settings.api_key_env = ""
        settings.persist_api_key = False
    if ask_if_missing and not (settings.api_key or settings.api_key_env):
        _ask_key(console, settings, provider)
    _validate_key(console, settings, provider)


def _validate_key(console, settings, provider):
    """Reject-and-retry an API key that the server explicitly refuses."""
    if not settings.api_key:
        return
    for attempt in range(3):
        valid = settings.key_is_valid()
        if valid is not False:
            return
        if attempt == 2:
            console.note(
                "The API key is still rejected; continuing with it.", "warn")
            return
        console.note(
            "That API key was rejected by the server. Enter it again, or "
            "press Enter to skip.", "warn")
        settings.forget_key()
        settings.api_key = ""
        settings.api_key_env = ""
        settings.persist_api_key = False
        if not _ask_key(console, settings, provider):
            return
        settings.remember_key()


def first_run(settings, console):
    """Ask only the minimum questions for a new (or invalidated) model setup.

    The current provider is the default, so re-asking after an invalid model
    only needs Enter when the provider itself is unchanged.
    """
    current_url = str(settings.values.get("url") or DEFAULT_MODEL_URL)
    current_model = settings.values.get("model")
    default_provider = _provider_index(settings)
    # ``choose`` renders the numbered list on the plain console and an
    # arrow-key overlay inside the fixed TUI.
    choice = console.choose(
        "First-run settings: choose a model provider",
        ["localhost", "deepseek", "custom URL"],
        default=default_provider,
    ) or default_provider
    if choice == 1:
        settings.values["url"] = DEFAULT_MODEL_URL
        settings.values["model"] = None
        detected = utils.determine_vllm_model(
            DEFAULT_MODEL_URL, timeout=10)
        if detected:
            settings.values["model"] = detected
            console.note(f"Detected local model: {detected}", "ok")
        else:
            console.note("No local model detected; set model later with /settings.",
                         "warn")
        _finish_key_for_model(settings, console, "localhost")
    elif choice == 2:
        settings.values["url"] = DEEPSEEK_URL
        # Ask for the key first so the live model list can be queried with it;
        # if the key is skipped the stable aliases are offered instead.
        _ask_key(console, settings, "deepseek")
        _ask_deepseek_model(settings, console)
        # The key prompt already happened above; now load/remember the key
        # for the selected model and verify it.
        _finish_key_for_model(
            settings, console, "deepseek",
            ask_if_missing=not settings.key_asked)
    else:
        settings.values["url"] = (
            _ask(console, f"URL [{current_url}]: ") or current_url)
        label = current_model or "server default"
        settings.values["model"] = (
            _ask(console, f"Model [{label}]: ") or current_model)
        _finish_key_for_model(settings, console, "custom")
    return settings


def resolve_for_startup(args, console=None):
    """Load previous settings, fill missing values, and apply them to ``args``."""
    console = console or ui.console
    if getattr(args, "no_settings", False):
        settings = Settings.from_args(args)
        settings.enabled = False
        settings.apply_to_args(args)
        return settings

    source = _source_session(args)
    saved = Settings.load(source) if source else Settings()
    settings = Settings.resolve(args, saved)

    interactive = interactive_startup(args)
    missing_model = not (settings.values.get("model")
                         or settings.values.get("url") != DEFAULT_MODEL_URL)
    invalid_model = False
    # An explicit --model is authoritative; do not second-guess it (and avoid
    # an extra network probe on a normal CLI run).
    if (interactive and not missing_model and not getattr(args, "model", None)):
        invalid_model = not model_is_valid(settings)
    if interactive and (missing_model or invalid_model):
        if invalid_model:
            console.note(
                f"Model {settings.values.get('model')!r} is not served at "
                f"{settings.values.get('url')}; choosing again.", "warn")
        first_run(settings, console)
        settings.startup_asked = True
    elif (interactive
          and settings.values.get("url") != DEFAULT_MODEL_URL
          and not settings.api_key):
        _ask_key(console, settings, "custom")
    if interactive and not settings.startup_asked:
        # Resuming (or a valid model on an explicit URL): the model was not
        # re-selected above, so check the stored/loaded key for it now.
        _validate_key(console, settings, provider_name(settings))

    settings.apply_to_args(args)
    return settings
