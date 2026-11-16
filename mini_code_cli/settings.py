"""Session-local settings: load, merge, validate and persist.

A session owns one small JSON file next to its history::

    <cache-dir>/<session-id>/settings.json

There is deliberately no global profile.  A bare ``mini-code`` imports the
settings of the most recent session (the existing ``.mini/last_session.json``
pointer), asks only for values that cannot be inferred, and writes them into
the new session.  The file is mode 0600 because it may contain an API key.

Precedence is: explicit CLI arguments, environment, the previous session's
settings, then built-in defaults.
"""
from __future__ import annotations

import getpass
import json
import os
import stat
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
    SESSION_DIRNAME,
)
from .persistence import find_last_session_dir, resolve_session_dir

SETTINGS_FILENAME = "settings.json"
SETTINGS_VERSION = 1


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
    label: str = ""
    mutable: bool = True
    cache: str = "none"

    def display(self):
        return self.label or self.name


def _specs():
    return (
        SettingSpec("model", "model", "str", None, cache="model",
                    label="model"),
        SettingSpec("url", "url", "str", DEFAULT_MODEL_URL, cache="model",
                    label="url"),
        SettingSpec("api-key", "api_key", "str", "", mutable=True,
                    cache="model", label="API key"),
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
                    cache="tools", label="shell"),
        SettingSpec("enable-subagents", "enable_subagents", "bool", False,
                    cache="tools", label="sub-agents"),
        SettingSpec("enable-skills", "enable_skills", "bool", False,
                    cache="tools", label="skills"),
        SettingSpec("sudo", "sudo", "bool", False),
        SettingSpec("ask-permission", "ask_permission", "bool", False,
                    label="ask permission"),
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
    enabled: bool = True

    def __post_init__(self):
        for spec in SPECS:
            self.values.setdefault(spec.attr, spec.default)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_args(cls, args):
        values = {}
        for spec in SPECS:
            if spec.attr == "api_key":
                continue
            values[spec.attr] = getattr(args, spec.attr, spec.default)
        if getattr(args, "plan_strategy", None):
            values["plan"] = True
        key = getattr(args, "api_key", None) or ""
        env_name = ""
        if not key:
            key = os.environ.get("MINI_CODE_API_KEY", "")
            if key:
                env_name = "MINI_CODE_API_KEY"
        return cls(values=values, api_key=key, api_key_env=env_name)

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
        if payload.get("api_key"):
            result.api_key = str(payload["api_key"])
            result.persist_api_key = True
        if payload.get("api_key_env"):
            result.api_key_env = str(payload["api_key_env"])
            result.api_key = os.environ.get(result.api_key_env, result.api_key)
        return result

    def save(self, session_dir):
        """Persist this session's settings with mode 0600."""
        if not self.enabled:
            return None
        payload = {"version": SETTINGS_VERSION, "values": dict(self.values)}
        if self.api_key and self.persist_api_key:
            payload["api_key"] = self.api_key
        elif self.api_key_env:
            payload["api_key_env"] = self.api_key_env
        path = os.path.join(session_dir, SETTINGS_FILENAME)
        _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    # -- merge ------------------------------------------------------------
    @classmethod
    def resolve(cls, args, saved=None):
        """Merge CLI, environment and the previous session's settings."""
        saved = saved or cls()
        result = cls()
        explicit = {
            "model": bool(getattr(args, "model", None)),
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
                result.values[spec.attr] = saved.values[spec.attr]
            else:
                result.values[spec.attr] = spec.default

        # A plan strategy without plan mode is meaningless (the CLI parser
        # already implies --plan when the strategy is explicit).
        if result.values.get("plan_strategy") != planning.DEFAULT_PLAN_STRATEGY:
            result.values["plan"] = True

        # API key: CLI > environment > previous session > empty.
        if getattr(args, "api_key", None):
            result.api_key = str(args.api_key)
            result.persist_api_key = False
        else:
            env_name = "MINI_CODE_API_KEY"
            env_key = os.environ.get(env_name, "")
            if env_key:
                result.api_key = env_key
                result.api_key_env = env_name
            elif saved.api_key:
                result.api_key = saved.api_key
                result.persist_api_key = saved.persist_api_key
            elif saved.api_key_env:
                result.api_key_env = saved.api_key_env
                result.api_key = os.environ.get(saved.api_key_env, "")
            elif saved.persist_api_key:
                result.persist_api_key = True
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
        """Refresh values after the session resolved derived values."""
        for spec in SPECS:
            if spec.attr in ("api_key", "max_tokens"):
                continue
            setattr_value = getattr(args, spec.attr, spec.default)
            self.values[spec.attr] = setattr_value

    # -- UI ---------------------------------------------------------------
    def rows(self):
        rows = []
        for index, spec in enumerate(SPECS, 1):
            value = self.api_key if spec.name == "api-key" else self.values.get(
                spec.attr, spec.default)
            rows.append({
                "number": index,
                "name": spec.name,
                "label": spec.display(),
                "value": format_value(spec, value),
                "raw": value,
                "kind": spec.kind,
                "choices": spec.choices,
                "mutable": spec.mutable,
                "cache": spec.cache,
                "spec": spec,
            })
        return rows

    def set_value(self, name, raw):
        """Validate and store one value; returns ``(spec, parsed)``."""
        canonical = BY_NAME.get(name) or BY_NAME.get(ALIASES.get(name, ""))
        if canonical is None:
            raise ValueError(f"unknown setting {name!r}")
        if not canonical.mutable:
            raise ValueError(f"{canonical.name} cannot be changed in this session")
        parsed = parse_value(canonical, raw)
        if canonical.attr == "api_key":
            self.api_key = parsed
            self.persist_api_key = True
        else:
            self.values[canonical.attr] = parsed
        return canonical, parsed

    def by_number(self, number):
        for row in self.rows():
            if row["number"] == int(number):
                return row
        raise ValueError(f"no setting number {number}")


def _interactive(args):
    return (args.prompt is None and sys.stdin.isatty()
            and not getattr(args, "no_settings", False))


def _source_session(args):
    if getattr(args, "restore_session", None):
        return resolve_session_dir(args.restore_session, args.cache_dir)
    return find_last_session_dir(args.allowed_dir)


def _ask(console, prompt_text):
    answer = console.read_line(prompt_text)
    return "" if answer is None else answer.strip()


def _ask_key(console, settings, provider):
    """Prompt for a key and ask whether to persist it (default: no)."""
    if provider == "localhost" and not settings.api_key:
        # Local vLLM normally ignores the key; an env/CLI key is optional.
        return
    if settings.api_key:
        return
    key = getpass.getpass("API key: ").strip()
    if not key:
        return
    settings.api_key = key
    answer = _ask(console, "Store this key in this session's settings.json "
                           "(mode 0600)? [y/N] ").lower()
    settings.persist_api_key = answer in ("y", "yes")


def _first_run(settings, console):
    """Ask only the minimum questions for a new user."""
    console.rule("first-run settings", fg="cyan")
    console.emit("Choose a model provider:")
    console.emit("  1. localhost")
    console.emit("  2. deepseek")
    console.emit("  3. custom URL")
    choice = _ask(console, "> ")
    if choice not in ("1", "2", "3"):
        choice = "1"
    if choice == "1":
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
        _ask_key(console, settings, "localhost")
    elif choice == "2":
        settings.values["url"] = "https://api.deepseek.com"
        settings.values["model"] = "deepseek-chat"
        _ask_key(console, settings, "deepseek")
    else:
        settings.values["url"] = _ask(console, "URL: ") or DEFAULT_MODEL_URL
        settings.values["model"] = _ask(console, "Model: ") or None
        _ask_key(console, settings, "custom")
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

    need_questions = (
        _interactive(args)
        and not (settings.values.get("model")
                 or settings.values.get("url") != DEFAULT_MODEL_URL)
    )
    if need_questions:
        _first_run(settings, console)
    elif (_interactive(args)
          and settings.values.get("url") != DEFAULT_MODEL_URL
          and not settings.api_key):
        _ask_key(console, settings, "custom")

    settings.apply_to_args(args)
    return settings
