"""mini_code_cli -- package metadata (the version shown in the welcome banner).

The version is read from ``pyproject.toml`` first: that file is the source of
truth when running from a checkout, and the *installed* metadata is often stale
(a plain ``pip install -e .`` keeps the version it was installed at, so bumping
``pyproject.toml`` would leave the banner reporting an old number). If the file
cannot be read, fall back to the installed distribution's metadata and finally
to ``"unknown"`` so importing the package can never fail over a version lookup.
"""

__all__ = ["__version__"]


def _read_version():
    """Best-effort package version: pyproject.toml, then installed metadata."""
    try:
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            import tomllib  # Python >= 3.11
        except ImportError:  # pragma: no cover - older interpreters
            tomllib = None
        if tomllib is not None:
            try:
                with open(pyproject, "rb") as f:
                    return str(tomllib.load(f)["project"]["version"])
            except Exception:
                pass
    except Exception:
        pass
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("mini-code-cli")
    except Exception:
        return "unknown"


__version__ = _read_version()
