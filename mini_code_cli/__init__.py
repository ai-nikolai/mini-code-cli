"""mini_code_cli -- package metadata (the version shown in the welcome banner).

The version is read from ``pyproject.toml`` first: that file is the source of
truth when running from a checkout, and the *installed* metadata is often stale
(a plain ``pip install -e .`` keeps the version it was installed at, so bumping
``pyproject.toml`` would leave the banner reporting an old number). If the file
cannot be read, fall back to the installed distribution's metadata and finally
to ``"unknown"`` so importing the package can never fail over a version lookup.
"""

__all__ = ["__version__"]


def _version_from_file(path):
    """Return the ``[project] version`` in ``path``.

    ``tomllib`` only exists from Python 3.11, and the scan below is not a
    nicety for 3.10: without it this function used to skip the file entirely on
    that interpreter and report the *installed* metadata instead, so a checkout
    with a stale editable install showed the wrong version in the banner (found
    on the 3.10 CI leg). Both paths read the same key, and the section tracking
    matters: a ``version`` key in some other table must not win.
    """
    try:
        import tomllib  # Python >= 3.11
    except ImportError:  # pragma: no cover - exercised on the 3.10 CI leg
        tomllib = None
    if tomllib is not None:
        with open(path, "rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    section = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                section = stripped
            elif section == "[project]" and stripped.startswith("version"):
                return stripped.split("=", 1)[1].strip().strip('"')
    raise KeyError(f"no version in the [project] table of {path}")


def _read_version():
    """Best-effort package version: pyproject.toml, then installed metadata."""
    try:
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        return _version_from_file(pyproject)
    except Exception:
        pass
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("mini-code-cli")
    except Exception:
        return "unknown"


__version__ = _read_version()
