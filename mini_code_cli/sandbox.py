"""mini_code_cli.sandbox -- enforcement of the ``--allowed-dir`` sandbox.

The agent's filesystem tools may only touch paths inside the directory the user
passed as ``--allowed-dir``. We use :func:`os.path.realpath` to resolve symlinks
and ``..`` so path-traversal or symlink tricks cannot escape, and we fail closed
when the answer is ambiguous (e.g. different drives on Windows).
"""
import os


def is_within_allowed_dir(file_path, allowed_dir):
    """Return True if the resolved ``file_path`` lives inside ``allowed_dir``.

    Uses ``os.path.realpath`` to resolve symlinks and ``..`` so path-traversal
    or symlink tricks cannot escape the allowed directory.
    """
    target = os.path.realpath(file_path)
    root = os.path.realpath(allowed_dir)
    try:
        return os.path.commonpath([target, root]) == root
    except (ValueError, TypeError):
        # Different drives (Windows) or a non-absolute path: fail closed.
        return False


# Maps each filesystem tool to the names of its path-bearing arguments, so the
# --allowed-dir sandbox can be enforced uniformly. Keeping this in one place
# means a future tool cannot silently bypass the sandbox by using a different
# argument name (e.g. ``glob``'s ``pattern`` instead of ``file_path``).
_FILE_TOOL_PATH_ARGS = {
    "read_file": ["file_path"],
    "write_file": ["file_path"],
    "grep": ["file_path"],
    "glob": ["pattern"],
}


def tool_paths_are_allowed(name, tool_params, allowed_dir):
    """Return True if every filesystem path argument of ``name`` stays inside
    ``allowed_dir``.

    Tools not listed in ``_FILE_TOOL_PATH_ARGS`` have no path to check and pass
    through (return True). A single path argument may itself be a list (e.g. a
    batch of files), so each entry is validated.
    """
    path_args = _FILE_TOOL_PATH_ARGS.get(name, [])
    for arg_name in path_args:
        value = tool_params.get(arg_name, "")
        values = value if isinstance(value, (list, tuple)) else [value]
        for path in values:
            if path and not is_within_allowed_dir(path, allowed_dir):
                return False
    return True
