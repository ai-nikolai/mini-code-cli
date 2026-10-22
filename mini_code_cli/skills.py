"""mini_code_cli.skills -- the ``.mini`` *skills* feature.

A *skill* is a small, reusable, self-describing capability the agent can load on
demand, kept in a local ``.mini`` directory inside the allowed dir:

    <allowed_dir>/.mini/
        SKILLS.md                 # high-level summary of every skill
        <skill-name>/
            skill.md              # what the skill is / how to use it
            <script>.py           # optional executable Python helper(s)

The agent "works with skills" through four tools (only present when ``--skills``
is enabled):

  * ``list_skills``  -- what skills exist (names + one-line summaries).
  * ``read_skill``   -- read a single skill's ``skill.md`` by name.
  * ``create_skill`` -- create a new skill (its dir + ``skill.md``); this also
                        refreshes the high-level ``SKILLS.md`` summary.
  * ``run_skill``    -- execute one of a skill's Python scripts, with optional
                        command-line args.

Names are sandboxed *by construction*: skill / script names are validated to be
a single safe path segment and then joined under the known ``.mini`` root inside
the allowed dir, so a skill tool can never escape ``--allowed-dir``. A final
``realpath`` containment check enforces that anyway.
"""

import os
import sys
import shlex
import shutil
import subprocess
import functools

from .config import SESSION_DIRNAME

# Default timeout (seconds) for a script run via run_skill.
DEFAULT_TOOL_TIMEOUT = 120

# Layout constants. Keeping them here (and re-exported) means prompts.py and the
# agent loop can refer to the same names rather than hard-coding them twice. The
# directory name itself comes from :mod:`mini_code_cli.config`, the one place
# that defines it (the latest-session pointer lives in the same directory).
SKILLS_DIRNAME = SESSION_DIRNAME
SUMMARY_FILENAME = "SKILLS.md"
SKILL_FILENAME = "skill.md"
SCRIPT_SUFFIX = ".py"

# The tool names this feature contributes to the agent (used by the loop to
# register them and by the sandbox/help wiring).
SKILL_TOOL_NAMES = ("list_skills", "read_skill", "create_skill", "run_skill")


# -- paths -----------------------------------------------------------------
def skills_dir(allowed_dir):
    """The ``.mini`` directory for ``allowed_dir``."""
    return os.path.join(allowed_dir, SKILLS_DIRNAME)


def summary_path(allowed_dir):
    """Path to the high-level ``SKILLS.md`` summary."""
    return os.path.join(skills_dir(allowed_dir), SUMMARY_FILENAME)


def skill_dir(allowed_dir, skill_name):
    """Path to a single skill's sub-directory."""
    return os.path.join(skills_dir(allowed_dir), skill_name)


def skill_md_path(allowed_dir, skill_name):
    """Path to a skill's ``skill.md``."""
    return os.path.join(skill_dir(allowed_dir, skill_name), SKILL_FILENAME)


# -- name safety -----------------------------------------------------------
def _safe_name(name, what="name"):
    """Return ``name`` if it is a safe single path segment, else ``None``.

    Rejects empty names, absolute paths, any path separator and the ``.`` /
    ``..`` special segments, so skill / script names cannot traverse out of the
    ``.mini`` root or address sibling directories. Best-effort: never raises.
    """
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if name in (".", ".."):
        return None
    # Any separator at all means "this is trying to be a path" -> refuse.
    # Reject any path separator on either platform (plus a bare backslash), so
    # a name can never contain a directory component -- it cannot traverse.
    # Any separator (on either platform) plus a bare backslash means
    # "this is trying to be a path" -> refuse, so a name can never hold a
    # directory component and cannot traverse.
    if any(sep in name for sep in (os.sep, "\\", "/")):
        return None
    return name


# -- discovery -------------------------------------------------------------
def _list_skill_names(allowed_dir):
    """Return the sorted names of skills (dirs holding a ``skill.md``)."""
    sd = skills_dir(allowed_dir)
    if not os.path.isdir(sd):
        return []
    out = []
    for entry in sorted(os.listdir(sd)):
        if os.path.isdir(os.path.join(sd, entry)) and \
                os.path.isfile(os.path.join(sd, entry, SKILL_FILENAME)):
            out.append(entry)
    return out


def list_scripts(allowed_dir, skill_name):
    """Return the executable Python script names (without ``.py``) in a skill."""
    safe = _safe_name(skill_name)
    if not safe:
        return []
    base = skill_dir(allowed_dir, safe)
    if not os.path.isdir(base):
        return []
    out = []
    for entry in sorted(os.listdir(base)):
        if entry.endswith(SCRIPT_SUFFIX):
            out.append(entry[: -len(SCRIPT_SUFFIX)])
    return out


def ensure_skills_dir(allowed_dir):
    """Create ``.mini`` and a skeleton ``SKILLS.md`` if they do not exist yet.

    Best-effort and never raises: a skills hiccup must never break the loop.
    """
    try:
        sd = skills_dir(allowed_dir)
        os.makedirs(sd, exist_ok=True)
        if not os.path.exists(summary_path(allowed_dir)):
            _write_summary(allowed_dir, [])
    except OSError:
        pass


def _skill_metadata(allowed_dir, skill_name):
    """Return ``(title, description)`` for a skill read from its ``skill.md``.

    ``description`` is the ``Summary:`` line if the skill has one, else the first
    non-heading line of the file. ``title`` is the first ``#`` heading, else the
    skill name. Best-effort: a missing/unreadable file yields (name, "").
    """
    path = skill_md_path(allowed_dir, skill_name)
    if not os.path.isfile(path):
        return skill_name, ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return skill_name, ""

    description = ""
    title = skill_name
    for line in content.splitlines():
        stripped = line.strip()
        if not title and stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or skill_name
            continue
        if stripped.lower().startswith("summary:"):
            description = stripped.split(":", 1)[1].strip()
            break
        if not stripped or stripped.startswith("#"):
            continue
        description = stripped
        break
    return (title or skill_name), description


# -- tools -----------------------------------------------------------------
def list_skills(allowed_dir):
    """List available skills (name + one-line summary). Creates ``.mini`` first.

    No parameters from the model (``allowed_dir`` is bound at wiring time).
    """
    ensure_skills_dir(allowed_dir)
    skills = _list_skill_names(allowed_dir)
    # Keep the high-level SKILLS.md summary in sync with what is on disk (e.g.
    # scripts dropped into a skill after it was created), so list_skills always
    # reflects the current state.
    if skills:
        _write_summary(allowed_dir, skills)
    if not skills:
        return "No skills yet. Use create_skill to add one."
    lines = ["Available skills:", ""]
    for name in skills:
        _title, desc = _skill_metadata(allowed_dir, name)
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def read_skill(allowed_dir, skill_name):
    """Return the contents of a skill's ``skill.md`` by name.

    If the skill does not exist, returns an error listing the available skills
    so the agent can recover.
    """
    safe = _safe_name(skill_name)
    if not safe:
        return "Error: invalid skill name."
    path = skill_md_path(allowed_dir, safe)
    if not os.path.isfile(path):
        available = _list_skill_names(allowed_dir)
        listing = ", ".join(available) if available else "none"
        return f"Error: no skill named {skill_name!r}. Available skills: {listing}."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"


def create_skill(allowed_dir, skill_name, description, title=None, overwrite=False):
    """Create a new skill: its sub-directory and ``skill.md``.

    Refreshes the high-level ``SKILLS.md`` summary afterwards so the new skill
    shows up in list_skills. Refuses to clobber an existing skill unless
    ``overwrite`` is set. Best-effort: never raises, returns an Error string.
    """
    safe = _safe_name(skill_name)
    if not safe:
        return "Error: invalid skill name. Use a short, single-segment name."
    if not description:
        return "Error: a description is required."

    sdir = skill_dir(allowed_dir, safe)
    mpath = skill_md_path(allowed_dir, safe)
    if os.path.exists(mpath) and not overwrite:
        return (f"Error: skill {safe!r} already exists (use overwrite to replace "
                f"it).")

    try:
        os.makedirs(sdir, exist_ok=True)
        with open(mpath, "w", encoding="utf-8") as f:
            f.write(_skill_md_template(title or safe, description))
        _write_summary(allowed_dir, _list_skill_names(allowed_dir))
        return f"Created skill {safe!r} at {mpath}"
    except OSError as e:
        return f"Error: {e}"


def delete_skill(allowed_dir, skill_name):
    """Delete a skill's sub-directory (and drop it from SKILLS.md).

    Kept minimal and only surfaced if ever wanted as a tool; best-effort.
    """
    safe = _safe_name(skill_name)
    if not safe:
        return "Error: invalid skill name."
    sdir = skill_dir(allowed_dir, safe)
    if not os.path.isdir(sdir):
        return f"Error: no skill named {skill_name!r}."
    try:
        shutil.rmtree(sdir)
        _write_summary(allowed_dir, _list_skill_names(allowed_dir))
        return f"Deleted skill {safe!r}."
    except OSError as e:
        return f"Error: {e}"


def run_skill(allowed_dir, skill_name, script_name, args=None):
    """Execute one of a skill's Python scripts and return its output.

    ``args`` is an optional string of command-line arguments, split with
    :func:`shlex.split` and passed as ``argv`` (never through a shell, so there
    is no shell-injection surface). The script is run with the same interpreter
    that runs mini-code (``sys.executable``). The script path is resolved under
    the skill's directory and checked to stay inside it.
    """
    safe = _safe_name(skill_name)
    script = _safe_name(script_name)
    if not safe or not script:
        return "Error: invalid skill or script name."
    if not script.endswith(SCRIPT_SUFFIX):
        script += SCRIPT_SUFFIX

    base = os.path.realpath(skill_dir(allowed_dir, safe))
    script_path = os.path.realpath(os.path.join(base, script))
    # Containment: the script must live directly inside the skill dir.
    if os.path.dirname(script_path) != base:
        return "Error: script path escapes the skill directory."
    if not os.path.isfile(script_path):
        available = ", ".join(list_scripts(allowed_dir, safe)) or "none"
        return (f"Error: no script {script!r} in skill {safe!r}. "
                f"Available scripts: {available}.")

    argv = shlex.split(args) if args else []
    cmd = [sys.executable, script_path] + argv
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=DEFAULT_TOOL_TIMEOUT, check=False)
    except Exception as e:
        return f"Error: {e}"
    out = result.stdout
    if result.stderr:
        out += "\nSTDERR:\n" + result.stderr
    return out


# -- summary ---------------------------------------------------------------
def _skill_md_template(title, description):
    return (
        f"# {title}\n\n"
        f"Summary: {description}\n\n"
        "## How to use\n"
        f"{description}\n\n"
        "## Scripts\n"
        "- (none yet)\n"
    )


def _write_summary(allowed_dir, skill_names):
    """Write the high-level ``SKILLS.md`` summary from the given skill names."""
    lines = [
        "# Skills",
        "",
        "This is the high-level summary of the skills kept in the local "
        "`.mini` directory. Each skill lives in its own sub-directory with a "
        "`skill.md` (what it is / how to use it) and optional executable "
        "Python scripts. Use `list_skills` / `read_skill` to consult them, "
        "`create_skill` to add one, and `run_skill` to run a skill's scripts.",
        "",
    ]
    if not skill_names:
        lines += ["No skills yet.", ""]
    for name in skill_names:
        _title, desc = _skill_metadata(allowed_dir, name)
        lines.append(f"## {name}")
        lines.append(desc or "(no summary)")
        scripts = list_scripts(allowed_dir, name)
        if scripts:
            lines.append("")
            lines.append(f"Scripts: {', '.join(scripts)}")
        lines.append("")
    try:
        with open(summary_path(allowed_dir), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        pass


# -- wiring (tool schemas + bound functions) -------------------------------
def _skill_functions(allowed_dir):
    """{name: functools.partial} for the four skill tools, allowed_dir bound."""
    return {
        "list_skills": functools.partial(list_skills, allowed_dir),
        "read_skill": functools.partial(read_skill, allowed_dir),
        "create_skill": functools.partial(create_skill, allowed_dir),
        "run_skill": functools.partial(run_skill, allowed_dir),
    }


def get_skills_tools(allowed_dir):
    """Return the OpenAI tool schemas for the four skill tools."""
    from . import tools
    return tools.util_get_tools(functions=_skill_functions(allowed_dir))


def get_skills_tools_dict(allowed_dir):
    """Return {name: bound-function} for the four skill tools."""
    from . import tools
    return tools.util_get_tools_dict(functions=_skill_functions(allowed_dir))


# ---------------------------------------------------------------------------
# Tools: read_skill_file / write_skill_file
#
# These let the agent read and write files *inside* the ``.mini`` skills
# directory (skill.md files, helper scripts, etc.). They are deliberately
# separate from the named skill_* tools above: they take an explicit path inside
# the ``.mini`` directory and are confined to it via the shared sandbox
# primitive (see :mod:`mini_code_cli.sandbox`), so a skill file can never be
# read from or written outside ``.mini``. This is what guarantees skill files
# always land in the correct place even when ``--allowed-dir`` points elsewhere.
# ---------------------------------------------------------------------------


def _skill_rel_path(file_path, allowed_dir):
    """Return the on-disk path for ``file_path`` resolved under ``.mini``.

    ``file_path`` is treated as a path *relative* to the ``.mini`` skills
    directory (e.g. ``my_skill/skill.md``) and is joined onto the ``.mini``
    root, then resolved. Returns ``None`` when the resolved path does not stay
    inside ``.mini`` (path traversal / absolute paths / escaping the skills
    root).
    """
    if not file_path:
        return None
    root = os.path.realpath(skills_dir(allowed_dir))
    resolved = os.path.realpath(os.path.join(root, file_path))
    # Containment: the resolved target must live inside the ``.mini`` root.
    if resolved != root and not resolved.startswith(root + os.sep):
        return None
    return resolved


def read_skill_file(file_path, allowed_dir):
    """Read a file that lives inside the ``.mini`` skills directory.

    Parameters: file_path -- a path relative to the ``.mini`` directory (e.g.
    ``my_skill/skill.md``). Only files inside ``.mini`` are honored; anything
    outside returns an error.
    """
    target = _skill_rel_path(file_path, allowed_dir)
    if target is None:
        return f"Error: file must be inside the .mini skills directory: {skills_dir(allowed_dir)}"
    if not os.path.isfile(target):
        return f"Error: no such file inside .mini: {file_path}"
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"


def write_skill_file(file_path, content, allowed_dir):
    """Write a file inside the ``.mini`` skills directory.

    Parameters: file_path (a path relative to the ``.mini`` directory, e.g.
    ``my_skill/skill.md``), content. Only files inside ``.mini`` are honored, so
    a skill file can never be relocated outside the skills directory.
    """
    target = _skill_rel_path(file_path, allowed_dir)
    if target is None:
        return f"Error: file must be inside the .mini skills directory: {skills_dir(allowed_dir)}"
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Wrote to {target}"
    except OSError as e:
        return f"Error: {e}"


def _skill_file_functions(allowed_dir):
    """{name: functools.partial} for the two scoped skill-file tools."""
    return {
        "read_skill_file": functools.partial(read_skill_file, allowed_dir=allowed_dir),
        "write_skill_file": functools.partial(write_skill_file, allowed_dir=allowed_dir),
    }


def get_skills_file_tools_dict(allowed_dir):
    """Return {name: {function, description}} for the two scoped skill-file tools."""
    from . import tools
    return tools.util_get_tools_dict(functions=_skill_file_functions(allowed_dir))


def get_skills_file_tools(allowed_dir):
    """Return the OpenAI tool schemas for the two scoped skill-file tools."""
    from . import tools
    return tools.util_get_tools(functions=_skill_file_functions(allowed_dir))
