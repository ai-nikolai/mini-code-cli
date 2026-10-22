"""mini_code_cli.tools -- the tool functions the agent loop can expose to the LLM.

The plain functions in this module (``read_file``, ``edit``, ``grep``, ...) are
the default tools. They can be discovered automatically (see
:func:`util_get_tools` / :func:`util_get_tools_dict`). A caller may also pass an
explicit ``functions`` mapping so another feature (e.g. :mod:`mini_code_cli.skills`,
whose tools already have ``allowed_dir`` bound) can reuse the schema/formatting.

Every tool returns a string (never raises): the runner also shields the loop
from exceptions, but a tool that reports its own, specific error tells the model
what to do next, instead of an opaque "tool raised an error".

The parameter sets are deliberately *small but useful*: the handful of knobs a
model actually needs (a line range, a recursive search, an include filter, an
exact-match edit) and nothing more. Every optional parameter has a default, so a
call that omits it behaves exactly as a call that never had it.
"""
import fnmatch
import functools
import glob as glob_module
import inspect
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request

# Default timeout (seconds) for a single tool invocation. Mirrors the CLI's
# --tool-timeout default so a tool that is slow/hangs gets aborted.
DEFAULT_TOOL_TIMEOUT = 120

# How much of a fetched web page to hand back by default. A raw HTML page can
# be megabytes, which would fill the context window in one call; the default is
# generous but bounded, and read_website exposes it as ``max_chars``.
DEFAULT_MAX_WEBPAGE_CHARS = 100_000

# Matched grep lines longer than this are truncated (a minified JS/CSS file is
# one line, and echoing it whole would drown the conversation).
MAX_GREP_LINE_CHARS = 1_000

# Directories that are never worth recursively searching: noise, huge, or
# generated. Only used by grep's recursive walk.
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules"}

# A browser-like User-Agent. Many sites (e.g. Wikipedia's API) reject the
# default "Python-urllib/..." client, so we identify as a normal browser.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)


# ---------------------------------------------------------------------------
# parameter coercion helpers
# ---------------------------------------------------------------------------

def _coerce_timeout(value, default=DEFAULT_TOOL_TIMEOUT):
    """Best-effort: turn a tool-call ``timeout`` into a number of seconds.

    Models do not always respect the schema's ``integer`` type and routinely
    send ``"30"`` (a string) or ``None`` instead of ``30``. Passing that
    straight to :func:`subprocess.run` / :func:`urllib.request.urlopen` fails
    deep inside the stdlib with an opaque
    ``unsupported operand type(s) for +: 'float' and 'str'``, so a perfectly
    good tool call would be reported back to the model as a crash. Coercing
    here keeps the failure (if any) about the *command*, not about its type.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value if value > 0 else default
    try:
        coerced = float(str(value).strip().rstrip("sSmM"))
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _coerce_int(value, default=0):
    """Best-effort int coercion for model-supplied arguments (never raises)."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default=False):
    """Best-effort bool coercion for model-supplied arguments.

    ``bool("false")`` is ``True`` in Python, so a model that sends booleans as
    strings (``"false"``, ``"no"``, ``"0"``) would silently flip the flag. This
    reads those spellings correctly and never raises.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _coerce_text(value):
    """Best-effort string coercion for file contents / search patterns."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _fetch(url, timeout=DEFAULT_TOOL_TIMEOUT):
    """GET `url` and return its decoded body, sending a browser User-Agent.

    Undecodable bytes are replaced rather than raising, so a page with a stray
    non-UTF-8 byte is still readable instead of failing the whole fetch.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=_coerce_timeout(timeout)) as response:
        return response.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# filesystem tools
# ---------------------------------------------------------------------------

def read_file(file_path, offset: int = 0, limit: int = 0, *args, **kwargs):
    """Read the contents of a file and return as a string.

    Param: file_path (required); offset (1-based first line to read, 0 = start
    of the file); limit (maximum number of lines, 0 = to the end of the file).

    Use offset/limit to page through a large file instead of reading it all.
    A partial read is prefixed with a ``[lines X-Y of N]`` header.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return f"Error: no such file: {file_path}"
    except IsADirectoryError:
        return (f"Error: {file_path} is a directory "
                f"(use glob to list its files, or grep to search it)")
    except Exception as e:
        return f"Error: {e}"

    offset = max(_coerce_int(offset, 0), 0)
    limit = max(_coerce_int(limit, 0), 0)
    if not offset and not limit:
        return text

    lines = text.splitlines(keepends=True)
    start = max(offset - 1, 0) if offset else 0
    end = start + limit if limit else len(lines)
    if start >= len(lines):
        return (f"Error: offset {offset} is past the end of {file_path} "
                f"({len(lines)} line(s)).")
    chunk = "".join(lines[start:end])
    header = f"[lines {start + 1}-{min(end, len(lines))} of {len(lines)}]\n"
    return header + chunk


def write_file(file_path, content, append: bool = False, create_dirs: bool = True,
               *args, **kwargs):
    """Write content to a file. Overwrites if it exists (unless append=True).

    Param: file_path (required); content (required, the text to write);
    append (bool, default false: add to the end instead of overwriting);
    create_dirs (bool, default true: create missing parent directories).
    """
    content = _coerce_text(content)
    append = _coerce_bool(append, False)
    create_dirs = _coerce_bool(create_dirs, True)
    try:
        if create_dirs:
            parent = os.path.dirname(os.path.abspath(file_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)
        verb = "Appended to" if append else "Wrote to"
        return f"{verb} file: {file_path} ({len(content)} chars)"
    except IsADirectoryError:
        return f"Error: {file_path} is a directory, not a file."
    except Exception as e:
        return f"Error: {e}"


def edit(file_path, old_string, new_string, replace_all: bool = False,
         *args, **kwargs):
    """Replace an exact string in a text file. Prefer this over rewriting a whole file.

    Param: file_path (required); old_string (required: the exact existing text
    to find, including indentation); new_string (required: the replacement
    text); replace_all (bool, default false: replace every occurrence instead
    of requiring a unique match).

    The match is literal (no regex). If old_string is missing, or matches more
    than once without replace_all, the edit is refused and an error explains
    what to do, so a wrong edit is never silently applied.
    """
    old_string = _coerce_text(old_string)
    new_string = _coerce_text(new_string)
    replace_all = _coerce_bool(replace_all, False)

    if not old_string:
        return ("Error: old_string must not be empty "
                "(use write_file to create or overwrite a file).")
    if old_string == new_string:
        return "Error: old_string and new_string are identical; nothing to do."

    try:
        # newline="" keeps CRLF files byte-faithful (no line-ending rewriting).
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
    except FileNotFoundError:
        return f"Error: no such file: {file_path}"
    except IsADirectoryError:
        return f"Error: {file_path} is a directory, not a file."
    except Exception as e:
        return f"Error: {e}"

    count = text.count(old_string)
    if count == 0:
        return (f"Error: old_string was not found in {file_path}. Read the file "
                f"first and copy the exact text (including whitespace).")
    if count > 1 and not replace_all:
        return (f"Error: old_string matches {count} times in {file_path}, so the "
                f"edit is ambiguous. Add more surrounding context to make it "
                f"unique, or pass replace_all=True to replace every occurrence.")

    updated = text.replace(old_string, new_string, -1 if replace_all else 1)
    try:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
    except Exception as e:
        return f"Error: {e}"

    replaced = count if replace_all else 1
    return f"Edited {file_path}: replaced {replaced} occurrence(s) of old_string."


def _grep_files(path, recursive, include):
    """Yield the files to search under ``path`` (sorted, noise dirs skipped).

    Symbolic links are skipped. The sandbox validates the *directory* a search
    is rooted in, so following a link could read a file from outside
    ``--allowed-dir`` -- a link inside the allowed dir is the one way a walk
    could escape it. Skipping links keeps the sandbox intact.
    """
    def wanted(full, name):
        return (os.path.isfile(full) and not os.path.islink(full)
                and (not include or fnmatch.fnmatch(name, include)))

    if not recursive:
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return
        for name in names:
            full = os.path.join(path, name)
            if wanted(full, name):
                yield full
        return

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if wanted(full, name):
                yield full


def grep(pattern, file_path, recursive: bool = False, include=None,
         ignore_case: bool = False, line_numbers: bool = True,
         max_results: int = 200, *args, **kwargs):
    """Search a file or directory for a regular expression and return matching lines.

    Param: pattern (required, a Python regular expression); file_path (required:
    a file, or a directory to search); recursive (bool, default false: also
    search sub-directories); include (optional glob for file names when
    searching a directory, e.g. "*.py"); ignore_case (bool, default false);
    line_numbers (bool, default true: prefix each match with its line number);
    max_results (max matching lines to return, 0 = unlimited, default 200).

    Directories that are pure noise (.git, __pycache__, node_modules, ...) are
    skipped. Returns "No matches ..." when nothing matched, so an empty result
    is never ambiguous.
    """
    pattern = _coerce_text(pattern)
    recursive = _coerce_bool(recursive, False)
    ignore_case = _coerce_bool(ignore_case, False)
    line_numbers = _coerce_bool(line_numbers, True)
    max_results = max(_coerce_int(max_results, 200), 0)

    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"Error: invalid regular expression {pattern!r}: {e}"

    path = _coerce_text(file_path) or "."
    if os.path.isdir(path):
        files = list(_grep_files(path, recursive, include))
        if not files:
            where = "recursively " if recursive else ""
            return f"No files found to search {where}in {path}."
        multiple = True
    elif os.path.isfile(path):
        files = [path]
        multiple = False
    else:
        return f"Error: no such file or directory: {path}"

    matches = []
    truncated = False
    for current in files:
        try:
            with open(current, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except (OSError, UnicodeError):
            continue
        if "\0" in text[:4096]:        # a binary file: not greppable
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not regex.search(line):
                continue
            if max_results and len(matches) >= max_results:
                truncated = True
                break
            shown = line
            if len(shown) > MAX_GREP_LINE_CHARS:
                shown = shown[:MAX_GREP_LINE_CHARS] + "...[line truncated]"
            prefix = ""
            if multiple:
                prefix += f"{current}:"
            if line_numbers:
                prefix += f"{line_number}:"
            matches.append(prefix + shown)
        if truncated:
            break

    if not matches:
        scope = "recursively " if (recursive and os.path.isdir(path)) else ""
        return f"No matches for {pattern!r} {scope}in {path}."
    if truncated:
        matches.append(f"...[stopped after {max_results} matching lines; "
                       f"raise max_results to see more]")
    return "\n".join(matches)


def glob(pattern, root=None, *args, **kwargs):
    """Return the file paths matching a glob pattern, one per line.

    Param: pattern (required, a glob such as "**/*.py"; "**" matches nested
    directories); root (optional base directory the pattern is resolved
    against; default: the current working directory).
    """
    pattern = _coerce_text(pattern)
    if not pattern:
        return "Error: pattern must not be empty."
    root = _coerce_text(root) or None
    try:
        full = pattern
        if root and not os.path.isabs(pattern):
            full = os.path.join(root, pattern)
        matches = sorted(glob_module.glob(full, recursive=True))
    except Exception as e:
        return f"Error: {e}"

    if not matches:
        where = f" under {root}" if root else ""
        return f"No files matched {pattern!r}{where}."
    return "\n".join(matches)


# ---------------------------------------------------------------------------
# web tools
# ---------------------------------------------------------------------------

def web_search(query, *args, **kwargs):
    """Search the web for `query` and return a plain-text summary of results.

    Param: query (required)

    This is a best-effort, dependency-free lookup. It tries, in order:

    1. DuckDuckGo's autocomplete API, which returns popular related queries and
       the top related website for the term.
    2. The Wikipedia search API, which returns matching article titles plus a
       short plain-text extract of the best match.

    Every source is wrapped in its own try/except so one failing endpoint can't
    mask the others, and the function always returns a string (results joined by
    newlines, or an "Error: ..." message).
    """
    results = []
    query = _coerce_text(query)
    if not query.strip():
        return "Error: query must not be empty."

    # 1) DuckDuckGo autocomplete: fast, no-keyword related queries.
    try:
        ddg_url = f"https://duckduckgo.com/ac/?q={urllib.parse.quote(query)}&type=list"
        data = json.loads(_fetch(ddg_url))
        suggestions = data[1] if (isinstance(data, list) and len(data) > 1) else []
        if suggestions:
            results.append("Related queries: " + ", ".join(suggestions[:10]))
    except Exception as e:
        results.append(f"(DuckDuckGo suggestion lookup failed: {e})")

    # 2) Wikipedia: matching article titles + a short extract of the best match.
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            f"{urllib.parse.quote(query)}&format=json&srlimit=5"
        )
        data = json.loads(_fetch(search_url))
        hits = data.get("query", {}).get("search", [])
        if hits:
            best = hits[0]
            results.append(f"Wikipedia: {best.get('title')}")
            extract_url = (
                "https://en.wikipedia.org/w/api.php?action=query&pageids="
                f"{best.get('pageid')}&prop=extracts&exintro&explaintext&format=json"
            )
            extract = json.loads(_fetch(extract_url))
            pages = extract.get("query", {}).get("pages", {})
            for page in pages.values():
                snippet = page.get("extract", "")
                if snippet:
                    snippet = "\n".join(line.strip() for line in snippet.splitlines() if line.strip())
                    results.append("  " + snippet[:500])
                    break
    except Exception as e:
        results.append(f"(Wikipedia lookup failed: {e})")

    return '\n'.join(results) if results else "No results found."


def read_website(url, timeout: int = DEFAULT_TOOL_TIMEOUT,
                 max_chars: int = DEFAULT_MAX_WEBPAGE_CHARS, *args, **kwargs):
    """Fetch and return the text content of a given URL.

    Param: url (required); timeout (seconds before the fetch is aborted);
    max_chars (maximum characters to return, 0 = unlimited, default 100000 --
    a raw page can be enormous, so it is truncated rather than flooding the
    context).

    Returns the page's raw text/HTML (tags included); it is not rendered.
    """
    url = _coerce_text(url)
    if not url:
        return "Error: url must not be empty."
    max_chars = max(_coerce_int(max_chars, DEFAULT_MAX_WEBPAGE_CHARS), 0)
    try:
        text = _fetch(url, timeout=timeout)
    except Exception as e:
        return f"Error: {e}"

    if max_chars and len(text) > max_chars:
        total = len(text)
        return (text[:max_chars]
                + f"\n...[truncated: showing {max_chars} of {total} characters; "
                  f"pass a larger max_chars (or 0) to see more]")
    return text


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------

def run_shell_command(command, timeout: int = DEFAULT_TOOL_TIMEOUT, cwd=None,
                      *args, **kwargs):
    """Run a shell command and return its output.

    Param: command (required, the shell command line); timeout (seconds before
    the command is killed); cwd (optional working directory for the command).

    Captures stdout and stderr; a non-zero exit code is reported so a failure
    is visible rather than looking like empty success.
    """
    command = _coerce_text(command)
    if not command.strip():
        return "Error: command must not be empty."
    cwd = _coerce_text(cwd) or None
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                timeout=_coerce_timeout(timeout), cwd=cwd, check=False)
        output = result.stdout or ""
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        output = output.strip()
        if result.returncode != 0:
            output += f"\nEXIT CODE: {result.returncode}"
        return output if output else f"Command produced no output (exit code {result.returncode})."
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_coerce_timeout(timeout)} seconds: {command!r}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# schema generation
# ---------------------------------------------------------------------------

def _local_function_names():
    """Names of the real tool functions defined in this module (not util_/_)."""
    return [name for name, obj in globals().items()
            if callable(obj) and obj.__module__ == __name__
            and not name.startswith('util_') and not name.startswith("_")]


def _param_type(annotation):
    """Map a type annotation to an OpenAI-schema type string."""
    param_type = str(annotation) if annotation != inspect.Parameter.empty else "string"
    if 'int' in param_type.lower() or 'integer' in param_type.lower():
        return "integer"
    if 'float' in param_type.lower() or 'number' in param_type.lower():
        return "number"
    if 'bool' in param_type.lower():
        return "boolean"
    if 'list' in param_type.lower() or 'array' in param_type.lower():
        return "array"
    return "string"


def _schema_params(func):
    """Return ``(properties, required)`` for a tool callable's signature.

    Handles both plain module functions and :func:`functools.partial` objects
    (used by skills, which bind ``allowed_dir`` up front): parameters that are
    already bound on a partial are omitted from the schema, and remaining
    non-default parameters are marked required. ``*args`` / ``**kwargs``
    catch-alls are always skipped.
    """
    if isinstance(func, functools.partial):
        underlying = func.func
        bound_kwargs = dict(func.keywords or {})
        bound_positional = list(func.args)
    else:
        underlying = func
        bound_kwargs = {}
        bound_positional = []

    sig = inspect.signature(underlying)
    # Bind leading positional args (in signature order) to the partial's
    # positional args, so a positional bound arg hides the right parameter.
    positional_names = [
        pn for pn, p in sig.parameters.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD)
    ]
    positional_bound = {}
    for i, pname in enumerate(positional_names):
        if i < len(bound_positional):
            positional_bound[pname] = bound_positional[i]

    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name in bound_kwargs or param_name in positional_bound:
            # Already bound on the partial -- the LLM must not supply it.
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty and param.default is not inspect.Parameter.empty \
                and param.default is not None:
            # No annotation, but a typed default (e.g. ``timeout=120``): infer
            # from it, so the schema advertises an integer/boolean instead of
            # defaulting everything to "string".
            annotation = type(param.default)
        param_type = _param_type(annotation)
        properties[param_name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    return properties, required


def _callable_description(func, name):
    """Return a human-friendly one-line description for a tool callable.

    The default tools are plain functions whose ``__doc__`` is the description
    the model sees. Tools contributed by the skills / plan features are built
    with :func:`functools.partial` (to bind the base directory), and ``partial``
    *replaces* ``__doc__`` with a generic "partial(func, ...)" string -- so for
    a partial we look at the underlying function's ``__doc__`` instead, exactly
    as ``_schema_params`` does when building the parameter schema. Fall back to
    a generated name only when there is no docstring at all.
    """
    underlying = func.func if isinstance(func, functools.partial) else func
    doc = (getattr(underlying, "__doc__", "") or "").strip()
    return doc or f"Function {name}"


def util_get_tools(allowed_tools=None, forbidden_tools=None, functions=None):
    """Return a list of OpenAI-style tool schemas describing the tool functions.

    By default scans this module's own tool functions. When ``functions`` is a
    {name: callable} mapping, those are described instead (in ``functions``
    order) -- letting other features (e.g. skills, with ``allowed_dir`` already
    bound) contribute their own tools while reusing this formatting.
    """
    tools = []

    if functions:
        items = list(functions.items())
    else:
        items = [(name, globals()[name]) for name in _local_function_names()]

    if allowed_tools:
        items = [(n, f) for n, f in items if n in allowed_tools]
    if forbidden_tools:
        items = [(n, f) for n, f in items if n not in forbidden_tools]

    for name, func in items:
        properties, required = _schema_params(func)
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": _callable_description(func, name),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        tools.append(tool)
    return tools


def util_get_tools_dict(allowed_tools=None, forbidden_tools=None, functions=None):
    """Return a dictionary mapping tool names to their function objects.

    ``functions`` is an optional {name: callable} mapping used in place of this
    module's own functions (see :func:`util_get_tools`)."""
    result = {}

    if functions:
        items = list(functions.items())
    else:
        items = [(name, globals()[name]) for name in _local_function_names()]

    if allowed_tools:
        items = [(n, f) for n, f in items if n in allowed_tools]
    if forbidden_tools:
        items = [(n, f) for n, f in items if n not in forbidden_tools]

    for name, func in items:
        desc = _callable_description(func, name)
        result[name] = {"function": func, "description": desc}
    return result


shell_tools = [
    "run_shell_command"
]

if __name__ == "__main__":
    print(json.dumps(util_get_tools(), indent=2))
