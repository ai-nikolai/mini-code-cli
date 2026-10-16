"""mini_code_cli.tools -- the tool functions the agent loop can expose to the LLM.

The plain functions in this module (``read_file``, ``grep``, ...) are the
default tools. They can be discovered automatically (see
:func:`util_get_tools` / :func:`util_get_tools_dict`). A caller may also pass an
explicit ``functions`` mapping so another feature (e.g. :mod:`mini_code_cli.skills`,
whose tools already have ``allowed_dir`` bound) can reuse the schema/formatting.
"""
import glob as glob_module
import subprocess
import urllib.parse
import urllib.request
import json
import inspect
import functools

# Default timeout (seconds) for a single tool invocation. Mirrors the CLI's
# --tool-timeout default so a tool that is slow/hangs gets aborted.
DEFAULT_TOOL_TIMEOUT = 120

# A browser-like User-Agent. Many sites (e.g. Wikipedia's API) reject the
# default "Python-urllib/..." client, so we identify as a normal browser.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)


def _fetch(url, timeout=DEFAULT_TOOL_TIMEOUT):
    """GET `url` and return its decoded body, sending a browser User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode('utf-8')


def read_file(file_path, *args, **kwargs):
    """Read the contents of a file and return as a string. Param: file_path"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(file_path, content, *args, **kwargs):
    """Write content to a file. Overwrites if file exists. Param: file_path, content"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Wrote to file: {file_path}"
    except Exception as e:
        return f"Error: {e}"


def grep(pattern, file_path, *args, **kwargs):
    """Search for a pattern in a file using grep and return matching lines. Param: pattern, file_path"""
    try:
        result = subprocess.run(['grep', pattern, file_path], capture_output=True, text=True, check=False)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def glob(pattern, *args, **kwargs):
    """Return list of file paths matching the given glob pattern. Param: pattern"""
    try:
        return glob_module.glob(pattern)
    except Exception as e:
        return f"Error: {e}"


def web_search(query, *args, **kwargs):
    """Search the web for `query` and return a plain-text summary of results.

    Param: query

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


def read_website(url, *args, **kwargs):
    """Fetch and return the text content of a given URL.

    Param: url

    `timeout` (seconds) bounds how long the fetch may run before it is aborted,
    mirroring the CLI's --tool-timeout so a slow or unresponsive site cannot
    hang the agent loop.
    """
    timeout = kwargs.get("timeout", DEFAULT_TOOL_TIMEOUT)
    try:
        return _fetch(url, timeout=timeout)
    except Exception as e:
        return f"Error: {e}"


def run_shell_command(command, timeout=DEFAULT_TOOL_TIMEOUT, *args, **kwargs):
    """Run a shell command and return its output. Param: command (string)"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, check=False)
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return output
    except Exception as e:
        return f"Error: {e}"


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
        param_type = _param_type(param.annotation)
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
    """Return a JSON string describing tool functions for LLM usage.

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
    print(util_get_tools())
    # print(json.dumps(tools_dict, indent=4))
