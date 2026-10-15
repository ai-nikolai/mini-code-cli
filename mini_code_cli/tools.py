import glob as glob_module
import subprocess
import urllib.parse
import urllib.request
import json

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


def util_get_tools(allowed_tools=None, forbidden_tools=None):
    """Return a JSON string describing all tool functions in this file for LLM usage."""
    import inspect
    tools = []
    func_names = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ and not name.startswith('util_') and not name.startswith("_")]
    
    if allowed_tools:
        func_names = [name for name in func_names if name in allowed_tools]
    if forbidden_tools:
        func_names = [name for name in func_names if name not in forbidden_tools]

    for name in func_names:
        func = globals()[name]
        sig = inspect.signature(func)
        parameters = {}
        required = []
        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs: they are catch-alls, not real inputs the
            # LLM should (or can) supply, so they must not appear in the schema.
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "string"
            # Simplify common types
            if 'int' in param_type.lower() or 'integer' in param_type.lower():
                param_type = "integer"
            elif 'float' in param_type.lower() or 'number' in param_type.lower():
                param_type = "number"
            elif 'bool' in param_type.lower():
                param_type = "boolean"
            elif 'list' in param_type.lower() or 'array' in param_type.lower():
                param_type = "array"
            else:
                param_type = "string"
            parameters[param_name] = {"type": param_type}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": func.__doc__ if func.__doc__ else f"Function {name}",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
        }
        tools.append(tool)
    return tools


def util_get_tools_dict(allowed_tools=None, forbidden_tools=None):
    """Return a dictionary mapping tool names to their function objects."""
    result = {}
    func_names = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ and not name.startswith('util_') and not name.startswith("_")]
    if allowed_tools:
        func_names = [name for name in func_names if name in allowed_tools]
    if forbidden_tools:
        func_names = [name for name in func_names if name not in forbidden_tools]

    for name in func_names:
        func = globals()[name]
        desc = func.__doc__ if func.__doc__ else f"Function {name}"
        result[name] = {"function": func, "description": desc}
    return result

shell_tools = [
    "run_shell_command"
]

if __name__=="__main__":
    print(util_get_tools())
    # print(json.dumps(tools_dict, indent=4))