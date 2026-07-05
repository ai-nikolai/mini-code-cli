import os
import glob as glob_module
import subprocess
import urllib.parse
import urllib.request
import json

def read_file(filepath):
    """Read the contents of a file and return as a string. Param: filepath"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error: {e}")


def write_file(filepath, content):
    """Write content to a file. Overwrites if file exists. Param: filepath, content"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error: {e}")
    return f"Wrote to file: {filepath}, content: {content}."

def grep(pattern, filepath):
    """Search for a pattern in a file using grep and return matching lines. Param: pattern, filepath"""
    try:
        result = subprocess.run(['grep', pattern, filepath], capture_output=True, text=True, check=False)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def glob(pattern):
    """Return list of file paths matching the given glob pattern. Param: pattern"""
    try:
        return glob_module.glob(pattern)
    except Exception as e:
        return f"Error: {e}"

def web_search(query):
    """Perform a simple web search using DuckDuckGo's API and return top results. Param: query"""
    query_encoded = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={query_encoded}&format=json&no_html=1"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        # Extract relevant text from the response
        results = []
        if 'AbstractText' in data and data['AbstractText']:
            results.append(data['AbstractText'])
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics']:
                if 'Text' in topic:
                    results.append(topic['Text'])
        return '\n'.join(results) if results else "No results found."
    except Exception as e:
        return f"Error: {e}"

def read_website(url):
    """Fetch and return the text content of a given URL (simplified, returns raw HTML). Param: url"""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        return f"Error: {e}"

def util_get_tools(tool_names=None):
    """Return a JSON string describing all tool functions in this file for LLM usage."""
    import inspect
    tools = []
    if tool_names:
        func_names = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ and not name.startswith('util_') and not name.startswith("_") and name in tool_names]
    else:
        func_names = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ and not name.startswith('util_') and not name.startswith("_")]

    for name in func_names:
        func = globals()[name]
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
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
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": func.__doc__ if func.__doc__ else f"Function {name}",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [p for p in sig.parameters if sig.parameters[p].default == inspect.Parameter.empty]
                }
            }
        }
        tools.append(tool)
    return tools


def util_get_tools_dict():
    """Return a dictionary mapping tool names to their function objects."""
    result = {}
    func_names = [name for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ and not name.startswith('util_') and not name.startswith("_")]
    for name in func_names:
        func = globals()[name]
        desc = func.__doc__ if func.__doc__ else f"Function {name}"
        result[name] = {"function": func, "description": desc}
    return result


if __name__=="__main__":
    # print(util_get_tools())
    print(json.dumps(tools_dict, indent=4))