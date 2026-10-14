"""
Usage:
    mini-code

Prompt:

write a similar model.py file but for a LSTM based language model. Make it minimal and call it model_lstm.py it should still be compatible with eval.py

"""

import os
import json
import signal

import time
import uuid

import argparse
import sys

from . import prompts
from . import tools
from . import utils


# ANSI escape codes for colors and bold
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

DEFAULT_CACHE_LOCATION=os.path.expanduser("~/.cache/mini_code/sessions/")
DEFAULT_MODEL_URL="http://0.0.0.0:30000"
DEFAULT_TEMPERATURE=1.0
DEFAULT_TOOL_TIMEOUT=120  # seconds before a single tool call is aborted
DEFAULT_HTTP_TIMEOUT=300  # seconds before an LLM HTTP request is aborted
DEFAULT_MAX_MESSAGES=100  # messages in history before it is compacted to a summary

def save_history(messages, args, session_name=""):
    # Save messages to history.json
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = DEFAULT_CACHE_LOCATION
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir,session_name), exist_ok=True)

    history_path = os.path.join(cache_dir, session_name, f"history.json")
    with open(history_path, 'w') as f:
        json.dump(messages, f, indent=2)


def accumulate_usage(token_stats, usage):
    """Add a single LLM call's token usage into the running session totals."""
    if not usage:
        return
    token_stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    token_stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    token_stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    token_stats["llm_calls"] += 1


def format_usage(usage):
    """Human-readable one-line summary of a single call's token usage."""
    if not usage:
        return "tokens: n/a (server did not report usage)"
    return (f"tokens: input={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} "
            f"total={usage.get('total_tokens', 0)}")


def save_usage(token_stats, cache_dir, session_name=""):
    """Persist the accumulated token usage to the session folder."""
    os.makedirs(cache_dir, exist_ok=True)
    usage_path = os.path.join(cache_dir, session_name, f"usage.json")
    with open(usage_path, 'w') as f:
        json.dump(token_stats, f, indent=2)
    return usage_path


def print_usage_summary(token_stats):
    """Print the accumulated token usage for the session so far."""
    print("-"*20)
    print(f"{BOLD}{CYAN}Token usage (this session):{RESET}")
    print(f"  LLM calls      : {token_stats['llm_calls']}")
    print(f"  Input tokens   : {token_stats['prompt_tokens']}")
    print(f"  Output tokens  : {token_stats['completion_tokens']}")
    print(f"  Total tokens   : {token_stats['total_tokens']}")
    print("-"*20)

def generate_unique_id():
    """Generate a unique ID based on current time and a random component."""
    timestamp = int(time.time() * 1000000)  # microseconds
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    # MODEL / LLM related
    parser.add_argument("--url", type=str, default=f"{DEFAULT_MODEL_URL}", help=f"API host, default={DEFAULT_MODEL_URL}")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM response, default=None")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature for LLM, default={DEFAULT_TEMPERATURE}")
    parser.add_argument("--model", type=str, help="Model name, default=None")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.")

    # Other settings:
    parser.add_argument("--cache-dir", type=str, help=f"The cache dir for the session histories. default={DEFAULT_CACHE_LOCATION}")

    # Agent behaviour related
    parser.add_argument("--system-prompt", type=str, help="Replace system prompt, with a custom system prompt.")
    parser.add_argument("--auto-mode", action="store_true", help="Whether to run the agent in `auto-mode'. Or default: `manual-mode'.")
    parser.add_argument("--agent-md", type=str, help="If the agent should use an agent-md file... (it will added after system message.)")
    parser.add_argument("--prompt", type=str, help="An initial prompt from the user...")

    # Agent permissions related
    parser.add_argument("--ask-permission", action="store_true", help="Ask for permission before any tool call.")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations, default='./'")
    parser.add_argument("--enable-shell", action="store_true", help="Allow shell execution. default= False") 
    parser.add_argument("--llm-timeout", type=int, default=DEFAULT_HTTP_TIMEOUT, help=f"HTTP timeout in seconds for LLM requests (reasoning models can be slow). default={DEFAULT_HTTP_TIMEOUT}")
    parser.add_argument("--tool-timeout", type=int, default=DEFAULT_TOOL_TIMEOUT, help=f"Timeout in seconds for a single tool call. default={DEFAULT_TOOL_TIMEOUT}")


    args = parser.parse_args()
    return args


def print_lines_with_border(lines):
    """Prints any list of strings with a border."""
    max_len = max(len(line) for line in lines)
    margin = 2  # Space on each side inside the box
    inner_width = max_len + 2 * margin
    # Build the box
    top_bottom = "╔" + "═" * inner_width + "╗"
    print(top_bottom)
    for line in lines:
        # Pad the line to inner_width characters
        padded = line.ljust(inner_width)
        print(f"║{padded}║")
    bottom = "╚" + "═" * inner_width + "╝"
    print(bottom)


def print_welcome(allowed_dir, server_url, args, unique_id, cache_dir, model_name=None):
    # Calculate the length of the longest line inside the box
    version = __import__('mini_code_cli').__version__
    
    permission_mode = "All tool calls require permission." if args.ask_permission else "Tool calls will execute without asking for permission."
    agent_mode = "Agent is in manual-mode." if not args.auto_mode else "Agent is in auto-mode."
    shell_mode = "Shell disabled." if not args.enable_shell else "Shell enabled."

    lines = [
        f"  MiniCodeCLI (v{version})",
        f"  The session-id is: {unique_id}" ,
        f"  It will be saved under: `{cache_dir}`",
         "  Type 'quit' or 'exit' to exit the loop.",
        f"  Allowed dir: {allowed_dir}",
        f"  Server URL: {server_url}",
        f"  Model Name: {model_name}",
        f"  {permission_mode}",
        f"  {agent_mode}",
        f"  {shell_mode}"
    ]

    print_lines_with_border(lines)

def print_help(allowed_dir, server_url, args, unique_id, cache_dir, model_name):
    print_welcome(allowed_dir, server_url, args, unique_id, cache_dir, model_name)
    
    help_message = ""
    help_message += "Manual-mode means the agent will ask for user input after each LLM prediction." if not args.auto_mode else "Auto-mode means the agent will keep calling the LLM until no further LLM calls are possible."

    print(f"{BOLD}{GREEN}HELP:{RESET} {help_message}")

def call_function_with_timeout(func, tool_params, name, timeout_seconds=DEFAULT_TOOL_TIMEOUT):
    """calls function with params, aborting after `timeout_seconds`."""
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Function call timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(**tool_params)
    except TimeoutError:
        print(f"{BOLD}{RED}Error:{RESET} Function {name} timed out after {timeout_seconds} seconds.")
        result = "Error:Function {name} timed out after {timeout_seconds} seconds."
    finally:
        signal.alarm(0)  # Disable the alarm
    
    return result



def is_within_allowed_dir(filepath, allowed_dir):
    """Return True if the resolved `filepath` lives inside `allowed_dir`.

    Uses os.path.realpath to resolve symlinks and '..' so path-traversal or
    symlink tricks cannot escape the allowed directory.
    """
    target = os.path.realpath(filepath)
    root = os.path.realpath(allowed_dir)
    try:
        return os.path.commonpath([target, root]) == root
    except (ValueError, TypeError):
        # Different drives (Windows) or a non-absolute path: fail closed.
        return False


def maybe_compact(messages, server_url, model, api_key, llm_timeout, max_messages=DEFAULT_MAX_MESSAGES):
    """Compact the conversation history once it grows beyond `max_messages`.

    Long (especially auto-mode) sessions can overflow the model's context window.
    When the history gets too large we ask the LLM to summarize it and replace the
    old turns with that summary, keeping the original system prompt as the first
    message. The compaction call itself is a single, tool-free request, so it
    cannot trigger further tool calls or recurse into compaction.
    """
    if len(messages) <= max_messages:
        return messages

    try:
        print(f"{BOLD}{YELLOW}System:{RESET} Compacting conversation history ({len(messages)} messages -> summary)...")
        summary_prompt = prompts.compaction_prompt(messages)
        summary, _, compact_usage = utils.call_openai_server(
            [{"role": "user", "content": summary_prompt}],
            max_tokens=1024,
            temperature=0.0,
            model=model,
            server_url=server_url,
            api_key=api_key,
            timeout=llm_timeout,
        )
        accumulate_usage(token_stats, compact_usage)
        if summary:
            system_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
            compacted = []
            if system_content:
                compacted.append({"role": "system", "content": system_content})
            compacted.append({
                "role": "user",
                "content": f"Here is a summary of our earlier conversation so far:\n{summary}",
            })
            print(f"{BOLD}{YELLOW}System:{RESET} History compacted to {len(compacted)} messages.")
            return compacted
    except Exception as e:
        print(f"{BOLD}{RED}Warning:{RESET} Failed to compact history: {e}")

    return messages


def main():
    args = parse_args()

    # Get session unique ID:
    unique_id = generate_unique_id()
    cache_dir = args.cache_dir if args.cache_dir else DEFAULT_CACHE_LOCATION

    # LLM Related
    server_url = f"{args.url}"
    new_model_name = utils.determine_vllm_model(server_url)

    if new_model_name:
        actual_model = new_model_name
    else:
        actual_model = args.model

    api_key = None
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.environ.get("MINI_CODE_API_KEY", "")
        if api_key:
            print(f"{YELLOW}API Key found at MINI_CODE_API_KEY{RESET}")
        else:
            print(f"{YELLOW}NO API Key found at MINI_CODE_API_KEY. Set it with export MINI_CODE_API_KEY='sk-api-key'{RESET}")

    
    # Permissions related
    allowed_dir = os.path.abspath(args.allowed_dir)

    if not args.enable_shell:
        forbidden_tools = tools.shell_tools
    else:
        forbidden_tools = None

    if args.enable_shell and not args.ask_permission:
        print(f"{BOLD}{RED}WARNING:{RESET} Shell execution is ENABLED and will run WITHOUT any permission "
              f"checks. The agent can execute ANY command on this system and can escape --allowed-dir. "
              f"Use with extreme caution.")

    # TOOLs
    tools_dict = tools.util_get_tools_dict(forbidden_tools=forbidden_tools)
    llm_tools_dict = tools.util_get_tools(forbidden_tools=forbidden_tools)


    # System Message and Agent.md
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = prompts.system_prompt(tools_dict=tools_dict, allowed_dir=allowed_dir)

    messages = [{"role": "system", "content": system_prompt}]

    if args.agent_md:
        print(f"{YELLOW}AGENT.md '{args.agent_md}' specified.{RESET}")

        # Check if the file exists
        if not os.path.isfile(args.agent_md):
            print(f"{RED}Error: The AGENT.md '{args.agent_md}' does not exist.{RESET}")
        else:
            # Load the content of agent-md file and add it after system message
            with open(args.agent_md, 'r', encoding='utf-8') as f:
                agent_md_content = f.read()
            print(f"{YELLOW}AGENT.md '{args.agent_md}' loaded.{RESET}")

        # Append the content as a user message after the system prompt
        messages.append({"role": "user", "content": agent_md_content})

    print_welcome(allowed_dir, server_url, args, unique_id, cache_dir, actual_model)

    # Token accounting
    token_stats = {
        "prompt_tokens": 0,      # input tokens
        "completion_tokens": 0,  # output tokens
        "total_tokens": 0,
        "llm_calls": 0,
    }

    # Agent Logic Flags
    llm_response_flag = False
    llm_tool_response_flag = False
    llm_repeat_flag = False
    tool_count = 0
    
    user_input = args.prompt

    while True:
        try:

            if llm_tool_response_flag and args.auto_mode: #skip user input and try to call the model again until there no more tool calls.
                print(f"{BOLD}{YELLOW}System:{RESET} Skipping user input. Continuing with LLM calls until no more tool calls. [Tool count={tool_count}]")
                llm_repeat_flag = True
            else:
            # get user input
                if llm_repeat_flag:
                    print(f"{BOLD}{YELLOW}System:{RESET} Called {tool_count} tools in total. Exiting tool loop.")

                tool_count = 0
                if not user_input:
                    try:
                        print("="*30)
                        user_input = input(f"{BOLD}{CYAN}You:{RESET} ")
                        if user_input.lower() in ["quit", "exit", "/quit", "/exit"]:
                            print("\nThank you. Goodbye!")
                            print(f"Saving history to: {cache_dir}/history_{unique_id}.json")
                            save_history(messages, args, unique_id)
                            usage_path = save_usage(token_stats, cache_dir, unique_id)
                            print_usage_summary(token_stats)
                            print(f"Saving usage to: {usage_path}")
                            break
                    except KeyboardInterrupt:
                        print("\nThank you. Goodbye!")
                        print(f"Saving history to: {cache_dir}/history_{unique_id}.json")
                        save_history(messages, args, unique_id)
                        usage_path = save_usage(token_stats, cache_dir, unique_id)
                        print_usage_summary(token_stats)
                        print(f"Saving usage to: {usage_path}")
                        break
                else:
                    print("="*30)
                    print(f"{BOLD}{CYAN}You (preset): {RESET}{user_input}")
                
                if user_input == "/help":
                    print_help(allowed_dir, server_url, args, unique_id, cache_dir, actual_model)
                    user_input = ""
                    continue

                messages.append({"role": "user", "content": user_input})
                user_input = ""

            llm_response_flag = False
            llm_tool_response_flag = False

            # Compact the history if it's getting long, to avoid overflowing the context window.
            messages = maybe_compact(messages, server_url, actual_model, api_key, args.llm_timeout)

            response_text, tool_calls, usage = utils.call_openai_server(
                messages, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature, 
                model=actual_model, 
                server_url=server_url, 
                tools=llm_tools_dict,
                api_key=api_key,
                timeout=args.llm_timeout,
            )

            # Record and show the token usage for this LLM call.
            accumulate_usage(token_stats, usage)
            print(f"{BOLD}{CYAN}[{format_usage(usage)}]{RESET}")

            if response_text:
                llm_response_flag = True
                # response_lines=[]
                # response_lines.extend(response_text.split('\n'))
                # print_lines_with_border(response_lines)
                print("-"*20)
                print(f"{BOLD}{GREEN}Assistant:{RESET}")
                print("-"*20)
                print(response_text)
                print("="*30)


            if tool_calls:
                llm_tool_response_flag = True
                
                messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": tool_calls
                })
                # Tool call parsing and execution
                print(f"{BOLD}{YELLOW}Tool Call:{RESET} LLM is trying to call {len(tool_calls)} tools.")

                for call in tool_calls:
                    tool_count += 1

                    call_id = call.get("id", "")
                    name = call.get("function", {}).get("name", "")
                    params_str = call.get("function", {}).get("arguments", "{}")
                    try:
                        tool_params = json.loads(params_str)
                    except json.JSONDecodeError:
                        print(f"{BOLD}{RED}Warning:{RESET} Error parsing arguments for tool {name}")
                        continue

                    if name in tools_dict:
                        # Check allowed directory for file operations
                        if name in ["read_file", "write_file", "grep", "glob"]:
                            filepath = tool_params.get("filepath", "")
                            if name == "write_file":
                                filepath = tool_params.get("filepath", "")
                            if filepath and not is_within_allowed_dir(filepath, allowed_dir):
                                print(f"{BOLD}{RED}Warning:{RESET} File operation not allowed outside allowed directory: {allowed_dir}")
                                continue

                        func = tools_dict[name]["function"]
                        # asking permissions
                        if args.ask_permission:
                            print(f"{BOLD}{YELLOW}Tool permission:{RESET} Are you sure you want to call the tool?")
                            print(f"name: {name}, parameters: {params_str}")
                            try:
                                permission_input = input("Type: 'y' or 'yes'\n")
                            except KeyboardInterrupt:
                                print("\nThank you. Goodbye! Saving history...")
                                save_history(messages, args, unique_id)
                                sys.exit(0)

                            if not permission_input in ['y', 'yes']:
                                print(f"{BOLD}{YELLOW}Tool permission denied:{RESET}.")
                                messages.append({
                                    "role": "user",
                                    "content": f"User denied permission for tool {name} and params {params_str}."
                                })
                                continue

                        print(f"{BOLD}{YELLOW}Tool Call:{RESET} LLM is trying to call: {name} with {tool_params}.")
                        result = call_function_with_timeout(func, tool_params, name, timeout_seconds=args.tool_timeout)
                        # Limiting output to 300 characters for readability
                        result_str = str(result)
                        if len(result_str) > 300:
                            result_str = result_str[:300] + "... (truncated to 300 chars)"
                        print(f"{BOLD}{MAGENTA}Tool '{name}' returned:{RESET} {result_str}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": str(result)
                        })
                    else:
                        print(f"{BOLD}{RED}Error:{RESET} Unknown tool: {name}, params {params_str}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"Error: Unknown tool: {name}, params {params_str}"
                        })
            else:
                messages.append({
                        "role": "assistant",
                        "content": response_text,
                        # "tool_calls": tool_calls
                })
            
            if not llm_response_flag and not llm_tool_response_flag:
                print(f"{BOLD}{RED}Warning:{RESET} No response was given by the LLM.")

            save_history(messages, args, unique_id)

        except KeyboardInterrupt:
            print(f"{BOLD}{RED}INTERRUPTING:{RESET} Mini Code Exiting...")

        except Exception as e:
            print(f"{BOLD}{RED}FATAL ERROR:{RESET} Mini Code Crashed...:\n{e}")
        
        finally:
            print(f"Saving history to: {cache_dir}/history_{unique_id}.json")
            save_history(messages, args, unique_id)
            usage_path = save_usage(token_stats, cache_dir, unique_id)
            print(f"Saving usage to: {usage_path}")
            print_usage_summary(token_stats)

if __name__ == "__main__":
    main()