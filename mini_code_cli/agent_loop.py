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

def save_history(messages, args, session_name=""):
    # Save messages to history.json
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = os.path.expanduser("~/.cache/mini_code/sessions/")
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, f"history_{session_name}.json")
    with open(history_path, 'w') as f:
        json.dump(messages, f, indent=2)

def generate_unique_id():
    """Generate a unique ID based on current time and a random component."""
    timestamp = int(time.time() * 1000000)  # microseconds
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    # MODEL / LLM related
    parser.add_argument("--url", type=str, default="http://0.0.0.0:30000", help="API host")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--model", type=str, default="default", help="Model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.")

    # Other settings:
    parser.add_argument("--cache-dir", type=str, help="The cache dir for the session histories.")

    # Agent behaviour related
    parser.add_argument("--system-prompt", type=str, help="Replace system prompt, with a custom system prompt.")
    parser.add_argument("--auto-mode", action="store_true", help="Whether to run the agent in `auto-mode'. Or default: `manual-mode'.")
    parser.add_argument("--agent-md", type=str, help="If the agent should use an agent-md file... (it will added after system message.)")
    parser.add_argument("--prompt", type=str, help="An initial prompt from the user...")

    # Agent permissions related
    parser.add_argument("--ask-permission", action="store_true", help="Ask for permission before any tool call.")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations")
    parser.add_argument("--enable-shell", action="store_true", help="Allow shell execution. Default: False") #todo


    args = parser.parse_args()
    return args


def print_welcome(allowed_dir, server_url, args, unique_id, cache_dir):
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
        f"  {permission_mode}",
        f"  {agent_mode}",
        f"  {shell_mode}"
    ]
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

def print_help(allowed_dir, server_url, args, unique_id, cache_dir):
    print_welcome(allowed_dir, server_url, args, unique_id, cache_dir)
    
    help_message = ""
    help_message += "Manual-mode means the agent will ask for user input after each LLM prediction." if not args.auto_mode else "Auto-mode means the agent will keep calling the LLM until no further LLM calls are possible."

    print(f"{BOLD}{GREEN}HELP:{RESET} {help_message}")

def call_function_with_timeout(func, tool_params, name):
    """calls function with params"""
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Function call timed out")

    timeout_seconds = 30  # Set your desired timeout in seconds
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



def main():
    args = parse_args()

    # Get session unique ID:
    unique_id = generate_unique_id()
    cache_dir = args.cache_dir if args.cache_dir else "~/.cache/mini_code/sessions/"

    # LLM Related
    server_url = f"{args.url}"
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

    print_welcome(allowed_dir, server_url, args, unique_id, cache_dir)

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
                        print("-"*30)
                        user_input = input(f"{BOLD}{CYAN}You:{RESET} ")
                        if user_input.lower() in ["quit", "exit", "/quit", "/exit"]:
                            print("\nThank you. Goodbye!")
                            print(f"Saving history to: {cache_dir}/{unique_id}")
                            save_history(messages, args, unique_id)
                            break
                    except KeyboardInterrupt:
                        print("\nThank you. Goodbye!")
                        print(f"Saving history to: {cache_dir}/{unique_id}")
                        save_history(messages, args, unique_id)
                        break
                else:
                    print("-"*30)
                    print(f"{BOLD}{CYAN}You (preset): {RESET}{user_input}")
                
                if user_input == "/help":
                    print_help(allowed_dir, server_url, args, unique_id, cache_dir)
                    user_input = ""
                    continue

                messages.append({"role": "user", "content": user_input})
                user_input = ""

            llm_response_flag = False
            llm_tool_response_flag = False

            response_text, tool_calls = utils.call_openai_server(
                messages, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature, 
                model=args.model, 
                server_url=server_url, 
                tools=llm_tools_dict,
                api_key=api_key,
            )

            if response_text:
                llm_response_flag = True
                print(f"{BOLD}{GREEN}Assistant:{RESET} {response_text}")

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
                            if filepath and not os.path.abspath(filepath).startswith(allowed_dir):
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
                        result = call_function_with_timeout(func, tool_params, name)
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
            print(f"Saving history to: {cache_dir}/{unique_id}")
            save_history(messages, args, unique_id)

if __name__ == "__main__":
    main()