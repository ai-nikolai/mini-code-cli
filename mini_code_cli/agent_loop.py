"""
Usage:
    mini-code

Prompt:

write a similar model.py file but for a LSTM based language model. Make it minimal and call it model_lstm.py it should still be compatible with eval.py

"""

import os
import json

from . import prompts
from . import tools
from . import utils

import argparse
import sys

# ANSI escape codes for colors and bold
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def save_history(messages):
    # Save messages to history.json
    with open("history.json", "w") as f:
        json.dump(messages, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    parser.add_argument("--url", type=str, default="http://0.0.0.0:30000", help="API host")
    parser.add_argument("--system_prompt", type=str, help="API host")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--model", type=str, default="default", help="Model name")
    parser.add_argument("--step_by_step", action="store_true", help="Whether to run the agent step by step. Default is to run in 'Permission' mode.")
    parser.add_argument("--ask_permission", action="store_true", help="Ask for permission before any tool call.")

    args = parser.parse_args()
    return args

def print_welcome(allowed_dir, server_url, args):
    # Calculate the length of the longest line inside the box
    version = __import__('mini_code_cli').__version__
    
    permission_mode = "All tool calls require permission." if args.ask_permission else "Tool calls will execute without asking for permission."
    agent_mode = "Agent is in manual-mode." if args.step_by_step else "Agent is in auto-mode."

    lines = [
        f"  MiniCodeCLI (v{version})",
         "  Type 'quit' or 'exit' to exit the loop.",
        f"  Allowed dir: {allowed_dir}",
        f"  Server URL: {server_url}",
        f"  {permission_mode}",
        f"  {agent_mode}",
         "  Shell disabled."
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

def print_help(allowed_dir, server_url, args):
    print_welcome(allowed_dir, server_url, args)
    
    help_message = ""
    help_message += "Manual-mode means the agent will ask for user input after each LLM prediction." if args.step_by_step else "Auto-mode means the agent will keep calling the LLM until no further LLM calls are possible."

    print(f"{BOLD}{GREEN}HELP:{RESET} {help_message}")

def main():
    args = parse_args()

    allowed_dir = os.path.abspath(args.allowed_dir)
    server_url = f"{args.url}"

    tools_dict = tools.util_get_tools_dict()
    llm_tools_dict = tools.util_get_tools()

    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = prompts.system_prompt(tools_dict=tools_dict, allowed_dir=allowed_dir)

    messages = [{"role": "system", "content": system_prompt}]

    print_welcome(allowed_dir, server_url, args)

    llm_response_flag = False
    llm_tool_response_flag = False
    llm_repeat_flag = False
    tool_count = 0
    
    while True:
        if llm_tool_response_flag and not args.step_by_step: #skip user input and try to call the model again until there no more tool calls.
            print(f"{BOLD}{YELLOW}System:{RESET} Skipping user input. Continuing with LLM calls until no more tool calls.")
            llm_repeat_flag = True
        else:
        # get user input
            if llm_repeat_flag:
                print(f"{BOLD}{YELLOW}System:{RESET} Called {tool_count} tools in total. Exiting tool loop.")

            tool_count = 0
            try:
                print("-"*30)
                user_input = input(f"{BOLD}{CYAN}You:{RESET} ")
                if user_input.lower() in ["quit", "exit", "/quit", "/exit"]:
                    print("\nThank you. Goodbye! Saving history...")
                    save_history(messages)
                    break
            except KeyboardInterrupt:
                print("\nThank you. Goodbye! Saving history...")
                save_history(messages)
                break
            
            if user_input == "/help":
                print_help(allowed_dir, server_url, args)
                continue

            messages.append({"role": "user", "content": user_input})

        llm_response_flag = False
        llm_tool_response_flag = False

        response_text, tool_calls = utils.call_openai_server(
            messages, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            model=args.model, 
            server_url=server_url, 
            tools=llm_tools_dict
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
                            save_history(messages)
                            sys.exit(0)

                        if not permission_input in ['y', 'yes']:
                            print(f"{BOLD}{YELLOW}Tool permission denied:{RESET}.")
                            messages.append({
                                "role": "user",
                                "content": f"User denied permission for tool {name} and params {params_str}."
                            })
                            continue
                    result = func(**tool_params)
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
                    print(f"{BOLD}{RED}Warning:{RESET} Unknown tool: {name}, params {params_str}")
        else:
            messages.append({
                    "role": "assistant",
                    "content": response_text,
                    # "tool_calls": tool_calls
            })
        
        if not llm_response_flag and not llm_tool_response_flag:
            print(f"{BOLD}{RED}Warning:{RESET} No response was given by the LLM.")

        save_history(messages)

if __name__ == "__main__":
    main()