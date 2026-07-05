import os
import json

import prompts
import tools
import utils

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
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens for LLM response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--model", type=str, default="default", help="Model name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    allowed_dir = os.path.abspath(args.allowed_dir)
    server_url = f"{args.url}"

    tools_dict = tools.util_get_tools_dict()
    llm_tools_dict = tools.util_get_tools()

    system_prompt = prompts.system_prompt(tools_dict=tools_dict, allowed_dir=allowed_dir)

    messages = [{"role": "system", "content": system_prompt}]

    print(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║       Coding Agent Loop Started\t\t\t\t\t║
    ║  Type 'quit' or 'exit' to exit the loop.\t\t\t\t║
    ║  Allowed dir: {allowed_dir}\t║    
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    while True:
        llm_response_flag = False
        llm_tool_response_flag = False
        try:
            print("-"*30)
            user_input = input(f"{BOLD}{CYAN}You:{RESET} ")
            if user_input.lower() in ["quit", "exit"]:
                print("\nThank you. Goodbye! Saving history...")
                save_history(messages)
                break
        except KeyboardInterrupt:
            print("\nThank you. Goodbye! Saving history...")
            save_history(messages)
            break

        messages.append({"role": "user", "content": user_input})

        response_text, tool_calls = utils.call_openai_server(
            messages, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            model=args.model, 
            server_url=server_url, 
            tools=llm_tools_dict
        )

        if response_text:
            print(f"{BOLD}{GREEN}Assistant:{RESET} {response_text}")
            messages.append({"role": "assistant", "content": response_text})
        else:
            llm_response_flag = True

        if tool_calls:
            # Tool call parsing and execution
            print(f"{BOLD}{YELLOW}Tool Call:{RESET} LLM is trying to call {len(tool_calls)} tools.")
            if tool_calls:
                for call in tool_calls:
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
                        print(f"{BOLD}{RED}Warning:{RESET} Unknown tool: {name}")
        else:
            llm_tool_response_flag = True
        
        if llm_response_flag and llm_tool_response_flag:
            print(f"{BOLD}{RED}Warning:{RESET} No response was given by the LLM.")

        save_history(messages)

if __name__ == "__main__":
    main()