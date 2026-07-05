import os
import json

import prompts
import tools
import utils

import argparse
import sys

def save_history(messages):
    # Save messages to history.json
    with open("history.json", "w") as f:
        json.dump(messages, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=30000, help="API port")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens for LLM response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--model", type=str, default="default", help="Model name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    allowed_dir = os.path.abspath(args.allowed_dir)
    sglang_url = f"http://{args.host}:{args.port}"

    tools_dict = tools.util_get_tools_dict()
    system_prompt = prompts.system_prompt(tools_dict)

    messages = [{"role": "system", "content": system_prompt}]

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       Coding Agent Loop Started                               ║
    ║  Type 'quit' or 'exit' to exit the loop.                      ║
    ║  Allowed dir: {allowed_dir}                                   ║    
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            save_history(messages)
            break

        messages.append({"role": "user", "content": user_input})

        response_text, tool_calls = utils.call_sglang(messages, max_tokens=args.max_tokens, temperature=args.temperature, model=args.model, sglang_url=sglang_url)

        if response_text:
            print(f"Assistant: {response_text}")
            messages.append({"role": "assistant", "content": response_text})

            # Tool call parsing and execution
            if tool_calls:
                for call in tool_calls:
                    call_id = call.get("id", "")
                    name = call.get("function", {}).get("name", "")
                    params_str = call.get("function", {}).get("arguments", "{}")
                    try:
                        tool_params = json.loads(params_str)
                    except json.JSONDecodeError:
                        print(f"Error parsing arguments for tool {name}")
                        continue

                    if name in tools_dict:
                        # Check allowed directory for file operations
                        if name in ["read_file", "write_file", "grep", "glob"]:
                            filepath = tool_params.get("filepath", "")
                            if name == "write_file":
                                filepath = tool_params.get("filepath", "")
                            if filepath and not os.path.abspath(filepath).startswith(allowed_dir):
                                print(f"Error: File operation not allowed outside allowed directory: {allowed_dir}")
                                continue

                        func = tools_dict[name]["function"]
                        result = func(**tool_params)
                        print(f"Tool '{name}' returned: {result}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": str(result)
                        })
                    else:
                        print(f"Unknown tool: {name}")
        
        save_history(messages)

if __name__ == "__main__":
    main()