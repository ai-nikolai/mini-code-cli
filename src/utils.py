import requests
import json
import argparse


def call_sglang(messages, tools=None, max_tokens=2048, temperature=0.0, top_p=0.95, model="default", sglang_url=None):
    """
    Querying sglang using requests library.
    """
    # global SGLANG_URL
    url = f"{sglang_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json().get("choices", [{}])[0].get("message", {})
        content = result.get("content", "")
        tool_call = msg.get("tool_calls", [])
        return content, tool_call

    except requests.exceptions.RequestException as e:
        print(f"Error calling SGLang: {e}")
        return None, None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None, None

# Alternative: use the newer SGLang API format (v0.3+)
def call_sglang_prompt(prompt, max_tokens=1024, temperature=0.0, top_p=0.95, model="default", sglang_url=None):
    """
    Alternative implementation using the chat completion API (SGLang v0.3+)
    """
    messages = [
            {"role": "user", "content": prompt}
        ]

    return call_sglang(messages, max_tokens, temperature, top_p, model, sglang_url=sglang_url)

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=30000, help="API port")
    parser.add_argument("--prompt", type=str, default="What is the purpose of life?", help="prompt")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Configuration for local SGLang deployment
    # global SGLANG_URL
    # SGLANG_URL = f"http://{args.host}:{args.port}"  # Default SGLang port, adjust if needed 
    sglang_url=f"http://{args.host}:{args.port}" 

    print("="*60)
    print("Entering LLM call:")
    print("-"*30)
    prompt = args.prompt

    print(f"Prompt:\n{prompt}")
    print("---")
 
    response_text, tools = call_sglang_prompt(prompt, sglang_url=sglang_url)
    if response_text:
        print(f"Response:\n{response_text}")
        print("---")