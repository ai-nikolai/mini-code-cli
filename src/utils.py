import requests
import json
import argparse


def call_openai_server(messages, tools=None, max_tokens=2048, temperature=0.0, top_p=0.95, model="default", server_url=None, api_key=None):
    """
    Querying an OpenAI-compatible server (local or remote) using the requests library.
    Supports authentication via API key if provided.
    """
    url = f"{server_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

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
        tool_calls = result.get("tool_calls", [])
        return content, tool_calls

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI Server...: {e}")
        return None, None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None, None

# Alternative: use the newer SGLang API format (v0.3+)
def call_openai_server_prompt(prompt, max_tokens=1024, temperature=0.0, top_p=0.95, model="default", server_url=None, api_key=None):
    """
    OpenAI compatible call, pre-creating the messages object.
    """
    messages = [
            {"role": "user", "content": prompt}
        ]

    return call_openai_server(messages, max_tokens, temperature, top_p, model, server_url=server_url, api_key=api_key)

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=30000, help="API port")
    parser.add_argument("--prompt", type=str, default="What is the purpose of life?", help="prompt")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication (optional)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Configuration for local SGLang deployment
    server_url=f"http://{args.host}:{args.port}" 

    print("="*60)
    print("Entering LLM call:")
    print("-"*30)
    prompt = args.prompt

    print(f"Prompt:\n{prompt}")
    print("---")
 
    response_text, tools = call_openai_server_prompt(prompt, server_url=server_url, api_key=args.api_key)
    if response_text:
        print(f"Response:\n{response_text}")
        print("---")

    # Test call to DeepSeek endpoint
    deepseek_server_url = "https://api.deepseek.com"
    deepseek_model = "deepseek-v4-flash"
    print("="*60)
    print("Entering DeepSeek call:")
    print("-"*30)
    response_text_ds, tools_ds = call_openai_server_prompt(prompt, server_url=deepseek_server_url, api_key=args.api_key, model=deepseek_model)
    if response_text_ds:
        print(f"DeepSeek Response:\n{response_text_ds}")
        print("---")
