# System prompt defining available tools and compaction behavior
def system_prompt(tools_dict=None, allowed_dir=None):
    """
    Takes a list of tool names and returns a system prompt string.
    """
    allowed_dir_prompt = ""
    if allowed_dir:
        allowed_dir_prompt = f"\nThe path you are allowed to operate in is called: {allowed_dir}"
    if tools_dict:
        tool_descriptions = "\n".join([f"{name} : {data['description']}" for name, data in tools_dict.items()])
        base_prompt = "You are a helpful AI coding assistant with access to the following tools:\n"
        end_prompt = "\n\nWhen answering, first use available tools if needed, then provide a compact summary of the results."
    else:
        tool_descriptions = ""
        base_prompt = "You are a helpful AI coding assistant without any tools."
        end_prompt = ""
    
    prompt = f"""
{base_prompt}
{tool_descriptions}{end_prompt}{allowed_dir_prompt}
Keep responses concise and focused.
"""
    return prompt.strip()

def compaction_prompt(messages):
    """
    Takes a messages list and returns a compact prompt string summarizing the conversation.
    """
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt = f"""
Please analyze the following conversation and output a concise summary covering:
1. Key topics discussed
2. Any decisions made
3. Important information or facts provided
4. Action items or next steps (if any)

Keep the summary to 2-3 sentences.

Conversation:
{conversation}
"""
    return prompt.strip()

if __name__=="__main__":
    from . import tools
    tools_dict = tools.util_get_tools_dict()
    print(system_prompt(tools_dict))

    print("="*30)

    print(system_prompt())