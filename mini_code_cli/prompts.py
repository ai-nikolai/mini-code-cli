# System prompt defining available tools and compaction behavior
def system_prompt(tools_dict=None, allowed_dir=None):
    """
    Takes a list of tool names and returns a system prompt string.
    """

    if tools_dict:
        tool_descriptions = "\n".join([f"{name} : {data['description']}" for name, data in tools_dict.items()])
        base_prompt = "You are a helpful AI assistant with access to the following tools (only):\n"
        end_prompt = (
        "Choose the most appropriate tool for the task at hand. "
        "If a tool fails or you hit a limit, analyze the error and try a different approach. "
        "Be concise and efficient in your tool usage."
        )
    else:
        tool_descriptions = ""
        base_prompt = "You are a helpful AI assistant without any tools."
        end_prompt = ""

        
    allowed_dir_prompt = ""
    if allowed_dir:
        allowed_dir_prompt = f"\nThe path you are allowed to operate in is called: {allowed_dir}"
    
    
    prompt = f"""
{base_prompt}
{tool_descriptions}{end_prompt}{allowed_dir_prompt}
Important: Only follow the user instruction. Do not create additional .md files (unless asked to do so). Do not create unit tests (unless asked to do so), etc.
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