# System prompt defining available tools and compaction behavior
def system_prompt(tools_dict=None, allowed_dir=None, plan_mode=False):
    """
    Takes a list of tool names and returns a system prompt string.

    When ``plan_mode`` is True the prompt also instructs the agent to maintain
    a running goal (CURRENT_GOAL.md) and plan (CURRENT_PLAN.md) in the session
    directory -- see plan.py.
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

    plan_mode_prompt = ""
    if plan_mode:
        plan_mode_prompt = "\n" + plan_instructions()
    
    
    prompt = f"""
{base_prompt}
{tool_descriptions}{end_prompt}{allowed_dir_prompt}{plan_mode_prompt}
Important: Follow the user instruction. Do not create additional .md files (unless asked to do so). Do not create unit tests (unless asked to do so), etc.
"""
    return prompt.strip()


def plan_instructions():
    """The extra system-prompt block appended when ``--plan`` mode is enabled.

    Tells the agent to maintain two tracking files in the session directory
    (see plan.py): ``CURRENT_GOAL.md`` holding the user's latest instruction,
    and ``CURRENT_PLAN.md`` holding the agent's plan for reaching it. The plan
    file must be updated as the task progresses and marked finished when done.
    """
    return (
        "PLAN MODE: You must keep your work organized in two files in the "
        "current session directory.\n"
        "1. CURRENT_GOAL.md: at the very start, write the user's current "
        "instruction / goal (verbatim) to this file. Always keep it up to date "
        "with the user's latest instruction.\n"
        "2. CURRENT_PLAN.md: first, write a clear, step-by-step plan for how you "
        "will accomplish the goal. As you work, update this file to reflect "
        "progress. Once you have completely finished the current goal, you MUST "
        "append an entry to CURRENT_PLAN.md clearly marking the task as finished "
        "(e.g. 'Task finished: <goal>').\n"
        "These files must always be kept in the session directory so the work is "
        "easy to track and resume."
    )

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
