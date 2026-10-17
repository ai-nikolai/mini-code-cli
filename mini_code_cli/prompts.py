# System prompt defining available tools and compaction behavior
def system_prompt(tools_dict=None, allowed_dir=None, plan_mode=False, skills_dir=None, session_dir=None):
    """
    Takes a list of tool names and returns a system prompt string.

    When ``plan_mode`` is True the prompt also explains that the loop keeps a
    running goal (CURRENT_GOAL.md) and plan (CURRENT_PLAN.md) in the session
    directory (see plan.py) for the human to inspect; the model does not author
    those files itself.
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

    skills_prompt = ""
    if skills_dir is not None:
        skills_prompt = "\n" + skills_instructions(skills_dir)

    plan_mode_prompt = ""
    if plan_mode:
        plan_mode_prompt = "\n" + plan_instructions(session_dir)
    
    
    prompt = f"""
{base_prompt}
{tool_descriptions}{end_prompt}{allowed_dir_prompt}{plan_mode_prompt}{skills_prompt}
Important: Follow the user instruction. Do not create additional .md files (unless asked to do so). Do not create unit tests (unless asked to do so), etc.
"""
    return prompt.strip()


def plan_instructions(session_dir=None):
    """The extra system-prompt block appended when ``--plan`` mode is enabled.

    Explains that the loop -- not the agent -- maintains two human-readable
    tracking files in the session directory (see plan.py): CURRENT_GOAL.md
    (the user's latest instruction) and CURRENT_PLAN.md (a running log of the
    work, finished when the goal is complete). They exist so the human can
    inspect progress in the session directory; the model should just do the
    work and not try to maintain these files itself (there are no tools for it,
    and the files live outside ``--allowed-dir`` anyway).
    """
    where = session_dir if session_dir else "the session directory (in the cache dir)"
    return (
        "PLAN MODE: Two human-readable tracking files are kept in the "
        f"session directory ({where}) and maintained by the loop, not by you, "
        "so they can be used to track progress:\n"
        "1. CURRENT_GOAL.md -- the user's latest instruction / goal.\n"
        "2. CURRENT_PLAN.md -- a running log of the work performed, ending with "
        "a 'Task finished: <goal>' marker once the goal is complete.\n"
        "You do NOT write or edit these files (no tool does, and they live "
        "outside the directory you are allowed to operate in); just carry out "
        "the work. The current plan + goal are re-sent to you after any history "
        "compaction so you can always see what you are still working toward."
    )

def skills_instructions(skills_dir):
    """The extra system-prompt block when the skills feature is enabled.

    Tells the agent it has a local ``.mini`` directory of reusable skills and
    explains the skill tools (see :mod:`mini_code_cli.skills`). ``skills_dir``
    is the parent allowed dir; the concrete ``.mini`` path is computed for the
    prompt. The agent must maintain skill files with the scoped
    read_skill_file / write_skill_file tools, which write/read ONLY inside the
    ``.mini`` directory, so skill files always land in the right place.
    """
    from . import skills as skills_mod
    return (
        "SKILLS MODE: You can work with reusable skills kept in the local "
        f"`.mini` directory at: {skills_mod.skills_dir(skills_dir)}.\n"
        "Each skill is a sub-directory containing a `skill.md` (what the skill is "
        "/ how to use it) and optional executable Python helper scripts. There is "
        "also a high-level `SKILLS.md` summary of all skills. All skill files live "
        "inside the ``.mini`` directory above.\n"
        "Use these tools to work with skills:\n"
        "  * `list_skills`  - see which skills exist (name + one-line summary).\n"
        "  * `read_skill`   - read a skill's `skill.md` by name, to learn how to "
        "use it or which scripts it provides.\n"
        "  * `create_skill` - create a new skill (its sub-directory and `skill.md`) "
        "when you build a reusable capability. Give a clear, single-segment name "
        "and a short summary. It is normal to then add Python helper scripts into "
        "the skill's directory. Use the scoped tools below (not the generic "
        "read_file / write_file) so the skill files are written into the ``.mini`` "
        "directory correctly.\n"
        "  * `write_skill_file` - write a file into a skill's directory (e.g. "
        "a new skill's `skill.md` or a helper script), giving its path inside "
        "the ``.mini`` directory (e.g. \"my_skill/skill.md\"). This tool is "
        "scoped so it can only write inside the ``.mini`` directory.\n"
        "  * `read_skill_file`  - read a file that already lives inside the "
        "``.mini`` directory (e.g. to check what an existing skill's script "
        "contains). This tool is scoped so it can only read inside ``.mini``.\n"
        "  * `run_skill`    - run one of a skill's Python scripts, passing any "
        "command-line arguments via its `args`. Read the skill first to know "
        "which script to run and what arguments it expects.\n"
        "Prefer turning reusable helpers into skills (with a `skill.md` plus "
        "scripts) so they can be reused later, rather than repeating one-off work.\n"
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
