"""Allow ``python -m mini_code_cli`` to start the agent.

The console script ``mini-code`` (see ``[project.scripts]``) is the normal entry
point; this module exists so the package also runs straight from a checkout or
an installed wheel without the script on ``PATH`` -- which is what a reader
tries first after ``pip install -e .``.
"""

from .agent_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
