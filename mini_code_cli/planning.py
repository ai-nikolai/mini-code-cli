"""Planning strategies for ``--plan`` mode.

The loop's built-in plan handling is :func:`plan_verify_step`, the original
implementation:

  * the loop asks the model to author a plan;
  * after the work settles it asks the model to review the plan;
  * it runs one more step so the review can drive further work.

Alternative strategies:

  * ``plan-verify`` -- same plan + review, but the post-review step is only
    appended to the transcript when it calls a tool. A no-tool answer is shown
    to the user and ends the turn without being stored.
  * ``open-plan`` -- the agent owns a plan document in ``--allowed-dir/.mini``
    and reads/writes it with the normal file tools; the loop neither authors
    nor reviews a plan.

Keeping the names here (rather than inline in ``agent_loop`` or ``argparse``)
means a future strategy only has to be added in one place plus the loop branch
that implements it.
"""

PLAN_VERIFY_STEP = "plan-verify-step"
PLAN_VERIFY = "plan-verify"
OPEN_PLAN = "open-plan"

PLAN_STRATEGIES = (PLAN_VERIFY_STEP, PLAN_VERIFY, OPEN_PLAN)
DEFAULT_PLAN_STRATEGY = PLAN_VERIFY_STEP
