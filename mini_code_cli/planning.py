"""Planning strategies for ``--plan`` mode.

The default is ``plan-verify``, the strategy that won the clean benchmark
comparison:

  * the loop asks the model to author a plan;
  * after the work settles it asks the model to review the plan;
  * the post-review step is appended to the transcript only when it calls a
    tool. A no-tool answer is shown to the user and ends the turn without being
    stored.

Alternative strategies:

  * ``plan-verify-step`` -- the original behaviour: same plan + review, then
    always run one extra step and keep whatever that step says.
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
DEFAULT_PLAN_STRATEGY = PLAN_VERIFY
