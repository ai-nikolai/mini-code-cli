"""Deterministic ``mock`` agent: exercises the harness without a model.

Every orchestration bug this benchmark package can have -- a wrong mount, a lost
patch, a mis-parsed result, a broken resume -- shows up with the mock agent too,
for free. It is the only agent that is safe to run in CI.

Behaviour is chosen with ``--behaviour`` so a test can ask for a known outcome:

================  =========================================================
``empty``         do nothing, exit 0 (a patch that changes nothing: "unresolved")
``touch``         create ``mock_solution.txt`` in the workspace, so the
                  captured diff is non-empty ("resolved", in a toy repo)
``fail``          exit 2 with a message on stderr (an agent that crashed)
``timeout``       sleep far longer than any sane budget
================  =========================================================
"""
import argparse
import os
import sys
import time

BEHAVIOURS = ("empty", "touch", "fail", "timeout")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="benchmarks.agents.mock",
                                     description="Deterministic stand-in agent")
    parser.add_argument("--prompt", default="", help="the task instruction (ignored)")
    parser.add_argument("--workspace", default=".", help="directory to act in")
    parser.add_argument("--behaviour", default="empty", choices=BEHAVIOURS)
    parser.add_argument("--file", default="mock_solution.txt",
                        help="file to create for behaviour=touch")
    args = parser.parse_args(argv)

    print(f"mock agent: behaviour={args.behaviour} workspace={args.workspace}")
    if args.behaviour == "empty":
        return 0
    if args.behaviour == "touch":
        target = os.path.join(args.workspace, args.file)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("mock solution\n")
        print(f"mock agent: wrote {target}")
        return 0
    if args.behaviour == "fail":
        print("mock agent: deliberate failure", file=sys.stderr)
        return 2
    # timeout: long enough to trip any per-instance budget
    time.sleep(86_400)
    return 0


if __name__ == "__main__":
    sys.exit(main())
