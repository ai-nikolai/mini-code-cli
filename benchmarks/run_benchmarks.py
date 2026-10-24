#!/usr/bin/env python3
"""``benchmarks/run_benchmarks.py`` -- the one command the docs tell you to run.

    python benchmarks/run_benchmarks.py --dataset swe-bench --agent mini-code --mode short_eval
    python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent all --mode short

It is a three-line wrapper: the logic lives in :mod:`benchmarks.campaign`, which is
also reachable as ``python -m benchmarks run -c <config> --suite <suite>`` (the
lower-level form) and as ``python -m benchmarks``. Keeping the wrapper *inside*
the package directory means the repository root holds no benchmark-specific
files, and ``sys.path`` only needs the parent of ``benchmarks/``.
"""
import os
import sys

# benchmarks/ -> repository root: the package is imported from source here, not
# from site-packages, so the parent directory has to be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.campaign import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
