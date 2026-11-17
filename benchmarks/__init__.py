"""Benchmark harness for mini-code-cli: Terminal-Bench, SWE-bench and friends.

This package is an *orchestration layer*, not a reimplementation: every suite is
executed by its own official harness (``terminal-bench``'s ``tb`` CLI and the
``swebench`` evaluation harness), and every agent is driven through its own CLI.
What lives here is the part those projects do not provide -- one configuration
format, one agent/setting/model matrix, reproducible run directories with a
manifest, artifact capture, and a report that can be compared across agents.

Nothing in this package is imported by ``mini_code_cli`` itself, the heavyweight
dependencies are optional (see ``benchmarks/requirements.txt``), and no dataset, log
or run artifact is ever installed with the CLI (``pyproject.toml`` only packages
``mini_code_cli*``).

Typical use, from a checkout::

    python -m benchmarks list
    python -m benchmarks plan   --config benchmarks/configs/default.json
    python -m benchmarks run    --config benchmarks/configs/default.json --limit 2
    python -m benchmarks report --run benchmarks/runs/<run-id>

``plan`` is a pure dry run: it prints the exact commands (and Docker actions) that
``run`` would execute, without downloading a dataset or starting a container.
"""

__all__ = ["__version__"]

# Kept separate from mini_code_cli.__version__ on purpose: the benchmarks are
# versioned by the harness/dataset they wrap (recorded per run in manifest.json),
# not by the CLI release they happen to live next to.
__version__ = "0.1.0"
