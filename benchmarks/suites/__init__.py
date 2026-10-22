"""Suite registry: benchmark name -> implementation.

Registering a new benchmark is one entry here plus one module that answers the
three questions in :class:`benchmarks.suites.base.Suite`.
"""
from .base import PlannedCommand, Suite, SuiteError, Trial, TrialOutcome
from .swebench import SweBenchSuite
from .terminal_bench import TerminalBenchSuite

SUITES = {
    TerminalBenchSuite.name: TerminalBenchSuite,
    SweBenchSuite.name: SweBenchSuite,
}

#: Friendly aliases people actually type.
ALIASES = {
    "terminalbench": TerminalBenchSuite.name,
    "tb": TerminalBenchSuite.name,
    "terminal-bench-2": TerminalBenchSuite.name,
    "swe-bench": SweBenchSuite.name,
    "swebench-verified": SweBenchSuite.name,
    "swe-bench-verified": SweBenchSuite.name,
}

__all__ = ["ALIASES", "SUITES", "PlannedCommand", "Suite", "SuiteError", "Trial",
           "TrialOutcome", "get_suite"]


def get_suite(spec):
    """Build the suite for a :class:`~benchmarks.config.SuiteSpec`.

    Aliases are resolved here (not in the config) so the config file can use
    whichever spelling the user prefers while the manifest records the canonical
    name -- reports stay comparable.
    """
    name = ALIASES.get(spec.name, spec.name)
    suite_class = SUITES.get(name)
    if suite_class is None:
        raise SuiteError(
            f"unknown suite {spec.name!r}; known suites: "
            f"{', '.join(sorted(SUITES))} (aliases: {', '.join(sorted(ALIASES))})")
    return suite_class(spec)
