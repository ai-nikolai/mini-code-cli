"""Rebuild the three TB task SIFs under the names harbor actually reuses.

harbor names its cached image ``<docker_image with / -> _, : -> _>.sif`` and
converts the published image as-is, so the baked extras (python3, curl, tmux,
/opt/harbor-server with uvicorn) have to be put in *that* file -- a
``<task>__<task>.sif`` from ``tb_apptainer prepare --tasks-dir`` is never read by
the campaign, because a local task checkout skips the prepare step entirely.
"""

import os
import sys

sys.path.insert(0, os.getcwd())
from benchmarks.suites import tb_apptainer as T

TASKS = ["adaptive-rejection-sampler", "bn-fit-modify", "break-filter-js-from-html"]
CACHE = os.path.join("benchmarks", ".cache", "sif")
# Build logs next to the scratch tree when one is configured, so a 30-minute apt
# run can be watched without writing into the repository.
LOGS = os.environ.get("BENCHMARK_SCRATCH") or os.path.join(CACHE, "logs")

os.makedirs(CACHE, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
for task in TASKS:
    image = f"alexgshaw/{task}:20251031"
    sif = os.path.join(CACHE, image.replace("/", "_").replace(":", "_") + ".sif")
    print(f"[build] {task} -> {sif}", flush=True)
    # build_sif takes a *path* to a definition file (prepare() writes one first),
    # and apptainer rejects a definition passed as an argument.
    definition = os.path.join(LOGS, f"{task}.def")
    with open(definition, "w", encoding="utf-8") as handle:
        handle.write(T.render_def(image))
    ok, message = T.build_sif(
        "apptainer", definition, sif,
        os.path.join(LOGS, f"build-{task}.log"), force=True, timeout=5400)
    print(f"[{task}] ok={ok} {message}", flush=True)
