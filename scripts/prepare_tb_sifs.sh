#!/usr/bin/env bash
#
# Prepare every Terminal-Bench 2.1 SIF, verify it, and write the staged dataset
# the launcher can run from (task.toml points at the SIF).
#
#     scripts/prepare_tb_sifs.sh
#     LIMIT=3 VERIFY=test scripts/prepare_tb_sifs.sh
#
# The derived SIFs include the harbor server + python and the baked
# mini_code_cli package; the staged copy is what `TB_STAGED` in
# submit_benchmark_slurm.sh consumes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- edit these -----------------------------------------------------------
BENCH_ENV="${BENCH_ENV:-hb}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PYTHON="${PYTHON:-python}"

SIF_CACHE="${SIF_CACHE:-}"                 # shared: TB goes to $SIF_CACHE/tb
TB_SIF_CACHE="${TB_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/tb}}"
TB_SIF_CACHE="${TB_SIF_CACHE:-benchmarks/.cache/sif}"
TB_STAGED="${TB_STAGED:-experiments/tb-staged}"
TB_TASKS_DIR="${TB_TASKS_DIR:-external_repos/terminal-bench-2-1}"
TB_LIMIT="${TB_LIMIT:-}"                   # empty = all tasks
TB_TASK_IDS="${TB_TASK_IDS:-}"             # space-separated subset, e.g. "bn-fit-modify"
JOBS="${JOBS:-2}"                          # parallel apptainer builds
FORCE="${FORCE:-0}"                        # 1 = rebuild existing SIFs
VERIFY="${VERIFY:-inspect}"                # inspect | test | none
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-}"
# -------------------------------------------------------------------------

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -d "$TB_TASKS_DIR" ]] || die "TB_TASKS_DIR not found: $TB_TASKS_DIR"
command -v apptainer >/dev/null || command -v singularity >/dev/null \
    || die "no apptainer/singularity on PATH"
[[ -f "$CONDA_SH" ]] || die "conda.sh not found: $CONDA_SH"
source "$CONDA_SH"
conda activate "$BENCH_ENV" || die "cannot activate $BENCH_ENV"

mkdir -p "$TB_SIF_CACHE" "$TB_STAGED" "$(dirname "$TB_STAGED")"
[[ -n "$APPTAINER_CACHEDIR" ]] && mkdir -p "$APPTAINER_CACHEDIR" \
    && export APPTAINER_CACHEDIR
[[ -n "$APPTAINER_TMPDIR" ]] && mkdir -p "$APPTAINER_TMPDIR" \
    && export APPTAINER_TMPDIR

say "TB task source : $TB_TASKS_DIR"
say "TB SIF cache   : $TB_SIF_CACHE"
say "TB staged set  : $TB_STAGED"
say "build jobs     : $JOBS   verify: $VERIFY"

CMD=("$PYTHON" -m benchmarks.suites.tb_apptainer prepare
     --tasks-dir "$TB_TASKS_DIR"
     --out "$TB_STAGED"
     --sif-cache "$TB_SIF_CACHE"
     --mini-code "$REPO_ROOT/mini_code_cli"
     --jobs "$JOBS")
[[ -n "$TB_LIMIT" ]] && CMD+=(--limit "$TB_LIMIT")
for task in $TB_TASK_IDS; do
    CMD+=(--task-id "$task")
done
[[ "$FORCE" == "1" ]] && CMD+=(--force)

say "building SIFs ..."
set +e
"${CMD[@]}"
build_rc=$?
set -e

VERIFY_FAILED=0
if [[ "$VERIFY" != "none" ]]; then
    mapfile -t SIFS < <(find "$TB_SIF_CACHE" -maxdepth 1 -type f -name '*.sif' | sort)
    say "verifying ${#SIFS[@]} SIF(s) ($VERIFY)"
    for sif in "${SIFS[@]}"; do
        if ! apptainer inspect "$sif" >/dev/null 2>&1; then
            echo "FAIL inspect: $sif" >&2
            VERIFY_FAILED=$((VERIFY_FAILED + 1))
            continue
        fi
        if [[ "$VERIFY" == "test" ]]; then
            if ! apptainer test "$sif" >/dev/null 2>&1; then
                echo "FAIL test:    $sif" >&2
                VERIFY_FAILED=$((VERIFY_FAILED + 1))
            fi
        fi
    done
fi

SIF_COUNT=$(find "$TB_SIF_CACHE" -maxdepth 1 -type f -name '*.sif' | wc -l)
say "SIFs on disk: $SIF_COUNT"
du -sh "$TB_SIF_CACHE" "$TB_STAGED" 2>/dev/null || true

if (( build_rc != 0 )); then
    echo "error: SIF build reported failures (see output above)" >&2
    exit 1
fi
if (( VERIFY_FAILED > 0 )); then
    echo "error: $VERIFY_FAILED SIF(s) failed verification" >&2
    exit 1
fi
say "TB SIFs ready"
