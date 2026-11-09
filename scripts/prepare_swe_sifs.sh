#!/usr/bin/env bash
#
# Prepare every SWE-bench SIF for one split, verify the images, and keep the
# instance list the grader needs.
#
#     scripts/prepare_swe_sifs.sh
#     SWE_DATASET=SWE-bench/SWE-bench_Verified scripts/prepare_swe_sifs.sh
#     SWE_LIMIT=5 VERIFY=exec scripts/prepare_swe_sifs.sh
#
# Pulling is serial (one `apptainer pull` per instance) and a full Lite split is
# ~385 GB at the measured mean; point SIF_CACHE at scratch storage.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- edit these -----------------------------------------------------------
BENCH_ENV="${BENCH_ENV:-hb}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PYTHON="${PYTHON:-python}"

SIF_CACHE="${SIF_CACHE:-}"                 # shared: SWE goes to $SIF_CACHE/swe
SWE_SIF_CACHE="${SWE_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/swe}}"
SWE_SIF_CACHE="${SWE_SIF_CACHE:-benchmarks/runs/_cache/swe-sif}"
SWE_DATASET="${SWE_DATASET:-SWE-bench/SWE-bench_Lite}"
SWE_SPLIT="${SWE_SPLIT:-test}"
SWE_INSTANCES="${SWE_INSTANCES:-experiments/swe-instances/instances.jsonl}"
SWE_PULL_LOGS="${SWE_PULL_LOGS:-experiments/swe-pulls}"
SWE_LIMIT="${SWE_LIMIT:-}"                 # empty = all instances
SWE_INSTANCE_IDS="${SWE_INSTANCE_IDS:-}"   # comma-separated subset
REFRESH="${REFRESH:-0}"                    # 1 = rewrite instances.jsonl
FORCE="${FORCE:-0}"                        # 1 = re-pull existing SIFs
VERIFY="${VERIFY:-inspect}"                # inspect | exec | none
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-}"
# -------------------------------------------------------------------------

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

command -v apptainer >/dev/null || command -v singularity >/dev/null \
    || die "no apptainer/singularity on PATH"
[[ -f "$CONDA_SH" ]] || die "conda.sh not found: $CONDA_SH"
source "$CONDA_SH"
conda activate "$BENCH_ENV" || die "cannot activate $BENCH_ENV"

mkdir -p "$SWE_SIF_CACHE" "$SWE_PULL_LOGS" "$(dirname "$SWE_INSTANCES")"
[[ -n "$APPTAINER_CACHEDIR" ]] && mkdir -p "$APPTAINER_CACHEDIR" \
    && export APPTAINER_CACHEDIR
[[ -n "$APPTAINER_TMPDIR" ]] && mkdir -p "$APPTAINER_TMPDIR" \
    && export APPTAINER_TMPDIR

say "SWE dataset    : $SWE_DATASET ($SWE_SPLIT)"
say "instance list  : $SWE_INSTANCES"
say "SWE SIF cache  : $SWE_SIF_CACHE"
say "pull logs      : $SWE_PULL_LOGS"
say "verify         : $VERIFY"

if [[ "$REFRESH" == "1" || ! -f "$SWE_INSTANCES" ]]; then
    CMD=("$PYTHON" -m benchmarks.suites.swebench_env instances
         --dataset "$SWE_DATASET" --split "$SWE_SPLIT"
         --out "$SWE_INSTANCES")
    [[ -n "$SWE_LIMIT" ]] && CMD+=(--limit "$SWE_LIMIT")
    [[ -n "$SWE_INSTANCE_IDS" ]] && CMD+=(--instance-ids "$SWE_INSTANCE_IDS")
    say "materialising instances ..."
    "${CMD[@]}"
else
    say "reusing instance list (REFRESH=0)"
fi

PULL=("$PYTHON" -m benchmarks.suites.swebench_apptainer pull
      --instances "$SWE_INSTANCES"
      --sif-cache "$SWE_SIF_CACHE"
      --log-dir "$SWE_PULL_LOGS")
[[ -n "$SWE_LIMIT" ]] && PULL+=(--limit "$SWE_LIMIT")
[[ "$FORCE" == "1" ]] && PULL+=(--force)

say "pulling SIFs (serial) ..."
set +e
"${PULL[@]}"
pull_rc=$?
set -e

VERIFY_FAILED=0
if [[ "$VERIFY" != "none" ]]; then
    mapfile -t SIFS < <(find "$SWE_SIF_CACHE" -maxdepth 1 -type f -name 'swe-*.sif' | sort)
    say "verifying ${#SIFS[@]} SIF(s) ($VERIFY)"
    for sif in "${SIFS[@]}"; do
        [[ -f "$sif.complete" ]] || echo "WARN no .complete marker: $sif" >&2
        if ! apptainer inspect "$sif" >/dev/null 2>&1; then
            echo "FAIL inspect: $sif" >&2
            VERIFY_FAILED=$((VERIFY_FAILED + 1))
            continue
        fi
        if [[ "$VERIFY" == "exec" ]]; then
            if ! apptainer exec --cleanenv "$sif" /bin/true >/dev/null 2>&1; then
                echo "FAIL exec:    $sif" >&2
                VERIFY_FAILED=$((VERIFY_FAILED + 1))
            fi
        fi
    done
fi

SIF_COUNT=$(find "$SWE_SIF_CACHE" -maxdepth 1 -type f -name 'swe-*.sif' | wc -l)
say "SIFs on disk: $SIF_COUNT"
du -sh "$SWE_SIF_CACHE" "$SWE_INSTANCES" 2>/dev/null || true

if (( pull_rc != 0 )); then
    echo "error: SIF pull reported failures (see $SWE_PULL_LOGS)" >&2
    exit 1
fi
if (( VERIFY_FAILED > 0 )); then
    echo "error: $VERIFY_FAILED SIF(s) failed verification" >&2
    exit 1
fi
say "SWE SIFs ready"
