#!/usr/bin/env bash
#
# Generate (and optionally submit) one Slurm job for one model, one or more
# benchmarks, and one or more agents. vLLM only.
#
#     benchmarks/scripts/submit_benchmark_slurm.sh
#     DRY_RUN=1 SUBMIT=0 benchmarks/scripts/submit_benchmark_slurm.sh
#
# The generated job lives under experiments/slurm/ (git-ignored). It starts vLLM
# on NUM_GPUS GPUs, waits for /health, then runs the benchmarks x agents
# sequentially. For a cross-product of models/benchmarks/agents, use
# benchmarks/scripts/submit_benchmark_matrix.sh, which calls this script once per cell.

set -euo pipefail

# --- edit these -----------------------------------------------------------
VERSION="${VERSION:-v1}"
WALLTIME="${WALLTIME:-48:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"                  # 1..4 (tensor-parallel size)
SCRATCH_GRES="${SCRATCH_GRES:-}"            # e.g. 100g; local NVMe at $TMPDIR

MODEL_NAME="${MODEL_NAME:-ornith-ai/Ornith-1.5-35B-A3B}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"   # seconds to wait for /health
MODEL_ENV="${MODEL_ENV:-env_vllm30}"       # conda env with vLLM
BENCH_ENV="${BENCH_ENV:-hb}"               # conda env with harbor + swebench
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"         # empty = server default
SERVER_EXTRA="${SERVER_EXTRA:-}"           # extra vLLM flags, word-split

TB_CONFIG="${TB_CONFIG:-benchmarks/configs/terminal_bench_21.json}"
SWE_CONFIG="${SWE_CONFIG:-benchmarks/configs/ornith_30b.json}"

# Where the large artefacts live. Empty = the repository defaults.
SIF_CACHE="${SIF_CACHE:-}"                 # e.g. /scratch/$USER/sif  (TB + SWE)
TB_CACHE="${TB_CACHE:-}"                   # e.g. /scratch/$USER/cache (datasets/TB)
SCRATCH_DIR="${SCRATCH_DIR:-}"             # e.g. /scratch/$USER/tmp (apptainer unpack)
TB_STAGED="${TB_STAGED:-experiments/tb-staged}"  # prepared Terminal-Bench staged dataset
TB_USE_STAGED="${TB_USE_STAGED:-1}"        # 1 = run from TB_STAGED when prepared.json exists
TB_SIF_CACHE="${TB_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/tb}}"  # default: repo
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-}"

# Full sets: terminalbench-2.1 (90 tasks) and swe-bench-lite-full (300
# instances). Use terminalbench-2.1-3 or swe-bench-lite for the smoke sets.
BENCHMARKS="${BENCHMARKS:-terminalbench-2.1 swe-bench-lite-full}"
AGENTS="${AGENTS:-mini-code pi claude-code}"
MINI_CODE_SETTINGS="${MINI_CODE_SETTINGS:-swe}"  # space-separated; "all" = every configured setting
MODE="${MODE:-full}"                       # full | short_eval | short
LIMIT="${LIMIT:-}"                         # empty = full; e.g. 5 for smoke
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"  # e.g. 1 (900 s/task)
N_CONCURRENT="${N_CONCURRENT:-1}"          # inside a benchmark, not across agents
FULL="${FULL:-1}"                          # 1 = drop smoke limits/task_ids
RUN_PREFIX="${RUN_PREFIX:-$VERSION}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"               # extra run_benchmarks.py flags

PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
SUBMIT="${SUBMIT:-1}"                      # 1 = sbatch; 0 = only write the file
DRY_RUN="${DRY_RUN:-0}"                    # 1 = no sbatch
CONFIG_PYTHON="${CONFIG_PYTHON:-python3}"
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

[[ "$NUM_GPUS" =~ ^[1-4]$ ]] || die "NUM_GPUS must be 1..4 (got $NUM_GPUS)"
[[ -n "$MODEL_NAME" ]] || die "MODEL_NAME is empty"
[[ -n "$BENCHMARKS" ]] || die "BENCHMARKS is empty"
[[ -n "$AGENTS" ]] || die "AGENTS is empty"
[[ -n "$LIMIT" ]] && FULL=0

SHORT_MODEL_NAME="$(printf '%s' "$MODEL_NAME" | cut -d/ -f2 \
    | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SIGNATURE="${VERSION}_${SHORT_MODEL_NAME}_${NUM_GPUS}gpu"
SLURM_DIR="$EXPERIMENTS_DIR/slurm"
CONFIG_DIR="$EXPERIMENTS_DIR/configs"
LOG_DIR="$EXPERIMENTS_DIR/logs"
mkdir -p "$SLURM_DIR" "$CONFIG_DIR" "$LOG_DIR"

FULL_FLAG=""
[[ "$FULL" == "1" ]] && FULL_FLAG="--full"

declare -a BENCH_LIST=() CONFIG_LIST=()
for bench in $BENCHMARKS; do
    case "$bench" in
        swe-bench*|swebench*) template="$SWE_CONFIG" ;;
        terminal*|tb*)        template="$TB_CONFIG" ;;
        *) die "unknown benchmark '$bench'" ;;
    esac
    slug="$(printf '%s' "$bench" | sed 's/[^a-zA-Z0-9]/_/g')"
    config="$CONFIG_DIR/${SIGNATURE}_${slug}.json"
    SIF_FLAG=""
    [[ -n "$SIF_CACHE" ]] && SIF_FLAG="--sif-cache $SIF_CACHE/swe"
    TB_STAGED_FLAG=""
    if [[ "$TB_USE_STAGED" == "1" && -n "$TB_STAGED" \
            && -f "$TB_STAGED/prepared.json" ]]; then
        TB_STAGED_FLAG="--tb-staged-dir $(printf '%q' "$TB_STAGED")"
    fi
    TB_SIF_FLAG=""
    [[ -n "$TB_SIF_CACHE" ]] && TB_SIF_FLAG="--tb-sif-cache $TB_SIF_CACHE"
    # shellcheck disable=SC2086
    "$CONFIG_PYTHON" "$SCRIPT_DIR/make_benchmark_config.py" "$template" "$config" \
        --model "$MODEL_NAME" \
        --base-url "http://127.0.0.1:${SERVER_PORT}" $FULL_FLAG $SIF_FLAG \
        $TB_STAGED_FLAG $TB_SIF_FLAG
    BENCH_LIST+=("$bench")
    CONFIG_LIST+=("$config")
done

JOB_FILE="$SLURM_DIR/bench_${TIMESTAMP}_${SIGNATURE}.sh"
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"
TIMEOUT_ARG=""
[[ -n "$AGENT_TIMEOUT_MULTIPLIER" ]] && \
    TIMEOUT_ARG="--agent-timeout-multiplier $AGENT_TIMEOUT_MULTIPLIER"
SCRATCH_ARG=""
[[ -n "$SCRATCH_DIR" ]] && SCRATCH_ARG="--scratch-dir $SCRATCH_DIR"
AGENT_LIST="$(printf '%q ' $AGENTS)"
MINI_SETTING_ARGS=""
if [[ "$MINI_CODE_SETTINGS" != "all" ]]; then
    for setting in $MINI_CODE_SETTINGS; do
        MINI_SETTING_ARGS+="--setting $(printf '%q' "$setting") "
    done
fi
BENCH_LIST_Q="$(printf '%q ' "${BENCH_LIST[@]}")"
CONFIG_LIST_Q="$(printf '%q ' "${CONFIG_LIST[@]}")"

GRES="gpu:${NUM_GPUS}"
[[ -n "$SCRATCH_GRES" ]] && GRES+=",scratch:${SCRATCH_GRES}"

cat <<EOF > "$JOB_FILE"
#!/bin/bash
# Generated by benchmarks/scripts/submit_benchmark_slurm.sh -- do not edit by hand.
#SBATCH --job-name=bench_${SIGNATURE}
#SBATCH --output=${LOG_DIR}/bench_%j_${SIGNATURE}.out
#SBATCH --error=${LOG_DIR}/bench_%j_${SIGNATURE}.err
#SBATCH --time=${WALLTIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --mem=${SBATCH_MEM}
$( [[ -n "$QOS" ]] && echo "#SBATCH --qos=${QOS}" )

set -uo pipefail
cd $(printf '%q' "$REPO_ROOT")
source $(printf '%q' "$CONDA_SH")
mkdir -p $(printf '%q' "$LOG_DIR")
# Keep the large caches off the (1 TB) home filesystem when configured.
$( [[ -n "$TB_CACHE" ]] && echo "export BENCH_CACHE=$(printf '%q' "$TB_CACHE")" )
$( [[ -n "$SIF_CACHE" ]] && echo "export BENCH_SIF_CACHE=$(printf '%q' "$SIF_CACHE/tb")" )
$( [[ -n "$APPTAINER_CACHEDIR" ]] && echo "export APPTAINER_CACHEDIR=$(printf '%q' "$APPTAINER_CACHEDIR")" )
$( [[ -n "$APPTAINER_TMPDIR" ]] && echo "export APPTAINER_TMPDIR=$(printf '%q' "$APPTAINER_TMPDIR")" )

# --- vLLM server on all NUM_GPUS GPUs ------------------------------------
conda activate $(printf '%q' "$MODEL_ENV") || exit 1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
SERVER_LOG=$(printf '%q' "$LOG_DIR")/server_\${SLURM_JOB_ID}_${SIGNATURE}.log
SERVER_PID=""

cleanup() {
    if [[ -n "\${SERVER_PID}" ]]; then
        echo "stopping vLLM (pid \${SERVER_PID})"
        kill "\${SERVER_PID}" 2>/dev/null || true
        wait "\${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "starting vLLM $(printf '%q' "$MODEL_NAME") on ${NUM_GPUS} GPU(s), port ${SERVER_PORT}"
VLLM_ARGS=(serve $(printf '%q' "$MODEL_NAME")
    --tensor-parallel-size ${NUM_GPUS}
    --host 0.0.0.0 --port ${SERVER_PORT}
    --enable-auto-tool-choice
    --tool-call-parser $(printf '%q' "$TOOL_CALL_PARSER")
    --trust-remote-code)
[[ -n "${REASONING_PARSER}" ]] && VLLM_ARGS+=(--reasoning-parser $(printf '%q' "$REASONING_PARSER"))
[[ -n "${MAX_MODEL_LEN}" ]] && VLLM_ARGS+=(--max-model-len ${MAX_MODEL_LEN})
# shellcheck disable=SC2206
[[ -n "${SERVER_EXTRA}" ]] && VLLM_ARGS+=(${SERVER_EXTRA})
vllm "\${VLLM_ARGS[@]}" > "\$SERVER_LOG" 2>&1 &
SERVER_PID=\$!

echo "waiting for http://127.0.0.1:${SERVER_PORT}/health (up to ${SERVER_TIMEOUT}s)"
ready=0
for ((i = 0; i < ${SERVER_TIMEOUT}; i++)); do
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if (( ready == 0 )); then
    echo "ERROR: server not ready; see \$SERVER_LOG" >&2
    exit 1
fi
echo "server ready"

# --- benchmarks x agents, one agent at a time ----------------------------
conda deactivate
conda activate $(printf '%q' "$BENCH_ENV") || exit 1
export CUDA_VISIBLE_DEVICES=""   # the runner and its containers are CPU-side

BENCHMARKS=(${BENCH_LIST_Q})
CONFIGS=(${CONFIG_LIST_Q})
AGENTS=(${AGENT_LIST})
SETTING_ARGS="${MINI_SETTING_ARGS}"
FAILED=0

for i in "\${!BENCHMARKS[@]}"; do
    for agent in "\${AGENTS[@]}"; do
        run_id="${RUN_PREFIX}_\${BENCHMARKS[\$i]}_\${agent}"
        echo
        echo "=== \$run_id ==="
        run_settings=""
        [[ "\$agent" == "mini-code" ]] && run_settings="\$SETTING_ARGS"
        python benchmarks/run_benchmarks.py \\
            --dataset "\${BENCHMARKS[\$i]}" \\
            --agent "\$agent" \\
            --mode ${MODE} \\
            --config "\${CONFIGS[\$i]}" \\
            --run-id "\$run_id" \\
            --results-dir $(printf '%q' "$RESULTS_DIR") \\
            --n-concurrent ${N_CONCURRENT} \\
            ${LIMIT_ARG} ${TIMEOUT_ARG} ${SCRATCH_ARG} ${EXTRA_ARGS} \
            \${run_settings}
        rc=\$?
        if (( rc != 0 )); then
            echo "WARNING: \$run_id exited \$rc; continuing" >&2
            FAILED=\$((FAILED + 1))
        fi
    done
done

echo
echo "campaign summaries:"
for i in "\${!BENCHMARKS[@]}"; do
    for agent in "\${AGENTS[@]}"; do
        echo "  ${RESULTS_DIR}/${RUN_PREFIX}_\${BENCHMARKS[\$i]}_\${agent}/summary_results.csv"
    done
done

if (( FAILED > 0 )); then
    echo "WARNING: \$FAILED campaign(s) exited non-zero" >&2
    exit 1
fi
EOF

chmod +x "$JOB_FILE"

echo "=== Benchmark Slurm configuration ==="
echo "job file   : $JOB_FILE"
echo "model      : $MODEL_NAME"
echo "vLLM       : tp=$NUM_GPUS port=$SERVER_PORT"
echo "benchmarks : ${BENCH_LIST[*]}"
[[ -n "$TB_STAGED" ]] && echo "TB staged  : $TB_STAGED (used=$TB_USE_STAGED)"
echo "agents     : $AGENTS (one at a time)"
echo "mini-code settings: $MINI_CODE_SETTINGS"
echo "mode / full: $MODE / $FULL"
echo "walltime   : $WALLTIME"
echo "configs    :"
for i in "${!BENCH_LIST[@]}"; do
    echo "  ${BENCH_LIST[$i]} -> ${CONFIG_LIST[$i]}"
done
echo "===================================="

if [[ "$DRY_RUN" == "1" || "$SUBMIT" != "1" ]]; then
    echo "not submitting (DRY_RUN=$DRY_RUN SUBMIT=$SUBMIT). Submit with:"
    echo "  sbatch $JOB_FILE"
    exit 0
fi

SUBMITTED="$(sbatch --parsable "$JOB_FILE")"
echo "submitted: $SUBMITTED"
echo "${SUBMITTED},submitted,${SIGNATURE}" >> "$EXPERIMENTS_DIR/submitted.csv"
