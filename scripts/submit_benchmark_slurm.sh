#!/usr/bin/env bash
#
# Generate and submit one Slurm batch job that runs the full benchmark matrix
# for one model: every benchmark x every agent, one agent at a time.
#
# The file this script writes lives under experiments/slurm/ (git-ignored), so a
# run is reproducible without committing a job script. Edit the variables below,
# then run it on the login node:
#
#     scripts/submit_benchmark_slurm.sh
#
# GPU split follows the kernel-benchmark launcher: NUM_GPUS is the whole job,
# BENCH_GPUS (default 0) is reserved for the runner; the model server gets the
# rest and uses them as its tensor-parallel world size. For this harness the
# benchmark runner itself is CPU/container-bound, so BENCH_GPUS=0 is normal.
# For a different model, set MODEL_NAME / MODEL_ENV / parsers; the configs are
# regenerated for that endpoint (see scripts/make_benchmark_config.py).

set -euo pipefail

# --- edit these -----------------------------------------------------------
VERSION="${VERSION:-v1}"
WALLTIME="${WALLTIME:-48:00:00}"

NUM_GPUS="${NUM_GPUS:-1}"                  # 1..4
BENCH_GPUS="${BENCH_GPUS:-0}"              # GPUs reserved for the runner (0 = model uses all)

MODEL_NAME="${MODEL_NAME:-ornith-ai/Ornith-1.5-35B-A3B}"
SERVER_BACKEND="${SERVER_BACKEND:-vllm}"   # vllm | sglang
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"   # seconds to wait for /health
MODEL_ENV="${MODEL_ENV:-env_vllm30}"       # conda env with vLLM/SGLang
BENCH_ENV="${BENCH_ENV:-hb}"               # conda env with harbor + swebench
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"         # empty = server default
SERVER_EXTRA="${SERVER_EXTRA:-}"           # extra server flags, word-split

TB_CONFIG="${TB_CONFIG:-benchmarks/configs/terminal_bench_21.json}"
SWE_CONFIG="${SWE_CONFIG:-benchmarks/configs/ornith_30b.json}"

# Two benchmarks, fully. Use terminalbench-2.1-3 for the three cached tasks;
# swe-bench-lite-full is the 300-instance Lite split (swe-bench-lite is the
# five-instance smoke set).
BENCHMARKS="${BENCHMARKS:-terminalbench-2.1 swe-bench-lite-full}"
AGENTS="${AGENTS:-mini-code pi claude-code}"   # always run one at a time
MODE="${MODE:-full}"                            # full | short_eval | short
LIMIT="${LIMIT:-}"                              # empty = full; e.g. 5 for a smoke run
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"  # e.g. 1 (900 s/task)
N_CONCURRENT="${N_CONCURRENT:-1}"               # inside a benchmark, not across agents
FULL="${FULL:-1}"                               # 1 = drop smoke limits/task_ids
RUN_PREFIX="${RUN_PREFIX:-$VERSION}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                    # extra run_benchmarks.py flags

PARTITION="${PARTITION:-gpu}"                   # set to your cluster partition
QOS="${QOS:-}"                                  # optional
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
SUBMIT="${SUBMIT:-1}"                           # 1 = sbatch; 0 = only write the file
DRY_RUN="${DRY_RUN:-0}"                         # 1 = no sbatch, no server started
CONFIG_PYTHON="${CONFIG_PYTHON:-python3}"       # stdlib-only config rewrite
# -------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

[[ "$NUM_GPUS" =~ ^[1-4]$ ]] || die "NUM_GPUS must be 1..4 (got $NUM_GPUS)"
[[ "$BENCH_GPUS" =~ ^[0-9]+$ ]] || die "BENCH_GPUS must be an integer"
(( BENCH_GPUS < NUM_GPUS )) || die "BENCH_GPUS must be smaller than NUM_GPUS"
MODEL_GPUS=$((NUM_GPUS - BENCH_GPUS))
(( MODEL_GPUS >= 1 )) || die "at least one GPU must be left for the model"
[[ "$SERVER_BACKEND" == "vllm" || "$SERVER_BACKEND" == "sglang" ]] \
    || die "SERVER_BACKEND must be vllm or sglang"
[[ -n "$LIMIT" ]] && FULL=0

MODEL_VISIBLE_DEVICES="$(seq -s, 0 $((MODEL_GPUS - 1)))"
if (( BENCH_GPUS > 0 )); then
    BENCH_VISIBLE_DEVICES="$(seq -s, "$MODEL_GPUS" $((NUM_GPUS - 1)))"
else
    BENCH_VISIBLE_DEVICES=""
fi

SHORT_MODEL_NAME="$(printf '%s' "$MODEL_NAME" | cut -d/ -f2 \
    | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SIGNATURE="${VERSION}_${SHORT_MODEL_NAME}_${NUM_GPUS}gpu"
SLURM_DIR="$EXPERIMENTS_DIR/slurm"
CONFIG_DIR="$EXPERIMENTS_DIR/configs"
LOG_DIR="$EXPERIMENTS_DIR/logs"
mkdir -p "$SLURM_DIR" "$CONFIG_DIR" "$LOG_DIR"

# Rebuild the config for this model/endpoint. FULL removes the checked-in smoke
# limits and task ids so the run actually covers the whole split.
FULL_FLAG=""
[[ "$FULL" == "1" ]] && FULL_FLAG="--full"

declare -a BENCH_LIST=() CONFIG_LIST=()
for bench in $BENCHMARKS; do
    case "$bench" in
        swe-bench*|swebench*) template="$SWE_CONFIG" ;;
        terminal*|tb*)        template="$TB_CONFIG" ;;
        *) die "unknown benchmark '$bench' (use terminalbench-2.1 or swe-bench-lite-full)" ;;
    esac
    slug="$(printf '%s' "$bench" | sed 's/[^a-zA-Z0-9]/_/g')"
    config="$CONFIG_DIR/${SIGNATURE}_${slug}.json"
    # shellcheck disable=SC2086
    "$CONFIG_PYTHON" scripts/make_benchmark_config.py "$template" "$config" \
        --model "$MODEL_NAME" --base-url "http://127.0.0.1:${SERVER_PORT}" \
        $FULL_FLAG
    BENCH_LIST+=("$bench")
    CONFIG_LIST+=("$config")
done

JOB_FILE="$SLURM_DIR/bench_${TIMESTAMP}_${SIGNATURE}.sh"
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"
TIMEOUT_ARG=""
[[ -n "$AGENT_TIMEOUT_MULTIPLIER" ]] && \
    TIMEOUT_ARG="--agent-timeout-multiplier $AGENT_TIMEOUT_MULTIPLIER"

cat <<EOF > "$JOB_FILE"
#!/bin/bash
# Generated by scripts/submit_benchmark_slurm.sh -- do not edit by hand.
#SBATCH --job-name=bench_${SIGNATURE}
#SBATCH --output=${LOG_DIR}/bench_%j_${SIGNATURE}.out
#SBATCH --error=${LOG_DIR}/bench_%j_${SIGNATURE}.err
#SBATCH --time=${WALLTIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --mem=${SBATCH_MEM}
$( [[ -n "$QOS" ]] && echo "#SBATCH --qos=${QOS}" )

set -uo pipefail
cd $(printf '%q' "$REPO_ROOT")
source $(printf '%q' "$CONDA_SH")
mkdir -p $(printf '%q' "$LOG_DIR")

# --- model server on the first MODEL_GPUS GPUs ---------------------------
conda activate $(printf '%q' "$MODEL_ENV") || exit 1
export CUDA_VISIBLE_DEVICES=${MODEL_VISIBLE_DEVICES}
SERVER_LOG=$(printf '%q' "$LOG_DIR")/server_\${SLURM_JOB_ID}_${SIGNATURE}.log
SERVER_PID=""

cleanup() {
    if [[ -n "\${SERVER_PID}" ]]; then
        echo "stopping model server (pid \${SERVER_PID})"
        kill "\${SERVER_PID}" 2>/dev/null || true
        wait "\${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "starting ${SERVER_BACKEND} ${MODEL_NAME} on GPUs ${MODEL_VISIBLE_DEVICES} (tp=${MODEL_GPUS})"
if [[ "${SERVER_BACKEND}" == "vllm" ]]; then
    VLLM_ARGS=(serve $(printf '%q' "$MODEL_NAME")
        --tensor-parallel-size ${MODEL_GPUS}
        --host 0.0.0.0 --port ${SERVER_PORT}
        --enable-auto-tool-choice
        --tool-call-parser $(printf '%q' "$TOOL_CALL_PARSER")
        --trust-remote-code)
    [[ -n "${REASONING_PARSER}" ]] && VLLM_ARGS+=(--reasoning-parser $(printf '%q' "$REASONING_PARSER"))
    [[ -n "${MAX_MODEL_LEN}" ]] && VLLM_ARGS+=(--max-model-len ${MAX_MODEL_LEN})
    # shellcheck disable=SC2206
    [[ -n "${SERVER_EXTRA}" ]] && VLLM_ARGS+=(${SERVER_EXTRA})
    vllm "\${VLLM_ARGS[@]}" > "\$SERVER_LOG" 2>&1 &
else
    SGLANG_ARGS=(-m sglang.launch_server
        --model-path $(printf '%q' "$MODEL_NAME")
        --tensor-parallel-size ${MODEL_GPUS}
        --host 0.0.0.0 --port ${SERVER_PORT}
        --trust-remote-code)
    [[ -n "${TOOL_CALL_PARSER}" ]] && SGLANG_ARGS+=(--tool-call-parser $(printf '%q' "$TOOL_CALL_PARSER"))
    [[ -n "${MAX_MODEL_LEN}" ]] && SGLANG_ARGS+=(--context-length ${MAX_MODEL_LEN})
    # shellcheck disable=SC2206
    [[ -n "${SERVER_EXTRA}" ]] && SGLANG_ARGS+=(${SERVER_EXTRA})
    python "\${SGLANG_ARGS[@]}" > "\$SERVER_LOG" 2>&1 &
fi
SERVER_PID=\$!

echo "waiting for http://127.0.0.1:${SERVER_PORT}/health (up to ${SERVER_TIMEOUT}s)"
ready=0
for ((i = 0; i < SERVER_TIMEOUT; i++)); do
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

# --- benchmark runner on the remaining GPUs (or none) --------------------
conda deactivate
conda activate $(printf '%q' "$BENCH_ENV") || exit 1
export CUDA_VISIBLE_DEVICES="${BENCH_VISIBLE_DEVICES}"

BENCHMARKS=($(printf '%q ' "${BENCH_LIST[@]}"))
CONFIGS=($(printf '%q ' "${CONFIG_LIST[@]}"))
AGENTS=($(printf '%q ' ${AGENTS}))
FAILED=0

for i in "\${!BENCHMARKS[@]}"; do
    for agent in "\${AGENTS[@]}"; do
        run_id="${RUN_PREFIX}_\${BENCHMARKS[\$i]}_\${agent}"
        echo
        echo "=== \$run_id ==="
        python benchmarks/run_benchmarks.py \\
            --dataset "\${BENCHMARKS[\$i]}" \\
            --agent "\$agent" \\
            --mode ${MODE} \\
            --config "\${CONFIGS[\$i]}" \\
            --run-id "\$run_id" \\
            --results-dir $(printf '%q' "$RESULTS_DIR") \\
            --n-concurrent ${N_CONCURRENT} \\
            ${LIMIT_ARG} ${TIMEOUT_ARG} ${EXTRA_ARGS}
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
echo "job file       : $JOB_FILE"
echo "model          : $MODEL_NAME"
echo "backend        : $SERVER_BACKEND  tp=$MODEL_GPUS  port=$SERVER_PORT"
echo "GPUs           : $NUM_GPUS total; model=$MODEL_VISIBLE_DEVICES bench=${BENCH_VISIBLE_DEVICES:-none}"
echo "benchmarks     : ${BENCH_LIST[*]}"
echo "agents         : $AGENTS (one at a time)"
echo "mode / full    : $MODE / $FULL"
echo "walltime       : $WALLTIME"
echo "configs        :"
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
