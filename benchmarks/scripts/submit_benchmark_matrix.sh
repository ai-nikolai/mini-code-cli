#!/usr/bin/env bash
#
# Submit the benchmark matrix: one Slurm job per (model, benchmark, agent) cell.
# Each cell calls benchmarks/scripts/submit_benchmark_slurm.sh with single values, so every
# job gets its own vLLM server and cannot contend with another agent.
#
#     benchmarks/scripts/submit_benchmark_matrix.sh
#     DRY_RUN=1 SUBMIT=0 benchmarks/scripts/submit_benchmark_matrix.sh
#
# Lists are plain bash arrays below. A model entry may be just a name, or
#   "name|NUM_GPUS|MODEL_ENV|TOOL_CALL_PARSER|REASONING_PARSER"
# and an agent entry may be just a name, or
#   "name|extra run_benchmarks.py flags"

set -euo pipefail

# --- the matrix -----------------------------------------------------------
MODELS=(
    "ornith-ai/Ornith-1.5-35B-A3B"
    # "ornith-ai/Ornith-1.5-397B-FP8|4|env_vllm30|qwen3_xml|qwen3"
)

BENCHMARKS=(
    "terminalbench-2.1"
    "swe-bench-lite-full"
)

AGENTS=(
    "mini-code"
    "pi"
    "claude-code"
    # "claude-code|--agent-timeout-multiplier 2"
)
# -------------------------------------------------------------------------

# Defaults for a model entry that does not spell out its own fields.
DEFAULT_NUM_GPUS="${NUM_GPUS:-1}"
DEFAULT_MODEL_ENV="${MODEL_ENV:-env_vllm30}"
DEFAULT_TOOL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
DEFAULT_REASONING_PARSER="${REASONING_PARSER:-qwen3}"
BENCH_ENV="${BENCH_ENV:-hb}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
SERVER_EXTRA="${SERVER_EXTRA:-}"
SIF_CACHE="${SIF_CACHE:-}"
TB_CACHE="${TB_CACHE:-}"
SCRATCH_DIR="${SCRATCH_DIR:-}"
TB_STAGED="${TB_STAGED:-experiments/tb-staged}"
TB_USE_STAGED="${TB_USE_STAGED:-1}"
TB_SIF_CACHE="${TB_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/tb}}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-}"

VERSION="${VERSION:-v1}"
WALLTIME="${WALLTIME:-48:00:00}"
MODE="${MODE:-full}"
FULL="${FULL:-1}"
LIMIT="${LIMIT:-}"
N_CONCURRENT="${N_CONCURRENT:-1}"
MINI_CODE_SETTINGS="${MINI_CODE_SETTINGS:-swe}"  # space-separated; "all" = every configured setting
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SUBMIT="${SUBMIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_benchmark_slurm.sh"
[[ -x "$SUBMIT_SCRIPT" ]] || { echo "error: missing $SUBMIT_SCRIPT" >&2; exit 2; }
(( ${#MODELS[@]} > 0 )) || { echo "error: MODELS is empty" >&2; exit 2; }
(( ${#BENCHMARKS[@]} > 0 )) || { echo "error: BENCHMARKS is empty" >&2; exit 2; }
(( ${#AGENTS[@]} > 0 )) || { echo "error: AGENTS is empty" >&2; exit 2; }

slug() { printf '%s' "$1" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-40; }

TOTAL=$(( ${#MODELS[@]} * ${#BENCHMARKS[@]} * ${#AGENTS[@]} ))
echo "matrix: ${#MODELS[@]} model(s) x ${#BENCHMARKS[@]} benchmark(s) x " \
     "${#AGENTS[@]} agent(s) = $TOTAL job(s)"
echo "mode=$MODE full=$FULL submit=$SUBMIT dry_run=$DRY_RUN"

CELL=0
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model model_gpus model_env tool_parser reasoning_parser \
        <<< "$model_entry"
    [[ -n "$model" ]] || { echo "error: empty model entry" >&2; exit 2; }
    model_gpus="${model_gpus:-$DEFAULT_NUM_GPUS}"
    model_env="${model_env:-$DEFAULT_MODEL_ENV}"
    tool_parser="${tool_parser:-$DEFAULT_TOOL_PARSER}"
    reasoning_parser="${reasoning_parser:-$DEFAULT_REASONING_PARSER}"

    for bench in "${BENCHMARKS[@]}"; do
        for agent_entry in "${AGENTS[@]}"; do
            IFS='|' read -r agent agent_extra <<< "$agent_entry"
            [[ -n "$agent" ]] || { echo "error: empty agent entry" >&2; exit 2; }
            cell_version="${VERSION}_$(slug "${model##*/}")_$(slug "$bench")_$(slug "$agent")"
            CELL=$((CELL + 1))
            echo
            echo "[$CELL/$TOTAL] model=$model gpus=$model_gpus bench=$bench agent=$agent"
            env \
                VERSION="$cell_version" \
                RUN_PREFIX="$cell_version" \
                MODEL_NAME="$model" \
                NUM_GPUS="$model_gpus" \
                MODEL_ENV="$model_env" \
                BENCH_ENV="$BENCH_ENV" \
                CONDA_SH="$CONDA_SH" \
                TOOL_CALL_PARSER="$tool_parser" \
                REASONING_PARSER="$reasoning_parser" \
                MAX_MODEL_LEN="$MAX_MODEL_LEN" \
                SERVER_EXTRA="$SERVER_EXTRA" \
                SERVER_PORT="$SERVER_PORT" \
                SERVER_TIMEOUT="$SERVER_TIMEOUT" \
                SIF_CACHE="$SIF_CACHE" \
                TB_CACHE="$TB_CACHE" \
                SCRATCH_DIR="$SCRATCH_DIR" \
                TB_STAGED="$TB_STAGED" \
                TB_USE_STAGED="$TB_USE_STAGED" \
                TB_SIF_CACHE="$TB_SIF_CACHE" \
                APPTAINER_CACHEDIR="$APPTAINER_CACHEDIR" \
                APPTAINER_TMPDIR="$APPTAINER_TMPDIR" \
                BENCHMARKS="$bench" \
                AGENTS="$agent" \
                MINI_CODE_SETTINGS="$MINI_CODE_SETTINGS" \
                EXTRA_ARGS="$agent_extra" \
                MODE="$MODE" \
                FULL="$FULL" \
                LIMIT="$LIMIT" \
                N_CONCURRENT="$N_CONCURRENT" \
                AGENT_TIMEOUT_MULTIPLIER="$AGENT_TIMEOUT_MULTIPLIER" \
                RESULTS_DIR="$RESULTS_DIR" \
                EXPERIMENTS_DIR="$EXPERIMENTS_DIR" \
                PARTITION="$PARTITION" \
                QOS="$QOS" \
                SBATCH_CPUS="$SBATCH_CPUS" \
                SBATCH_MEM="$SBATCH_MEM" \
                WALLTIME="$WALLTIME" \
                SUBMIT="$SUBMIT" \
                DRY_RUN="$DRY_RUN" \
                "$SUBMIT_SCRIPT"
        done
    done
done

echo
echo "matrix complete: $CELL cell(s)"
[[ "$DRY_RUN" == "1" || "$SUBMIT" != "1" ]] && \
    echo "job files are under $EXPERIMENTS_DIR/slurm/"
