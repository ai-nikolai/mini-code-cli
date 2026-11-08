#!/usr/bin/env bash
#
# Run the benchmark campaigns on an existing Slurm allocation, one agent at a
# time. Designed for a single interactive GPU job:
#
#     squeue -u "$USER"                 # find the interactive job id
#     scripts/run_benchmark_slurm.sh    # uses JOBID below
#
# Everything the script needs is a variable in the block below. The defaults
# match the H200 setup used in benchmarks/README.md: one vLLM server on
# 127.0.0.1:30000 and one campaign per agent (never concurrently -- running
# several campaigns at once is what produced the 3.7 h timeouts).
#
# The script is deliberately a thin wrapper: it starts/monitors the model
# server, then calls the same `benchmarks/run_benchmarks.py` a human would.
# It never edits the checkout and never pushes anything.

set -euo pipefail

# --- edit these -----------------------------------------------------------
JOBID="${JOBID:-}"                         # empty = run on the current host
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${DATASET:-terminalbench-2.1-3}"  # swe-bench | swe-bench-lite | terminalbench-2.1-3
AGENTS="${AGENTS:-mini-code pi claude-code}"  # space-separated; run one by one
MODE="${MODE:-short}"                      # short | short_eval | full
LIMIT="${LIMIT:-}"                         # empty = mode default
N_CONCURRENT="${N_CONCURRENT:-1}"          # inside one campaign (tasks), not across agents
RUN_PREFIX="${RUN_PREFIX:-serial}"         # run-id prefix, e.g. tb21-serial-mini-code
RESULTS_DIR="${RESULTS_DIR:-results}"      # git-ignored campaign directory
EXTRA_ARGS="${EXTRA_ARGS:-}"               # extra args passed to run_benchmarks.py

PYTHON="${PYTHON:-$HOME/miniconda3/envs/hb/bin/python}"        # has harbor+swebench
VLLM="${VLLM:-$HOME/miniconda3/envs/env_vllm30/bin/vllm}"      # vLLM executable
MODEL="${MODEL:-ornith-ai/Ornith-1.5-35B-A3B}"
PORT="${PORT:-30000}"
START_SERVER="${START_SERVER:-1}"          # 1 = start vLLM, 0 = use a running one
STOP_SERVER="${STOP_SERVER:-1}"            # stop the server we started on exit
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"    # seconds to wait for /health
LOG_DIR="${LOG_DIR:-/tmp/opencode/bench-slurm}"
# -------------------------------------------------------------------------

SERVER_PID=""
SERVER_LOG="$LOG_DIR/vllm-$PORT.log"

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Run one shell command on the allocation (or locally when no JOBID is set).
# Everything is passed as a single login-shell command so PATH/cd behave as on
# the GPU node; the benchmark harness itself uses argv lists, never a shell.
on_node() {
    if [[ -n "$JOBID" ]]; then
        srun --overlap --jobid="$JOBID" --ntasks=1 bash -lc "$*"
    else
        bash -lc "$*"
    fi
}

check_job() {
    if [[ -n "$JOBID" ]]; then
        squeue -h -j "$JOBID" -o '%T' | grep -q RUNNING \
            || die "Slurm job $JOBID is not RUNNING (squeue -u $USER)"
    fi
    [[ -x "$PYTHON" ]] || die "python not executable: $PYTHON"
}

empty_health() {
    on_node "curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health" 2>/dev/null || true
}

wait_health() {
    local waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        if [[ "$(empty_health)" == "200" ]]; then
            say "model server is healthy (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    die "model server did not answer /health within ${HEALTH_TIMEOUT}s (see $SERVER_LOG)"
}

start_server() {
    if [[ "$(empty_health)" == "200" ]]; then
        say "model server already healthy on port $PORT; not starting another"
        return 0
    fi
    if [[ "$START_SERVER" != "1" ]]; then
        wait_health
        return 0
    fi
    say "starting vLLM ($MODEL) on port $PORT"
    local bin_dir
    bin_dir="$(dirname "$VLLM")"
    # `ninja` lives in the vLLM env; without it flashinfer's JIT fails at
    # engine start ("No such file or directory: 'ninja'").
    if [[ -n "$JOBID" ]]; then
        srun --overlap --jobid="$JOBID" --ntasks=1 bash -lc \
            "export PATH=$(printf '%q' "$bin_dir"):\$PATH; \
             cd $(printf '%q' "$REPO_ROOT") && exec $(printf '%q' "$VLLM") serve \
             $(printf '%q' "$MODEL") --tensor-parallel-size 1 --host 0.0.0.0 \
             --port $PORT --enable-auto-tool-choice --tool-call-parser qwen3_xml \
             --reasoning-parser qwen3 --trust-remote-code" \
            >"$SERVER_LOG" 2>&1 &
    else
        (
            export PATH="$bin_dir:$PATH"
            cd "$REPO_ROOT"
            exec "$VLLM" serve "$MODEL" --tensor-parallel-size 1 --host 0.0.0.0 \
                --port "$PORT" --enable-auto-tool-choice \
                --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
                --trust-remote-code
        ) >"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!
    wait_health
}

stop_server() {
    if [[ -n "$SERVER_PID" && "$STOP_SERVER" == "1" ]]; then
        say "stopping vLLM (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

run_agent() {
    local agent="$1"
    local run_id="${RUN_PREFIX}-${agent}"
    local log="$LOG_DIR/${run_id}.log"
    local args=(--dataset "$DATASET" --agent "$agent" --mode "$MODE"
                --run-id "$run_id" --results-dir "$RESULTS_DIR"
                --n-concurrent "$N_CONCURRENT")
    [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")
    # EXTRA_ARGS is intentionally word-split: it is a convenience for flags such
    # as "--run-id ..." and must not be used to pass quoted task text.
    # shellcheck disable=SC2206
    [[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

    say "running $agent -> $run_id (log: $log)"
    mkdir -p "$LOG_DIR"
    if on_node "cd $(printf '%q' "$REPO_ROOT") && exec $(printf '%q' "$PYTHON") \
            benchmarks/run_benchmarks.py $(printf '%q ' "${args[@]}")" \
            2>&1 | tee "$log"; then
        say "$agent finished"
    else
        # Do not abort the remaining agents: one broken agent is a result too.
        say "$agent failed (see $log); continuing"
    fi
}

main() {
    check_job
    mkdir -p "$LOG_DIR"
    trap stop_server EXIT INT TERM

    if [[ -n "$JOBID" ]]; then
        say "allocation $JOBID on $(squeue -h -j "$JOBID" -o '%N')"
    fi
    say "dataset=$DATASET mode=$MODE agents=$AGENTS"
    start_server

    for agent in $AGENTS; do
        run_agent "$agent"
    done

    say "done. Campaign summaries:"
    for agent in $AGENTS; do
        printf '  %s\n' "$RESULTS_DIR/${RUN_PREFIX}-${agent}/summary_results.csv"
    done
}

main "$@"
