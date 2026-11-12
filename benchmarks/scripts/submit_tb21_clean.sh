#!/usr/bin/env bash
#
# Prepare (and, when explicitly requested, submit) the clean Terminal-Bench 2.1
# rerun: mini-code's six strategy settings, Pi and Claude Code, all on the same
# 87-task set and with the fixed resource policy.
#
#     benchmarks/scripts/submit_tb21_clean.sh              # generate only
#     SUBMIT=1 benchmarks/scripts/submit_tb21_clean.sh     # sbatch the 5 jobs
#
# The suite template already carries the clean-run policy:
#   - override_memory_mb=16384
#   - exclude_tasks = qemu-startup, qemu-alpine-ssh
#   - verifier_timeout_multiplier=2
#   - environment_build_timeout_multiplier=12
# The runner also records vLLM token deltas per trial.
#
# This script does NOT submit by default: reviewing the generated job files and
# configs first is deliberate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SUBMIT="${SUBMIT:-0}"
VERSION_PREFIX="${VERSION_PREFIX:-tb21-clean1}"
N_CONCURRENT="${N_CONCURRENT:-2}"
WALLTIME="${WALLTIME:-72:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-16}"
SBATCH_MEM="${SBATCH_MEM:-256G}"
# SLURM scratch is private per job and exposed as $TMPDIR. Request 100 GB of
# NVMe for Apptainer's unpack/overlay work; without --gres=scratch a job is
# capped at 1 GB of local storage.
SCRATCH_GRES="${SCRATCH_GRES:-100g}"
SCRATCH_DIR="${SCRATCH_DIR:-\${TMPDIR:-/tmp}}"

# Two settings per mini-code job keeps the number of GPU allocations (and the
# fairshare cost) down while still finishing each pair inside 72h.
CELLS=(
  "a|mini-code|swe swe-compaction-predict-files|31200"
  "b|mini-code|swe-compaction-predict-messages swe-plan|31201"
  "c|mini-code|swe-plan-verify swe-open-plan|31202"
  "pi|pi|all-tools|31203"
  "claude|claude-code|bypass-permissions|31204"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r suffix agent settings port <<< "$cell"
  name="${VERSION_PREFIX}-${suffix}"

  BENCHMARKS=terminalbench-2.1 \
  AGENTS="$agent" \
  MINI_CODE_SETTINGS="$settings" \
  MODE=full FULL=1 LIMIT= \
  N_CONCURRENT="$N_CONCURRENT" NUM_GPUS=1 SERVER_PORT="$port" \
  WALLTIME="$WALLTIME" PARTITION=gpu \
  SBATCH_CPUS="$SBATCH_CPUS" SBATCH_MEM="$SBATCH_MEM" \
  TB_USE_STAGED=1 TB_STAGED=experiments/tb-staged \
  SCRATCH_GRES="$SCRATCH_GRES" SCRATCH_DIR="$SCRATCH_DIR" \
  VERSION="$name" RUN_PREFIX="$name" \
  SUBMIT="$SUBMIT" \
  benchmarks/scripts/submit_benchmark_slurm.sh
done

if [[ "$SUBMIT" != "1" ]]; then
  cat <<EOF

Prepared ${#CELLS[@]} job file(s); nothing was submitted (SUBMIT=$SUBMIT).
Review them under experiments/slurm/, then launch with:

  SUBMIT=1 benchmarks/scripts/submit_tb21_clean.sh
EOF
fi
