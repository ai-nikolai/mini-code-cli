#!/usr/bin/env bash
#
# Prune benchmark storage by degree. Dry-run by default; set
# DRY_RUN=0 CONFIRM=yes to actually delete.
#
#   MODE=light    remove heavy per-trial intermediates (testbeds, agent-tools,
#                 eval/pull workdirs, .scratch). Keeps SIFs and all results.
#   MODE=partial  light + delete the SIFs for BENCH (keep summaries/patches/logs)
#   MODE=sifs     delete only the SIFs for BENCH
#   MODE=scratch  delete only Apptainer scratch/cache contents
#   MODE=full     partial + remove results/, benchmarks/runs/, generated job
#                 configs/logs and scratch contents
#
# Examples for a 1 TB home:
#   scripts/prune_benchmark_storage.sh                       # show light plan
#   MODE=partial BENCH=tb DRY_RUN=0 CONFIRM=yes \
#       scripts/prune_benchmark_storage.sh                   # free TB SIFs
#   MODE=full BENCH=all DRY_RUN=0 CONFIRM=yes \
#       scripts/prune_benchmark_storage.sh                   # nuke caches/results

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-light}"                      # light | partial | sifs | scratch | full
BENCH="${BENCH:-all}"                      # all | tb | swe (SIF selection)
DRY_RUN="${DRY_RUN:-1}"                    # 1 = print only
CONFIRM="${CONFIRM:-no}"                   # yes = allow DRY_RUN=0

SIF_CACHE="${SIF_CACHE:-}"
TB_SIF_CACHE="${TB_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/tb}}"
TB_SIF_CACHE="${TB_SIF_CACHE:-benchmarks/.cache/sif}"
SWE_SIF_CACHE="${SWE_SIF_CACHE:-${SIF_CACHE:+$SIF_CACHE/swe}}"
SWE_SIF_CACHE="${SWE_SIF_CACHE:-benchmarks/runs/_cache/swe-sif}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUNS_DIR="${RUNS_DIR:-benchmarks/runs}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
SCRATCH_DIR="${SCRATCH_DIR:-}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-}"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

case "$MODE" in light|partial|sifs|scratch|full) ;; *)
    die "MODE must be light|partial|sifs|scratch|full" ;;
esac
case "$BENCH" in all|tb|swe) ;; *)
    die "BENCH must be all|tb|swe" ;;
esac
if [[ "$DRY_RUN" != "1" && "$CONFIRM" != "yes" ]]; then
    die "refusing to delete without CONFIRM=yes (or keep DRY_RUN=1)"
fi

say "mode=$MODE bench=$BENCH dry_run=$DRY_RUN"
echo "results     : $RESULTS_DIR"
echo "runs        : $RUNS_DIR"
echo "experiments : $EXPERIMENTS_DIR"
echo "TB SIFs     : $TB_SIF_CACHE"
echo "SWE SIFs    : $SWE_SIF_CACHE"
[[ -n "$SCRATCH_DIR" ]] && echo "scratch     : $SCRATCH_DIR"
[[ -n "$APPTAINER_CACHEDIR" ]] && echo "apptainer   : $APPTAINER_CACHEDIR"

# --- helpers -------------------------------------------------------------
prune_intermediates() {
    local roots=() root
    for root in "$RESULTS_DIR" "$RUNS_DIR"; do
        [[ -d "$root" ]] && roots+=("$root")
    done
    (( ${#roots[@]} )) || return 0
    local find_args=(-type d \( -name testbed -o -name agent-tools -o -name eval
                      -o -name pulls -o -name .scratch \) -prune)
    if [[ "$DRY_RUN" == "1" ]]; then
        find "${roots[@]}" "${find_args[@]}" -print
    else
        find "${roots[@]}" "${find_args[@]}" -exec rm -rf {} +
    fi
}

remove_sifs() {
    local remove_tb=0 remove_swe=0
    [[ "$BENCH" == "all" || "$BENCH" == "tb" ]] && remove_tb=1
    [[ "$BENCH" == "all" || "$BENCH" == "swe" ]] && remove_swe=1
    if (( remove_tb )) && [[ -d "$TB_SIF_CACHE" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            find "$TB_SIF_CACHE" -maxdepth 1 -type f \
                \( -name '*.sif' -o -name '*.sif.key' \) -print
        else
            find "$TB_SIF_CACHE" -maxdepth 1 -type f \
                \( -name '*.sif' -o -name '*.sif.key' \) -delete
        fi
    fi
    if (( remove_swe )) && [[ -d "$SWE_SIF_CACHE" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            find "$SWE_SIF_CACHE" -maxdepth 1 -type f \
                \( -name 'swe-*.sif' -o -name 'swe-*.sif.complete' \) -print
        else
            find "$SWE_SIF_CACHE" -maxdepth 1 -type f \
                \( -name 'swe-*.sif' -o -name 'swe-*.sif.complete' \) -delete
        fi
    fi
}

safe_wipe() {
    local dir="$1"
    [[ -n "$dir" && "$dir" != "/" && "$dir" != "$HOME" ]] \
        || die "refusing to wipe $dir"
    [[ -d "$dir" && ! -L "$dir" ]] || return 0
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "would wipe: $dir"
        find "$dir" -mindepth 1 -maxdepth 1 -print
    else
        find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
}

remove_scratch() {
    [[ -n "$SCRATCH_DIR" ]] && safe_wipe "$SCRATCH_DIR"
    [[ -n "$APPTAINER_CACHEDIR" ]] && safe_wipe "$APPTAINER_CACHEDIR"
}

# --- modes ---------------------------------------------------------------
case "$MODE" in
    light)
        say "heavy per-trial intermediates"
        prune_intermediates
        ;;
    sifs)
        say "SIFs for $BENCH"
        remove_sifs
        ;;
    scratch)
        say "Apptainer scratch/cache"
        remove_scratch
        ;;
    partial)
        say "heavy per-trial intermediates"
        prune_intermediates
        say "SIFs for $BENCH (summaries/patches/logs stay)"
        remove_sifs
        ;;
    full)
        say "heavy per-trial intermediates"
        prune_intermediates
        say "SIFs for $BENCH"
        remove_sifs
        say "results, runs, generated jobs, scratch"
        safe_wipe "$RESULTS_DIR"
        safe_wipe "$RUNS_DIR"
        safe_wipe "$EXPERIMENTS_DIR/logs"
        safe_wipe "$EXPERIMENTS_DIR/slurm"
        safe_wipe "$EXPERIMENTS_DIR/configs"
        safe_wipe "$EXPERIMENTS_DIR/tb-staged"
        safe_wipe "$EXPERIMENTS_DIR/swe-instances"
        safe_wipe "$EXPERIMENTS_DIR/swe-pulls"
        remove_scratch
        ;;
esac

say "done (dry_run=$DRY_RUN)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Nothing was deleted. Re-run with DRY_RUN=0 CONFIRM=yes."
fi
