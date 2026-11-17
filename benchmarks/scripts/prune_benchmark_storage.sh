#!/usr/bin/env bash
#
# Prune benchmark storage by degree. Dry-run by default; set
# DRY_RUN=0 CONFIRM=yes to actually delete.
#
# STAGES AND WHAT THEY HOLD
#   prepare:  source tasks/images -> SIFs; instance list -> SWE SIFs;
#             TB staged task dirs; generated configs/jobs. Small except SIFs.
#   run:      results/<campaign>/ (summaries, patches, predictions, reports)
#             plus heavy trial workspaces: testbed/, agent-tools/, eval/, pulls/,
#             bench logs, and Apptainer scratch/cache.
#   after:    summaries + patches + manifests are the reusable output;
#             SIFs / testbeds / eval / pulls / scratch are reproducible.
#
# USE CASES
#   # show what a light cleanup would remove (default: no deletion)
#   benchmarks/scripts/prune_benchmark_storage.sh
#
#   # keep scores/patches/SIFs, free the copied testbeds and eval workdirs
#   MODE=keep-results DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
#   # after a TB run, before a SWE run: free TB SIFs + TB trial workspaces
#   MODE=after-tb DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
#   # after a SWE run: free SWE SIFs, keep summaries/patches
#   MODE=after-swe DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
#   # remove only the SIFs for one benchmark (keep all results)
#   MODE=sifs BENCH=swe DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
#   # clear Apptainer cache/tmp only
#   MODE=scratch DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
#   # full reset: results/, runs/, generated jobs/configs/logs, SIFs, scratch
#   MODE=nuke DRY_RUN=0 CONFIRM=yes \
#       benchmarks/scripts/prune_benchmark_storage.sh
#
# MODES: keep-results (=light) | light | sifs | scratch | partial |
#        after-tb | after-swe | full | nuke

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-light}"                      # see the use cases above
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

# Friendly aliases for the two ends of the usual workflow.
case "$MODE" in
    keep-results) MODE=light ;;
    after-tb)     MODE=partial; BENCH=tb ;;
    after-swe)    MODE=partial; BENCH=swe ;;
    nuke)         MODE=full; BENCH=all ;;
esac
case "$MODE" in light|partial|sifs|scratch|full) ;; *)
    die "MODE must be keep-results|light|sifs|scratch|partial|after-tb|after-swe|full|nuke" ;;
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

say "sizes after plan (before deletion they are the current sizes):"
for d in "$RESULTS_DIR" "$RUNS_DIR" "$TB_SIF_CACHE" "$SWE_SIF_CACHE" \
         "$SCRATCH_DIR" "$APPTAINER_CACHEDIR"; do
    [[ -n "$d" && -d "$d" ]] && du -sh "$d" 2>/dev/null || true
done
say "done (dry_run=$DRY_RUN)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Nothing was deleted. Re-run with DRY_RUN=0 CONFIRM=yes."
fi
