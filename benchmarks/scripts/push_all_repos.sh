#!/usr/bin/env bash
# Push the three repositories a benchmark change usually spans, in one go.
#
#   benchmarks/scripts/push_all_repos.sh --dry-run    # print the commands, push nothing
#   benchmarks/scripts/push_all_repos.sh              # push
#
#   mini-code-cli        @ feat/benchmarks      this repository
#   harbor               @ mini-code/apptainer  Terminal-Bench harness
#   terminal-bench-2-1   @ mini-code/bench      the 2.1 task set
#
# Checkouts are looked up in external_repos/ (what setup_external_repos.sh
# creates) and then next to this repository. Every one must be on its expected
# branch, clean, and have an origin -- pushing half-finished work by accident is
# worse than not pushing. The two forks use --force-with-lease: both histories
# were rewritten once, so a plain push would be rejected, while the lease still
# refuses if somebody else pushed in the meantime.
set -euo pipefail

export GIT_TERMINAL_PROMPT=0

# name : branch : push flags
REPOS=(
    "mini-code-cli:feat/benchmarks:"
    "harbor:mini-code/apptainer:--force-with-lease"
    "terminal-bench-2-1:mini-code/bench:--force-with-lease"
)

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    printf '\nEnv: EXTERNAL_REPOS_DIR (default <repo>/external_repos)\n'
}

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --dry-run|-n) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${EXTERNAL_REPOS_DIR:-$ROOT/external_repos}"

failed=0
for entry in "${REPOS[@]}"; do
    IFS=: read -r name branch flags <<<"$entry"

    path="$ROOT"
    [ "$name" = "mini-code-cli" ] || path="$DEST/$name"
    [ -d "$path/.git" ] || path="$ROOT/$name"
    if [ ! -d "$path/.git" ]; then
        echo "skip: $name not found in $DEST/ or $ROOT/" >&2
        failed=1
        continue
    fi

    problem=""
    current="$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    [ "$current" = "$branch" ] || problem="is on '$current', expected '$branch'"
    [ -n "$problem" ] || [ -z "$(git -C "$path" status --porcelain)" ] \
        || problem="has uncommitted changes (commit or stash them first)"
    [ -n "$problem" ] || git -C "$path" remote get-url origin >/dev/null 2>&1 \
        || problem="has no 'origin' remote"
    if [ -n "$problem" ]; then
        echo "skip: $name $problem" >&2
        failed=1
        continue
    fi

    echo "$name @ $branch: $(git -C "$path" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo '?') commit(s) to push"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  git -C $path push $flags -u origin $branch"
    elif git -C "$path" push $flags -u origin "$branch"; then
        echo "  pushed"
    else
        echo "  ! push failed (authentication, protection rules, or non-fast-forward)" >&2
        failed=1
    fi
done

[ "$DRY_RUN" = "0" ] || echo "
dry run: nothing was pushed"
exit "$failed"
