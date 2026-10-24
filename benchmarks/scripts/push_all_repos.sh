#!/usr/bin/env bash
# Push the three repositories that make up a benchmark run, in one go.
#
#   benchmarks/scripts/push_all_repos.sh --dry-run    # print the commands only
#   benchmarks/scripts/push_all_repos.sh              # actually push
#
# A run spans three histories, each on its own branch:
#
#   mini-code-cli @ feat/benchmarks        this repository
#   harbor        @ mini-code/apptainer    Terminal-Bench harness, Apptainer backend
#   terminal-bench-2-1 @ mini-code/bench   the 2.1 task set
#
# The forks are looked up in `external_repos/` first (what setup_external_repos.sh
# creates) and then next to this repository (the `harbor/`,
# `terminal-bench-2-1/` checkouts the export workflow used).
#
# Safety: every checkout must be on its expected branch and clean before anything
# is pushed -- pushing somebody's half-finished work by accident is worse than
# not pushing. `terminal-bench-2-1` is pushed with --force-with-lease because its
# history was rewritten once (a secret was removed from a test fixture); the other
# two are fast-forward pushes and are rejected if they are not.
set -euo pipefail

export GIT_TERMINAL_PROMPT=0

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Environment:
  EXTERNAL_REPOS_DIR  where the fork clones live (default: <repo>/external_repos)

Exit status: 0 when every push succeeded (or --dry-run printed them), 1 otherwise.
EOF
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

# Find a fork: external_repos/<name> wins, the sibling checkout is the fallback.
locate() {
    local name="$1"
    if [ -d "$DEST/$name/.git" ]; then echo "$DEST/$name"; return 0; fi
    if [ -d "$ROOT/$name/.git" ]; then echo "$ROOT/$name"; return 0; fi
    return 1
}

status=0
for entry in "mini-code-cli:feat/benchmarks:" "harbor:mini-code/apptainer:--force-with-lease" \
             "terminal-bench-2-1:mini-code/bench:--force-with-lease"; do
    IFS=: read -r name branch force <<<"$entry"

    if [ "$name" = "mini-code-cli" ]; then
        path="$ROOT"
    else
        if ! path="$(locate "$name")"; then
            echo "skip: $name not found in $DEST/ or $ROOT/" >&2
            status=1
            continue
        fi
    fi

    current="$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
    if [ "$current" != "$branch" ]; then
        echo "skip: $path is on '$current', expected '$branch'" >&2
        status=1
        continue
    fi
    if [ -n "$(git -C "$path" status --porcelain)" ]; then
        echo "skip: $path has uncommitted changes (commit or stash them first)" >&2
        status=1
        continue
    fi
    if ! git -C "$path" remote get-url origin >/dev/null 2>&1; then
        echo "skip: $path has no 'origin' remote" >&2
        status=1
        continue
    fi

    ahead="$(git -C "$path" rev-list --count "origin/$branch..$branch" 2>/dev/null || echo "?")"
    echo "$name @ $branch: $ahead commit(s) to push ($path)"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  git -C $path push $force -u origin $branch"
        continue
    fi

    if git -C "$path" push $force -u origin "$branch"; then
        echo "  pushed"
    else
        echo "  ! push failed for $name (authentication, protection rules, or a" >&2
        echo "    non-fast-forward without --force-with-lease)" >&2
        status=1
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "dry run: nothing was pushed"
fi

exit "$status"
