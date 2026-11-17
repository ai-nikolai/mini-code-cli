#!/usr/bin/env bash
# Clone/fetch the checkouts a Terminal-Bench run needs, at the right branches.
#
#   benchmarks/scripts/setup_external_repos.sh              # harbor + terminal-bench-2-1
#   benchmarks/scripts/setup_external_repos.sh --with-main  # ... and mini-code-cli
#
# The clones land in external_repos/ (git-ignored) at the repository root:
#
#   external_repos/harbor/                @ mini-code/apptainer
#   external_repos/terminal-bench-2-1/    @ mini-code/bench
#
# Why a clone at all, when benchmarks/requirements.txt already installs harbor
# from its branch? Because the 2.1 task set is *data* (89 task directories, no
# setup.py/pyproject.toml), so pip cannot provide it: `harbor run -p` needs the
# directory on disk, and the 2.1 config's dataset_dir names exactly the path this
# script creates. The harbor checkout is then the extra one, for reading,
# changing and pushing the harness itself.
#
# Safe to re-run: an existing clone is fetched, never re-created, and a checkout
# with uncommitted changes is reported instead of being switched out from under
# you. No prompting, no sudo, no destructive commands.
set -euo pipefail

export GIT_TERMINAL_PROMPT=0            # never block on a password prompt
OWNER="${EXTERNAL_REPOS_OWNER:-ai-nikolai}"

# name : branch : one-line description
REPOS=(
    "harbor:mini-code/apptainer:Terminal-Bench harness with the Apptainer backend"
    "terminal-bench-2-1:mini-code/bench:Terminal-Bench 2.1 task set (data)"
)

usage() {
    sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    printf '\nEnv: EXTERNAL_REPOS_DIR (default <repo>/external_repos), EXTERNAL_REPOS_OWNER (default %s)\n' "$OWNER"
}

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --with-main) REPOS+=("mini-code-cli:feat/benchmarks:the agent and the wrappers") ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${EXTERNAL_REPOS_DIR:-$ROOT/external_repos}"
mkdir -p "$DEST"
echo "external repos: $DEST"

failed=0
for entry in "${REPOS[@]}"; do
    IFS=: read -r name branch what <<<"$entry"
    path="$DEST/$name"

    if [ -d "$path/.git" ]; then
        git -C "$path" fetch --quiet origin "$branch" \
            || { echo "  ! $name: fetch failed, using what is on disk" >&2; failed=1; }
    else
        git clone --quiet --branch "$branch" --single-branch \
            "https://github.com/$OWNER/$name.git" "$path" \
            || { echo "  ! $name: clone failed (network, or missing branch $branch)" >&2; failed=1; continue; }
    fi

    current="$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [ "$current" != "$branch" ]; then
        if [ -n "$(git -C "$path" status --porcelain)" ]; then
            echo "  ! $name is on '$current' with uncommitted changes; not switching" >&2
            failed=1
            continue
        fi
        git -C "$path" checkout --quiet --track "origin/$branch" 2>/dev/null \
            || git -C "$path" checkout --quiet "$branch"
    fi
    git -C "$path" branch --quiet --set-upstream-to="origin/$branch" >/dev/null 2>&1 || true

    printf 'ok: %s @ %s (%s, %s ahead) -- %s\n' "$name" "$branch" \
        "$(git -C "$path" rev-parse --short=12 HEAD)" \
        "$(git -C "$path" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo '?')" \
        "$what"
done

echo
echo "next: python -m pip install -r $ROOT/benchmarks/requirements.txt"
echo "push: benchmarks/scripts/push_all_repos.sh --dry-run"
exit "$failed"
