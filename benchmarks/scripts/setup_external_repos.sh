#!/usr/bin/env bash
# Put the checkouts the benchmark needs where it expects to find them.
#
#   benchmarks/scripts/setup_external_repos.sh              # clone/fetch the forks
#   benchmarks/scripts/setup_external_repos.sh --with-main  # ... and mini-code-cli
#   benchmarks/scripts/setup_external_repos.sh --help
#
# Three repositories take part in a run:
#
#   mini-code-cli       this repository: the agent, the harness wrappers, the docs
#   harbor              Terminal-Bench 2.x harness, with the Apptainer backend
#                       on branch `mini-code/apptainer`
#   terminal-bench-2-1  the 2.1 task set (data, not a package)
#                       on branch `mini-code/bench`
#
# harbor is *also* installable straight from its branch, which is what
# benchmarks/requirements.txt does:
#
#     pip install -r benchmarks/requirements.txt
#
# so a run on someone else's machine needs no clone at all. This script exists
# for local development, where you want the harness sources next to the agent
# sources -- to read them, to change them, and to push back (see
# push_all_repos.sh). It is deliberately boring: no prompting, no interactive
# git, no destructive commands, and re-runnable.
#
# The clones go to `external_repos/` at the repository root. That directory is
# git-ignored; the branches on github.com/ai-nikolai are the source of truth.
set -euo pipefail

# Never stop for a password: a CI job or a pipe has no terminal to type into.
export GIT_TERMINAL_PROMPT=0

OWNER="${EXTERNAL_REPOS_OWNER:-ai-nikolai}"

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Environment:
  EXTERNAL_REPOS_DIR    where the clones live (default: <repo>/external_repos)
  EXTERNAL_REPOS_OWNER  GitHub owner to clone from (default: ai-nikolai)

Exit status: 0 when every checkout is present and on its branch, 1 otherwise.
EOF
}

WITH_MAIN=0
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --with-main) WITH_MAIN=1 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${EXTERNAL_REPOS_DIR:-$ROOT/external_repos}"

# repo name : branch : what it is
REPOS=(
    "harbor:mini-code/apptainer:Terminal-Bench harness with the Apptainer backend"
    "terminal-bench-2-1:mini-code/bench:the Terminal-Bench 2.1 task set (data)"
)
if [ "$WITH_MAIN" = "1" ]; then
    REPOS+=("mini-code-cli:feat/benchmarks:the agent and the benchmark wrappers")
fi

mkdir -p "$DEST"
echo "external repos: $DEST"

status=0
for entry in "${REPOS[@]}"; do
    IFS=: read -r name branch what <<<"$entry"
    path="$DEST/$name"
    if [ ! -d "$path/.git" ]; then
        url="https://github.com/$OWNER/$name.git"
        echo "clone: $name <- $url"
        if ! git clone --quiet --branch "$branch" --single-branch "$url" "$path"; then
            echo "  ! could not clone $url (network, or the branch does not exist)" >&2
            status=1
            continue
        fi
    else
        echo "fetch: $name (already cloned)"
        if ! git -C "$path" fetch --quiet origin "$branch"; then
            echo "  ! fetch failed for $name; using what is on disk" >&2
            status=1
        fi
    fi

    if ! git -C "$path" rev-parse --verify --quiet "$branch" >/dev/null; then
        echo "  ! $name has no branch $branch" >&2
        status=1
        continue
    fi

    # Checkout only when we are not already there: `switch` on a dirty tree is
    # how a half-finished experiment gets lost.
    current="$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
    if [ "$current" != "$branch" ]; then
        if [ -n "$(git -C "$path" status --porcelain)" ]; then
            echo "  ! $name is on '$current' with uncommitted changes; not switching" >&2
            status=1
            continue
        fi
        echo "checkout: $name -> $branch"
        git -C "$path" checkout --quiet "$branch"
    fi

    # Track the remote branch so a later `push` has an upstream.
    git -C "$path" branch --quiet --set-upstream-to="origin/$branch" "$branch" 2>/dev/null || true

    sha="$(git -C "$path" rev-parse --short=12 "$branch")"
    ahead="$(git -C "$path" rev-list --count "origin/$branch..$branch" 2>/dev/null || echo "?")"
    echo "ok: $name @ $branch ($sha, $ahead commit(s) ahead of origin) -- $what"
done

cat <<EOF

Next:
  python -m pip install -r "$ROOT/benchmarks/requirements.txt"
  python "$ROOT/benchmarks/run_benchmarks.py" --dataset swe-bench --agent mini-code --mode short_eval

To publish local changes in these checkouts, see
  benchmarks/scripts/push_all_repos.sh --dry-run
EOF

exit "$status"
