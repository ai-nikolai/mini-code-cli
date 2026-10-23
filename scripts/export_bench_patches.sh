#!/bin/bash
# Export (or verify) the patches we carry on top of the three upstream projects.
#
# The forks live as git checkouts next to this repository -- `harbor/`,
# `SWE-bench/`, `terminal-bench-2-1/` -- on branches named `mini-code/*`. They are
# git-ignored: the branches are the source of truth, and the exported patches are
# what makes the changes reviewable here (and re-appliable to a fresh clone of
# upstream).
#
#   scripts/export_bench_patches.sh            # write benchmarks/patches/<repo>/
#   scripts/export_bench_patches.sh --check    # report what is missing, change nothing
#
# The patch file name carries the base commit, so two exports from different bases
# cannot silently collide.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

# repo-dir : branch : base ref : patch sub-directory
REPOS=(
  "harbor:mini-code/apptainer:v0.24.0:harbor"
  "SWE-bench:mini-code/apptainer:02e7a74ffd0b707aab73d203fe87bdc7c76afc8e:swebench"
  "terminal-bench-2-1:mini-code/bench:69671fbaac6d67a7ef0dfec016cc38a64ef7a77c:terminal-bench-2"
)

status=0
for entry in "${REPOS[@]}"; do
  IFS=: read -r dir branch base sub <<<"$entry"
  path="$ROOT/$dir"
  if [ ! -d "$path/.git" ]; then
    echo "missing: $dir/ (clone https://github.com/<you>/$dir.git into $dir/)"
    status=1
    continue
  fi
  if ! git -C "$path" rev-parse --verify --quiet "$branch" >/dev/null; then
    echo "missing: $dir has no branch $branch (git -C $dir switch -c $branch $base)"
    status=1
    continue
  fi
  if ! git -C "$path" rev-parse --verify --quiet "$base" >/dev/null; then
    echo "missing: $dir does not have the base commit $base (fetch it from upstream)"
    status=1
    continue
  fi
  base_sha="$(git -C "$path" rev-parse --short=12 "$base")"
  ahead="$(git -C "$path" rev-list --count "$base..$branch")"
  dirty=""
  [ -n "$(git -C "$path" status --porcelain)" ] && dirty=" (uncommitted changes)"

  if [ "$CHECK" = "1" ]; then
    echo "ok: $dir @ $branch, $ahead commit(s) on $base ($base_sha)$dirty"
    continue
  fi

  out="$ROOT/benchmarks/patches/$sub"
  mkdir -p "$out"
  find "$out" -maxdepth 1 -name '*.patch' -delete
  if [ "$ahead" = "0" ]; then
    echo "nothing to export: $dir @ $branch is level with $base"
    continue
  fi
  git -C "$path" format-patch --no-numbered --zero-commit \
      --output-directory "$out" "$base..$branch" >/dev/null
  echo "exported $ahead patch(es): $dir -> ${out#"$ROOT"/}"
done

if [ "$CHECK" = "1" ] && [ "$status" != "0" ]; then
  echo
  echo "hint: the checkouts are the user's forks; clone them next to this repo and"
  echo "      re-run without --check to export. Missing patches only matter for"
  echo "      reproducing the run elsewhere -- the suites fall back to the in-repo"
  echo "      adapters (benchmarks/suites/{tb,swebench}_apptainer.py)."
fi
exit "$status"
