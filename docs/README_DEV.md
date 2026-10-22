# mini-code-cli — Developer Guide

This is the developer-facing companion to [README.md](../README.md). It covers local setup,
linting, the test suite, CI, and how to cut a release. User-facing documentation (install,
CLI flags, tools, sub-agents, in-session commands) stays in the [README](../README.md);
the design of the agent loop is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

- [Local development setup](#local-development-setup)
- [Repository layout](#repository-layout)
- [Linting](#linting)
- [Testing](#testing)
- [Continuous integration](#continuous-integration)
- [Releasing (tag `main` → GitHub release)](#releasing-tag-main--github-release)
- [Manual / PyPI upload](#manual--pypi-upload)

## Local development setup

```bash
git clone https://github.com/ai-nikolai/mini-code-cli.git
cd mini-code-cli

# optional, but recommended: an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# editable install (the only runtime dependency is `requests`)
pip install -e .

mini-code --help
```

Requirements: Python `>= 3.10` (see `pyproject.toml`). The dev loop itself needs **no extra
dependencies** — linting and testing are standard-library only. `build` and `twine` are only
needed when you actually build/upload distributions.

## Repository layout

```
mini_code_cli/            the package (agent loop, tools, prompts, plan mode, sub-agents, skills)
  subagents/              sub-agent base class + the four built-in sub-agents
scripts/lint.py           stdlib-only linter used by CI (and by you)
scripts/run_model.sh      helper for serving a local model (a small checkpoint; see README)
scripts/push_version_tag.py  tags HEAD with the pyproject version and pushes the tag
tests/                    stdlib unittest suite
.github/workflows/ci.yml      lint + test on every push / pull request
.github/workflows/release.yml build + GitHub release, triggered by pushing a `v*` tag
docs/README_DEV.md        this developer guide
docs/ARCHITECTURE.md      design of the agent loop, plan mode, compaction and sub-agents
pyproject.toml            package metadata, version, entry point
```

[ARCHITECTURE.md](ARCHITECTURE.md) describes the design of the agent loop, plan mode,
compaction and sub-agents.

## Linting

There is no third-party linter to install: `scripts/lint.py` is itself standard-library only
and checks that every `*.py` under `mini_code_cli/`, `tests/` and `scripts/` compiles, has
unambiguous indentation (`tabnanny`), and is tidy (no trailing whitespace, no tab characters,
a newline at end of file). It exits non-zero if it reports anything:

```bash
python3 scripts/lint.py          # lint the defaults
python3 scripts/lint.py tests    # ...or specific files / directories
```

## Testing

The project ships with a small standard-library test suite (no extra dependencies to install):

```bash
# Run the whole suite from the repo root:
python3 -m unittest discover -s tests -p "test_*.py"

# ...or run a single test file:
python3 tests/test_tokens.py
python3 tests/test_e2e_agent_loop.py
python3 tests/test_interactive.py
python3 tests/test_subagents.py
```

The tests focus on the parts of the agent that are easy to get wrong and worth guarding:

The suite is hermetic: every test builds its own temporary `--allowed-dir` and cache dir, so
nothing is written into your real `~/.cache/mini/sessions/` and — importantly — the session
pointers the tests create go into a `.mini_tests` directory inside the test's temporary allowed
dir, never into a real `.mini` (that one is yours: your skills folder and `last_session.json`).

- **`test_e2e_agent_loop.py`** — the whole loop, driven the way a user drives it
  (`AgentSession(args).run(...)`), with only the LLM stubbed: real tools run on disk, the
  `--allowed-dir` sandbox is exercised end to end, plan mode's tracking files are written,
  sub-agent delegation runs a nested loop, history compaction rolls the segment over
  (keeping per-segment usage), sessions restore in place while a folder that does not
  satisfy the restore contract (no recorder, or no segment for it) is refused in favour of a
  fresh session, and the `edit` tool / richer
  tool schemas are covered.
- **`test_tokens.py`** — the compaction *policy*: the model → context-window table, the
  budget resolution (explicit / model / conservative fallback), the token estimator, and
  the trigger itself — which fires on projected tokens (+ the reply reserve) and **never**
  on the message count.
- **`test_interactive.py`** — the shell-safety classifier (DENY / ASK / ALLOW tiers) and
  compaction edge cases (an empty summary is retried, then abandoned without losing the
  transcript; a failing summariser call never corrupts the segment).
- **`test_subagents.py`** — sub-agent delegation, isolation of the child session, the
  `report.md` channel and the intersection of child permissions with the parent's.

This keeps the tool honest as it grows, and makes it safe to experiment with riskier features
(like `--enable-shell`) without regressions.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs exactly these two commands on every
push and pull request:

```bash
python3 scripts/lint.py
python3 -m unittest discover -s tests -p "test_*.py"
```

A green CI run is the baseline for merging to `main`; the release workflow below does not re-run
the tests, so make sure `main` is green before you tag it.

## Releasing (tag `main` → GitHub release)

Releases are cut **from `main` by pushing a tag**, never from a feature branch. The tag is what
fires [`.github/workflows/release.yml`](../.github/workflows/release.yml).

### 1. Bump the version

`version` in `pyproject.toml` is the single source of truth for the package version:

```toml
[project]
name = "mini-code-cli"
version = "0.1.1.0"
```

Bump it, and (optionally) the `version` line in the citation block of `README.md`. Commit and
merge to `main` through a pull request.

### 2. Tag `main` and push the tag

```bash
git checkout main
git pull --ff-only origin main
git tag -a v0.1.1.1 -m "release 0.1.1.1"     # tag must match the pyproject version, prefixed with "v"
git push origin v0.1.1.1                     # <- this triggers the release workflow
```

Or let `scripts/push_version_tag.py` do it: it reads the version from `pyproject.toml`, tags
`HEAD` and pushes the tag, and is a no-op if the tag is already on the remote. It **refuses**
unless `HEAD` is exactly the tip of `origin/main` (fetched fresh) and `pyproject.toml` is
committed — the workflow would reject an off-`main` tag anyway, so the script fails first,
before a red CI run. Use `--dry-run` to see what it would do.

### 3. What the release workflow does

`.github/workflows/release.yml` is triggered by `push` of any `v*` tag, and runs a single
`ubuntu-latest` job:

1. **Checks out the tagged commit** (full history, so the ancestry check below works).
2. **Verifies the tag is on `main`** — it fetches `origin/main` and fails unless the tagged
   commit is an ancestor of `main`. Tagging a feature branch therefore never produces a release.
3. **Verifies the tag matches the package version** — `v0.1.1.0` must correspond to
   `version = "0.1.1.0"` in `pyproject.toml`, so the tag can never lie about what was built.
4. **Builds the distributions** with `python -m build` (sdist + wheel in `dist/`) — exactly the
   artifacts PyPI/twine expect.
5. **Validates them** with `python -m twine check dist/*` (metadata, README rendering, etc.),
   failing the workflow if the packages are not uploadable.
6. **Uploads `dist/` as a workflow artifact** (`dist`, 14-day retention) so the build is
   inspectable/downloadable from the Actions run.
7. **Creates a GitHub release** named after the tag, with generated release notes, and attaches
   the sdist and the wheel to it. Re-running the workflow for an existing release re-uploads
   the files (`--clobber`) instead of failing.

The result: every `v*` tag pushed on `main` produces a new GitHub **release** carrying the
PyPI-ready build.

### 4. If the workflow fails

- *"tag is not on main"* — you tagged a branch that is not merged. Delete the tag
  (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), merge, re-tag.
- *"does not match pyproject version"* — bump `version` in `pyproject.toml`, or use the tag that
  matches it. Delete the bad tag before pushing the corrected one.
- *`twine check` failure* — almost always bad metadata or an unrenderable `README.md`; fix and
  re-tag (a re-run of a successful build keeps the same tag).

## Manual / PyPI upload

The release workflow publishes to **GitHub**, not PyPI: publishing to an index is a deliberate,
separate step (it is a one-way action). Download the artifacts from the GitHub release (or build
locally) and upload them when you are ready:

```bash
# build the sdist + wheel locally (needs: pip install build)
python3 -m build

# sanity-check the metadata (needs: pip install twine)
python3 -m twine check dist/*

# upload to PyPI with an API token (needs: pip install twine)
TWINE_USERNAME="__token__" \
TWINE_PASSWORD="$PYPI_API_KEY" \
python3 -m twine upload dist/* --verbose
```

To publish automatically on a tag instead, add a job to `release.yml` that either uses
`pypa/gh-action-pypi-publish` with [trusted publishing](https://docs.pypi.org/trusted-publishers/)
or fills in the `twine upload` command above from a `PYPI_API_TOKEN` secret; give it
`environment: pypi` and `permissions: id-token: write` if you use trusted publishing.
