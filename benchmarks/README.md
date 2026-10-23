# Benchmarks (academic evaluation)

`benchmarks/` runs coding agents against **Terminal-Bench 2.x** and **SWE-bench Verified** and
compares them: mini-code with several settings, Claude Code, Pi, plus a deterministic `mock`
agent for plumbing checks. It is an orchestration layer, not a reimplementation — every
benchmark is executed by its own official harness (`harbor` for Terminal-Bench 2.x, `tb` for the
1.x core datasets, `swebench.harness.run_evaluation` for SWE-bench), and every agent is driven
through its own CLI.

Everything in this directory is driven by one command line, `python -m benchmarks`, plus a
reader for the results (`python -m benchmarks.analyze`).

---

## 1. Prerequisites

| What | Why | Install |
| --- | --- | --- |
| Python ≥ 3.11 | the harness itself (`tomllib`, dataclasses) | — |
| `harbor` (Python ≥ 3.12) | executes Terminal-Bench 2.x | `pip install harbor` |
| `swebench` ≥ 5 | grades SWE-bench predictions | `pip install 'swebench>=5'` |
| `datasets` | materialises the SWE-bench instance list | `pip install datasets` |
| **a container runtime** | every task/instance runs in a container | Docker (with a daemon) **or** Apptainer (`apptainer`, formerly Singularity) |
| a model endpoint | the agents need an LLM | see §5 |
| `claude` / `pi` CLIs | only for host-side runs; harbor installs its own copies inside the container | `npm install -g @anthropic-ai/claude-code` / `curl -fsSL https://pi.dev/install.sh \| sh` |

The benchmark dependencies are deliberately *not* part of the CLI install:

```bash
python -m pip install -r requirements-bench.txt   # terminal-bench, swebench, vllm, litellm
```

`harbor` needs Python 3.12 while mini-code runs on 3.11, so keeping it in its own environment is
the normal case, e.g.:

```bash
conda create -y -n hb python=3.12 && conda activate hb && pip install harbor
PATH="$HOME/miniconda3/envs/hb/bin:$PATH" python -m benchmarks run ...   # harbor visible, mini-code still local
```

Check what this machine can actually do before anything else:

```bash
python -m benchmarks list
python -m benchmarks validate -c benchmarks/configs/ornith_30b.json
```

`validate` is the honest command: it checks the config *and* the host (harness binaries, agent
CLIs, container runtime, API keys) and lists every problem it finds with exit code 2.

---

## 2. The commands, in the order you use them

```bash
# 0. what exists
python -m benchmarks list

# 1. is this host able to run it?
python -m benchmarks validate -c benchmarks/configs/ornith_30b.json

# 2. dry run -- prints the exact argv of every command, downloads and starts nothing
python -m benchmarks plan -c benchmarks/configs/ornith_30b.json --limit 3

# 3. materialise datasets that need it (SWE-bench instance list; needs `datasets`)
python -m benchmarks prepare -c benchmarks/configs/ornith_30b.json --suite swebench --limit 3

# 4. run it (writes benchmarks/runs/<run-id>/ and, with --csv, one summary CSV)
python -m benchmarks run -c benchmarks/configs/ornith_30b.json \
    --run-id ornith-30b-smoke --limit 3 \
    --csv benchmarks/results/ornith-30b-smoke.csv

# 5. rebuild the summary of a finished run at any time
python -m benchmarks report --run benchmarks/runs/ornith-30b-smoke --csv

# 6. read the numbers back
python -m benchmarks analyze benchmarks/results/ornith-30b-smoke.csv

# 7. the serving commands for whatever the config points at
python -m benchmarks models -c benchmarks/configs/ornith_30b.json
```

Useful flags: `--suite`, `--agent`, `--model` filter the matrix (all repeatable), `--limit N`
caps tasks/instances per trial, `--dry-run` forces `plan` even after `run`, `--allow-host`
permits host-side agent execution (see §7), `--no-resume` redoes trials that already finished.

### The fixed-model smoke matrix

`benchmarks/configs/ornith_30b.json` is the "one model, three agents, three samples" config:
`terminal-bench@2.0` and `SWE-bench/SWE-bench_Verified`, `limit: 3`, agents
`mini-code:default`, `claude-code:bypass-permissions` and `pi:default`, all against
`ornith-ai/Ornith-1.5-35B-A3B` on `http://127.0.0.1:30000`. It is the config every example on
this page uses. Two model entries exist because the same vLLM server is reached over two
protocols: `openai` (mini-code, Pi) and `anthropic` (Claude Code, `/v1/messages`).

Config layout (all keys are validated; a typo is an error, not a silent default):

| Key | Meaning |
| --- | --- |
| `suites[]` | `name`, `dataset`, `version`, `split`, `limit`, `task_ids`, `options` (suite-specific) |
| `agents[]` | `name`, `setting`/`settings[]`, `options`, `models` (restricts the cross-product) |
| `models[]` | `name`, `model`, `base_url`, `api` (`openai`\|`anthropic`), `api_key_env` |
| `runtime` | `auto` (default), `docker` or `apptainer` |
| `docker` | `cpus`, `memory`, `pids_limit`, `network` |
| `timeouts` | `instance_seconds`, `agent_seconds`, `test_seconds` |
| `env` | variables exported to every harness/agent command |
| `output_dir`, `concurrency` | where runs land, how many trials in parallel |

---

## 3. Docker or Apptainer

Both are supported, and `runtime: "auto"` picks whichever is installed (Docker first). Pin it
when a comparison has to be reproducible across machines.

```bash
# Docker (a daemon is required; this is what harbor and SWE-bench assume)
python -m benchmarks run -c benchmarks/configs/ornith_30b.json --limit 3 --csv

# Apptainer (no daemon, no root -- HPC clusters; images are pulled as .sif)
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer      # not /tmp: images are GBs
export TMPDIR=/scratch/$USER/tmp
python -m benchmarks run -c benchmarks/configs/ornith_30b.json --limit 3 --csv
```

`plan` shows the translation explicitly, so nothing is guessed:

```bash
# docker
$ docker run --rm --name bench-swebench-astropy-12907 -v $PWD/out:/out -w /testbed \
    -e OPENAI_API_KEY=... swebench/sweb.eval.x86_64.astropy__astropy-12907 bash -lc '...'
# apptainer
$ apptainer exec --cleanenv --pwd /testbed --bind $PWD/out:/out \
    --env OPENAI_API_KEY=... docker://swebench/sweb.eval.x86_64.astropy__astropy-12907 bash -lc '...'
```

What Apptainer cannot do (printed as notes by `plan`, never silently dropped): no per-container
`--cpus`/`--memory`/`--pids-limit`, and containers share the host network namespace, so
`network=` is advisory. Point `APPTAINER_CACHEDIR` and `TMPDIR` at a large filesystem — the
default `/tmp` is frequently a few GB, and a task image needs ~2 GB while being converted.

**Where Apptainer is not enough** (measured, not assumed — see §8):

* **Terminal-Bench 2.x** — harbor drives containers through Docker. Its Apptainer backend
  (`harbor run --env singularity`) does convert the task images to `.sif`, but it then needs a
  `/usr/bin/python3` *inside* the image to bootstrap its RPC server, and Terminal-Bench images do
  not ship one. Use a Docker host for this suite.
* **SWE-bench grading** — `swebench.harness.run_evaluation` drives the Docker SDK. On an
  Apptainer-only machine only the agent/patch half can run; set `options.skip_evaluation: true`
  to make that explicit (`validate` warns, and does not pretend the numbers are comparable).

---

## 4. Datasets

| Suite | Dataset id | Executed by |
| --- | --- | --- |
| `terminal-bench` | `terminal-bench@2.0` (`harbor datasets list --legacy` for what the registry publishes; pin `2.1` as soon as it appears) | `harbor run -d <dataset>@<version>` |
| `terminal-bench` (`options.harness: "tb"`) | `terminal-bench-core==0.1.1` | `tb run --dataset ...` (Terminal-Bench 1.x) |
| `swebench` | `SWE-bench/SWE-bench_Verified`, split `test` | `python -m swebench.harness.run_evaluation` |

Claude Code and Pi are harbor's own agents for Terminal-Bench; mini-code goes through the
installed-agent bridge in `benchmarks/harbor_agents.py` (harbor needs the repository on
`PYTHONPATH` for that, which the suite sets for you).

---

## 5. Serving the model

vLLM, tool calling enabled (one H200 is enough for the 35B-A3B):

```bash
vllm serve ornith-ai/Ornith-1.5-35B-A3B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

Current vLLM also serves the Anthropic Messages API at the same port (`/v1/messages`), which is
all Claude Code needs:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:30000
export ANTHROPIC_API_KEY=local-vllm            # value is ignored; it only has to be set
```

Older vLLM builds expose `/v1/messages` incompletely; put the LiteLLM bridge in front then
(`benchmarks/models.py` prints the exact command) and point `ANTHROPIC_BASE_URL` at the proxy:

```bash
litellm --config litellm.yaml --port 40000     # model_list -> openai/ornith… -> 127.0.0.1:30000
```

Containers must be able to reach the endpoint: bind vLLM to `0.0.0.0` and use an address the
container can route to (on Docker, `--network host` or a LAN IP; Apptainer shares the host
network already).

---

## 6. What gets produced

```
benchmarks/
├── runs/<run-id>/                  # one directory per run, git-ignored
│   ├── manifest.json               # git SHA (+dirty), runtime + version, resolved config, per-trial status
│   ├── trials/<trial>/trial.json   # the exact argv of every command, the agent's settings
│   ├── trials/<trial>/logs/*.log   # stdout/stderr of each command, credentials redacted
│   ├── trials/<trial>/results.jsonl# outcomes of this trial, one JSON object per line
│   ├── summary.json                # the cross-agent table, machine-readable
│   └── summary.md                  # the same table as markdown
└── results/<run-id>.csv            # --csv: the table as CSV, git-ignored
```

Per-suite extras live next to the trial: harbor writes `trials/<trial>/harbor/<job>/result.json`
plus one directory per task (with `trial.log`, `exception.txt`); SWE-bench writes
`instances/<instance_id>/{model.patch,agent.log}`, `predictions.jsonl` and the grader's
`<report_dir>/<model>.<run-id>.json`.

`summary.md` looks like this (all six cells, one row each):

| suite | agent | setting | model | resolved | n | rate | errors | timeouts | cmd fail | mean s |
|---|---|---|---|---|---|---|---|---|---|---|
| terminal-bench | mini-code | default | ornith-35b-a3b | 0 | 3 | 0.00% | 3 | 0 | 0 | - |

* `n` is the number of instances that produced an outcome; `n = 0` means the trial never ran.
* `cmd fail` counts commands that did not finish cleanly — a missing harness lands there, never
  in the score. A cell with `n = 0, cmd fail = 1` is a *broken* run, not a bad agent.
* Runs are resumable: a trial with a `DONE` marker is skipped unless `--no-resume` is passed.

---

## 7. Reading the numbers back

```bash
python -m benchmarks analyze benchmarks/results/ornith-30b-smoke.csv
python -m benchmarks analyze benchmarks/runs/ornith-30b-smoke            # same, straight from the run
python -m benchmarks analyze results/*.csv --md report.md --json report.json --csv combined.csv
```

The analyser reads a CSV, a `summary.json` or a whole run directory, merges rows that share
`(suite, agent, setting, model)` (so several runs of the same configuration add up), and prints:

```
suite           agent        setting             model                    n  ok  rate  err  to  cmdfail  mean s
--------------  -----------  ------------------  -----------------------  -  --  ----  ---  --  -------  ------
swebench        claude-code  bypass-permissions  ornith-35b-a3b-anthropic 0  0   -     0    0   1        -
terminal-bench  mini-code    default             ornith-35b-a3b           3  0   0%    3    0   0        -
ALL                                                                       9  0   0%    9    0   3        0
```

`ok` = resolved instances, `rate` = `ok/n`, `to` = timeouts, `cmdfail` = commands that did not
finish, `mean s` = mean wall-clock per outcome (pooled, not a mean of means). `--md`, `--json`
and `--csv` write the same table for a README, a script or a spreadsheet.

---

## 8. Results on the H200 used to validate this

Fixed model `ornith-ai/Ornith-1.5-35B-A3B` (vLLM on `:30000`), `--limit 3`,
`runtime: auto`. The host has **no Docker** — Apptainer 1.5.0 only — so this is the
Docker-free path from §10 producing the numbers:

```bash
python -m benchmarks run -c benchmarks/configs/ornith_30b.json \
    --suite terminal-bench --limit 3 --run-id tb-app-3 \
    --csv benchmarks/results/tb-app-3.csv
python -m benchmarks analyze benchmarks/results/tb-app-3.csv
```

| suite | agent | setting | n | resolved | errors | note |
| --- | --- | --- | --- | --- | --- | --- |
| terminal-bench | mini-code | default | 3 | 0 | 0 | three tasks completed; none solved |
| terminal-bench | claude-code | bypass-permissions | 3 | 0 | 1 | installs and runs in the container; one task hit `AgentAuthenticationError` against the local endpoint |
| terminal-bench | pi | default | 3 | 0 | 0 | three tasks completed; none solved |
| **ALL** | | | **9** | **0** | **1** | |

Three tasks each (`adaptive-rejection-sampler`, `bn-fit-modify`,
`break-filter-js-from-html`), 9 agent runs, every container built and torn down by
harbor through Apptainer. That 0/9 is a *score*, not a failure: the tasks are
hours-long for a human and the model is a 35B MoE. Before this path existed the
same command reported `n = 3, errors = 3` per agent (harbor died installing
`/usr/bin/python3`) — a broken run, which the report would have shown as a zero.

What this run exercised: config expansion, the SIF preparation step (3 derived
images, cached afterwards), harbor's singularity backend, agent installation
inside the task image, verifier execution, per-task logs, outcome collection
(including harbor's `reward_stats`, deduplicated across retries), `summary.{json,md}`,
the CSV export and the analyser.

**SWE-bench now grades without Docker too** (see §11). The instance images are public
OCI (`swebench/sweb.eval.x86_64.<instance_id>`, `__` -> `_1776_` in the tag), and
grading is backend-agnostic, so the harness only needed its transport replaced. The
acceptance test: the *gold* patch for `astropy__astropy-12907` scores
`resolved: true` through `--backend apptainer` with no Docker daemon on the machine.

### What the Docker-free path actually costs (measured)

The first end-to-end run of a *Dockerfile-only* task on this host — the case that
needs an image built without a Docker daemon at all — finished like this:

```bash
harbor run -p benchmarks/runs/localtask/bn-fit-modify -a oracle \
    --env singularity --environment-kwarg singularity_image_cache_dir=$SIF_CACHE
# reward 1.0, 9/9 verifier tests, 74 s wall clock
```

Getting there exposed three bugs worth recording, because each one *looked* like a
harness failure and was not:

- **`WORKDIR` is a working directory, not a `mkdir`.** The Dockerfile translation
  emitted `mkdir -p /app` and nothing else, so the task's own
  `RUN curl -o bn_sample_10k.csv` wrote the dataset to `/`, and a perfectly correct
  oracle solution then failed 9 verifier tests on the missing file. `%post` is one
  shell, so the translation now emits `cd /app` too. The image cache key is the
  *translated definition*, so a translation fix cannot silently reuse a stale `.sif`.
- **A quiet trial is not a stalled trial.** uvicorn's access log is off (it floods
  the trial log), so "server up, nothing happens" was unattributable. The server now
  appends one line per request to `/staging/harbor-requests.log` — on the host, read-
  able from outside a trial that is still running — and `HARBOR_SERVER_ACCESS_LOG=1`
  turns the uvicorn log back on.
- **Apptainer's writable overlay is 64 MiB.** `apt-get install nodejs` inside a
  `.sif` fails with `unable to move aside './etc/logrotate.d' ... Invalid
  cross-device link` (dpkg renames a directory that lives in the read-only lower
  layer), and `npm install -g` needs ~300 MiB. Claude Code is therefore installed
  from the static Node tarball into `/opt/node`, and for SWE-bench instances the
  agent's npm tree is installed into a **bind-mounted** host directory mounted at
  `/opt/agent`, with only the `/usr/local/bin` symlinks written to the overlay. Those
  symlinks do not survive between `apptainer exec` calls, so the agent command puts
  `/opt/agent/bin` on `PATH` itself.

**A `.sif` cannot be written in place, so the checkout has to be copied out.** The
first SWE-bench smoke reported `n = 0` for every agent, and the reason was not the
agents: `--writable-tmpfs` is discarded when `apptainer exec` exits, so the agent's
edits were gone before the patch was captured (mini-code edited files, the patch was
0 bytes, the harness had nothing to grade). `apptainer overlay create` is the
textbook answer and needs privileges this host does not grant (`mkfs.ext3: Operation
not supported`), so the suite copies `/testbed` to the host once per instance and
bind-mounts it back for both the agent and the capture. With that, the same run
reports `n = 1`, `errors = 0`, and the evaluation actually executes the tests (53 s)
instead of returning an empty verdict.

Claude Code against vLLM needs no proxy: vLLM 0.30 answers `POST /v1/messages`
(Anthropic Messages API) directly, so `ANTHROPIC_BASE_URL=http://127.0.0.1:30000`
and any `ANTHROPIC_API_KEY` are enough — Apptainer shares the host network namespace,
which is why the loopback address works here and would not under Docker. One
correction was needed in the fork: `--model anthropic/<id>` sent the routing prefix
to the endpoint (`... may not exist or you may not have access`), and the prefix is
now stripped when a custom base URL is set. With that, a terminal-bench run streamed
49 assistant turns and 8.8 MB of transcript before hitting its own 900 s budget — an
authentication failure, at last, is not what stops it.

## 9. Safety and habits

* **`plan` first.** It prints the exact argv (never a shell string), the environment variable
  *names* it sets and the container calls, without downloading a dataset or starting a container.
* **Agent execution is refused outside a container** unless `--allow-host` is passed: several
  settings (`--sudo`, `bypassPermissions`) are only safe in a disposable environment.
* **Nothing secret is written.** Logs, results and summaries pass through a redactor that keeps
  environment variable *names* and replaces values; only `api_key_env` names ever enter a config.
* **Start small.** `--limit 1` on `terminal-bench` is the cheapest end-to-end check; harbor is
  young and moves fast, so re-read `harbor run --help` when a flag stops being accepted.

---

## 10. Apptainer without Docker (Docker-free path)

On a cluster with no Docker daemon, Terminal-Bench 2.x can still be run — not by
reimplementing it, but by preparing the task images so harbor's existing Apptainer
backend (`--env singularity`) can use them. Three things were missing, and all
three are handled by `benchmarks/suites/tb_apptainer.py`:

1. **The task images have no `/usr/bin/python3`.** harbor's backend runs a small
   FastAPI server inside the container; on a bare image it fails with
   `FATAL: cannot install /usr/bin/python3`. Each task image is therefore
   converted **once** into a derived `.sif` that already contains `python3`,
   `python3-venv`, `tmux`, `asciinema`, `git`, `curl` and — because the SIF is
   read-only and harbor's `--writable-tmpfs` layer does not survive its agent
   setup step — a ready `/opt/harbor-server` venv with `uvicorn`/`fastapi`.
2. **`task.toml` must name an image.** The staged copy of each task points
   `docker_image` at its derived `.sif`, so nothing is pulled or converted at run
   time.
3. **harbor's default mount policy kills DNS.** Its default is
   `singularity_no_mount=home,tmp,bind-paths`, and `bind-paths` also suppresses the
   `/etc/resolv.conf` bind — the container then cannot resolve anything, which
   surfaces far away as `apt`/`curl`/verifier failures (`Could not resolve host:
   astral.sh`). The suite passes `singularity_no_mount=home,tmp` instead.

```bash
# 1. derive one .sif per task and stage a local dataset (downloads the task files
#    with `harbor download <dataset> --export` on first use; cached under
#    benchmarks/.cache/ which is git-ignored)
export BENCH_SIF_CACHE=/scratch/$USER/sif        # GBs: never point this at /tmp
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer TMPDIR=/scratch/$USER/tmp
python -m benchmarks prepare -c benchmarks/configs/ornith_30b.json \
    --suite terminal-bench --limit 3

# 2. run it: harbor is invoked with -p <staged dataset> --env singularity
python -m benchmarks run -c benchmarks/configs/ornith_30b.json \
    --suite terminal-bench --limit 3 --run-id tb-apptainer \
    --csv benchmarks/results/tb-apptainer.csv
```

`plan` prints both commands even before the images exist, so the cost is visible:
first run ≈ 5–10 min per task image (pull + `apt-get` + `pip install` in
`apptainer build --fakeroot`), later runs are cached and cost seconds.

Verified on an H200 with Apptainer 1.5.0, no Docker, no root:

| Check | Result |
| --- | --- |
| `harbor run -p <staged task> -a oracle --env singularity` | **reward 1.0**, `1 passed` |
| `-a claude-code` in the same container | installs and runs; needs the model id without harbor's `anthropic/` prefix |
| `apptainer pull docker://swebench/sweb.eval.x86_64.<id>` | works — SWE-bench instance images are public OCI |

Requirements and limits of this path:

* `apptainer build --fakeroot` must be permitted on the machine (it is here:
  `unshare -Ur` works). Rootless podman is *not* an alternative — it needs
  `/etc/subuid` ranges this account does not have.
* budgets: `cpus`/`memory`/`pids_limit` are advisory under Apptainer (printed as
  notes by `plan`), and containers share the host network namespace.
* SWE-bench grading under Apptainer is §11, not this section: `tb_apptainer.py`
  derives images for Terminal-Bench only.

---

## 11. Our forks, and when they are used

Two behaviours cannot be expressed from outside the tools, so we carry branches of
them. They are cloned **next to** this repository and git-ignored; the diffs are
exported into `benchmarks/patches/` so the changes are reviewable here.

| checkout | fork | branch | base |
| --- | --- | --- | --- |
| `harbor/` | `ai-nikolai/harbor` | `mini-code/apptainer` | tag `v0.24.0` (`bf8996ef6d97`) |
| `SWE-bench/` | `ai-nikolai/SWE-bench` | `mini-code/apptainer` | `02e7a74ffd0b` (`swebench.__version__ == "5.0.2"`; upstream has no `v5.0.2` tag) |
| `terminal-bench-2-1/` | `ai-nikolai/terminal-bench-2-1` | `mini-code/bench` | `69671fbaac6d` (the commit harbor's `registry.json` pins for `terminal-bench@2.0`) |

What each patch changes:

* **harbor** — the singularity environment builds a task's `.sif` from
  `environment/Dockerfile` when `task.toml` names no image (content-addressed
  name, `apptainer build --fakeroot`, cached), reports a fakeroot refusal as such,
  uses `apptainer` when installed (`HARBOR_CONTAINER_BIN` overrides), defaults
  `--no-mount` to `home,tmp` (keeping `/etc/resolv.conf`, so DNS works), and its
  `bootstrap.sh` prefers a pre-baked `/opt/harbor-server` venv and stops apt's
  sandbox from rejecting the fakeroot user.
* **SWE-bench** — `swebench/harness/container_backends/`: a backend interface
  (`create`/`copy`/`exec-with-timeout`/`cleanup`), the Docker backend as a pure
  adapter over the existing helpers, and an Apptainer one, plus
  `--backend docker|apptainer`, `--sif-cache`, `--apptainer-bin`. Apptainer has no
  long-lived container, so the backend replays state-changing commands in their
  working directory before each exec, under `--fakeroot --writable-tmpfs`.
* **terminal-bench-2** — no changes yet; it is pinned so a run cannot silently pick
  up a different task set.

Install them into the environments that run them (editable, `--no-deps` so the
already-installed dependencies are not churned):

```bash
~/miniconda3/envs/hb/bin/python -m pip install -e ./harbor --no-deps
python -m pip install -e ./SWE-bench --no-deps           # the env that runs benchmarks
python -c "import harbor, swebench; print(harbor.__file__, swebench.__file__)"
```

The suites decide by **what is imported**, not by what exists on disk
(`benchmarks/forks.py`), because a stale site-packages copy and a fresh checkout
look identical otherwise:

| detected | Terminal-Bench | SWE-bench |
| --- | --- | --- |
| patched harbor imported | `harbor run -d terminal-bench@2.0 --env singularity` builds each `.sif` on demand; no staging (`tb_apptainer.py` is skipped) | — |
| unpatched harbor | `tb_apptainer.py` stages a SIF-pointing dataset (§10) | — |
| patched swebench imported | — | `python -m swebench.harness.run_evaluation --backend apptainer` |
| unpatched swebench | — | `benchmarks/suites/swebench_apptainer.py` (same report shape) |

`plan` prints which one will run, and `manifest.json` records every checkout's
branch, SHA and dirty flag, so a result cannot be reproduced against different code
by accident.

Export the patches (and check the checkouts) with:

```bash
scripts/export_bench_patches.sh --check     # what is missing, changes nothing
scripts/export_bench_patches.sh             # benchmarks/patches/<repo>/*.patch
git -C harbor am benchmarks/patches/harbor/*.patch    # re-apply to a fresh clone
```

Push the branches once, so the patches have a home:

```bash
git -C harbor            push -u origin mini-code/apptainer
git -C SWE-bench         push -u origin mini-code/apptainer
git -C terminal-bench-2-1 push -u origin mini-code/bench
```

### Evidence from this host (no Docker, Apptainer 1.5.0, no root)

| Check | Result |
| --- | --- |
| `harbor run -p <task with only a Dockerfile> -a oracle --env singularity` | the `.sif` is built from the Dockerfile through a generated definition file (`harbor-local_bn-fit-modify_<hash>.sif`), and the FastAPI server starts inside it |
| `harbor run -d terminal-bench@2.0 -a oracle -i bn-fit-modify --env singularity` | registry images pull and boot, but the *trial* then stalls on this shared host (no agent output, no verifier output), so no reward was collected; the CLI reports `n_errored_trials: 1` |
| `python -m swebench.harness.run_evaluation --backend apptainer` with the **gold** patch | `Instances resolved: 1` / `unresolved: 0` — `astropy__astropy-12907`, 53 s, no Docker daemon |
| `pytest tests/test_container_backends.py` (SWE-bench fork) | 8 passed |
| `pytest tests/unit/environments/test_singularity_*.py` (harbor fork) | 13 passed |

Two things this host corrected, worth knowing before debugging the same symptoms:

* **`apptainer build out.sif <directory>` is not a Dockerfile build.** It writes a
  40 kB image *of the directory* and exits 0; the container then has no `bash` and
  no `WORKDIR`, and harbor dies at `failed to set working directory: chdir /app`.
  The fork translates the Dockerfile into a definition file for exactly this
  reason. Apptainer 1.5.0-1.el9 has no `--buildkit`.
* **`--pwd` must exist.** Apptainer fails before the bootstrap script runs if the
  working directory is missing, so a `WORKDIR` has to become `mkdir -p` at build
  time (which the translation does).

### Image cache: what this host taught us

* `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` must point at a **large** filesystem.
  Here `/tmp` and `/var/tmp` are 1 GB and `/dev/shm` is RAM: extracting a 1.2 GB
  SWE-bench image into `/dev/shm` is OOM-killed (`signal 9`), which looks like a
  corrupt image because the `.sif` exists at its pre-allocated size long before it
  is valid. Use the cluster scratch (`/hx2-weka/...` here), accept ~15-25 min for
  the first SWE-bench instance image, and never probe the file mid-build.
* Weka has no user xattrs (`ENOTSUP`, warn-only for `mksquashfs`), and the
  SWE-bench `.sif` for `astropy__astropy-12907` ends up at **1.09 GB** — budget
  ~1-4 GB per instance, cached and reused.

### Claude Code against a local endpoint

`AgentAuthenticationError` is the CLI's catch-all: a missing key, an unreachable
host, a non-Anthropic route and a genuine 401 all look the same. The adapter now
sets the whole set the CLI needs for a local endpoint (`ANTHROPIC_BASE_URL`,
`ANTHROPIC_AUTH_TOKEN` *and* `ANTHROPIC_API_KEY`, `ANTHROPIC_SMALL_FAST_MODEL`, and
the telemetry switches that stop it calling out from a compute node) and
`preflight` probes `/v1/messages`, translating the answer into an instruction
(404 -> "not an Anthropic-compatible endpoint, put LiteLLM in front of vLLM";
401/403 -> "a key must exist, its value is not checked"; unreachable -> "start the
endpoint, and check routing from inside the container").

---
