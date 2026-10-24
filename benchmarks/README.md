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
| `harbor` (Python ≥ 3.12) | executes Terminal-Bench 2.x | `pip install -r benchmarks/requirements.txt` (our fork's branch) |
| `swebench` ≥ 5 | grades SWE-bench predictions | `pip install 'swebench>=5'` |
| `datasets` | materialises the SWE-bench instance list | `pip install datasets` |
| **a container runtime** | every task/instance runs in a container | Docker (with a daemon) **or** Apptainer (`apptainer`, formerly Singularity) |
| a model endpoint | the agents need an LLM | see §5 |
| `claude` / `pi` CLIs | only for host-side runs; harbor installs its own copies inside the container | `npm install -g @anthropic-ai/claude-code` / `curl -fsSL https://pi.dev/install.sh \| sh` |

The benchmark dependencies are deliberately *not* part of the CLI install:

```bash
python -m pip install -r benchmarks/requirements.txt   # harbor, terminal-bench, swebench, vllm, litellm
benchmarks/scripts/setup_external_repos.sh             # the TB 2.1 tasks: data, not a package (§13.1)
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

## 2. The one command: `benchmarks/run_benchmarks.py`

Two commands cover the two datasets, for one agent or all three:

```bash
python benchmarks/run_benchmarks.py --dataset swe-bench        --agent mini-code --mode short_eval
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent all       --mode short
```

It is executable, so `./benchmarks/run_benchmarks.py ...` works too; the file is a
three-line wrapper around `python -m benchmarks` (same commands, lower-level form).

`--agent all` runs mini-code, pi and claude-code one after another and writes a
single CSV covering all three. Everything lands in one campaign directory:

```
results/<dataset>-<agent>-<mode>-<timestamp>/
    summary_results.csv      # one row per agent -- the file you open first
    mini-code/   pi/   claude-code/   # each agent's own run dir, manifest, logs
```

`summary_results.csv` has a fixed header:

```
dataset,agent,mode,trials,resolved,pass_rate,mean_reward,mean_seconds,mean_tokens,run_id,status
```

`status` is `ok`, `partial_failures`, `all_failed`, or `dry-run`; an empty cell means
"this suite does not report that number", never zero.

Modes:

| `--mode` | limit | graders | use it for |
|---|---|---|---|
| `short` | 3 | off | does this agent work end to end at all |
| `short_eval` | 3 | on | the smallest run whose number is a real score |
| `full` | all | on | the real measurement (agents get an hour) |

`--limit`, `--n-concurrent`, `--run-id` and `--results-dir` override the mode;
`--dry-run` prints every command it would run and the summary path, and starts
nothing.

### The same thing one layer down

`benchmarks/run_benchmarks.py` is a thin front end over `python -m benchmarks`; the flags below are
what it composes, and they are still there when you need to change something it does
not expose:

#### The commands, in the order you use them

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

## 4. What every agent is told (one file)

`benchmarks/agents/agent.md` is the preamble every agent receives -- where the
checkout is, reproduce before fixing, leave the fix in the worktree, do not commit,
handle the documented edge cases. It is *one* file, so a comparison cannot be
explained by three different prompts:

| agent | how the file reaches it |
|---|---|
| mini-code | `--agent-md <file>` (the CLI places it right after the system prompt) |
| claude-code | `--append-system-prompt "$(cat file)"` (the flag takes text, so it is inlined) |
| pi | at the head of the instruction (its one-shot mode has no system-prompt flag) |

Host runs read the file directly. Terminal-Bench containers have no bind mount, so
`CommandAgent` uploads it to `/opt/agent/agent.md` and the argv template names that
path. Harbor's own pi and claude-code agents build their own prompt internally, so on
Terminal-Bench those two get the text from harbor and not from this file -- SWE-bench
is where all three are guaranteed identical.

Edit that file, and all three agents change together. Nothing else in the harness
hard-codes a preamble.

---

## 5. Datasets

| Suite | Dataset id | Executed by |
| --- | --- | --- |
| `terminal-bench` 2.0 | `terminal-bench@2.0` -- what the harbor registry publishes (`harbor datasets list --legacy`) | `harbor run -d <dataset>@<version>` |
| `terminal-bench` 2.1 | the local `external_repos/terminal-bench-2-1/` checkout (the registry has no `@2.1` yet) | `harbor run -p external_repos/terminal-bench-2-1`, via `options.dataset_dir` in `benchmarks/configs/terminal_bench_21.json` |
| `terminal-bench` (`options.harness: "tb"`) | `terminal-bench-core==0.1.1` | `tb run --dataset ...` (Terminal-Bench 1.x) |
| `swebench` | `SWE-bench/SWE-bench_Verified`, split `test` | `python -m swebench.harness.run_evaluation` |

Claude Code and Pi are harbor's own agents for Terminal-Bench; mini-code goes through the
installed-agent bridge in `benchmarks/harbor_agents.py` (harbor needs the repository on
`PYTHONPATH` for that, which the suite sets for you).

---

## 6. Serving the model

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

## 7. What gets produced

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

## 8. Reading the numbers back

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

## 9. Results on the H200 used to validate this

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
| terminal-bench | mini-code | default | 3 | 0 | 1 | `python3 -m mini_code_cli` is not in the image: 2 tasks died with `NonZeroAgentExitCodeError` (`No module named mini_code_cli`), 1 with `AgentAuthenticationError` |
| terminal-bench | claude-code | bypass-permissions | 3 | 0 | 1 | installs and runs in the container; one task hit `AgentAuthenticationError` against the local endpoint (fixed later, see above) |
| terminal-bench | pi | default | 3 | 0 | 1 | same failure class as mini-code: the CLI is not in the image |
| **ALL** | | | **9** | **0** | **3** | |

Three tasks each (`adaptive-rejection-sampler`, `bn-fit-modify`,
`break-filter-js-from-html`), 9 agent runs, every container built and torn down by
harbor through Apptainer. **Re-read of the same artifacts:** two of those rows are
not scores at all — `mini-code` and `pi` were never *in* the images, so their runs
died with `No module named mini_code_cli` / a missing `pi` binary, and the earlier
"three tasks completed" note here was wrong. Only the claude-code row was a real
attempt. The lesson is the one this section keeps repeating: an exception is not a
zero, and the per-trial `exception.txt` — not the summary count — is what says so.
For a TB task image the fix is to install the agent's code into the image (the
Dockerfile path in §8 can bake it in); harbor's TB path has no bind mount for host
code, unlike the SWE-bench suite's `/opt/agent` mount.

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

### Measured with the unified entry point (last run)

Same model and host, `--mode short_eval` on three SWE-bench instances
(`astropy__astropy-12907`, `13033`, `13236`), run id `swe-short-eval4`:

| agent | resolved | instances | note |
| --- | --- | --- | --- |
| mini-code | 2/3 | 12907, 13033 | 13236 applies cleanly but fails its tests |
| pi | 2/3 | 12907, 13236 | 13033 applies cleanly but fails its tests |
| claude-code | 2/3 awarded | 12907, 13033 | third patch applies cleanly; grading for it aborted when the suite exited 1 before the report was written |

All nine trials produced a non-empty `model.patch` and every patch applied, so
these are real attempts, not missing-agent failures -- the earlier 0/9 rows were
the agent never existing inside the image. Two bugs were behind the previous
1/3: the SWE path never wrote `/opt/agent/agent.md` (so every mini-code trial
died on `The AGENT.md ... does not exist`) and the endpoint had been started
without `--tool-call-parser qwen3_xml`, which turned every tool call into plain
prose and ended the run after one turn.

**Terminal-Bench 2.1 is still not scored.** The trials start the container, the
in-container server comes up and then the environment times out. The last line
of `trial.log` was `[harbor] Installing tmux...`: `bootstrap.sh` ran an
unbounded `apt-get install` on an image whose apt lists were cleared and whose
writable overlay is 64 MiB, before `exec`ing the server. That call is now
bounded (`timeout`), the failure is reported instead of swallowed, and
`HARBOR_SKIP_TOOL_INSTALL=1` skips it entirely -- a session without tmux already
degrades to a plain shell. With the fix the start phase gets through
(`tmux unavailable; continuing without it`, then `Uvicorn running`), so the
remaining blocker is the first call harbor makes *after* the server is up, which
is where the reruns still park.

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

## 10. Safety and habits

* **`plan` first.** It prints the exact argv (never a shell string), the environment variable
  *names* it sets and the container calls, without downloading a dataset or starting a container.
* **Agent execution is refused outside a container** unless `--allow-host` is passed: several
  settings (`--sudo`, `bypassPermissions`) are only safe in a disposable environment.
* **Nothing secret is written.** Logs, results and summaries pass through a redactor that keeps
  environment variable *names* and replaces values; only `api_key_env` names ever enter a config.
* **Start small.** `--limit 1` on `terminal-bench` is the cheapest end-to-end check; harbor is
  young and moves fast, so re-read `harbor run --help` when a flag stops being accepted.

---

## 11. Apptainer without Docker (Docker-free path)

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

| Check | Result | Cost, measured |
| --- | --- | --- |
| `harbor run -p <Dockerfile-only task> -a oracle --env singularity` | **reward 1.0**, 9/9 verifier tests | 74 s end to end, 257 MB `.sif` |
| `-a claude-code` in a pulled task image | installs, authenticates, streams turns | Node tarball ~50 MB, CLI ~15 s |
| `apptainer pull docker://swebench/sweb.eval.x86_64.<id>` | works — the images are public OCI | **1.09 GB, ~20 s** from a warm cache |
| repeated task image | reused from `BENCH_SIF_CACHE` | 0.1 s (hash of the *definition*) |

Requirements and limits of this path:

* `apptainer build --fakeroot` must be permitted on the machine (it is here:
  `unshare -Ur` works). Rootless podman is *not* an alternative — it needs
  `/etc/subuid` ranges this account does not have.
* budgets: `cpus`/`memory`/`pids_limit` are advisory under Apptainer (printed as
  notes by `plan`), and containers share the host network namespace.
* the writable layer is **64 MiB**, and it is thrown away when `apptainer exec`
  exits. Two consequences worth knowing before debugging an agent: `apt-get install`
  of anything with a `dpkg` postinst fails with `Invalid cross-device link` (dpkg
  renames a directory that lives in the read-only lower layer), and an agent's edits
  do not survive into the next command unless they land on a bind mount. That is why
  Node is installed from the static tarball, agent tools live in a bind-mounted
  `/opt/agent`, and SWE-bench copies `/testbed` out to the host (§8).
* **a pulled image that ships no `/usr/bin/python3` pays for one at start-up.**
  harbor's in-container server needs an interpreter, so its bootstrap runs
  `apt-get install -y python3` — inside the 64 MiB overlay, under fakeroot, which
  took over ten minutes on `alexgshaw/gpt2-codegolf:20251031` and blew the 600 s
  environment-start budget with `EnvironmentStartTimeoutError` on the first attempt
  (a later attempt, with the apt lists warm, got through). Images that already
  carry python3 (`bn-fit-modify`, `cancel-async-tasks`) start in seconds. Bake the
  interpreter in at build time -- the Dockerfile path in §8 does exactly that --
  rather than raising the timeout.
* SWE-bench grading under Apptainer is §11, not this section: `tb_apptainer.py`
  derives images for Terminal-Bench only.

---

## 12. Our forks, and when they are used

Two behaviours cannot be expressed from outside the tools, so we carry branches of
them. Harbor is installed from its branch by `benchmarks/requirements.txt`; the
task set is cloned into `external_repos/` by
`benchmarks/scripts/setup_external_repos.sh` (git-ignored — the branches are the
source of truth, see §13).

| checkout | fork | branch | base |
| --- | --- | --- | --- |
| `external_repos/harbor/` | `ai-nikolai/harbor` | `mini-code/apptainer` | tag `v0.24.0` (`bf8996ef6d97`) |
| `SWE-bench/` | `ai-nikolai/SWE-bench` | `mini-code/apptainer` | `02e7a74ffd0b` (`swebench.__version__ == "5.0.2"`; upstream has no `v5.0.2` tag) |
| `external_repos/terminal-bench-2-1/` | `ai-nikolai/terminal-bench-2-1` | `mini-code/bench` | `69671fbaac6d` (the commit harbor's `registry.json` pins for `terminal-bench@2.0`) |

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
already-installed dependencies are not churned). Harbor is the one that matters
for Terminal-Bench; the SWE-bench fork is only needed for the `--backend
apptainer` route in §11 — the default SWE-bench path (`§11.1`) is the in-repo
`benchmarks/suites/swebench_apptainer.py` and needs no fork:

```bash
~/miniconda3/envs/hb/bin/python -m pip install -e ./harbor --no-deps   # or: pip install -r benchmarks/requirements.txt
python -c "import harbor; print(harbor.__file__)"
# only for the fork-based SWE-bench backend:
python -m pip install -e ./SWE-bench --no-deps && python -c "import swebench; print(swebench.__file__)"
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

## 13. Where the harness forks come from, and how to develop or publish them

A run needs two repositories that are not this one:

| repository | branch | what it is | how you get it |
| --- | --- | --- | --- |
| [`ai-nikolai/harbor`](https://github.com/ai-nikolai/harbor) | `mini-code/apptainer` | the Terminal-Bench 2.x harness, with the Docker-free (Apptainer) backend | `pip install -r benchmarks/requirements.txt` — the requirement pins the branch: `harbor @ git+https://github.com/ai-nikolai/harbor.git@mini-code/apptainer` |
| [`ai-nikolai/terminal-bench-2-1`](https://github.com/ai-nikolai/terminal-bench-2-1) | `mini-code/bench` | the 2.1 task set (data: 89 task directories, no packaging metadata) | **clone, not pip**: `benchmarks/scripts/setup_external_repos.sh` puts it at `external_repos/terminal-bench-2-1`, the path the 2.1 config's `dataset_dir` names (§13.1) |

**There is no patch step.** Both branches are pushed, so pip and git install the
exact code a run uses, and `manifest.json` records the branch, SHA and dirty flag
of every checkout that took part — a result cannot silently be reproduced against
different code. (The old workflow exported `benchmarks/patches/*.patch` and
re-applied them to a fresh upstream clone; that only existed because the branches
were not published yet.)

### 13.1 Local development: `benchmarks/scripts/setup_external_repos.sh`

**Running Terminal-Bench 2.1 needs this step; nothing else does.** The task set
is 89 task directories with no `setup.py` and no `pyproject.toml`, so pip cannot
install it (`pip install "terminal-bench @ git+…@mini-code/bench"` → *does not
appear to be a Python project*), and `harbor run -p` needs the directory on disk:

```bash
benchmarks/scripts/setup_external_repos.sh
```

creates (git-ignored, at the repository root)

```
external_repos/harbor/                @ mini-code/apptainer   # to read/change the harness
external_repos/terminal-bench-2-1/    @ mini-code/bench       # dataset_dir of the 2.1 config
```

`--with-main` adds `external_repos/mini-code-cli/` (`feat/benchmarks`). Re-running
is safe: an existing clone is fetched, and a checkout with uncommitted changes is
reported instead of being switched. Knobs: `EXTERNAL_REPOS_DIR`,
`EXTERNAL_REPOS_OWNER`; `--help` lists them.

`benchmarks/configs/terminal_bench_21.json` points at
`external_repos/terminal-bench-2-1`, resolved **against the current working
directory** — so run the entry point from the repository root, or pass an absolute
path:

```bash
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent mini-code --mode short
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent mini-code --mode short \
    --set options.dataset_dir=/abs/path/to/terminal-bench-2-1
```

### 13.2 Publishing: `benchmarks/scripts/push_all_repos.sh`

A benchmark change usually spans all three histories, so there is one command to
publish them:

```bash
benchmarks/scripts/push_all_repos.sh --dry-run   # print the commands, push nothing
benchmarks/scripts/push_all_repos.sh             # push
```

It finds the forks in `external_repos/` first and then in the sibling checkouts
(`external_repos/<name>`, else a `<name>` checkout next to this repository),
refuses to push a checkout that is on the wrong
branch or has uncommitted changes, and pushes:

```bash
git -C .                 push            -u origin feat/benchmarks
git -C <harbor>          push --force-with-lease -u origin mini-code/apptainer
git -C <terminal-bench>  push --force-with-lease -u origin mini-code/bench
```

`--force-with-lease` is deliberate on the two forks: their histories were rewritten
once (a secret had to be removed from a test fixture in `terminal-bench-2-1`), so a
plain push would be rejected — and the lease means it still refuses if someone else
pushed in the meantime.

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
