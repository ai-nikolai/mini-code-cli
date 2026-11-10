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
| Python ≥ 3.12 | the harness, `harbor`, and mini-code itself all run on one interpreter | — |
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

**One environment is enough.** `harbor` requires Python ≥ 3.12 and mini-code supports
3.10+, so a single 3.12 environment holds the CLI, the harness and the dependencies:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -e .                              # mini-code itself
python -m pip install -r benchmarks/requirements.txt    # harbor, swebench, vllm, litellm
python benchmarks/run_benchmarks.py --dataset swe-bench --agent mini-code --mode short_eval
```

Nothing in the harness needs mini-code to be installed anywhere else, and nothing in
mini-code needs harbor: the split only ever saved a few hundred MB when the two were
built for different interpreters. Separate environments still work (mini-code on 3.10
or 3.11 with harbor elsewhere), you just have to run the entry point with the
interpreter that has harbor.

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

### Running on a Slurm allocation, one agent at a time

Concurrent campaigns compete for the same model server; the first attempt at
this produced 3.7 h trials that were all timeouts, not model measurements. Use
the wrapper — it starts vLLM on the allocation (with the vLLM env's `bin` on
`PATH`, so flashinfer's JIT finds `ninja`), waits for `/health`, then runs one
campaign per agent:

```bash
benchmarks/scripts/run_benchmark_slurm.sh
```

Every knob is a variable at the top of the script (`JOBID`, `DATASET`,
`AGENTS="mini-code pi claude-code"`, `MODE`, `LIMIT`, `RUN_PREFIX`,
`START_SERVER`, `STOP_SERVER`, ...), and it can be overridden in the
environment. It writes `results/<RUN_PREFIX>-<agent>/summary_results.csv`
and a per-agent log under `/tmp/opencode/bench-slurm/`. It never edits the
checkout and never pushes.

### Full matrix with `sbatch` (1–4 GPUs, different models)

`benchmarks/scripts/submit_benchmark_slurm.sh` generates a batch script under
`experiments/slurm/` (git-ignored), regenerates the model configs under
`experiments/configs/`, and submits it. Every knob is at the top of the file:

```bash
NUM_GPUS=4                 # 1..4 -> model tensor-parallel size
BENCH_GPUS=0               # GPUs reserved for the runner (0 is normal here)
MODEL_NAME="ornith-ai/Ornith-1.5-35B-A3B"
SERVER_BACKEND="vllm"      # or sglang
MODEL_ENV="env_vllm30"     # conda env with the server
BENCH_ENV="hb"             # conda env with harbor + swebench
BENCHMARKS="terminalbench-2.1 swe-bench-lite-full"
AGENTS="mini-code pi claude-code"
MODE="full"                # full | short_eval | short
FULL=1                     # drop the checked-in smoke limits/task_ids
SUBMIT=1                   # 1 = sbatch; 0 = only write the file
```

`DRY_RUN=1 SUBMIT=0` writes everything without submitting. The generated job
starts the server on the first `NUM_GPUS - BENCH_GPUS` GPUs, waits for
`/health`, then runs every benchmark × agent **sequentially** (one agent at a
time) and writes `results/<RUN_PREFIX>_<benchmark>_<agent>/summary_results.csv`.
The model config is rewritten by `benchmarks/scripts/make_benchmark_config.py`, so changing
`MODEL_NAME` is enough.

The two "full" sets are large: the local Terminal-Bench 2.1 checkout has **90
tasks**, and SWE-bench Lite has **300 instances**. From the measured campaigns
(one H200, serial, `n_concurrent=1`):

| 1 agent / 1 model / 1 benchmark | Terminal-Bench 2.1 (90) | SWE-bench Lite (300) |
|---|---|---|
| mini-code | ~5–10 h | ~10–20 h |
| Pi | ~15–30 h | ~12–25 h |
| Claude Code | ~12–25 h | ~15–30 h |

So a full 3-agent × 2-benchmark matrix is roughly **3–6 days** serial. GPU
count mainly buys model throughput (and lets a larger checkpoint fit); the
agents are tool/container-bound. For a shorter result, run one benchmark or a
`LIMIT` subset, or submit one job per agent on separate allocations (do not
share a model server between jobs unless you accept contention).

**Storage is the other constraint.** The measured SIFs average ~0.33 GB per
TB task and ~1.28 GB per SWE-bench Lite instance:

| Artefact | Full TB 2.1 (90) | Full Lite (300) |
|---|---|---|
| persistent SIFs | ~30–90 GB | ~300–600 GB (≈385 GB at the measured mean) |
| Apptainer cache/tmp | several GB | ~200–400 GB unless pruned |
| results / testbeds / logs | ~2–10 GB | ~20–60 GB |

The full pair therefore needs roughly **0.6–1.0 TB** of scratch before model
weights. Point the launcher at real scratch storage, not a 1 TB home:

```bash
SIF_CACHE=/scratch/$USER/sif TB_CACHE=/scratch/$USER/cache \
SCRATCH_DIR=/scratch/$USER/tmp APPTAINER_CACHEDIR=/scratch/$USER/apcache \
APPTAINER_TMPDIR=/scratch/$USER/aptmp \
  benchmarks/scripts/submit_benchmark_matrix.sh
```

If only home is available, run one benchmark at a time, prune SIFs/testbeds
between chunks, and expect the full Lite split to be close to the quota.

### Prepare, run, prune: one benchmark at a time

For a 1 TB home, prebuild one benchmark's images, run it, then free that space
before preparing the other. The stream is:

```bash
# 1. Terminal-Bench images + staged task set; verify every SIF
benchmarks/scripts/prepare_tb_sifs.sh
#    VERIFY=test does `apptainer test` on each (slower); VERIFY=none skips.

# 2. run Terminal-Bench (the launcher detects experiments/tb-staged/prepared.json)
BENCHMARKS=terminalbench-2.1 MODE=full benchmarks/scripts/submit_benchmark_slurm.sh

# 3. keep summaries/patches/logs, drop TB SIFs and heavy trial workspaces
DRY_RUN=0 CONFIRM=yes MODE=partial BENCH=tb \
    benchmarks/scripts/prune_benchmark_storage.sh

# 4. SWE-bench instances + SIFs
benchmarks/scripts/prepare_swe_sifs.sh
#    SWE_DATASET=SWE-bench/SWE-bench_Verified for the 500-instance set.

# 5. run SWE-bench
BENCHMARKS=swe-bench-lite-full MODE=full benchmarks/scripts/submit_benchmark_slurm.sh

# 6. after it finishes, free the SWE SIFs too
DRY_RUN=0 CONFIRM=yes MODE=partial BENCH=swe \
    benchmarks/scripts/prune_benchmark_storage.sh
```

`benchmarks/scripts/prune_benchmark_storage.sh` is **dry-run by default**. Degrees:

| `MODE` | Removes | Keeps |
|---|---|---|
| `light` | testbeds, `agent-tools`, `eval`/`pulls`, `.scratch` | SIFs, all results/logs |
| `sifs` | SIFs for `BENCH=tb\|swe\|all` | results/logs |
| `scratch` | `SCRATCH_DIR` / `APPTAINER_CACHEDIR` contents | everything else |
| `partial` | `light` + selected SIFs | summaries, patches, `results.jsonl`, `trial.json`, logs |
| `full` | `partial` + `results/`, `benchmarks/runs/`, generated jobs/configs/logs, scratch | the checkout |

Do not prune while a campaign is running: `light`/`partial` touch active trial
workspaces.

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

### mini-code strategy presets

The benchmark configs list several mini-code settings. The `swe` setting is the
baseline used for real runs; the four strategy presets change exactly one
compaction or planning choice on top of it:

| Setting | Extra choice |
| --- | --- |
| `swe` | baseline: `summarise` compaction, no plan mode |
| `swe-compaction-predict-files` | `--compaction-strategy predict-files` |
| `swe-compaction-predict-messages` | `--compaction-strategy predict-messages` |
| `swe-plan` | `--plan` (default `plan-verify-step`) |
| `swe-plan-verify` | `--plan --plan-strategy plan-verify` |
| `swe-open-plan` | `--plan --plan-strategy open-plan` |

All five are present in `terminal_bench_21.json`, `ornith_30b.json` and
`mini_code_settings.json`. The Slurm launchers default to the `swe` baseline only;
select more with a space-separated `MINI_CODE_SETTINGS`:

```bash
MINI_CODE_SETTINGS="swe swe-compaction-predict-files swe-compaction-predict-messages" \
    benchmarks/scripts/submit_benchmark_slurm.sh

MINI_CODE_SETTINGS=all BENCHMARKS=terminalbench-2.1-3 \
    benchmarks/scripts/run_benchmark_slurm.sh
```

`run_benchmarks.py` exposes the same selection as repeatable `--setting` flags:

```bash
python benchmarks/run_benchmarks.py --dataset swe-bench --agent mini-code \
    --setting swe-open-plan --mode short_eval
```

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
| `swebench` | `SWE-bench/SWE-bench_Verified` (500 instances), split `test` | `python -m swebench.harness.run_evaluation` |
| `swebench` | `SWE-bench/SWE-bench_Lite` (300 instances), split `test` -- `--dataset swe-bench-lite` | same harness, same images |

Claude Code and Pi are harbor's own agents for Terminal-Bench; mini-code goes through the
installed-agent bridge in `benchmarks/harbor_agents.py` (harbor needs the repository on
`PYTHONPATH` for that, which the suite sets for you).

**Which SWE-bench splits are available?** *Verified* and *Lite* -- both today, with
nothing extra to install or clone:

```bash
python benchmarks/run_benchmarks.py --dataset swe-bench       --agent all --mode short_eval --limit 5
python benchmarks/run_benchmarks.py --dataset swe-bench-lite  --agent all --mode short_eval --limit 5
```

Lite is the *same* `swebench` packaging, the same grading path and the same published
instance images (`swebench/sweb.eval.x86_64.<id>`, which the dataset rows name
themselves); it is only a different HuggingFace dataset -- 300 instances instead of
500 -- so the whole difference is the `dataset` field, which
`campaign.DATASETS["swe-bench-lite"]` overrides:

```json
{"dataset": "SWE-bench/SWE-bench_Lite", "split": "test"}
```

Lite's rows are ordered the same way as Verified's, so `--limit 5` alone would score
the same five instances in both campaigns. `--dataset swe-bench-lite` therefore pins a
different five (`campaign.LITE_INSTANCES`: `django__django-11099`,
`sympy__sympy-20590`, `matplotlib__matplotlib-26020`, `pytest-dev__pytest-11148`,
`scikit-learn__scikit-learn-13496`) through the suite's `task_ids`; the same mechanism
takes `options.instance_ids` / `options.skip_ids` for an ad-hoc sample.

No forks, no extra repositories, no additional containers: `--dataset swe-bench-lite`
downloads the instance list through `datasets` (an existing requirement) and pulls the
same images the Verified runs use. **Measured, not assumed:** `swe-lite-5f` ran both
splits' full path on this Docker-less host and graded 12/15 (§9) -- Lite is *proven*,
not merely "should work". The 300 Lite instances overlap Verified (Lite is a
subset of the public train/test pool, Verified is human-adjudicated), so the two
campaigns share SIFs and are cheap to run back to back.

Running the *full* (Unverified) `SWE-bench/SWE-bench` split works the same way: point
the suite's `dataset` at it. The entry point takes no per-option override (there is no
`--set`), so that means a one-line config -- copy
`benchmarks/configs/ornith_30b.json`, change the `swebench` suite's `dataset`, and pass
it with `python -m benchmarks run -c <config> --suite swebench`.

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
* `errors`/`timeouts` are harness failures, not wrong answers. `rate` is
  `resolved / (resolved + unresolved)`; the error/timeout counts are in their own columns, and
  `summary_results.csv` appends them after `status`. A pure error cell has `rate = -`, not 0%.
* `status` is `ok` only when every selected instance produced a verdict; otherwise it is
  `no_results`, `partial_failures` or `all_failed`. `run`/`campaign` exit non-zero when a cell
  is not `ok`, so a shell script cannot mistake a broken run for a score.
* Runs are resumable, but `DONE` means "a verdict exists": a trial that failed a command or
  collected no outcomes is retried, and a `DONE` marker is only reused when the trial's
  content identity (dataset + sample + agent/model + git SHA) still matches. `--no-resume`
  starts the trial directory clean.
* `--agent` is run **one agent at a time**; concurrent campaigns share one model server and
  turn contention into timeouts. The Slurm wrapper above encodes this.

---

## 8. Reading the numbers back

```bash
python -m benchmarks analyze benchmarks/results/ornith-30b-smoke.csv
python -m benchmarks analyze benchmarks/runs/ornith-30b-smoke            # same, straight from the run
python -m benchmarks analyze results/*.csv --md report.md --json report.json --csv combined.csv
```

The analyser reads a CSV, a `summary.json` or a whole run directory, merges rows that share
`(dataset, suite, agent, setting, model)` (so several runs of the same configuration add up,
while Verified and Lite never collapse into one cell), and prints:

```
suite           agent        setting             model                    n  ok  rate  err  to  cmdfail  mean s
--------------  -----------  ------------------  -----------------------  -  --  ----  ---  --  -------  ------
swebench        claude-code  bypass-permissions  ornith-35b-a3b-anthropic 0  0   -     0    0   1        -
terminal-bench  mini-code    default             ornith-35b-a3b           3  0   0%    3    0   0        -
ALL                                                                       9  0   0%    9    0   3        0
```

`ok` = resolved instances, `rate` = `ok/(ok + unresolved)` (errors/timeouts are not wrong
answers), `to` = timeouts, `cmdfail` = commands that did not finish, `mean s` = mean wall-clock
per outcome that reported one (pooled, not a mean of means; `-` when none did). `--md`,
`--json` and `--csv` write the same table for a README, a script or a spreadsheet.

---

## 8.2 Per-instance metrics (`metrics.csv`)

Every run writes `<run>/metrics.csv` (one row per task/instance) and the
campaign merges the agent runs into `results/<campaign>/metrics.csv`. The
columns are stable:

```
dataset, suite, agent, setting, model, run_id, instance_id, outcome, resolved,
duration_seconds, prompt_tokens, completion_tokens, total_tokens, llm_calls,
tool_calls, turns, tool_names, cost_usd, detail
```

* `duration_seconds` — the agent execution time (harness timestamps or the
  command's `[exit … in Xs]` log line).
* `prompt_tokens` / `completion_tokens` / `total_tokens` — from the agent's own
  usage report. Empty when the agent/server does not report it; never 0 as a
  stand-in.
* `llm_calls` / `turns` — model turns (mini-code reports LLM calls; Claude Code
  reports `num_turns`; Pi may leave these empty).
* `tool_calls` / `tool_names` — number/names of tool invocations where the agent
  logs them (mini-code, Claude Code tool-use events).
* `cost_usd` — when the agent reports it (Claude Code).
* `outcome`/`resolved`/`detail` — the same verdict and reason as `results.jsonl`.

The aggregate `summary_results.csv` appends the corresponding means:
`mean_prompt_tokens`, `mean_completion_tokens`, `mean_llm_calls`,
`mean_tool_calls`, `mean_turns`, `mean_cost_usd`, and a `tool_names` column.

## 8.5 What the harness guarantees now

* **One outcome per task.** Harbor's per-attempt suffix (`task__AbC123`) is
  stripped, a reward and an exception for the same task are merged, and
  `resolved > error > timeout > unresolved` is applied across retries.
* **Errors are not zeros.** `summary_results.csv` carries `errors` and
  `timeouts`; `pass_rate` excludes them; `status` is `no_results` /
  `partial_failures` / `all_failed` when they occur; the CLI exits non-zero.
* **No empty patches by design.** SWE-bench seeds outcomes from the prepared
  instance list, marks a 0-byte `model.patch` `unresolved` with reason
  `empty patch`, and a missing report is an `error`, not `n = 0`.
* **Docker SWE-bench keeps edits.** The agent and patch-capture commands share
  a host working copy bind-mounted at `/testbed`, for both Docker and Apptainer.
* **Keys travel by value only where needed.** `{env:NAME}` is expanded in argv
  at execution time (and the rendered log line is redacted); the manifest
  records only `env` key names. Redaction no longer confuses `tokens : 2,741`
  with a credential, and covers `Authorization` / `x-api-key` headers.
  *Caveat:* harbor writes its own `trial.log`/agent transcripts directly and
  those are **not** passed through the redactor; treat a run directory that
  used real credentials as sensitive.
* **Timeouts are bounded.** A harness command gets
  `max(configured, instance_seconds × tasks)`; the runner streams output, kills
  the process group on timeout, and records a `timeout` outcome when nothing
  was collected.
* **Resume is content-addressed.** `DONE` is written only for a verdict, and it
  is reused only when `trial.identity()` (dataset + sample + agent/model + git
  SHA) matches; `--no-resume` clears the trial directory first.
* **Caches are labelled.** Derived TB SIFs carry a `.key` hash of the source
  image + definition + baked `mini_code_cli`; SWE SIFs carry a `.complete`
  marker written only after a successful pull.
* **Local endpoints are health-gated.** A run against a `127.0.0.1` /
  `localhost` OpenAI endpoint probes `/health` for 30 s before starting;
  `BENCH_SKIP_HEALTH=1` opts out.

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

**Terminal-Bench 2.1 is now scored end to end.** One cached task, one agent
(`--dataset terminalbench-2.1-bn`, `bn-fit-modify`, `mini-code`), run `tb21-bn3`:

| what | result |
| --- | --- |
| environment start | ok, ~2 min (SIF cached; a *rebuilt* SIF carries `/opt/harbor-server`) |
| agent execution | 3.5 min, `python3 -m mini_code_cli` inside the task image, exit 0 |
| verifier | pytest ran for 3 min: **9 tests, 0 passed, 9 failed** (`verifier/ctrf.json`) |
| reward | **0.0**, `exception_info: null` — a real score, not a crash |
| campaign CSV | `terminalbench-2.1-bn,mini-code,short,1,0,0.0,0.0,,,tb21-bn3/mini-code,ok` |

The zero is honest: the agent left the task's five output CSVs unwritten
(`test_bn_sample_exists: FileNotFoundError`, ...), i.e. on this task it did not
manage the work in the time it used -- the opposite failure mode from the earlier
"everything is an exception" runs. Two further fixes were needed to get here: the
suite now forwards harbor's `--agent-timeout-multiplier` (a 2.1 task's own 900 s
budget expired mid-work) and pins the requested tasks with `-i` (harbor otherwise
takes the dataset directory's own first tasks: asking for `bn-fit-modify` ran
`adaptive-rejection-sampler`).

The two blockers behind the old "parks at `Uvicorn running`" note were
fixed in this session. First the start phase: `bootstrap.sh` ran an
unbounded `apt-get install` on an image whose apt lists were cleared and whose
writable overlay is 64 MiB, before `exec`ing the server -- that call is bounded now
and a session without tmux degrades to a plain shell (`tmux unavailable; continuing
without it`). Then the "parks at `Uvicorn running`" symptom this section used to
blame on the post-start call: it was harbor's health loop, a counted
`for i in range(60)` + `sleep(1)` -- a fixed one-minute window that expires while the
container is still bootstrapping (2-4 min here), after which harbor kills the
container and reports `server failed to start`. With a wall-clock deadline
(§11, "Four Apptainer-only failures") the trial proceeds through agent setup, runs
`python3 -m mini_code_cli` inside the task image, and the verifier executes:
`trial.log` shows the agent's own argv, `agent_execution` has real timestamps and
`verifier_result.rewards.reward` is written (0.0 on the trial below).

What is left is scale: a *budget* question, not a plumbing one. On
`adaptive-rejection-sampler` the agent was still working when harbor's
`AgentTimeoutError` fired at the task's 900 s budget (raised 6x to 5400 s via
`agent_timeout_multiplier`); `bn-fit-modify` scored inside 7 minutes but scored
zero. A five-task, three-agent Terminal-Bench campaign therefore costs hours and
will not flatter a 30B model, which is why the campaign numbers in §9 are
SWE-bench; the path itself is proven, task by task.

### Terminal-Bench 2.1, three agents (`tb21-3j-mini`, `tb21-3j-pi`, `tb21-3j-cc2`)

Same model and server, `--mode short --limit 3`, the three cached 2.1 tasks
(`adaptive-rejection-sampler`, `break-filter-js-from-html`, `bn-fit-modify`):

| agent | resolved | tasks resolved | note |
| --- | --- | --- | --- |
| mini-code | 0/3 | -- | R installed but unusable (`libblas.so.3` missing); XSS bypass failed; bn outputs missing |
| pi | 2/3 | bn-fit-modify, break-filter-js-from-html | adaptive-rejection: no tool call, 65 536 output tokens spent on reasoning |
| claude-code | 2/3 | bn-fit-modify, break-filter-js-from-html | adaptive-rejection: 7/9 verifier tests pass |
| **all** | **4/9 (44%)** | | no exceptions, no timeouts |

All nine trials produced a real graded reward (`0.0`/`1.0`) with
`exception_info: null`; the zeros are attempts, not crashes. `mini-code` is the
cheapest per task (~4.5 min), `pi` and `claude-code` take 15-20 min each, and on
`adaptive-rejection-sampler` all three hinge on getting a working R into the
container: `claude-code` does (7/9), `mini-code`'s install cannot load
`libblas.so.3`, and `pi` never reaches a tool call.

Two harness bugs were found and fixed to get a real `claude-code` row:

* `benchmarks/agents/base.py`: harbor's factory already resolves the config
  `env` block and passes it to the agent constructor as `extra_env`; supplying
  it again through `--agent-kwarg` was a duplicate keyword, so no trial ran.
* `benchmarks/runner.py`: a command's `{env:NAME}` pass-through blanked a
  literal the config `env` block supplied, so Claude Code started with
  `apiKeySource: none` and every trial died with "Not logged in".

### Five data points x three agents, SWE-bench Verified (`swe-verified-5c`)

The run this section exists for: the same fixed model, `--mode short_eval`,
`--limit 5`, all three agents, every patch graded through Apptainer:

```bash
BENCHMARK_SCRATCH=/hx2-weka/home/nr1713/scratch/bench \
  python benchmarks/run_benchmarks.py --dataset swe-bench --agent all \
      --mode short_eval --limit 5 --run-id swe-verified-5c
```

| agent | resolved | instances resolved | instances not resolved | grading errors |
| --- | --- | --- | --- | --- |
| mini-code | **2/5** | 12907, 13453 | 13033, 13236, 13398 | 0 |
| pi | **4/5** | 12907, 13236, 13398, 13453 | 13033 | 0 |
| claude-code | **4/5** | 12907, 13236, 13398, 13453 | 13033 | 0 |
| **all** | **10/15 (66.7%)** | | | |

All fifteen trials produced a non-empty `model.patch`, all fifteen were *graded*
(`resolved_ids` / `unresolved_ids` in `swebench-*.json`, no `error_ids`), and the
CSV above is `results/swe-verified-5c/summary_results.csv`. `astropy__astropy-13033`
is the one instance no agent solves -- its FAIL_TO_PASS tests stay red even when the
patch applies. Wall clock: 14:11 -> 15:24 for the whole campaign (agent minutes are
dominated by `claude-code` on 13398: 647 s, and by `mini-code` on 13398: 350 s);
grading is ~285 s per agent group of five.

Reading the same run for *cost*: `mini-code` is the slowest agent per instance
(122/72/228/350/41 s) because it makes many small tool calls (14-41 LLM calls per
trial), while `pi` and `claude-code` solve the same instances in fewer, larger
turns. None of the three needed a reprompt, a retry or a restart, which is what the
four Apptainer fixes in §11 bought.

### The same campaign on SWE-bench Lite (`swe-lite-5f`)

```bash
BENCHMARK_SCRATCH=/hx2-weka/home/nr1713/scratch/bench \
  python benchmarks/run_benchmarks.py --dataset swe-bench-lite --agent all \
      --mode short_eval --limit 5 --run-id swe-lite-5f
```

Same model, same three agents, same grading path -- Lite needs *nothing* new (§5):
five more published instance images are pulled (`swe-django__django-11099.sif`,
`swe-sympy__sympy-20590.sif`, `swe-matplotlib__matplotlib-26020.sif`,
`swe-pytest-dev__pytest-11148.sif`, `swe-scikit-learn__scikit-learn-13496.sif`) and
graded with the same `swebench.harness.grading`.

| agent | resolved | instances resolved | instances not resolved | grading errors |
| --- | --- | --- | --- | --- |
| mini-code | **3/5** | 11099, 13496, 20590 | matplotlib-26020, pytest-11148 | 0 |
| pi | **4/5** | 11099, 26020, 13496, 20590 | pytest-11148 | 0 |
| claude-code | **5/5** | all five | -- | 0 |
| **all** | **12/15 (80%)** | | | |

`results/swe-lite-5f/summary_results.csv`:

```
swe-bench-lite,mini-code,short_eval,5,3,0.6,0.6,,,swe-lite-5f/mini-code,ok
swe-bench-lite,pi,short_eval,5,4,0.8,0.8,,,swe-lite-5f/pi,ok
swe-bench-lite,claude-code,short_eval,5,5,1.0,1.0,,,swe-lite-5f/claude-code,ok
```

Wall clock 15:47 -> 17:20 for all fifteen trials plus three grading passes, with
all ten instance images already cached by then. First-time cost is the pulls, not the
harness, and it is uneven: django 11 min, sympy 9 min, pytest 9 min, matplotlib
**32 min** (~1.4 GB and a conda env unpack), scikit-learn 12 min -- budget an hour
before the first agent turn on a cold cache, and note that `--limit` does not shrink
it because every selected instance is pulled before any agent runs.

Three of these numbers are worth more than the total: `matplotlib__matplotlib-26020`
is the instance `pi` and `claude-code` solve and `mini-code` does not, `pytest-11148`
is the one both cheaper agents miss, and `claude-code` cleared Lite where it had
missed `astropy__astropy-13033` on Verified.

### What Lite actually needed to work (two first-run-only bugs)

Both were invisible on Verified because its images were already cached, and both
cost a whole Lite campaign before they were found:

* **the agent command referenced the registry, not the `.sif`.** The per-instance
  commands used `docker://swebench/sweb.eval.x86_64.<id>` whenever the `.sif` did not
  exist *at plan time* -- but the plan's own first step pulls exactly that file. So a
  brand-new instance set ran `apptainer exec docker://...` and died in 1.7 s with
  `requested access to the resource is denied` (the docker tag needs the dataset's
  `__` -> `_1776_` escaping, which the template cannot reproduce). Apptainer now
  always execs the cache path; run `swe-lite-5d` is the failure, `swe-lite-5f` the fix.
* **the host's site-packages leaked into every image.** `--cleanenv` does not stop
  apptainer from bind-mounting `$HOME`, so `~/.local/lib/python3.11/site-packages`
  landed on the container's `sys.path` and injected this host's *dev* matplotlib into
  every image (`Error processing line 1 of ...matplotlib-3.8.0.dev...-nspkg.pth` at
  the top of every agent log). `PYTHONNOUSERSITE=1` is now set for every apptainer
  exec.

### Terminal-Bench, one scored trial (`tb21-bn3`)

```bash
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1-bn \
    --agent mini-code --mode short --limit 1 --run-id tb21-bn3
```

`bn-fit-modify`, one agent, one cached `.sif`: environment 2 min, agent 3.5 min,
verifier 3 min (pytest, 9 tests), **reward 0.0**, `exception_info: null`, and
`results/tb21-bn3/summary_results.csv` written by the same collector the SWE-bench
campaigns use. The zero is a measurement, not a failure of the harness: the agent
never wrote the task's output CSVs, so nine `FileNotFoundError`s is exactly what
the verifier should say. `results/tb21-fix4` is the counter-example worth keeping --
same task family, `AgentTimeoutError` at 5400 s, no reward at all.

### Terminal-Bench 2.1: three agents x three tasks (`tb21-3c`, `tb21-3d`)

```bash
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 \
    --agent all --mode short --limit 3 --run-id tb21-3c
```

Three agents, the same three tasks, one shared `.sif` cache. The rewards are only
half the information here: the `exception_info` column is what says whether a zero
is a measurement or a harness fault.

| Agent | Task | Reward | `exception_info` | What it means |
| --- | --- | --- | --- | --- |
| mini-code | `break-filter-js-from-html` | **1.0** | `null` | solved, verifier green |
| mini-code | `bn-fit-modify` | 0.0 | `null` | ran to the verifier; measured zero (§9 `tb21-bn3` explains why) |
| mini-code | `adaptive-rejection-sampler` | 0.0 | `AgentTimeoutError` | budget, not a harness fault |
| pi | `bn-fit-modify`, `break-filter-js-from-html` | 0.0 | `null` | measured zeros |
| pi | `adaptive-rejection-sampler` | -- | `NonZeroAgentExitCodeError` | install died: `apt-get install curl` hit the client's 600 s cap (§11) |
| claude-code | `adaptive-rejection-sampler` | -- | `AgentAuthenticationError` | `Not logged in` -- a dummy `ANTHROPIC_API_KEY` is enough to *launch* Claude Code but not to get past its own check |

`tb21-3d` repeated the two failing agents after the image cache was repaired and
reproduced the same split (`pi`: `RuntimeError` in setup, i.e. the install again;
mini-code: `NonZeroAgentExitCodeError` on the same task), which is why the fix went
into the install path rather than into a task config -- see §11.

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
### Four Apptainer-only failures, and what they actually were

Each of these looked like a model failure in the summary CSV and was not one. They
are listed together because the pattern is the same: the harness reported a zero
while the error text named the real cause.

| Symptom | Real cause | Fix |
| --- | --- | --- |
| Terminal-Bench trial sits at `Uvicorn running` for ten minutes, then `server failed to start` | harbor's health loop was `for i in range(60)` + `sleep(1)` -- a *fixed one-minute window*. On a network filesystem the in-container bootstrap (venv probe, bounded tmux install, uvicorn) takes 2-4 min, and every fast connection refusal burned an iteration, so the window expired before the server was up | `external_repos/harbor` (commit "singularity: wait for the server with a wall-clock deadline"): wait on a deadline, `HARBOR_SINGULARITY_START_TIMEOUT` (default 900 s) |
| every SWE-bench instance reported `the model patch did not apply` | the graded `/testbed` comes from the read-only `.sif`, so `git apply` died with `Unable to create '/testbed/.git/index.lock': Read-only file system` | `benchmarks/suites/swebench_apptainer.py` stages the checkout on the host and binds it back over `/testbed`, the same trick the agent path uses |
| a campaign ends with `no report file` although every agent ran | the per-trial instance list is copied from `_prepare/` **without** its `instances.grading.jsonl` sidecar, so the grader had no `eval_script` (`KeyError: 'eval_script'`) | `benchmarks/runner.py` copies the sidecar alongside the list |
| `grading failed: No module named 'swebench'` | the grader is spawned with the *launcher's* interpreter; on this host `swebench` lives in the harbor env, not in the CLI's env | `_grader_python()` in `benchmarks/suites/swebench.py` probes for an interpreter that can `import swebench` (`options.grader_python`, `$BENCHMARK_GRADER_PYTHON`, then `~/miniconda3/envs/hb`) |

Two more turned up only on a **cold instance set** (the first Lite run), because
every Verified campaign had its `.sif` files already cached:

| Symptom | Real cause | Fix |
| --- | --- | --- |
| every Lite agent dies in ~2 s with `reading manifest latest in docker.io/swebench/sweb.eval.x86_64.<id>: requested access to the resource is denied` | the instance commands fell back to the `docker://` reference whenever the `.sif` was not on disk *at plan time* -- but the plan's first step is the pull that creates it, and the docker tag needs the dataset's `__` -> `_1776_` escaping | `swebench.py`: under apptainer the image is always the cache path; the pull step guarantees it exists |
| every agent log opens with `Error processing line 1 of .../matplotlib-3.8.0.dev...-nspkg.pth` | `--cleanenv` does not stop apptainer from bind-mounting `$HOME`, so the host's `~/.local/lib/python3.11/site-packages` is on the container's `sys.path` (a dev matplotlib, injected into a matplotlib benchmark image) | `benchmarks/runtime.py`: `PYTHONNOUSERSITE=1` for every apptainer exec |

Two of these were found by *reading the exception text* rather than the reward
column, and both live in the install path, which no SWE-bench campaign exercises:

| Symptom | Real cause | Fix |
| --- | --- | --- |
| `pi` fails setup with `Agent install failed: HTTP request timed out after 600 seconds`, and `pi`-side tasks never even start | the singularity client had a fixed `_DEFAULT_HTTP_TIMEOUT = 600` fallback for any `exec` that named no budget, and `ensure_system_dependencies` named none -- so `apt-get install curl` on a slow mirror was cut at ten minutes by the *client*, not by any task budget | `external_repos/harbor` (commit "singularity: do not cap an unlimited exec at 600 s"): the fallback is `HARBOR_SINGULARITY_EXEC_TIMEOUT` (default 3600 s), and the package-manager call passes `SYSTEM_PACKAGE_TIMEOUT_SEC` (1800 s) so a wedged apt is still killed inside the container |
| `which curl` inside a derived Terminal-Bench `.sif` prints a usage error and then `NO_CURL`, which reads like "curl is not baked in" | `apptainer exec <sif> which curl` mangles the argv (`which` sees an argument it rejects), so the probe fails even when curl is present -- `curl --version` and `ls /usr/bin/curl` both work | probe with `curl --version`, never `which`; §11's image build already put `curl` in `POST_INSTALL`, but a `.sif` built **before** that line existed does not have it, so it has to be rebuilt |

The second one is worth keeping precisely because the wrong diagnosis was cheap:
`pi` needs `curl` for its installer, `curl` is already in `tb_apptainer`'s
`POST_INSTALL` list, and the actual missing piece was that the cached image predated
that list. Rebuilding is not optional after touching the package list, and a stale
`.sif` is invisible in the summary CSV -- it shows up as an install failure three
layers down.

Note that a local task checkout skips `tb_apptainer prepare` entirely (harbor builds
each `.sif` itself), so the file harbor reuses is the one named after the *image*
(`alexgshaw_<task>_20251031.sif`), not the `<task>__<task>.sif` that `prepare` would
write. `benchmarks/scripts/rebuild_tb_sifs.py` rebuilds the former with the same
definition `prepare` uses (baked `curl`, tmux, and the `/opt/harbor-server` venv).

A fifth one is not Apptainer-specific but bites the same way: with a thinking model,
an output cap that is too small is *not* a neutral safety net. At `--max-tokens 4096`
the model spent the whole budget on reasoning and returned neither text nor a tool
call; the CLI ended the session, the patch came back empty and the instance scored
unresolved. The `swe` setting now uses `8192`, and `agent_loop` nudges a truncated
(`finish_reason == "length"`) turn to continue up to three times instead of stopping.

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
directory** — so run the entry point from the repository root:

```bash
python benchmarks/run_benchmarks.py --dataset terminalbench-2.1 --agent mini-code --mode short
```

To run from elsewhere, use an absolute path instead of a relative one: the entry point
has no per-option override, so make the edit in the config (or a copy of it):

```json
"dataset_dir": "/abs/path/to/external_repos/terminal-bench-2-1"
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
