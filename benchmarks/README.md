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

## 8. Results on the H200 used to validate this (2026-10-22)

Fixed model `ornith-ai/Ornith-1.5-35B-A3B` (vLLM, port 30000), three agents, `--limit 3`,
`runtime: auto`:

| suite | agent | n | errors | cmd fail | why |
| --- | --- | --- | --- | --- | --- |
| terminal-bench | mini-code / claude-code / pi | 3 each | 3 each | 0 | harbor's `--env singularity` converted the task images to `.sif`, then failed: `[server] /usr/bin/python3 not found, installing... FATAL: cannot install /usr/bin/python3` |
| swebench | mini-code / claude-code / pi | 0 | 0 | 1 each | no instance list (`datasets` not installed) and no Docker SDK for the grader |

Both suites need Docker, which this host does not have (`docker: MISSING`, `apptainer:
/usr/bin/apptainer`, no rootless socket, no podman) — that is what `validate` reports, and the
CSV records it as `errors`/`cmd fail` instead of inventing a score. The remainder of the pipeline
is exercised end to end: config expansion, `plan`, harbor job creation, image conversion,
per-task logs, outcome collection, `summary.{json,md}`, the CSV export and the analyser.

To get real numbers, rerun the same command on a Docker host (or point `runtime` at Apptainer for
the suites whose tasks ship plain images):

```bash
python -m benchmarks run -c benchmarks/configs/ornith_30b.json \
    --run-id ornith-30b-docker --limit 3 --csv benchmarks/results/ornith-30b.csv
python -m benchmarks analyze benchmarks/results/ornith-30b.csv
```

---

## 9. Safety and habits

* **`plan` first.** It prints the exact argv (never a shell string), the environment variable
  *names* it sets and the container calls, without downloading a dataset or starting a container.
* **Agent execution is refused outside a container** unless `--allow-host` is passed: several
  settings (`--sudo`, `bypassPermissions`) are only safe in a disposable environment.
* **Nothing secret is written.** Logs, results and summaries pass through a redactor that keeps
  environment variable *names* and replaces values; only `api_key_env` names ever enter a config.
* **Start small.** `--limit 1` on `terminal-bench` is the cheapest end-to-end check; harbor is
  young and moves fast, so re-read `harbor run --help` when a flag stops being accepted.
