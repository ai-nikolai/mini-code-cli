# Benchmark scripts

Everything here drives the benchmark harness in `benchmarks/`. The local-model
helper `scripts/run_model.sh` lives in the repository-root `scripts/` together
with the mini-code-cli tooling (`lint.py`, `push_version_tag.py`), because it is
also useful when running mini-code against a local model.

Run every command from the repository root. Generated jobs, configs and logs
live under `experiments/`, which is git-ignored.

## Quick map

| Script | What it does |
|---|---|
| `setup_external_repos.sh` | Clone/update the fork checkouts (`harbor`, `terminal-bench-2-1`). |
| `push_all_repos.sh` | Push the fork branches (not used by a benchmark run). |
| `rebuild_tb_sifs.py` | Low-level rebuild of the derived TB SIFs; normally use `prepare_tb_sifs.sh`. |
| `run_benchmark_slurm.sh` | Run on an **existing interactive allocation** (`srun --overlap --jobid=…`): starts vLLM, waits, runs one agent at a time. |
| `submit_prepare_tb_sifs.sh` | Submit a CPU `sbatch` job that runs `prepare_tb_sifs.sh`. |
| `prepare_tb_sifs.sh` | Build all TB 2.1 SIFs with the baked `mini_code_cli`, stage the task set, verify each image. |
| `prepare_swe_sifs.sh` | Materialise the SWE instance list, pull all SIFs, verify them. |
| `submit_benchmark_slurm.sh` | Submit one GPU job for one model × benchmark list × agent list (sequential inside the job). |
| `submit_benchmark_matrix.sh` | Cross-product of `MODELS × BENCHMARKS × AGENTS`; one `sbatch` job per cell, each with its own model server. |
| `prune_benchmark_storage.sh` | Clean heavy/intermediate data by degree; dry-run by default. |
| `make_benchmark_config.py` | Helper used by the launchers to rewrite a config for another model/endpoint. |

## The one-benchmark-at-a-time workflow

For a 1 TB home, prepare one benchmark, run it, prune it, then prepare the
other. This is the intended order:

```bash
# 0. once, if the fork checkouts are missing
benchmarks/scripts/setup_external_repos.sh

# 1. Terminal-Bench images + staged task set
SIF_CACHE=/scratch/$USER/sif benchmarks/scripts/prepare_tb_sifs.sh

# 2. run TB (the launcher detects experiments/tb-staged/prepared.json)
SIF_CACHE=/scratch/$USER/sif BENCHMARKS=terminalbench-2.1 MODE=full \
    benchmarks/scripts/submit_benchmark_slurm.sh

# 3. free TB SIFs + heavy trial workspaces, keep scores/patches/logs
DRY_RUN=0 CONFIRM=yes MODE=after-tb \
    benchmarks/scripts/prune_benchmark_storage.sh

# 4. SWE-bench images + instance list
SIF_CACHE=/scratch/$USER/sif benchmarks/scripts/prepare_swe_sifs.sh

# 5. run SWE (Lite full = 300 instances)
SIF_CACHE=/scratch/$USER/sif BENCHMARKS=swe-bench-lite-full MODE=full \
    benchmarks/scripts/submit_benchmark_slurm.sh

# 6. free SWE SIFs, keep scores/patches/logs
DRY_RUN=0 CONFIRM=yes MODE=after-swe \
    benchmarks/scripts/prune_benchmark_storage.sh
```

`SIF_CACHE` must be the same for prepare and run. If it is unset, the
repositories' defaults are used (`benchmarks/.cache/sif` for TB,
`benchmarks/runs/_cache/swe-sif` for SWE).

For a smoke run, use `MODE=short_eval LIMIT=5` (SWE) or `MODE=short` with
`BENCHMARKS=terminalbench-2.1-3` (the three cached TB tasks).

## Stages and what is stored where

| Stage | Stores | Used by | Size / notes |
|---|---|---|---|
| **Source tasks** | `external_repos/terminal-bench-2-1/` (90 task dirs), SWE dataset cache | prepare | small (MBs); cloned by `setup_external_repos.sh` |
| **SIFs** | TB: `benchmarks/.cache/sif/<task>__<task>.sif` + `.key`; SWE: `benchmarks/runs/_cache/swe-sif/swe-<id>.sif` + `.complete` | run, grading, re-run | TB ~0.33 GB each (90 → ~30–90 GB); SWE ~1.28 GB each (Lite 300 → ~385 GB, Verified 500 → ~640 GB) |
| **Prepared inputs** | TB: `experiments/tb-staged/<task>/task.toml` pointing at the SIF + `prepared.json`; SWE: `experiments/swe-instances/instances.jsonl` + grading sidecar | run | MBs |
| **Generated config/jobs** | `experiments/configs/*.json`, `experiments/slurm/*.sh`, `experiments/submitted.csv`, `experiments/logs/` | run, audit | MBs |
| **Run artifacts** | `results/<campaign>/…`: `summary_results.csv`, per-run `summary.json/md`, `manifest.json`, `trial.json`, `results.jsonl`, SWE `instances/<id>/model.patch`, `agent.log`, `testbed/`, `agent-tools/`, `eval/`, `pulls/`, TB `harbor/` logs/artifacts | reporting, re-run, debugging | heavy workspaces (testbeds, agent-tools, eval/pulls); summaries/patches are small |
| **Apptainer scratch/cache** | `$APPTAINER_CACHEDIR`, `$APPTAINER_TMPDIR`, `$SCRATCH_DIR`, `.scratch/` | prepare/run only | ~200–400 GB for a full SWE pull if not pruned |
| **Model weights** | `~/.cache/huggingface` (not managed by these scripts) | run | current cache ~170 GB; add model-specific size |

What uses what:

* `prepare_*` reads source tasks/instance list and writes **SIFs** + prepared
  inputs. It needs no model server. `prepare_tb_sifs.sh` bakes `mini_code_cli`
  into the image.
* A **run** reads SIFs + prepared inputs + its config, and writes run artifacts.
  SWE copies the instance checkout to `instances/<id>/testbed/` so the agent and
  the patch capture share it; TB runs from the staged task set.
* **Grading** uses the SIF and the captured `model.patch` (SWE) or the verifier
  (TB). It does not need the agent workspace afterwards.
* A **re-run** can reuse the SIFs and prepared inputs. `--no-resume` (or a new
  campaign id) clears the trial directory first.
* **Summaries and patches are the output**; SIFs, testbeds, `agent-tools`,
  `eval`/`pulls` and Apptainer scratch are reproducible intermediates.

## Pruning degrees

`prune_benchmark_storage.sh` is **dry-run by default**. Deletion requires
`DRY_RUN=0 CONFIRM=yes`.

| `MODE` | Removes | Use when |
|---|---|---|
| `keep-results` / `light` | testbeds, `agent-tools`, `eval`, `pulls`, `.scratch` | keep SIFs for another agent, but free trial workspaces |
| `sifs` | SIFs for `BENCH=tb\|swe\|all` | free images but keep results |
| `scratch` | `SCRATCH_DIR` / `APPTAINER_CACHEDIR` contents | prepare/run finished and scratch is stale |
| `partial` | `light` + selected SIFs | general between-benchmark cleanup |
| `after-tb` | `partial BENCH=tb` | after TB, before SWE |
| `after-swe` | `partial BENCH=swe` | after SWE |
| `full` | `partial` + `results/`, `benchmarks/runs/`, generated jobs/configs/logs, scratch | reset the machine |
| `nuke` | `full BENCH=all` | full reset (asks for `CONFIRM=yes`) |

Never prune while a campaign is running: `light`/`partial` touch active trial
workspaces.

## Common environment variables

| Variable | Meaning |
|---|---|
| `SIF_CACHE` | shared root for both caches (`$SIF_CACHE/tb`, `$SIF_CACHE/swe`) |
| `TB_CACHE` / `SCRATCH_DIR` | TB dataset cache / Apptainer unpack dir |
| `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` | Apptainer caches (compute `/tmp` is only 1 GB) |
| `BENCH_ENV` / `CONDA_SH` | conda env with harbor+swebench, and its profile script |
| `MODEL_NAME`, `NUM_GPUS`, `MODEL_ENV`, `TOOL_CALL_PARSER`, `REASONING_PARSER` | model/server selection |
| `BENCHMARKS`, `AGENTS`, `MODE`, `FULL`, `LIMIT` | matrix/selection |
| `SUBMIT`, `DRY_RUN`, `CONFIRM` | write-only vs submit, dry-run, deletion confirmation |

## Sizes and expectations

Full sets: TB 2.1 = 90 tasks; SWE-bench Lite = 300 instances (Verified 500).
With cached images and serial tasks: TB ~5–10 h (mini-code), ~15–30 h (Pi),
~12–25 h (Claude Code); Lite ~10–20 h, ~12–25 h, ~15–30 h respectively. A full
3-agent × 2-benchmark matrix is roughly 3–6 days serial; the matrix launcher can
run one job per cell on separate allocations instead.

The first prepare is dominated by image pulls/builds and needs the storage in
the table above. Use `VERIFY=test` (TB, `apptainer test`) or `VERIFY=exec` (SWE)
for a stronger check than the default `apptainer inspect`.
