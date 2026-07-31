# Running Mini Code with Harbor

[Harbor](https://harborframework.com/) runs Mini Code in benchmark task
containers and records the results. The adapter uses your current working tree,
not the published package.

## Setup

Install Docker (and make sure Docker desktop is running) and Ollama. Then run this once from the repository root:

```shell
uv tool install --force --with-editable . harbor
ollama pull qwen2.5-coder:7b
```

This installs [Harbor](https://harborframework.com/docs/getting-started) with
the local benchmark adapter.

## Local smoke test

Run one Terminal-Bench 2 task on macOS:

```shell
harbor run \
  --dataset terminal-bench@2.0 \
  --include-task-name break-filter-js-from-html \
  --agent benchmarks.harbor_agent:MiniCodeAgent \
  --model qwen2.5-coder:7b \
  --ak base_url=http://host.docker.internal:11434 \
  --allow-agent-host host.docker.internal \
  -n 1
```

Add `--install-only` to test container setup without running the model.

## Full Terminal-Bench 2 run

Run the full Terminal-Bench 2 dataset with four concurrent tasks and three
attempts per task:

```shell
harbor run \
  --dataset terminal-bench@2.0 \
  --agent benchmarks.harbor_agent:MiniCodeAgent \
  --model qwen2.5-coder:7b \
  --ak base_url=http://host.docker.internal:11434 \
  --allow-agent-host host.docker.internal \
  -n 4 \
  -k 3
```

## Remote endpoint

Pass the API key through the agent environment:

```shell
harbor run \
  --dataset terminal-bench@2.0 \
  --include-task-name break-filter-js-from-html \
  --agent benchmarks.harbor_agent:MiniCodeAgent \
  --model provider/model-name \
  --ak base_url=https://api.example.com \
  --ae MINI_CODE_API_KEY="$MINI_CODE_API_KEY" \
  --allow-agent-host api.example.com \
  -n 1
```

Mini Code appends `/v1/chat/completions` to `base_url`.

## Results

Harbor saves runs under `jobs/<timestamp>/`. Useful files include:

- `result.json`: rewards and run summary
- `agent/mini-code.log`: Mini Code console output
- `agent/history_<session-id>.json`: conversation history



## Troubleshooting

- `No module named 'benchmarks'`: rerun the editable Harbor installation above.
- Model unreachable: check the endpoint and its `--allow-agent-host` entry.
- Zero reward: inspect `mini-code.log`, `trial.log`, verifier output, and
`result.json`.

To use another Harbor dataset, replace `--dataset terminal-bench@2.0` and
adjust or remove `--include-task-name`.
