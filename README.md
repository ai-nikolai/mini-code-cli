# A minimal coding agent cli

Purpose: no bloat, minimal coding cli, to enjoy experimenting and researching things...

## Getting started:
1. Installation (from pypi)
```bash
pip3 install mini-code-cli
```

2. Run agent: (get LLM running via SGLang or Vllm, expect)
```bash
mini_code_cli #run the agent loop
mini_code_cli --url "0.0.0.0" --allowed-dir ./experiments/ --api-key "sk-dummy" #localhost and 30000 is default, allowed_dir ./ is default
```

## (C) Nikolai Rozanov, 2026 - Present