#!/usr/bin/env python3
"""Write a copy of a benchmark config for another model / endpoint.

``submit_benchmark_slurm.sh`` uses this so a model name at the top of the
launcher is enough: the config's model entries, base URLs and the env block are
rewritten consistently, and a full run can drop the sample limit / task ids
that the checked-in smoke configs carry.

Usage:
    make_benchmark_config.py TEMPLATE OUT --model MODEL --base-url URL
        [--dataset DATASET] [--full]
"""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("out")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", default=None,
                        help="override the suite dataset (e.g. SWE-bench_Lite)")
    parser.add_argument("--full", action="store_true",
                        help="drop the smoke limits and task_ids")
    parser.add_argument("--sif-cache", default=None,
                        help="set options.sif_cache for SWE-bench")
    parser.add_argument("--tb-staged-dir", default=None,
                        help="run Terminal-Bench from this staged SIF dataset")
    parser.add_argument("--tb-sif-cache", default=None,
                        help="set the Terminal-Bench singularity image cache")
    args = parser.parse_args(argv)

    with open(args.template, encoding="utf-8") as handle:
        payload = json.load(handle)

    base = args.base_url.rstrip("/")
    for model in payload.get("models") or []:
        model["model"] = args.model
        model["base_url"] = base

    env = payload.setdefault("env", {})
    env["OPENAI_BASE_URL"] = f"{base}/v1"
    env["ANTHROPIC_BASE_URL"] = base
    env["ANTHROPIC_MODEL"] = args.model
    env.setdefault("OPENAI_API_KEY", "local-vllm")
    env.setdefault("ANTHROPIC_API_KEY", "local-vllm")

    for suite in payload.get("suites") or []:
        if args.dataset:
            suite["dataset"] = args.dataset
        options = suite.setdefault("options", {})
        if args.full:
            suite.pop("limit", None)
            options.pop("task_ids", None)
            options.pop("limit", None)
        if args.sif_cache and suite.get("name") == "swebench":
            options["sif_cache"] = args.sif_cache
        if args.tb_staged_dir and suite.get("name") == "terminal-bench":
            options["dataset_dir"] = args.tb_staged_dir
        if args.tb_sif_cache and suite.get("name") == "terminal-bench":
            options["singularity_image_cache_dir"] = args.tb_sif_cache

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
