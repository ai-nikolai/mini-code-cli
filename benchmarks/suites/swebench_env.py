"""Small CLI helpers for the SWE-bench adapter (run as ``python -m ...``).

Two jobs, both of which are pure file transformations and therefore testable
without Docker, a dataset or a model:

``instances``
    Materialise the pinned instance list: read a local JSONL (``--from-file``,
    the offline path) or load the HuggingFace dataset at an exact ``--revision``
    and copy the fields the run needs. Keeping this as a separate step is what
    lets ``plan`` print a run without downloading several gigabytes.

``predictions``
    Collect the per-instance ``model.patch`` files into the ``predictions.jsonl``
    the SWE-bench harness mandates: one object per instance with
    ``instance_id`` / ``model_name_or_path`` / ``model_patch``. An instance whose
    patch is missing becomes an empty patch -- explicitly, so the report says
    "unresolved" instead of the harness erroring out on a malformed file.
"""
import argparse
import json
import os
import sys

#: Fields kept from the dataset: enough to run the agent and to write a prediction,
#: and small enough that the JSONL stays a few MB rather than a few GB.
INSTANCE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement",
                   "environment_setup_commit", "version", "hints_text", "created_at")

DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
DEFAULT_SPLIT = "test"


def load_instances(dataset=DEFAULT_DATASET, split=DEFAULT_SPLIT, revision=None,
                   limit=None, from_file=None):
    """Return a list of instance dicts, from a file or the pinned dataset."""
    if from_file:
        return read_instances_file(from_file, limit)
    try:
        from datasets import load_dataset  # heavy, optional: the swebench extra
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "reading the SWE-bench dataset needs the 'datasets' package "
            "(pip install 'swebench>=5'); alternatively pass --from-file with a "
            "JSONL of instances") from e
    loaded = load_dataset(dataset, split=split, revision=revision)
    instances = []
    for row in loaded:
        instances.append({key: row.get(key) for key in INSTANCE_FIELDS
                          if row.get(key) is not None})
        if limit and len(instances) >= int(limit):
            break
    return instances


def read_instances_file(path, limit=None):
    """Read instances from a ``.jsonl`` (one object per line) or ``.json`` list."""
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
            instances = payload if isinstance(payload, list) else list(payload.values())
        else:
            instances = [json.loads(line) for line in handle if line.strip()]
    if limit:
        instances = instances[: int(limit)]
    for instance in instances:
        if "instance_id" not in instance:
            raise SystemExit(f"{path}: an instance has no instance_id")
    return instances


def write_instances(instances, path):
    """Write one JSON object per line, in the order given (stable for diffing)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for instance in instances:
            handle.write(json.dumps(instance, sort_keys=True) + "\n")
    return path


def write_predictions(instances_path, patches_dir, model_name, out_path):
    """Build ``predictions.jsonl`` from ``<patches_dir>/<instance_id>/model.patch``.

    A missing or empty patch is written as an empty ``model_patch``: the harness
    then counts the instance as unresolved, which is the honest outcome, instead
    of failing the whole evaluation on a broken file.
    """
    instances = read_instances_file(instances_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    missing = []
    with open(out_path, "w", encoding="utf-8") as handle:
        for instance in instances:
            instance_id = instance["instance_id"]
            patch_path = os.path.join(patches_dir, instance_id, "model.patch")
            patch = ""
            if os.path.isfile(patch_path):
                with open(patch_path, encoding="utf-8", errors="replace") as source:
                    patch = source.read()
            else:
                missing.append(instance_id)
            handle.write(json.dumps({
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }) + "\n")
    if missing:
        print(f"predictions: {len(missing)} instance(s) produced no patch "
              f"(written as empty): {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="benchmarks.suites.swebench_env",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    instances = sub.add_parser("instances", help="materialise the instance list")
    instances.add_argument("--dataset", default=DEFAULT_DATASET)
    instances.add_argument("--split", default=DEFAULT_SPLIT)
    instances.add_argument("--revision", default=None,
                           help="dataset revision/commit to pin")
    instances.add_argument("--limit", type=int, default=None)
    instances.add_argument("--from-file", default=None,
                           help="offline JSONL/JSON of instances instead of HF")
    instances.add_argument("--out", required=True)

    predictions = sub.add_parser("predictions",
                                 help="collect patches into predictions.jsonl")
    predictions.add_argument("--instances", required=True)
    predictions.add_argument("--patches-dir", required=True)
    predictions.add_argument("--model-name", required=True)
    predictions.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "instances":
        found = load_instances(dataset=args.dataset, split=args.split,
                               revision=args.revision, limit=args.limit,
                               from_file=args.from_file)
        write_instances(found, args.out)
        print(f"instances: wrote {len(found)} to {args.out}")
        return 0
    write_predictions(args.instances, args.patches_dir, args.model_name, args.out)
    print(f"predictions: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
