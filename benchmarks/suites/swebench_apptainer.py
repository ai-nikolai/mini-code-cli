"""Grade SWE-bench predictions with Apptainer instead of a Docker daemon.

``swebench.harness.run_evaluation`` is bound to ``docker-py``: it creates a
container, copies the patch and the eval script into it, runs ``/eval.sh``, copies
the log out and writes ``report.json``. None of that is SWE-bench's *scoring* --
the scoring is :func:`swebench.harness.grading.get_eval_report`, which takes a
``TestSpec``, a prediction and a log file.

This module keeps the scoring and replaces only the transport:

    apptainer exec --pwd /testbed --bind <logdir>:/out <image.sif> \\
        bash -lc 'git apply -v /out/patch.diff; bash /out/eval.sh'

The instance's own ``eval_script`` (shipped by the SWE-bench dataset) does the
checkout, the test patch, the conda activation and the test command; the report is
produced by ``get_eval_report`` and written in the same shape as the upstream
runner, so `SweBenchSuite.collect` reads it unchanged.

Why this exists: HPC machines have Apptainer and no Docker daemon, and every
SWE-bench Verified instance image is published as public OCI
(``swebench/sweb.eval.x86_64.<instance_id>`` with ``__`` escaped to ``_1776_``),
so ``apptainer pull`` / ``apptainer exec docker://...`` needs no daemon at all.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

#: Where the writable bits of an instance live inside the container.
LOGS_DIR = "/out"

#: Fallback flags for `git apply`, mirroring the upstream runner's chain: a plain
#: apply, then a 3-way merge (a rebased patch often applies cleanly that way).
GIT_APPLY_CHAIN = ("git apply -v", "git apply -v --3way")


def escaped_image(instance):
    """The OCI image for an instance, with Docker-tag-safe escaping.

    SWE-bench tags use ``_1776_`` where the instance id has ``__`` (a tag cannot
    contain ``__``). Datasets that ship an ``image`` field are used verbatim.
    """
    image = str(instance.get("image") or "").strip()
    if image:
        return image
    instance_id = str(instance["instance_id"])
    return ("swebench/sweb.eval.x86_64." +
            instance_id.replace("__", "_1776_"))


def image_ref(image):
    """``docker://`` unless it is already a native Apptainer image."""
    image = str(image)
    if image.endswith(".sif") or "://" in image:
        return image
    return f"docker://{image}"


def sif_path(sif_cache, instance_id):
    """Where an instance's image is cached (flat, collision-free name)."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(instance_id))
    return os.path.join(sif_cache, f"swe-{clean}.sif")


def pull(apptainer, image, path, log_path, env=None, force=False, timeout=7_200):
    """``apptainer pull`` one instance image; returns ``(ok, message)``."""
    if os.path.isfile(path) and not force:
        return True, "cached"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    argv = [apptainer, "pull", "--force", path, image_ref(image)]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(argv)}\n")
        handle.flush()
        try:
            result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"apptainer pull failed: {e}"
    if result.returncode != 0:
        return False, f"apptainer pull exited {result.returncode} (see {log_path})"
    return True, "pulled"


#: Fields `make_test_spec` needs; when they are absent the sidecar is consulted.
GRADING_FIELDS = ("image", "log_parser", "eval_type", "eval_script",
                  "FAIL_TO_PASS", "PASS_TO_PASS")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_instances(path, limit=None):
    """Read the instance list written by ``swebench_env instances``.

    The agent-facing list is deliberately small, so the grading fields (the
    instance's ``image``, its ``eval_script`` and the test lists) live in the
    ``<name>.grading.jsonl`` sidecar the same command writes; the two are joined
    on ``instance_id`` here. A list that already carries them (an older run, or a
    local ``--from-file``) is used as-is.
    """
    instances = read_jsonl(path)
    missing = [i for i in instances
               if any(field not in i for field in GRADING_FIELDS)]
    if missing:
        root, ext = os.path.splitext(path)
        sidecar = f"{root}.grading{ext or '.jsonl'}"
        if os.path.isfile(sidecar):
            by_id = {row["instance_id"]: row for row in read_jsonl(sidecar)}
            merged = []
            for instance in instances:
                extra = by_id.get(instance["instance_id"], {})
                merged.append({**extra, **instance})
            instances = merged
    return instances[:int(limit)] if limit else instances


def run_one(apptainer, instance, patch_text, work_dir, timeout, env=None,
            extra_args=()):
    """Run one instance's eval script in its image; return ``(log_path, seconds)``.

    The patch and the eval script are written next to the log and bind-mounted,
    so nothing is copied into a container that may be read-only -- the same
    contract the upstream runner gets from ``copy_to_container``.
    """
    instance_id = instance["instance_id"]
    os.makedirs(work_dir, exist_ok=True)
    patch_path = os.path.join(work_dir, "patch.diff")
    eval_path = os.path.join(work_dir, "eval.sh")
    log_path = os.path.join(work_dir, "run_tests.log")
    with open(patch_path, "w", encoding="utf-8") as handle:
        handle.write(patch_text or "")
    with open(eval_path, "w", encoding="utf-8") as handle:
        handle.write(instance["eval_script"])

    # A .sif is read-only, so `git apply` dies with "Unable to create
    # '/testbed/.git/index.lock': Read-only file system" and every instance is
    # reported as "the model patch did not apply" -- a grading artefact, not a
    # model failure. Stage the checkout on the host first and bind it back over
    # /testbed: the eval script (which does `cd /testbed` and a git checkout for
    # the test files) then runs against a writable copy, exactly as upstream's
    # container allows. The agent path already does this for the same reason.
    testbed = os.path.join(work_dir, "testbed")
    if not os.path.isdir(os.path.join(testbed, ".git")):
        os.makedirs(testbed, exist_ok=True)
        stage = [apptainer, "exec", "--cleanenv", "--pwd", "/",
                 "--bind", f"{os.path.abspath(work_dir)}:{LOGS_DIR}",
                 *extra_args, str(instance["_sif"]),
                 "bash", "-lc", "cp -a /testbed/. /out/testbed/"]
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(stage)}\n")
            handle.flush()
            subprocess.run(stage, stdout=handle, stderr=subprocess.STDOUT,
                           env=env, timeout=timeout)

    # `git apply` then the instance's own script: the script checks out the base
    # commit for the test files, so the patch must land first, exactly as upstream.
    commands = [f"{cmd} {LOGS_DIR}/patch.diff" for cmd in GIT_APPLY_CHAIN]
    script = " || ".join(commands) + f"; bash {LOGS_DIR}/eval.sh"
    argv = [apptainer, "exec", "--cleanenv", "--pwd", "/testbed",
            "--bind", f"{os.path.abspath(work_dir)}:{LOGS_DIR}",
            "--bind", f"{os.path.abspath(testbed)}:/testbed",
            *extra_args, str(instance["_sif"]), "bash", "-lc", script]

    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(argv)}\n\n")
        handle.flush()
        try:
            subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                           env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            with open(log_path, "a", encoding="utf-8") as append:
                append.write(f"\nTimeout error: {timeout} seconds exceeded.\n")
            return log_path, float(timeout)
    return log_path, time.monotonic() - started


def grade(instance, patch_text, log_path, include_tests_status=False):
    """Score one instance with swebench's own grading; returns the report entry."""
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    spec = make_test_spec(instance)
    prediction = {"instance_id": instance["instance_id"],
                  "model_name_or_path": "benchmarks",
                  "model_patch": patch_text or ""}
    report = get_eval_report(test_spec=spec, prediction=prediction,
                             test_log_path=str(log_path),
                             include_tests_status=include_tests_status)
    return report.get(instance["instance_id"], {})


def _infra_problem(entry, seconds, timeout):
    """Why this instance tells us nothing, or ``None`` when it is a real verdict."""
    if entry.get("error"):
        return str(entry["error"])
    if seconds >= float(timeout):
        return f"timed out after {timeout}s"
    if entry.get("resolved"):
        return None
    if entry.get("patch_exists") and not entry.get("patch_successfully_applied"):
        return "the model patch did not apply"
    return None


def evaluate(instances, patches_dir, report_path, work_root, sif_cache=None,
             timeout=1_800, apptainer=None, limit=None, env=None, only=()):
    """Evaluate every instance that has a patch; write an upstream-shaped report.

    An instance with no patch (or an image that absent) is recorded in
    ``error_ids``: "the agent produced nothing" and "the harness could not run"
    must not look like each other.
    """
    apptainer = apptainer or shutil.which("apptainer") or shutil.which("singularity")
    if apptainer is None:
        raise SystemExit("swebench_apptainer: no apptainer/singularity on PATH")
    env = dict(env or os.environ)
    sif_cache = sif_cache or os.path.join(work_root, "sif")
    wanted = {str(i) for i in only} if only else None

    resolved, unresolved, errors, completed = [], [], [], 0
    results = {}
    for instance in instances:
        instance_id = instance["instance_id"]
        if wanted and instance_id not in wanted:
            continue
        if limit and completed >= int(limit):
            break
        sif = sif_path(sif_cache, instance_id)
        if not os.path.isfile(sif):
            # No image means the harness never got as far as the agent's patch:
            # "we could not run it" must not be reported as "the agent failed".
            errors.append(instance_id)
            results[instance_id] = {"resolved": False, "infra_failure": True,
                                    "error": "no image"}
            continue
        patch_file = os.path.join(patches_dir, instance_id, "model.patch")
        patch_text = ""
        if os.path.isfile(patch_file):
            with open(patch_file, encoding="utf-8") as handle:
                patch_text = handle.read()
        instance = dict(instance, _sif=sif)
        log_path, seconds = run_one(apptainer, instance, patch_text,
                                    os.path.join(work_root, instance_id),
                                    timeout, env=env)
        try:
            entry = grade(instance, patch_text, log_path)
        except Exception as e:  # noqa: BLE001 - a grading crash is an outcome
            entry = {"resolved": False, "error": f"grading failed: {e}"}
        entry = dict(entry, duration_seconds=round(seconds, 2),
                     patch_bytes=len(patch_text or ""))
        # Grading only means something if the container ran and the patch landed;
        # otherwise the run is an infrastructure failure, not an unresolved patch.
        problems = _infra_problem(entry, seconds, timeout)
        if problems:
            entry = dict(entry, infra_failure=True, error=problems)
        results[instance_id] = entry
        completed += 1
        if entry.get("resolved"):
            resolved.append(instance_id)
        elif entry.get("infra_failure") or entry.get("error"):
            errors.append(instance_id)
        else:
            unresolved.append(instance_id)

    report = {
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "error_ids": sorted(set(errors)),
        "completed_instances": completed,
        "results": results,
        "backend": "apptainer",
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    env = dict(os.environ)
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if apptainer is None:
        print("error: no apptainer/singularity on PATH", file=sys.stderr)
        return 2

    if args.command == "pull":
        instances = load_instances(args.instances, args.limit)
        failed = 0
        for instance in instances:
            image = escaped_image(instance)
            path = sif_path(args.sif_cache, instance["instance_id"])
            ok, message = pull(apptainer, image, path,
                               os.path.join(args.log_dir, instance["instance_id"] +
                                            ".pull.log"), env=env,
                               force=args.force)
            print(f"{instance['instance_id']:<50} {message}")
            failed += 0 if ok else 1
        print(f"pull: {len(instances) - failed}/{len(instances)} ready "
              f"in {os.path.abspath(args.sif_cache)}")
        return 0 if not failed else 1

    report = evaluate(
        load_instances(args.instances, args.limit), args.patches_dir,
        args.report, args.work_dir, sif_cache=args.sif_cache,
        timeout=args.timeout, apptainer=apptainer, env=env,
        only=args.only or ())
    print(json.dumps({key: report[key] for key in
                      ("completed_instances", "resolved_ids", "unresolved_ids",
                       "error_ids")}, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.suites.swebench_apptainer",
        description="Run and grade SWE-bench instances with Apptainer.")
    sub = parser.add_subparsers(dest="command", required=True)

    pull_cmd = sub.add_parser("pull", help="cache the instance images as .sif")
    pull_cmd.add_argument("--instances", required=True)
    pull_cmd.add_argument("--sif-cache", required=True)
    pull_cmd.add_argument("--log-dir", default="logs")
    pull_cmd.add_argument("--limit", type=int, default=None)
    pull_cmd.add_argument("--force", action="store_true")

    run_cmd = sub.add_parser("evaluate", help="apply patches, run tests, grade")
    run_cmd.add_argument("--instances", required=True)
    run_cmd.add_argument("--patches-dir", required=True)
    run_cmd.add_argument("--report", required=True)
    run_cmd.add_argument("--work-dir", required=True)
    run_cmd.add_argument("--sif-cache", default=None)
    run_cmd.add_argument("--timeout", type=int, default=1_800)
    run_cmd.add_argument("--limit", type=int, default=None)
    run_cmd.add_argument("--only", action="append", default=None)
    return parser


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
