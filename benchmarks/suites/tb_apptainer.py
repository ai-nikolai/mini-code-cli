"""Prepare Terminal-Bench tasks for Apptainer: one derived ``.sif`` per task.

Harbor's Apptainer backend (``harbor run --env singularity``) can run a task whose
``task.toml`` points at an image, but Terminal-Bench images are *bare*: they have
no ``/usr/bin/python3`` and no writable ``/opt``, while harbor's backend needs a
Python RPC server inside the container (``/opt/harbor-server`` with
uvicorn/fastapi). Converting the published image as-is fails at
``FATAL: cannot install /usr/bin/python3``.

So each task image is converted **once**, into a derived SIF that already carries
harbor's prerequisites:

    Bootstrap: docker
    From: alexgshaw/<task>:20251031          # the image task.toml names
    %post                                    # python3 + venv + harbor server
    %environment                             # PYTHONPATH for mini-code

The task directory is then copied with ``docker_image`` rewritten to the SIF path,
and harbor runs the *local dataset* (``-p``) instead of pulling the registry copy.
Nothing in harbor or Terminal-Bench is modified: this is a preparation step, and
``plan``/``validate`` report it like any other.

Two Apptainer details are load-bearing and encoded here:

* the SIF cache must live on a filesystem with room for GB-sized images (``/tmp``
  is routinely a few GB);
* harbor's default ``singularity_no_mount=home,tmp,bind-paths`` suppresses the
  ``/etc/resolv.conf`` bind, so the container loses DNS — the suite passes
  ``singularity_no_mount=home,tmp`` instead (see ``terminal_bench.py``).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

#: Where the derived images go when a run does not name a cache.
DEFAULT_SIF_CACHE = os.path.join("benchmarks", ".cache", "sif")

_DOCKER_IMAGE_RE = re.compile(r'^\s*docker_image\s*=\s*["\']([^"\']+)["\']',
                              re.MULTILINE)

#: Packages harbor's backend shells out to, plus what a coding agent typically
#: needs. One apt call, because each `apt-get update` in a fresh layer is slow.
POST_INSTALL = (
    "python3", "python3-venv", "python3-pip", "curl", "ca-certificates", "git",
    "tmux", "asciinema", "procps",
)

#: Python packages the baked-in agent needs at runtime. mini-code is not
#: dependency-free after all: it imports ``requests`` for its HTTP client.
PIP_FOR_AGENT = ("requests",)


def render_def(image, mini_code_dir=None):
    """The Apptainer definition that turns a task image into a runnable SIF."""
    lines = [f"Bootstrap: docker", f"From: {image}", ""]
    if mini_code_dir:
        # The bridge starts `python3 -m mini_code_cli` inside the container; the
        # package is dependency-free, so copying the directory is enough.
        lines += ["%files",
                  f"    {mini_code_dir} /opt/mini-code/mini_code_cli", ""]
    lines += [
        "%post",
        "    export DEBIAN_FRONTEND=noninteractive",
        "    if command -v apt-get >/dev/null 2>&1; then",
        # APT::Sandbox::User=root matters under --fakeroot: apt otherwise drops
        # to the _apt user, which fakeroot cannot switch to, and the build hangs
        # on the first `apt-get update` (observed: a 35-minute stall).
        "        APT_OPTS='-o APT::Sandbox::User=root -o Acquire::http::Timeout=20 -o Acquire::Retries=1'",
        "        apt-get $APT_OPTS update -qq || true",
        f"        apt-get $APT_OPTS install -y -qq {' '.join(POST_INSTALL)} || true",
        "    elif command -v apk >/dev/null 2>&1; then",
        "        apk add --no-cache python3 py3-pip bash curl git tmux || true",
        "    fi",
        "    # harbor's singularity backend wants a ready server venv here; the SIF",
        "    # is read-only, and its --writable-tmpfs layer does not survive the",
        "    # agent setup step, so the venv has to be baked in.",
        "    if [ -x /usr/bin/python3 ]; then",
        # agent runtime deps into the *system* interpreter (the bridge starts
        # `python3 -m mini_code_cli`); --break-system-packages for PEP 668 images.
        f"        /usr/bin/python3 -m pip install -q --break-system-packages "
        f"{' '.join(PIP_FOR_AGENT)} 2>/dev/null || "
        f"/usr/bin/python3 -m pip install -q {' '.join(PIP_FOR_AGENT)} || true",
        "        /usr/bin/python3 -m venv /opt/harbor-server || true",
        "        /opt/harbor-server/bin/pip install -q --upgrade pip || true",
        "        /opt/harbor-server/bin/pip install -q uvicorn fastapi || true",
        "    fi",
        "",
    ]
    if mini_code_dir:
        lines += ["%environment",
                  "    export PYTHONPATH=/opt/mini-code:${PYTHONPATH:-}", ""]
    lines += ["%test",
              "    /opt/harbor-server/bin/python -c 'import uvicorn, fastapi' "
              "|| exit 1", ""]
    return "\n".join(lines)


def task_image(task_dir):
    """The ``docker_image`` named by a task's ``task.toml`` (or ``None``)."""
    path = os.path.join(task_dir, "task.toml")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        match = _DOCKER_IMAGE_RE.search(handle.read())
    return match.group(1) if match else None


def point_task_at_sif(task_dir, sif_path):
    """Rewrite ``docker_image`` in a copy of the task so harbor uses the SIF."""
    path = os.path.join(task_dir, "task.toml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text = _DOCKER_IMAGE_RE.sub(f'docker_image = "{os.path.abspath(sif_path)}"',
                                text, count=1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def sif_name(dataset, task_name):
    """A flat, collision-free file name for a task's derived image."""
    clean = f"{dataset}__{task_name}".replace("/", "_").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", clean) + ".sif"


def build_sif(apptainer, definition, sif_path, log_path, env=None, force=False,
              timeout=3_600):
    """``apptainer build --fakeroot`` one definition; returns ``(ok, message)``."""
    if os.path.isfile(sif_path) and not force:
        return True, "cached"
    os.makedirs(os.path.dirname(os.path.abspath(sif_path)) or ".", exist_ok=True)
    argv = [apptainer, "build", "--fakeroot"]
    if force:
        argv.append("--force")            # apptainer refuses to clobber otherwise
    argv += [sif_path, definition]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(argv)}\n")
        handle.flush()
        try:
            result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"apptainer build failed: {e}"
    if result.returncode != 0:
        return False, f"apptainer build exited {result.returncode} (see {log_path})"
    return True, "built"


def prepare(dataset_dir, out_dir, sif_cache, limit=None, task_ids=(),
            mini_code_dir=None, jobs=1, force=False, log_dir=None, env=None):
    """Derive one SIF per task and stage a local dataset harbor can run.

    Returns ``{"tasks": [...], "failed": [...], "out_dir": ...}``. Every task
    keeps its own status, because "one image would not build" must not hide the
    tasks that did.
    """
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if apptainer is None:
        raise SystemExit("tb_apptainer: no apptainer/singularity binary on PATH")
    names = sorted(name for name in os.listdir(dataset_dir)
                   if os.path.isdir(os.path.join(dataset_dir, name)))
    if task_ids:
        wanted = {str(t) for t in task_ids}
        names = [name for name in names if name in wanted]
    if limit:
        names = names[:int(limit)]
    if not names:
        raise SystemExit(f"tb_apptainer: no tasks under {dataset_dir!r}")

    log_dir = log_dir or os.path.join(out_dir, "_logs")
    os.makedirs(out_dir, exist_ok=True)

    def one(name):
        source = os.path.join(dataset_dir, name)
        image = task_image(source)
        if not image:
            return {"task": name, "status": "skipped",
                    "reason": "task.toml names no docker_image"}
        sif = os.path.join(sif_cache, sif_name(name, name))
        target = os.path.join(out_dir, name)
        if os.path.isdir(os.path.join(target, "task.toml")) and \
                os.path.isfile(sif) and not force:
            return {"task": name, "status": "cached", "sif": sif,
                    "dir": target, "image": image}
        definition = os.path.join(log_dir, f"{name}.def")
        os.makedirs(log_dir, exist_ok=True)
        with open(definition, "w", encoding="utf-8") as handle:
            handle.write(render_def(image, mini_code_dir))
        ok, message = build_sif(apptainer, definition, sif,
                                os.path.join(log_dir, f"{name}.build.log"),
                                env=env, force=force)
        if not ok:
            return {"task": name, "status": "failed", "reason": message,
                    "image": image}
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)
        point_task_at_sif(target, sif)
        return {"task": name, "status": "ready", "sif": sif, "dir": target,
                "image": image}

    with ThreadPoolExecutor(max_workers=max(int(jobs), 1)) as pool:
        results = list(pool.map(one, names))
    failed = [r for r in results if r["status"] == "failed"]
    manifest = {"out_dir": out_dir, "sif_cache": sif_cache,
                "tasks": results, "failed": failed}
    with open(os.path.join(out_dir, "prepared.json"), "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, sort_keys=True)
        h.write("\n")
    return manifest


def download_tasks(dataset, download_dir, harbor=None):
    """``harbor download <dataset> --export`` -- the only way to get task files."""
    harbor = harbor or shutil.which("harbor")
    if not harbor:
        raise SystemExit("tb_apptainer: harbor is not on PATH; cannot download "
                         f"{dataset} (pip install harbor)")
    os.makedirs(download_dir, exist_ok=True)
    argv = [harbor, "download", dataset, "--export", "-o", download_dir]
    print(f"$ {' '.join(argv)}")
    result = subprocess.run(argv, env=dict(os.environ))
    if result.returncode != 0:
        raise SystemExit(f"tb_apptainer: harbor download failed "
                         f"(exit {result.returncode})")
    # The export lands as <download-dir>/<dataset-name>/<task>/
    name = str(dataset).split("@")[0].replace("/", "_")
    candidate = os.path.join(download_dir, name)
    return candidate if os.path.isdir(candidate) else download_dir


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        tasks_dir = args.tasks_dir
        if not tasks_dir:
            if not args.dataset:
                raise SystemExit("tb_apptainer: pass --tasks-dir or --dataset")
            tasks_dir = download_tasks(
                args.dataset, args.download_dir or os.path.join(args.out, "_export"))
        manifest = prepare(
            dataset_dir=tasks_dir, out_dir=args.out,
            sif_cache=args.sif_cache, limit=args.limit,
            task_ids=args.task_id or (), mini_code_dir=args.mini_code,
            jobs=args.jobs, force=args.force, env=dict(os.environ))
        for task in manifest["tasks"]:
            detail = task.get("reason") or task.get("sif") or ""
            print(f"{task['task']:<40} {task['status']:<8} {detail}")
        if manifest["failed"]:
            print(f"tb_apptainer: {len(manifest['failed'])} task(s) failed",
                  file=sys.stderr)
            return 1
        print(f"tb_apptainer: {len(manifest['tasks'])} task(s) ready in "
              f"{os.path.abspath(args.out)}")
        return 0
    print(json.dumps(prepare_status(args.out), indent=2))
    return 0


def prepare_status(out_dir):
    path = os.path.join(out_dir, "prepared.json")
    if not os.path.isfile(path):
        return {"out_dir": out_dir, "prepared": False}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.suites.tb_apptainer",
        description="Derive Apptainer images for Terminal-Bench tasks.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="build SIFs and stage a local dataset")
    prep.add_argument("--tasks-dir", default=None,
                      help="an exported dataset: <dir>/<task>/task.toml")
    prep.add_argument("--dataset", default=None,
                      help="dataset label to download with `harbor download` "
                           "(e.g. terminal-bench@2.0) when --tasks-dir is absent")
    prep.add_argument("--download-dir", default=None,
                      help="where the export lands (default: alongside --out)")
    prep.add_argument("--out", required=True, help="where the staged dataset goes")
    prep.add_argument("--sif-cache", default=DEFAULT_SIF_CACHE,
                      help="where the .sif files are cached (needs GBs)")
    prep.add_argument("--limit", type=int, default=None)
    prep.add_argument("--task-id", action="append", default=None)
    prep.add_argument("--jobs", type=int, default=1)
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--mini-code", default=None,
                      help="path to the mini_code_cli package to bake in")
    status = sub.add_parser("status", help="what has been prepared so far")
    status.add_argument("--out", required=True)
    return parser


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
