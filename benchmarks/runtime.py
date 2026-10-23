"""Container runtimes: `docker` where it exists, `apptainer` where it does not.

Benchmarks run agents in disposable containers, but not every machine has a Docker
daemon -- HPC clusters (and the H200 this was validated on) ship Apptainer /
Singularity instead, where there is no daemon and no root. This module is the one
place that knows the difference, so a suite can say *what* to run inside a
container and let the runtime decide *how*:

    docker run --rm --name bench -v /host:/out -w /testbed -e K=V IMAGE CMD...
    apptainer exec --cleanenv --bind /host:/out --pwd /testbed --env K=V docker://IMAGE CMD...

The translation is intentionally shallow and honest about what cannot be
translated: Apptainer has no per-container CPU/memory limits (``--cpus`` /
``--memory`` are reported instead of silently dropped) and containers share the
host network namespace, so ``--network`` has no equivalent either.

Nothing here raises: a machine with no runtime at all must still be able to
``plan``, and a report must still be readable.
"""
import os
import shutil
import subprocess

#: Supported runtimes, in the order auto-detection prefers them.
RUNTIMES = ("docker", "apptainer")

#: What ``--runtime`` accepts besides a runtime name.
AUTO = "auto"

#: Flags that exist only in one runtime, so a plan can warn instead of pretending.
UNSUPPORTED = {
    "docker": {"--nv": "Apptainer's GPU flag has no Docker spelling (use --gpus)"},
    "apptainer": {
        "--cpus": "Apptainer has no per-container CPU quota",
        "--memory": "Apptainer has no per-container memory limit",
        "--pids-limit": "Apptainer has no per-container PID limit",
        "--network": "Apptainer containers share the host network namespace",
        "--gpus": "Apptainer takes GPU access through --nv",
    },
}


class RuntimeError_(RuntimeError):
    """The requested container runtime is unusable (missing binary, bad name)."""


def binaries():
    """``{runtime: path-or-None}`` for every supported runtime."""
    return {name: shutil.which(name) for name in RUNTIMES}


def available(name):
    """True when ``name``'s binary is on ``PATH`` (a daemon is not checked here)."""
    return bool(name) and shutil.which(str(name)) is not None


def resolve(preferred=None):
    """Pick the runtime to use: ``preferred`` if usable, else auto-detected.

    ``None``/``"auto"`` means "whichever is installed", Docker first because it is
    what the upstream harnesses (harbor, swebench) assume. An explicit choice that
    is not installed returns it unchanged -- the caller's preflight is what turns
    that into an error message naming the missing binary, which is more useful
    than silently switching runtime behind the user's back.
    """
    choice = str(preferred).strip().lower() if preferred else AUTO
    if choice in ("", AUTO):
        for name in RUNTIMES:
            if available(name):
                return name
        return None
    return choice


def version(name, timeout=20):
    """The runtime's version string, or ``None`` (missing binary / no daemon)."""
    if not available(name):
        return None
    if name == "docker":
        argv = ["docker", "version", "--format", "{{.Server.Version}}"]
    else:
        argv = ["apptainer", "--version"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip().splitlines()[0] if result.stdout.strip() else None


def image_ref(name, image):
    """How ``image`` is spelled for ``name``.

    Apptainer pulls OCI images through ``docker://``; a ``.sif`` path is passed
    through untouched, since that is already a native image.
    """
    image = str(image)
    if name == "apptainer" and not image.endswith(".sif") and "://" not in image:
        return f"docker://{image}"
    return image


def container_argv(name, image, command, container_name=None, env=None,
                   limits=None, mounts=(), workdir=None, network=None,
                   gpus=False, remove=True, overlays=()):
    """Build a one-shot ``argv`` that runs ``command`` inside ``image``.

    ``command`` is a list (never a shell string). ``mounts`` is a sequence of
    ``(host, container)`` pairs, ``limits`` the ``docker`` dict from the config
    (``cpus``/``memory``/``pids_limit``). Returned as a list so ``plan`` can print
    it verbatim and the runner can execute it without a shell.
    """
    command = [str(part) for part in command]
    env = dict(env or {})
    limits = dict(limits or {})
    # Commands run with cwd=the run directory, so a relative host path would be
    # resolved twice (apptainer: "mount .../trials/x/benchmarks/runs/..."), and
    # the mount would not exist.
    mounts = [(os.path.abspath(str(host)), str(target)) for host, target in mounts]
    name = resolve(name)

    if name == "docker":
        argv = ["docker", "run"]
        if remove:
            argv.append("--rm")
        if container_name:
            argv += ["--name", str(container_name)]
        if gpus:
            argv += ["--gpus", "all"]
        for host, target in mounts:
            argv += ["-v", f"{host}:{target}"]
        if workdir:
            argv += ["-w", str(workdir)]
        if limits.get("cpus"):
            argv += ["--cpus", str(limits["cpus"])]
        if limits.get("memory"):
            argv += ["--memory", str(limits["memory"])]
        if limits.get("pids_limit"):
            argv += ["--pids-limit", str(limits["pids_limit"])]
        network = network or limits.get("network")
        if network:
            argv += ["--network", str(network)]
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        return argv + [image_ref(name, image)] + command

    # apptainer (and, by name, singularity: the two CLIs are interchangeable)
    binary = "apptainer" if name in ("apptainer", None) else str(name)
    # A .sif is read-only: without a writable overlay every in-container write
    # fails ("/testbed/.git/index.lock: Read-only file system"), which is *every*
    # agent run -- it edits the checkout -- and every `git add -A` patch capture.
    argv = [binary, "exec", "--cleanenv", "--writable-tmpfs"]
    if gpus:
        argv.append("--nv")
    if workdir:
        argv += ["--pwd", str(workdir)]
    for host, target in mounts:
        argv += ["--bind", f"{host}:{target}"]
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    for overlay in overlays:
        # Persistent across execs, unlike --writable-tmpfs: an agent's edits have
        # to still be there when the next command captures the patch.
        argv += ["--overlay", os.path.abspath(str(overlay))]
    return argv + [image_ref("apptainer", image)] + command


def translation_notes(name, limits=None, network=None):
    """What the chosen runtime cannot honour, phrased for a ``plan``.

    Printed rather than enforced: a run with no CPU limit is still a valid run, it
    is just not the run the config describes, and the person reading ``plan``
    should know that before it costs GPU hours.
    """
    name = resolve(name)
    notes = []
    if name != "apptainer":
        return notes
    limits = dict(limits or {})
    for key in ("cpus", "memory", "pids_limit"):
        if limits.get(key):
            notes.append(f"apptainer: {key}={limits[key]} is not enforced "
                         f"(no per-container limits)")
    if network or limits.get("network"):
        notes.append("apptainer: containers share the host network, so "
                     "network=" + str(network or limits.get("network")) +
                     " is advisory")
    notes.append("apptainer: images are pulled as .sif into "
                 "$APPTAINER_CACHEDIR (set it to a large scratch filesystem)")
    return notes


def require(name, what="this run"):
    """Raise a message naming the missing binary, or return ``name``."""
    name = resolve(name)
    if name is None:
        raise RuntimeError_(
            f"{what} needs a container runtime: install docker or apptainer "
            f"(none of {', '.join(RUNTIMES)} is on PATH)")
    if not available(name):
        raise RuntimeError_(f"{what} needs the {name!r} binary, which is not on "
                            f"PATH")
    return name
