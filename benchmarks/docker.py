"""Docker inspection helpers: preflight, digests, and nothing that runs agents.

Container *execution* for SWE-bench is expressed as explicit ``docker run`` argv
in the suite (so ``plan`` can print it verbatim); this module only answers
questions about the daemon, which is what a preflight needs and what the manifest
records (image digests make a run reproducible even though images are mutable
tags).

Everything here degrades to "unknown" instead of raising: a benchmark run on a
machine without Docker is still allowed to *plan*, and the report is still
readable.
"""
import json
import shutil
import subprocess


def docker_available():
    """True when a ``docker`` binary exists and the daemon answers."""
    return docker_version() is not None


def docker_version(timeout=20):
    """The daemon's version string, or ``None`` (no binary / no daemon)."""
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def image_digest(image, timeout=30):
    """``sha256:...`` for a local image, or ``None`` when it is not present.

    Recorded in the manifest because the suites reference images by *tag*
    (``swebench/sweb.eval.x86_64...``, ``tb__task``): a tag can be rebuilt, so
    the digest is the only durable record of what actually ran.
    """
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}",
             image],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value.split("@")[-1] if "@" in value else (value or None)


def gpu_available(timeout=20):
    """Whether the host exposes an NVIDIA device to Docker (``--gpus all``)."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    try:
        runtimes = json.loads(result.stdout or "{}")
    except ValueError:
        return False
    return "nvidia" in {str(name).lower() for name in runtimes}
