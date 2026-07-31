"""Harbor installed-agent adapter for the current Mini Code CLI checkout."""

from __future__ import annotations

import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE_PATH = "/installed-agent/mini_code_cli.tar.gz"
_UV_ENV = {
    "UV_TOOL_BIN_DIR": "/usr/local/bin",
    "UV_TOOL_DIR": "/opt/uv-tools",
}


def create_source_archive(repository_root: Path, destination: Path) -> Path:
    """Create an installable source archive containing only project inputs."""
    repository_root = repository_root.resolve()
    required_files = [
        repository_root / "pyproject.toml",
        repository_root / "README.md",
        repository_root / "LICENSE",
    ]
    missing = [path.name for path in required_files if not path.is_file()]
    package_root = repository_root / "mini_code_cli"
    if not package_root.is_dir():
        missing.append("mini_code_cli/")
    if missing:
        raise FileNotFoundError(
            f"Cannot build Mini Code source archive; missing: {', '.join(missing)}"
        )

    version_match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        (repository_root / "pyproject.toml").read_text(encoding="utf-8"),
    )
    if version_match is None:
        raise ValueError("Could not determine project version from pyproject.toml")
    archive_root = f"mini_code_cli-{version_match.group(1)}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w:gz") as archive:
        for path in required_files:
            archive.add(path, arcname=f"{archive_root}/{path.name}")
        for path in sorted(package_root.rglob("*.py")):
            relative_path = path.relative_to(repository_root)
            archive.add(path, arcname=f"{archive_root}/{relative_path}")
    return destination


class MiniCodeAgent(BaseInstalledAgent):
    """Install and run the checked-out Mini Code CLI inside a Harbor task."""

    def __init__(self, *args, base_url: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url

    @staticmethod
    @override
    def name() -> str:
        return "mini-code"

    @override
    def get_version_command(self) -> str | None:
        return (
            "UV_TOOL_BIN_DIR=/usr/local/bin UV_TOOL_DIR=/opt/uv-tools "
            "uv tool list"
        )

    @override
    def parse_version(self, stdout: str) -> str:
        match = re.search(r"(?m)^mini-code-cli\s+v?(\S+)", stdout)
        return match.group(1) if match else stdout.strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        with tempfile.TemporaryDirectory(prefix="mini-code-harbor-") as temp_dir:
            archive_path = create_source_archive(
                _REPOSITORY_ROOT,
                Path(temp_dir) / "mini_code_cli.tar.gz",
            )
            await environment.upload_file(archive_path, _ARCHIVE_PATH)

        bootstrap_command = (
            "set -euo pipefail; "
            "if ! command -v curl >/dev/null 2>&1; then "
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache curl ca-certificates; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y curl ca-certificates; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum install -y curl ca-certificates; "
            "else echo 'No supported package manager found to install curl' >&2; "
            "exit 1; fi; fi; "
            "if ! command -v uv >/dev/null 2>&1; then "
            "curl -LsSf https://astral.sh/uv/install.sh "
            "| env UV_INSTALL_DIR=/usr/local/bin sh; "
            "fi; "
            f"uv tool install --force {_ARCHIVE_PATH}; "
            "uv tool list"
        )
        await self.exec_as_root(
            environment,
            command=bootstrap_command,
            env={**_UV_ENV, "PYTHONUNBUFFERED": "1"},
        )

    def _build_run_command(self, instruction: str) -> str:
        if not self.model_name:
            raise ValueError("Harbor must provide a model with --model")
        if not self.base_url:
            raise ValueError(
                "Harbor must provide an OpenAI-compatible endpoint with "
                "--ak base_url=<url>"
            )

        return shlex.join(
            [
                "mini-code",
                "--auto-mode",
                "--enable-shell",
                "--exit-after-run",
                "--cache-dir",
                "/logs/agent",
                "--model",
                self.model_name,
                "--url",
                self.base_url,
                "--prompt",
                instruction,
            ]
        )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self.exec_as_agent(
            environment,
            command=self._build_run_command(instruction),
            env={"PYTHONUNBUFFERED": "1"},
            cwd=environment.task_env_config.workdir,
        )
