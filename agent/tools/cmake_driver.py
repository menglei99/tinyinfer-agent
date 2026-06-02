"""Drive the C++ toolchain: cmake configure / build / ctest.

This module is intentionally minimal — Week 2 will lift it into an MCP server
so the Agent can call these tools over the standard MCP protocol.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> CommandResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def cmake_available() -> bool:
    return shutil.which("cmake") is not None


def configure(project_dir: Path, build_dir: Path) -> CommandResult:
    build_dir.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            "cmake",
            "-S", str(project_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )


def build(build_dir: Path, target: str | None = None) -> CommandResult:
    cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
    if target:
        cmd += ["--target", target]
    return _run(cmd)


def ctest(build_dir: Path, test_filter: str | None = None) -> CommandResult:
    cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
    if test_filter:
        cmd += ["-R", test_filter]
    return _run(cmd)
