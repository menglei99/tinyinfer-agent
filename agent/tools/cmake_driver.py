"""Drive the C++ toolchain: cmake configure / build / ctest.

By default we shell out to the host `cmake` / `ctest`. On machines that
lack a C++ compiler (this dev box, for example), set TINYINFER_DOCKER_IMAGE
to route every command through `docker run --rm -v <project_root>:/work`
against that image. See docs/DOCKER_BUILDER.md.

The switch is contained here so callers (agent.mcp.server, the LangGraph
execute_tests_node) don't have to know which mode is active.
"""

from __future__ import annotations

import os
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


# ---------- mode detection ----------


def _docker_image() -> str | None:
    """Active docker image name, or None for host-mode."""
    v = (os.getenv("TINYINFER_DOCKER_IMAGE") or "").strip()
    return v or None


def _mount_root() -> Path:
    """Host directory that gets bind-mounted to /work inside the container.

    Defaults to the resolved tinyinfer/ project dir. Override with
    TINYINFER_PROJECT_DIR (an absolute or repo-relative path).
    """
    raw = os.getenv("TINYINFER_PROJECT_DIR") or "tinyinfer"
    return Path(raw).resolve()


# ---------- runners ----------


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> CommandResult:
    """Host-side subprocess.run with text capture.

    `encoding="utf-8", errors="replace"` keeps Windows from blowing up when
    the child writes UTF-8 (e.g. cmake/ctest output relayed from docker)
    while the host code page is GBK.
    """
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )


def _to_container_path(host_path: Path, mount_root: Path) -> str:
    """Translate a host path to its /work/<rel> equivalent inside the container.

    Both args are resolved before comparison so symlinks / case-insensitive
    Windows paths don't trip the containment check.
    """
    host_abs = host_path.resolve()
    root_abs = mount_root.resolve()
    if host_abs == root_abs:
        return "/work"
    try:
        rel = host_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(
            f"path {host_abs} is outside the docker mount root {root_abs}; "
            f"set TINYINFER_PROJECT_DIR to a directory that contains all "
            f"cmake/build paths"
        ) from exc
    # Force forward slashes for the container (Linux).
    return "/work/" + rel.as_posix()


def _docker_run(
    cmd_inside: list[str], image: str, mount_root: Path, timeout: int = 600
) -> CommandResult:
    """Run a command inside a one-shot docker container."""
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount_root.as_posix()}:/work",
        "-w",
        "/work",
        image,
        *cmd_inside,
    ]
    return _run(docker_cmd, timeout=timeout)


# ---------- public API ----------


def cmake_available() -> bool:
    """True when cmake (or our chosen docker image) is reachable.

    Host mode: `shutil.which("cmake")`.
    Docker mode: `docker image inspect <image>` returns 0.
    """
    image = _docker_image()
    if image is None:
        return shutil.which("cmake") is not None
    if shutil.which("docker") is None:
        return False
    probe = _run(["docker", "image", "inspect", image])
    return probe.ok


def configure(project_dir: Path, build_dir: Path) -> CommandResult:
    """Equivalent to `cmake -S <project_dir> -B <build_dir> -DCMAKE_BUILD_TYPE=Release`."""
    image = _docker_image()
    if image is None:
        build_dir.mkdir(parents=True, exist_ok=True)
        return _run(
            [
                "cmake",
                "-S", str(project_dir),
                "-B", str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )

    mount = _mount_root()
    # Best-effort: create the build dir on host so the bind mount sees it
    # immediately (avoids cmake racing with directory creation in the container).
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd_inside = [
        "cmake",
        "-S", _to_container_path(project_dir, mount),
        "-B", _to_container_path(build_dir, mount),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    return _docker_run(cmd_inside, image=image, mount_root=mount)


def build(build_dir: Path, target: str | None = None) -> CommandResult:
    """Equivalent to `cmake --build <build_dir> --config Release [--target <target>]`."""
    image = _docker_image()
    if image is None:
        cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
        if target:
            cmd += ["--target", target]
        return _run(cmd)

    mount = _mount_root()
    cmd_inside = [
        "cmake", "--build", _to_container_path(build_dir, mount),
        "--config", "Release",
    ]
    if target:
        cmd_inside += ["--target", target]
    # Build can be slow on first FetchContent (clones googletest).
    return _docker_run(cmd_inside, image=image, mount_root=mount, timeout=1200)


def ctest(build_dir: Path, test_filter: str | None = None) -> CommandResult:
    """Equivalent to `ctest --test-dir <build_dir> --output-on-failure [-R <filter>]`."""
    image = _docker_image()
    if image is None:
        cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
        if test_filter:
            cmd += ["-R", test_filter]
        return _run(cmd)

    mount = _mount_root()
    cmd_inside = [
        "ctest", "--test-dir", _to_container_path(build_dir, mount),
        "--output-on-failure",
    ]
    if test_filter:
        cmd_inside += ["-R", test_filter]
    return _docker_run(cmd_inside, image=image, mount_root=mount)
