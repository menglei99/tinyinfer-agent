"""驱动 C++ toolchain：cmake configure / build / ctest。

默认直接调宿主的 `cmake` / `ctest`。如果机器上没 C++ 编译器（比如这台开发
机），设上 TINYINFER_DOCKER_IMAGE，所有命令会被包成
`docker run --rm -v <project_root>:/work` 跑在那个 image 里。详见
docs/DOCKER_BUILDER.md。

模式切换只在本文件内做，caller（agent.mcp.server、LangGraph 的
execute_tests_node）不需要知道当前是哪种模式。
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


# ---------- 模式判断 ----------


def _docker_image() -> str | None:
    """当前生效的 docker image 名；None 表示 host 模式。"""
    v = (os.getenv("TINYINFER_DOCKER_IMAGE") or "").strip()
    return v or None


def _mount_root() -> Path:
    """要 bind-mount 到容器内 /work 的宿主目录。

    默认 = 已 resolve 的 tinyinfer/ 项目目录。可通过 TINYINFER_PROJECT_DIR
    覆盖（绝对路径或仓库相对路径都行）。
    """
    raw = os.getenv("TINYINFER_PROJECT_DIR") or "tinyinfer"
    return Path(raw).resolve()


# ---------- runner ----------


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> CommandResult:
    """宿主侧 subprocess.run + 文本捕获。

    `encoding="utf-8", errors="replace"` 防止 Windows 在子进程写 UTF-8
    （比如从 docker relay 出来的 cmake/ctest 输出）而宿主 code page 是 GBK
    时炸掉。
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
    """把宿主路径翻译成容器内 /work/<rel> 形式。

    两个参数都先 resolve，避免 symlink / Windows 大小写不敏感把 containment
    判断绊掉。
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
    # 容器内是 Linux，路径用正斜杠
    return "/work/" + rel.as_posix()


def _docker_run(
    cmd_inside: list[str], image: str, mount_root: Path, timeout: int = 600
) -> CommandResult:
    """在一次性 docker 容器里跑一条命令。"""
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
    """cmake（或选定的 docker image）能不能用。

    Host 模式：`shutil.which("cmake")`。
    Docker 模式：`docker image inspect <image>` 返回 0。
    """
    image = _docker_image()
    if image is None:
        return shutil.which("cmake") is not None
    if shutil.which("docker") is None:
        return False
    probe = _run(["docker", "image", "inspect", image])
    return probe.ok


def configure(project_dir: Path, build_dir: Path) -> CommandResult:
    """等价于 `cmake -S <project_dir> -B <build_dir> -DCMAKE_BUILD_TYPE=Release`。"""
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
    # 提前在宿主侧建 build dir，让 bind mount 立刻能看到（避免容器里 cmake
    # 和目录创建赛跑）。
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd_inside = [
        "cmake",
        "-S", _to_container_path(project_dir, mount),
        "-B", _to_container_path(build_dir, mount),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    return _docker_run(cmd_inside, image=image, mount_root=mount)


def build(build_dir: Path, target: str | None = None) -> CommandResult:
    """等价于 `cmake --build <build_dir> --config Release [--target <target>]`。"""
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
    # 首次 build 可能很慢（FetchContent 要 clone googletest）
    return _docker_run(cmd_inside, image=image, mount_root=mount, timeout=1200)


def ctest(build_dir: Path, test_filter: str | None = None) -> CommandResult:
    """等价于 `ctest --test-dir <build_dir> --output-on-failure [-R <filter>]`。"""
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
