"""Tests for the docker-mode switch in cmake_driver.

We mock subprocess.run so the unit tests never need a real docker daemon
or compiler. The goal is to lock in command shape, mount layout, and path
translation; the actual "does docker build work" verification is the
manual sanity step in docs/DOCKER_BUILDER.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools import cmake_driver
from agent.tools.cmake_driver import CommandResult


def _ok_result(cmd: list[str], stdout: str = "") -> CommandResult:
    return CommandResult(cmd=cmd, returncode=0, stdout=stdout, stderr="")


# ---------- mode detection ----------


def test_docker_image_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("TINYINFER_DOCKER_IMAGE", raising=False)
    assert cmake_driver._docker_image() is None


def test_docker_image_returns_none_for_empty_string(monkeypatch):
    """Empty string env var should not flip mode — it usually means "I tried
    to disable this but didn't unset it." Treat as host mode."""
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "")
    assert cmake_driver._docker_image() is None


def test_docker_image_strips_whitespace(monkeypatch):
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "  tinyinfer-builder  ")
    assert cmake_driver._docker_image() == "tinyinfer-builder"


# ---------- path translation ----------


def test_to_container_path_root(tmp_path):
    assert cmake_driver._to_container_path(tmp_path, tmp_path) == "/work"


def test_to_container_path_subdir(tmp_path):
    sub = tmp_path / "build-docker"
    sub.mkdir()
    assert cmake_driver._to_container_path(sub, tmp_path) == "/work/build-docker"


def test_to_container_path_rejects_outside(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError, match="outside the docker mount root"):
        cmake_driver._to_container_path(outside, tmp_path)


# ---------- host-mode commands unchanged ----------


def test_host_mode_configure_uses_plain_cmake(monkeypatch, tmp_path):
    monkeypatch.delenv("TINYINFER_DOCKER_IMAGE", raising=False)
    proj = tmp_path / "tinyinfer"
    proj.mkdir()
    bld = tmp_path / "build"

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok_result(cmd)

    with patch.object(cmake_driver, "_run", side_effect=fake_run):
        cmake_driver.configure(proj, bld)

    assert captured["cmd"][0] == "cmake"
    assert "-S" in captured["cmd"]
    assert "docker" not in captured["cmd"][0]


def test_host_mode_ctest_uses_plain_ctest(monkeypatch, tmp_path):
    monkeypatch.delenv("TINYINFER_DOCKER_IMAGE", raising=False)
    bld = tmp_path / "build"

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok_result(cmd)

    with patch.object(cmake_driver, "_run", side_effect=fake_run):
        cmake_driver.ctest(bld, test_filter="matmul")

    assert captured["cmd"][0] == "ctest"
    assert "-R" in captured["cmd"]
    assert "matmul" in captured["cmd"]


# ---------- docker-mode command assembly ----------


def _project_layout(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "tinyinfer"
    project.mkdir()
    build = project / "build-docker"
    return project, build


def test_docker_mode_configure_wraps_in_docker_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "tinyinfer-builder")
    monkeypatch.setenv("TINYINFER_PROJECT_DIR", str(tmp_path / "tinyinfer"))
    project, build = _project_layout(tmp_path)

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok_result(cmd)

    with patch.object(cmake_driver, "_run", side_effect=fake_run):
        cmake_driver.configure(project, build)

    cmd = captured["cmd"]
    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    # mount: <host_project_dir>:/work
    mount_idx = cmd.index("-v") + 1
    assert cmd[mount_idx].endswith(":/work")
    # working directory must be /work
    assert cmd[cmd.index("-w") + 1] == "/work"
    # image follows the options
    assert "tinyinfer-builder" in cmd
    # cmake inside the container, container-rooted paths
    assert "cmake" in cmd
    src_idx = cmd.index("-S") + 1
    bld_idx = cmd.index("-B") + 1
    assert cmd[src_idx] == "/work"
    assert cmd[bld_idx] == "/work/build-docker"
    assert "-DCMAKE_BUILD_TYPE=Release" in cmd


def test_docker_mode_build_with_target(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "tinyinfer-builder")
    monkeypatch.setenv("TINYINFER_PROJECT_DIR", str(tmp_path / "tinyinfer"))
    _, build = _project_layout(tmp_path)

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok_result(cmd)

    with patch.object(cmake_driver, "_run", side_effect=fake_run):
        cmake_driver.build(build, target="test_matmul_baseline")

    cmd = captured["cmd"]
    assert cmd[0] == "docker"
    assert "cmake" in cmd
    assert "--build" in cmd
    assert cmd[cmd.index("--build") + 1] == "/work/build-docker"
    assert "--target" in cmd
    assert "test_matmul_baseline" in cmd


def test_docker_mode_ctest_with_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "tinyinfer-builder")
    monkeypatch.setenv("TINYINFER_PROJECT_DIR", str(tmp_path / "tinyinfer"))
    _, build = _project_layout(tmp_path)

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok_result(cmd)

    with patch.object(cmake_driver, "_run", side_effect=fake_run):
        cmake_driver.ctest(build, test_filter="softmax")

    cmd = captured["cmd"]
    assert cmd[0] == "docker"
    assert "ctest" in cmd
    assert "--test-dir" in cmd
    assert cmd[cmd.index("--test-dir") + 1] == "/work/build-docker"
    assert "--output-on-failure" in cmd
    assert "-R" in cmd
    assert "softmax" in cmd


# ---------- cmake_available ----------


def test_cmake_available_host_mode_checks_path(monkeypatch):
    monkeypatch.delenv("TINYINFER_DOCKER_IMAGE", raising=False)
    with patch("agent.tools.cmake_driver.shutil.which", return_value="/usr/bin/cmake"):
        assert cmake_driver.cmake_available() is True
    with patch("agent.tools.cmake_driver.shutil.which", return_value=None):
        assert cmake_driver.cmake_available() is False


def test_cmake_available_docker_mode_inspects_image(monkeypatch):
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "tinyinfer-builder")

    with patch("agent.tools.cmake_driver.shutil.which", return_value="/usr/bin/docker"):
        with patch.object(
            cmake_driver, "_run",
            return_value=_ok_result(["docker", "image", "inspect", "tinyinfer-builder"]),
        ):
            assert cmake_driver.cmake_available() is True

        # image missing -> non-zero returncode
        bad = CommandResult(cmd=["docker", "image", "inspect"], returncode=1,
                            stdout="", stderr="No such image")
        with patch.object(cmake_driver, "_run", return_value=bad):
            assert cmake_driver.cmake_available() is False


def test_cmake_available_docker_mode_returns_false_without_docker(monkeypatch):
    """Env asks for docker, but the docker CLI isn't installed at all."""
    monkeypatch.setenv("TINYINFER_DOCKER_IMAGE", "tinyinfer-builder")
    with patch("agent.tools.cmake_driver.shutil.which", return_value=None):
        assert cmake_driver.cmake_available() is False
