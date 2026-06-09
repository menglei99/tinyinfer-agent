"""Oracle 反馈式 reflexion：real oracle 报 missed 时，harness 让 generator 再跑一次。

这些测试 stub 掉 cmake_driver 和 graph，所以套件不需要 docker 也不需要真的
LLM。要验的是：harness 是否调了两次 generator？lesson 是否注入到第二次 attempt？
最终是否用第二次 attempt 的判决覆盖了第一次？
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.cmake_driver import CommandResult
from benchmark.harness import (
    FaultInjectionHarness,
    HarnessResult,
    ORACLE_MISS_LESSON,
)


_TINY_DIFF = """\
diff --git a/src.txt b/src.txt
--- a/src.txt
+++ b/src.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_mutated
 line3
"""


def _seed(tmp_path: Path) -> Path:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    seed = seeds_dir / "01_matmul_offbyone_inner.diff"
    seed.write_text(_TINY_DIFF, encoding="utf-8")
    return seed


def _project(tmp_path: Path) -> Path:
    """造一个含 diff 目标文件的 tmp 项目。"""
    (tmp_path / "src.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    return tmp_path


def _make_fake_graph(captured_initials: list[dict]):
    """记录每次跑被传入的 initial state 的 graph stub。

    返回一个 'square_2x2' matmul 测试，让 structural critic + heuristic 都能
    走通。我们不关心测试内容，只关心 graph 被跑了几次、第二次有没有带 lesson。
    """
    from agent.state import SkillKind, TestCase

    fake_tests = [
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
            test_name="square_2x2", cpp_source="// x", rationale="square",
        ),
    ]

    class StubGraph:
        def stream(self, initial, stream_mode="updates"):
            captured_initials.append(dict(initial))
            yield {"parse_diff": {"changed_ops": []}}
            yield {"generate_tests": {"generated_tests": fake_tests}}

    return StubGraph()


# ---------- env / constructor ----------


def test_oracle_reflexion_default_off(tmp_path):
    seed = _seed(tmp_path)
    h = FaultInjectionHarness(seed.parent, oracle="real")
    assert h.oracle_reflexion is False


def test_oracle_reflexion_env_on(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_REFLEXION", "on")
    h = FaultInjectionHarness(tmp_path, oracle="real")
    assert h.oracle_reflexion is True


def test_oracle_reflexion_constructor_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_REFLEXION", "off")
    h = FaultInjectionHarness(tmp_path, oracle="real", oracle_reflexion=True)
    assert h.oracle_reflexion is True


# ---------- behaviour ----------


def _stub_prepare(harness: FaultInjectionHarness) -> None:
    """跳过一次性 cmake configure+build 准备。"""
    harness._real_prepared = True


def test_reflexion_off_does_not_re_run_on_miss(monkeypatch, tmp_path):
    """标准 real oracle：missed -> 一次 attempt，没有第二轮。"""
    seed = _seed(tmp_path)
    project = _project(tmp_path)
    captured: list[dict] = []

    h = FaultInjectionHarness(
        seed.parent, oracle="real", oracle_reflexion=False,
        project_dir=project, build_dir=tmp_path / "build",
    )
    h._graph = _make_fake_graph(captured)
    _stub_prepare(h)

    # Oracle：build ok，ctest 全过（= missed）
    build_ok = CommandResult(cmd=["cmake"], returncode=0, stdout="", stderr="")
    ctest_pass = CommandResult(
        cmd=["ctest"], returncode=0,
        stdout="1/1 Test #1: test_matmul_fp32_generated .. Passed 0.01 sec\n",
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", return_value=ctest_pass):
        result = h.run_one(seed)

    assert result.fault_likely_caught is False
    assert result.attempts == 1
    assert result.oracle_reflexion_used is False
    assert len(captured) == 1  # graph 恰好跑了一次


def test_reflexion_on_retries_and_can_catch(monkeypatch, tmp_path):
    """oracle_reflexion=True + 第一次 miss + 第二次 catch -> caught + attempts=2。"""
    seed = _seed(tmp_path)
    project = _project(tmp_path)
    captured: list[dict] = []

    h = FaultInjectionHarness(
        seed.parent, oracle="real", oracle_reflexion=True,
        project_dir=project, build_dir=tmp_path / "build",
    )
    h._graph = _make_fake_graph(captured)
    _stub_prepare(h)

    build_ok = CommandResult(cmd=["cmake"], returncode=0, stdout="", stderr="")
    pass_then_fail = [
        # attempt 1：ctest 全过 -> missed
        CommandResult(cmd=["ctest"], returncode=0,
                      stdout="1/1 Test #1: test_matmul_fp32_generated .. Passed 0.01 sec\n",
                      stderr=""),
        # attempt 2：ctest 有失败 -> caught
        CommandResult(cmd=["ctest"], returncode=8,
                      stdout="1/1 Test #1: test_matmul_fp32_generated ..***Failed 0.02 sec\n",
                      stderr=""),
    ]
    ctest_call_idx = {"i": 0}

    def fake_ctest(*args, **kwargs):
        r = pass_then_fail[ctest_call_idx["i"]]
        ctest_call_idx["i"] += 1
        return r

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", side_effect=fake_ctest):
        result = h.run_one(seed)

    assert result.fault_likely_caught is True
    assert result.attempts == 2
    assert result.oracle_reflexion_used is True
    assert "test_matmul_fp32_generated" in result.failed_test_names
    assert "oracle-reflexion attempt 2" in result.detail

    # graph 跑了两次；第二次拿到了 lesson
    assert len(captured) == 2
    assert "reflexion_lessons" not in captured[0]
    assert captured[1].get("reflexion_lessons") == [ORACLE_MISS_LESSON]


def test_reflexion_on_records_persistent_miss(monkeypatch, tmp_path):
    """oracle_reflexion=True + 两次都 miss -> not caught，attempts=2。"""
    seed = _seed(tmp_path)
    project = _project(tmp_path)
    captured: list[dict] = []

    h = FaultInjectionHarness(
        seed.parent, oracle="real", oracle_reflexion=True,
        project_dir=project, build_dir=tmp_path / "build",
    )
    h._graph = _make_fake_graph(captured)
    _stub_prepare(h)

    build_ok = CommandResult(cmd=["cmake"], returncode=0, stdout="", stderr="")
    ctest_pass = CommandResult(
        cmd=["ctest"], returncode=0,
        stdout="1/1 Test #1: test_matmul_fp32_generated .. Passed 0.01 sec\n",
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", return_value=ctest_pass):
        result = h.run_one(seed)

    assert result.fault_likely_caught is False
    assert result.attempts == 2
    assert result.oracle_reflexion_used is True
    assert "still missed" in result.detail
    assert len(captured) == 2


def test_reflexion_skips_when_first_attempt_already_caught(tmp_path):
    """第一次就抓到，不再 re-run —— 省一次 graph + 一次 ctest。"""
    seed = _seed(tmp_path)
    project = _project(tmp_path)
    captured: list[dict] = []

    h = FaultInjectionHarness(
        seed.parent, oracle="real", oracle_reflexion=True,
        project_dir=project, build_dir=tmp_path / "build",
    )
    h._graph = _make_fake_graph(captured)
    _stub_prepare(h)

    build_ok = CommandResult(cmd=["cmake"], returncode=0, stdout="", stderr="")
    ctest_fail = CommandResult(
        cmd=["ctest"], returncode=8,
        stdout="1/1 Test #1: test_matmul_fp32_generated ..***Failed 0.02 sec\n",
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", return_value=ctest_fail):
        result = h.run_one(seed)

    assert result.fault_likely_caught is True
    assert result.attempts == 1
    assert result.oracle_reflexion_used is False
    assert len(captured) == 1


def test_reflexion_skips_when_build_failed(tmp_path):
    """Build 失败不是漏抓 fault，是 oracle infra 问题。重跑也没用，
    所以跳过第二轮。"""
    seed = _seed(tmp_path)
    project = _project(tmp_path)
    captured: list[dict] = []

    h = FaultInjectionHarness(
        seed.parent, oracle="real", oracle_reflexion=True,
        project_dir=project, build_dir=tmp_path / "build",
    )
    h._graph = _make_fake_graph(captured)
    _stub_prepare(h)

    build_fail = CommandResult(
        cmd=["cmake"], returncode=2, stdout="", stderr="undefined reference",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_fail), \
         patch("agent.tools.cmake_driver.ctest") as ctest_mock:
        result = h.run_one(seed)
        # 两次 attempt 都不该跑 ctest
        ctest_mock.assert_not_called()

    assert result.attempts == 1
    assert result.oracle_reflexion_used is False
    assert result.build_status == "build_failed"
    assert len(captured) == 1


def test_oracle_miss_lesson_does_not_reveal_diff():
    """lesson 不能把 seed diff 的具体内容回灌——那就是给 LLM 抄答案。"""
    # 不应该含 diff 的 `+`/`-` 行
    head = ORACLE_MISS_LESSON.split("Oracle 反馈")[0]
    assert "+" not in head and "-" not in head
    # 提示的是通用维度类别，不是具体源码：
    text = ORACLE_MISS_LESSON.lower()
    assert "output buffer" in text or "buffer 初值" in ORACLE_MISS_LESSON
    assert "alias" in text
    assert "shape" in text
