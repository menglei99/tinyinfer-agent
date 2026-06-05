"""Reflexion: critic FAIL -> lesson -> next-iteration generator prompt."""

import json

import pytest

from agent.graph.nodes import (
    _build_reflexion_lesson,
    _reflexion_enabled,
    critic_node,
)
from agent.llm.client import LLMResponse
from agent.skills.numerical import NumericalSkill
from agent.state import CriticVerdict, SkillKind, TestCase


# ---------- env gate ----------


def test_reflexion_enabled_by_default(monkeypatch):
    monkeypatch.delenv("REFLEXION", raising=False)
    assert _reflexion_enabled() is True


@pytest.mark.parametrize("val", ["off", "false", "0", "no", "OFF", "False"])
def test_reflexion_disabled_explicitly(monkeypatch, val):
    monkeypatch.setenv("REFLEXION", val)
    assert _reflexion_enabled() is False


# ---------- lesson formatter ----------


def test_lesson_includes_missing_dimensions_and_feedback():
    v = CriticVerdict(
        passed=False,
        missing_dimensions=["square_2d", "k1_outer"],
        feedback="Add a square 2D shape (e.g. 4x4).",
    )
    s = _build_reflexion_lesson(2, v)
    assert s.startswith("Lesson #2:")
    assert "square_2d" in s and "k1_outer" in s
    assert "Add a square 2D shape" in s


def test_lesson_empty_when_nothing_to_say():
    v = CriticVerdict(passed=False, missing_dimensions=[], feedback="")
    assert _build_reflexion_lesson(1, v) == ""


# ---------- critic emits lesson on FAIL ----------


def test_critic_emits_lesson_on_structural_fail():
    """Single-test set fails the matmul structural critic (missing many dims).
    The critic node must include reflexion_lessons in its return value."""
    bad = [TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                    test_name="square_2x2", cpp_source="x")]
    out = critic_node({"generated_tests": bad, "critic_iterations": 0}, llm=None)
    assert out["critic_verdict"].passed is False
    lessons = out.get("reflexion_lessons")
    assert lessons and len(lessons) == 1
    assert lessons[0].startswith("Lesson #1:")


def test_critic_no_lesson_on_pass():
    full = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="square_2x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="rect_3x4_4x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="thin_1x8_8x4", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="stability_large_magnitude", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="outer_product_k1", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="aligned_16x32_32x64", cpp_source="x"),
    ]
    out = critic_node({"generated_tests": full, "critic_iterations": 0}, llm=None)
    assert out["critic_verdict"].passed is True
    # No reflexion entry on PASS.
    assert "reflexion_lessons" not in out


def test_critic_skips_lesson_when_reflexion_off(monkeypatch):
    monkeypatch.setenv("REFLEXION", "off")
    bad = [TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                    test_name="square_2x2", cpp_source="x")]
    out = critic_node({"generated_tests": bad, "critic_iterations": 0}, llm=None)
    assert out["critic_verdict"].passed is False
    assert "reflexion_lessons" not in out


# ---------- generator consumes lessons ----------


class CapturingLLM:
    """LLM stub that records the last user prompt and returns an empty plan."""

    def __init__(self):
        self.last_user = ""

    def complete(self, system, user, *, json_mode=False) -> LLMResponse:
        self.last_user = user
        return LLMResponse(text=json.dumps({"shapes": []}))


def test_planner_includes_lessons_when_provided():
    llm = CapturingLLM()
    skill = NumericalSkill(
        llm=llm,
        lessons=["Lesson #1: missing square_2d. Feedback: add a 4x4 case."],
    )
    skill._plan_shapes("matmul_fp32")
    assert "Lessons from prior iterations" in llm.last_user
    assert "Lesson #1" in llm.last_user
    assert "square_2d" in llm.last_user


def test_planner_omits_lessons_block_when_empty():
    llm = CapturingLLM()
    skill = NumericalSkill(llm=llm, lessons=None)
    skill._plan_shapes("matmul_fp32")
    assert "Lessons from prior iterations" not in llm.last_user
