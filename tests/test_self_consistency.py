"""Tests for self-consistency shape voting in NumericalSkill."""

import json

from agent.llm.client import LLMResponse
from agent.skills.numerical import NumericalSkill, _majority_vote_shapes


class ScriptedLLM:
    """Returns a different shape plan on each call, cycling through a list."""

    def __init__(self, plans: list[list[dict]]):
        self._plans = plans
        self._i = 0

    def complete(self, system, user, *, json_mode=False) -> LLMResponse:
        plan = self._plans[self._i % len(self._plans)]
        self._i += 1
        return LLMResponse(text=json.dumps({"shapes": plan}))


# ---------- _majority_vote_shapes unit ----------


def test_majority_vote_keeps_shapes_in_majority():
    # 3 draws. (2,2,2) appears 3x, (3,4,2) 2x, (9,9,9) once.
    shapes = [
        {"m": 2, "k": 2, "n": 2}, {"m": 3, "k": 4, "n": 2},
        {"m": 2, "k": 2, "n": 2}, {"m": 3, "k": 4, "n": 2}, {"m": 9, "k": 9, "n": 9},
        {"m": 2, "k": 2, "n": 2},
    ]
    out = _majority_vote_shapes(shapes, op_name="matmul_fp32", n_draws=3)
    keys = {(s["m"], s["k"], s["n"]) for s in out}
    assert (2, 2, 2) in keys      # 3 >= ceil(3/2)=2
    assert (3, 4, 2) in keys      # 2 >= 2
    assert (9, 9, 9) not in keys  # 1 < 2


def test_majority_vote_drops_outliers():
    shapes = [
        {"m": 8, "k": 8, "n": 8},
        {"m": 1, "k": 1, "n": 1},  # appears once across 4 draws
    ]
    out = _majority_vote_shapes(shapes, op_name="matmul_fp32", n_draws=4)
    # threshold = ceil(4/2) = 2; nothing reaches it.
    assert out == []


def test_majority_vote_ignores_malformed():
    shapes = [{"m": 2, "k": 2, "n": 2}, {"bad": "shape"}, {"m": 2, "k": 2, "n": 2}]
    out = _majority_vote_shapes(shapes, op_name="matmul_fp32", n_draws=2)
    assert len(out) == 1
    assert (out[0]["m"], out[0]["k"], out[0]["n"]) == (2, 2, 2)


# ---------- end-to-end through the skill ----------


def test_skill_without_self_consistency_uses_single_call(monkeypatch):
    monkeypatch.delenv("SELF_CONSISTENCY_N", raising=False)
    llm = ScriptedLLM([[{"m": 2, "k": 2, "n": 2}]])
    skill = NumericalSkill(llm=llm)
    skill._plan_shapes("matmul_fp32")
    assert llm._i == 1  # exactly one LLM call


def test_skill_with_self_consistency_draws_n_times(monkeypatch):
    monkeypatch.setenv("SELF_CONSISTENCY_N", "3")
    # All three draws agree on (2,2,2); draw-specific noise differs.
    llm = ScriptedLLM([
        [{"m": 2, "k": 2, "n": 2}, {"m": 5, "k": 5, "n": 5}],
        [{"m": 2, "k": 2, "n": 2}, {"m": 6, "k": 6, "n": 6}],
        [{"m": 2, "k": 2, "n": 2}, {"m": 7, "k": 7, "n": 7}],
    ])
    skill = NumericalSkill(llm=llm)
    survivors = skill._plan_shapes("matmul_fp32")
    assert llm._i == 3  # three draws
    keys = {(s["m"], s["k"], s["n"]) for s in survivors}
    assert (2, 2, 2) in keys
    # The per-draw noise shapes each appeared only once -> dropped.
    assert (5, 5, 5) not in keys
    assert (6, 6, 6) not in keys


def test_skill_self_consistency_invalid_env_defaults_off(monkeypatch):
    monkeypatch.setenv("SELF_CONSISTENCY_N", "not-a-number")
    skill = NumericalSkill()
    assert skill._sc_n == 0
