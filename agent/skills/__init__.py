from agent.skills.base import Skill
from agent.skills.memory import MemorySkill
from agent.skills.numerical import NumericalSkill
from agent.skills.perf import PerformanceSkill
from agent.state import SkillKind


def get_skill(kind: SkillKind, *, llm=None) -> Skill:
    if kind == SkillKind.NUMERICAL:
        return NumericalSkill(llm=llm)
    if kind == SkillKind.PERFORMANCE:
        return PerformanceSkill(llm=llm)
    if kind == SkillKind.MEMORY:
        return MemorySkill(llm=llm)
    raise NotImplementedError(f"Skill {kind} not implemented — see ROADMAP")


__all__ = ["Skill", "NumericalSkill", "PerformanceSkill", "MemorySkill", "get_skill"]
