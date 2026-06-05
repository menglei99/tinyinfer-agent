from agent.skills.base import Skill
from agent.skills.memory import MemorySkill
from agent.skills.numerical import NumericalSkill
from agent.skills.perf import PerformanceSkill
from agent.state import SkillKind


def get_skill(kind: SkillKind, *, llm=None, lessons=None) -> Skill:
    """Build a Skill instance.

    Args:
        kind: which skill to build.
        llm: optional LLM client (only NumericalSkill uses this today).
        lessons: optional list of human-readable Reflexion lesson strings
                 from earlier critic-loop iterations. Each skill ignores
                 lessons it can't act on.
    """
    if kind == SkillKind.NUMERICAL:
        return NumericalSkill(llm=llm, lessons=lessons)
    if kind == SkillKind.PERFORMANCE:
        return PerformanceSkill(llm=llm)
    if kind == SkillKind.MEMORY:
        return MemorySkill(llm=llm)
    raise NotImplementedError(f"Skill {kind} not implemented — see ROADMAP")


__all__ = ["Skill", "NumericalSkill", "PerformanceSkill", "MemorySkill", "get_skill"]
