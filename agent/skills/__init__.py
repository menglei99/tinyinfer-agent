from agent.skills.base import Skill
from agent.skills.memory import MemorySkill
from agent.skills.numerical import NumericalSkill
from agent.skills.perf import PerformanceSkill
from agent.state import SkillKind


def get_skill(kind: SkillKind, *, llm=None, lessons=None) -> Skill:
    """构造一个 Skill 实例。

    Args:
        kind: 要建哪个 skill。
        llm: 可选 LLM client（目前只有 NumericalSkill 用得到）。
        lessons: 可选的 human-readable Reflexion lesson 字符串列表，来自前几轮
                 critic loop 迭代。每个 skill 自己决定是否能用上 lesson，用不上
                 就忽略。
    """
    if kind == SkillKind.NUMERICAL:
        return NumericalSkill(llm=llm, lessons=lessons)
    if kind == SkillKind.PERFORMANCE:
        return PerformanceSkill(llm=llm)
    if kind == SkillKind.MEMORY:
        return MemorySkill(llm=llm)
    raise NotImplementedError(f"Skill {kind} not implemented — see ROADMAP")


__all__ = ["Skill", "NumericalSkill", "PerformanceSkill", "MemorySkill", "get_skill"]
