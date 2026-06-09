"""Skill 基类。

一个 Skill 封装一个测试维度（numerical / perf / memory）。每个 skill 持有：
  - 偏置 planner 的 LLM prompt 片段
  - 测试生成逻辑（render C++ 源码）
  - 这个 skill 的 per-skill critic 维度
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.state import ChangedOp, SkillKind, TestCase


class Skill(ABC):
    kind: SkillKind
    activation_keywords: list[str] = []

    @abstractmethod
    def generate(self, op: ChangedOp) -> list[TestCase]: ...

    @abstractmethod
    def required_coverage_dimensions(self) -> list[str]:
        """critic 该针对这个 skill 验证的覆盖维度。"""
