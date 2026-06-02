"""Skill base class.

A Skill encapsulates a testing dimension (numerical / perf / memory). Each
skill owns:
  - the LLM prompt fragment that biases the planner
  - the test generation logic (renders C++ source)
  - the per-skill critic dimensions
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
        """Coverage dimensions the critic should verify for this skill."""
