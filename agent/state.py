"""Pydantic models and TypedDict for the LangGraph agent state.

The state is the single source of truth that flows between graph nodes.
Keep it small, serializable, and free of runtime objects (clients, sockets).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SkillKind(str, Enum):
    NUMERICAL = "numerical"
    PERFORMANCE = "performance"  # TODO Week 2
    MEMORY = "memory"  # TODO Week 2


class ChangedOp(BaseModel):
    """One operator detected in the diff."""

    name: str = Field(..., description="Operator name, e.g. 'matmul_fp32'")
    file_path: str = Field(..., description="Source file path containing the change")
    summary: str = Field("", description="One-line description of what changed")


class TestCase(BaseModel):
    """A single generated regression test case."""

    op_name: str
    skill: SkillKind
    test_name: str
    cpp_source: str = Field(..., description="Full GTest .cpp content")
    rationale: str = Field("", description="Why this case (shape/dtype/edge etc.)")
    inputs: dict = Field(default_factory=dict, description="Inputs used to compute reference")
    expected_outputs: dict = Field(default_factory=dict)


class CriticVerdict(BaseModel):
    passed: bool
    missing_dimensions: list[str] = Field(default_factory=list)
    feedback: str = ""


class ExecutionResult(BaseModel):
    """Result of compiling/running a generated test (optional in MVP)."""

    test_name: str
    compiled: bool = False
    ran: bool = False
    passed: bool = False
    stdout: str = ""
    stderr: str = ""


def _merge_list(left: list, right: list) -> list:
    """Reducer for state list fields — concat instead of overwrite."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """LangGraph state. Each field is optional to ease incremental updates."""

    # Inputs
    diff: str
    diff_path: str

    # parse_diff
    changed_ops: list[ChangedOp]

    # route_skill
    selected_skills: list[SkillKind]

    # generate_tests
    generated_tests: Annotated[list[TestCase], _merge_list]

    # critic
    critic_verdict: Optional[CriticVerdict]
    critic_iterations: int

    # install_tests (copy generated cpp into tinyinfer/tests/)
    installed_test_paths: Annotated[list[str], _merge_list]

    # execute
    execution_results: Annotated[list[ExecutionResult], _merge_list]

    # execute toggle (so the graph can short-circuit when execution is off)
    execute_requested: bool

    # report
    report_markdown: str

    # bookkeeping
    trace_id: str
    error: str
