"""LangGraph agent state 的 Pydantic 模型和 TypedDict。

state 是 graph 节点之间流转的唯一真值来源。保持小、可序列化、不要塞运行时
对象（client、socket 这类）进去。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SkillKind(str, Enum):
    NUMERICAL = "numerical"
    PERFORMANCE = "performance"  # TODO Week 2
    MEMORY = "memory"  # TODO Week 2


class ChangedOp(BaseModel):
    """diff 里识别出的一个算子。"""

    name: str = Field(..., description="算子名，例如 'matmul_fp32'")
    file_path: str = Field(..., description="改动所在的源文件路径")
    summary: str = Field("", description="一句话描述改了啥")


class TestCase(BaseModel):
    """一条生成出来的回归测试 case。"""

    op_name: str
    skill: SkillKind
    test_name: str
    cpp_source: str = Field(..., description="完整的 GTest .cpp 文件内容")
    rationale: str = Field("", description="为啥要这个 case（shape/dtype/edge 等）")
    inputs: dict = Field(default_factory=dict, description="算 reference 用到的输入")
    expected_outputs: dict = Field(default_factory=dict)


class CriticVerdict(BaseModel):
    passed: bool
    missing_dimensions: list[str] = Field(default_factory=list)
    feedback: str = ""
    coverage_report: dict[str, Any] = Field(
        default_factory=dict,
        description="开放 schema 的 dict，存 per-skill / per-dimension 的 critic 细节。",
    )


class ExecutionResult(BaseModel):
    """编译 / 跑生成测试的结果（MVP 里可选）。"""

    test_name: str
    compiled: bool = False
    ran: bool = False
    passed: bool = False
    stdout: str = ""
    stderr: str = ""


def _merge_list(left: list, right: list) -> list:
    """state 里 list 字段的 reducer —— concat 而不是 overwrite。"""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """LangGraph state。每个字段都 optional 方便增量更新。"""

    # 输入
    diff: str
    diff_path: str

    # parse_diff
    changed_ops: list[ChangedOp]

    # route_skill
    selected_skills: list[SkillKind]

    # retrieve_context（RAG 命中 —— 用 plain dict 让 state 保持 JSON-safe）
    retrieved_docs: list[dict]

    # generate_tests
    generated_tests: Annotated[list[TestCase], _merge_list]

    # critic
    critic_verdict: Optional[CriticVerdict]
    critic_iterations: int

    # reflexion（每次 critic FAIL 都加一条 human-readable lesson；
    # generator 下一轮会读这些 lesson 来偏置 shape 提议）。
    reflexion_lessons: Annotated[list[str], _merge_list]

    # install_tests（把生成的 cpp 拷到 tinyinfer/tests/）
    installed_test_paths: Annotated[list[str], _merge_list]

    # execute
    execution_results: Annotated[list[ExecutionResult], _merge_list]

    # execute 开关（关掉时 graph 可以短路）
    execute_requested: bool

    # report
    report_markdown: str

    # 记账
    trace_id: str
    error: str
