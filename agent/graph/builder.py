"""LangGraph StateGraph wiring.

Flow:
    parse_diff -> route_skill -> generate_tests -> critic
                                                    |
                                  (fail, iter < N) -+-> generate_tests
                                  (pass or iter>=N) -> install_tests
                                                    -> execute? (gated by state.execute_requested)
                                                    -> write_report -> END

execute_tests goes through the local MCP stdio server (agent.mcp.server) so
the same toolchain entry point is shared by Claude Desktop / Cursor / this agent.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import END, START, StateGraph

from agent.graph import nodes
from agent.llm import LLMClient
from agent.state import AgentState


_MAX_CRITIC_ITERS = 2


def _critic_router(state: AgentState) -> str:
    verdict = state.get("critic_verdict")
    iters = int(state.get("critic_iterations") or 0)

    if verdict is None:
        return "install"
    if verdict.passed:
        return "install"
    if iters >= _MAX_CRITIC_ITERS:
        return "install"
    return "regenerate"


def _execute_router(state: AgentState) -> str:
    return "execute" if state.get("execute_requested") else "skip"


def build_graph(llm: Optional[LLMClient] = None):
    sg: StateGraph = StateGraph(AgentState)

    sg.add_node("parse_diff", partial(nodes.parse_diff_node, llm=llm))
    sg.add_node("route_skill", nodes.route_skill_node)
    sg.add_node("generate_tests", partial(nodes.generate_tests_node, llm=llm))
    sg.add_node("critic", partial(nodes.critic_node, llm=llm))
    sg.add_node("install_tests", nodes.install_tests_node)
    sg.add_node("execute_tests", nodes.execute_tests_node)
    sg.add_node("write_report", nodes.write_report_node)

    sg.add_edge(START, "parse_diff")
    sg.add_edge("parse_diff", "route_skill")
    sg.add_edge("route_skill", "generate_tests")
    sg.add_edge("generate_tests", "critic")
    sg.add_conditional_edges(
        "critic",
        _critic_router,
        {"regenerate": "generate_tests", "install": "install_tests"},
    )
    sg.add_conditional_edges(
        "install_tests",
        _execute_router,
        {"execute": "execute_tests", "skip": "write_report"},
    )
    sg.add_edge("execute_tests", "write_report")
    sg.add_edge("write_report", END)

    return sg.compile()
