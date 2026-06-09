"""端到端 smoke test：mock 模式下用一份样例 diff 跑完整张 graph。"""

import os
from pathlib import Path

from agent.graph import build_graph
from agent.llm import get_llm_client
from agent.state import AgentState


def test_graph_runs_end_to_end_on_sample_diff(tmp_path):
    os.environ["LLM_PROVIDER"] = "mock"
    diff_path = Path(__file__).parent.parent / "demo" / "diffs" / "sample_matmul.diff"
    diff_text = diff_path.read_text(encoding="utf-8")

    llm = get_llm_client()
    graph = build_graph(llm=llm)

    initial: AgentState = {
        "diff": diff_text,
        "diff_path": str(diff_path),
        "trace_id": "test",
        "critic_iterations": 0,
    }

    final_state: dict = {}
    for chunk in graph.stream(initial, stream_mode="updates"):
        for _name, update in chunk.items():
            for k, v in update.items():
                if isinstance(v, list) and isinstance(final_state.get(k), list):
                    final_state[k] = final_state[k] + v
                else:
                    final_state[k] = v

    assert final_state.get("changed_ops"), "diff parser should detect at least one op"
    assert final_state.get("generated_tests"), "should generate at least one test"

    cpp_sources = {t.cpp_source for t in final_state["generated_tests"]}
    assert any("TEST(MatmulGenerated" in s for s in cpp_sources)
    assert any("EXPECT_NEAR" in s for s in cpp_sources)

    report = final_state.get("report_markdown", "")
    assert "tinyinfer-agent" in report
    assert "matmul_fp32" in report
