"""Tests for retrieve_context_node."""

from agent.graph.nodes import retrieve_context_node
from agent.rag.store import Document, ScoredDoc
from agent.state import ChangedOp


def test_retrieve_context_with_no_retriever_returns_empty():
    out = retrieve_context_node({"changed_ops": [ChangedOp(name="matmul_fp32", file_path="x", summary="x")]})
    assert out == {"retrieved_docs": []}


def test_retrieve_context_without_changed_ops_returns_empty():
    class StubR:
        def retrieve(self, q, top_k=5):
            raise AssertionError("should not be called when no ops")
    out = retrieve_context_node({"changed_ops": []}, retriever=StubR())
    assert out == {"retrieved_docs": []}


def test_retrieve_context_passes_query_to_retriever():
    seen = {}

    class StubR:
        def retrieve(self, query, top_k=5):
            seen["q"] = query
            seen["k"] = top_k
            return [ScoredDoc(doc=Document(doc_id="d1", text="t1"), score=0.9)]

    state = {
        "changed_ops": [ChangedOp(name="softmax_fp32", file_path="x", summary="adds max-subtract")],
        "diff": "+++ b/x.cpp\n+// reorder loops for SIMD\n",
    }
    out = retrieve_context_node(state, retriever=StubR())
    assert "softmax_fp32" in seen["q"]
    assert "max-subtract" in seen["q"]
    assert seen["k"] == 5
    assert len(out["retrieved_docs"]) == 1
    assert out["retrieved_docs"][0]["doc"]["doc_id"] == "d1"
    assert out["retrieved_docs"][0]["score"] == 0.9


def test_retrieve_context_swallows_retriever_failure():
    class ExplodingR:
        def retrieve(self, q, top_k=5):
            raise RuntimeError("milvus down")

    state = {"changed_ops": [ChangedOp(name="matmul_fp32", file_path="x", summary="x")]}
    out = retrieve_context_node(state, retriever=ExplodingR())
    assert out["retrieved_docs"] == []
    # Pipeline should keep going; the error is recorded but not raised.
    assert "milvus down" in out["error"]
