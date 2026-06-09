"""hybrid retriever、corpus 和 embedding cache 的测试。"""

from pathlib import Path

import pytest

from agent.llm.embedder import MockEmbeddingClient
from agent.rag.cache import cached_embed
from agent.rag.corpus import load_default_corpus
from agent.rag.retriever import HybridRetriever, build_default_retriever
from agent.rag.store import Document, NumpyVectorStore


# ---------- corpus ----------


def test_corpus_includes_handwritten_sources():
    corpus = load_default_corpus()
    sources = {d.metadata.get("source") for d in corpus}
    # Hand-written sources are guaranteed; src_comment depends on what's in the
    # in-tree cpp files (currently very few comments) — so we don't require it.
    assert "op_semantics" in sources
    assert "fault_rationale" in sources


def test_corpus_is_ascii_only():
    corpus = load_default_corpus()
    for d in corpus:
        # If a maintainer adds non-ASCII content to a hand-written entry, BM25
        # tokenization will silently drop terms — fail loudly here instead.
        d.text.encode("ascii")


# ---------- cache ----------


def test_cache_avoids_redundant_calls(tmp_path):
    class CountingEmbedder:
        _model = "test-counter"
        def __init__(self):
            self.calls = 0
        def embed(self, texts):
            self.calls += 1
            return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    e = CountingEmbedder()
    cache_dir = tmp_path / "ec"
    v1 = cached_embed(e, ["alpha", "beta"], cache_dir=cache_dir)
    v2 = cached_embed(e, ["alpha", "beta"], cache_dir=cache_dir)
    # Second call must read from disk: embedder.embed not called again.
    assert e.calls == 1
    assert v1 == v2


def test_cache_partial_hit(tmp_path):
    class CountingEmbedder:
        _model = "test-counter-2"
        def __init__(self):
            self.calls = []
        def embed(self, texts):
            self.calls.append(list(texts))
            return [[float(len(t))] * 4 for t in texts]

    e = CountingEmbedder()
    cache_dir = tmp_path / "ec"
    cached_embed(e, ["alpha"], cache_dir=cache_dir)
    cached_embed(e, ["alpha", "beta"], cache_dir=cache_dir)
    # Second invocation should only ask for "beta".
    assert e.calls[1] == ["beta"]


# ---------- retriever ----------


@pytest.fixture
def small_corpus():
    return [
        Document(doc_id="op.matmul.gemm", text="matmul row-major GEMM accumulator k loop"),
        Document(doc_id="op.softmax.def", text="softmax exp max subtract for stability"),
        Document(doc_id="op.layernorm.eps", text="layernorm divide by sqrt(var + eps)"),
        Document(doc_id="op.conv.stride", text="conv2d kernel stride padding output dims"),
        Document(doc_id="fault.matmul.offbyone", text="matmul off-by-one inner k loop bound"),
    ]


def test_retriever_returns_top_k(small_corpus, tmp_path):
    r = HybridRetriever(
        store=NumpyVectorStore(),
        embedder=MockEmbeddingClient(dim=64, seed=1),
        corpus=small_corpus,
        use_cache=False,
    )
    hits = r.retrieve("matmul GEMM accumulator", top_k=2)
    assert len(hits) == 2
    # The matmul docs should rank in the top 2 — BM25 alone guarantees it.
    ids = {h.doc.doc_id for h in hits}
    assert "op.matmul.gemm" in ids


def test_retriever_handles_empty_query(small_corpus):
    r = HybridRetriever(
        store=NumpyVectorStore(),
        embedder=MockEmbeddingClient(dim=64, seed=1),
        corpus=small_corpus,
        use_cache=False,
    )
    assert r.retrieve("", top_k=3) == []


def test_retriever_handles_empty_corpus():
    r = HybridRetriever(
        store=NumpyVectorStore(),
        embedder=MockEmbeddingClient(dim=64, seed=1),
        corpus=[],
        use_cache=False,
    )
    assert r.retrieve("anything", top_k=3) == []


def test_retriever_score_is_descending(small_corpus):
    r = HybridRetriever(
        store=NumpyVectorStore(),
        embedder=MockEmbeddingClient(dim=64, seed=1),
        corpus=small_corpus,
        use_cache=False,
    )
    hits = r.retrieve("softmax stability", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retriever_bm25_only_weights(small_corpus):
    """bm25_weight=1.0 -> BM25-only ranking. The exact-token query should win."""
    r = HybridRetriever(
        store=NumpyVectorStore(),
        embedder=MockEmbeddingClient(dim=64, seed=1),
        corpus=small_corpus,
        bm25_weight=1.0,
        use_cache=False,
    )
    hits = r.retrieve("conv2d stride padding", top_k=1)
    assert hits[0].doc.doc_id == "op.conv.stride"


def test_retriever_validates_weight():
    with pytest.raises(ValueError, match="bm25_weight"):
        HybridRetriever(
            store=NumpyVectorStore(),
            embedder=MockEmbeddingClient(dim=8, seed=1),
            corpus=[],
            bm25_weight=1.5,
        )


def test_build_default_retriever_runs():
    """End-to-end smoke: factory returns a working retriever using mock embedder."""
    r = build_default_retriever(use_cache=False)
    assert r.corpus_size > 10  # corpus has >= 30 entries from the three sources
    hits = r.retrieve("matmul k accumulator", top_k=3)
    assert hits
    # Top hit must be matmul-related.
    assert "matmul" in hits[0].doc.text.lower()
