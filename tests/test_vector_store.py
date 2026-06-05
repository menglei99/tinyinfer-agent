"""Tests for vector store backends."""

import importlib

import numpy as np
import pytest

from agent.rag.store import (
    ChromaVectorStore,
    Document,
    MilvusVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    get_vector_store,
)


def _make_corpus():
    docs = [
        Document(doc_id="a", text="matmul row-major GEMM"),
        Document(doc_id="b", text="softmax with max subtract"),
        Document(doc_id="c", text="layernorm mean-variance normalisation"),
        Document(doc_id="d", text="conv 2d kernel stride padding"),
    ]
    rng = np.random.default_rng(0)
    embs = [rng.standard_normal(64).astype(np.float32).tolist() for _ in docs]
    return docs, embs


# ---------- numpy ----------


def test_numpy_store_upsert_then_search_returns_docs():
    store = NumpyVectorStore()
    docs, embs = _make_corpus()
    store.upsert(docs, embs)
    assert store.size == 4

    hits = store.search(embs[1], top_k=2)
    assert len(hits) == 2
    assert hits[0].doc.doc_id == "b"
    assert hits[0].score > 0.99


def test_numpy_store_handles_empty_search_before_upsert():
    store = NumpyVectorStore()
    assert store.search([0.0] * 64, top_k=3) == []


def test_numpy_store_rejects_dim_mismatch():
    store = NumpyVectorStore()
    rng = np.random.default_rng(0)
    store.upsert([Document(doc_id="x", text="t")], [rng.standard_normal(8).astype(np.float32).tolist()])
    with pytest.raises(ValueError, match="dim mismatch"):
        store.upsert(
            [Document(doc_id="y", text="t")],
            [rng.standard_normal(16).astype(np.float32).tolist()],
        )


def test_numpy_store_top_k_clamped():
    store = NumpyVectorStore()
    docs, embs = _make_corpus()
    store.upsert(docs, embs)
    hits = store.search(embs[0], top_k=99)
    assert len(hits) == 4


# ---------- factory ----------


def test_factory_default_is_chroma(monkeypatch, tmp_path):
    """Chroma is the documented default; it must build without a backend arg."""
    monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    store = get_vector_store()
    assert isinstance(store, ChromaVectorStore)


def test_factory_numpy_via_env(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "numpy")
    assert isinstance(get_vector_store(), NumpyVectorStore)


def test_factory_chroma_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    assert isinstance(get_vector_store(), ChromaVectorStore)


def test_factory_qdrant_when_available(monkeypatch, tmp_path):
    if importlib.util.find_spec("qdrant_client") is None:
        pytest.skip("qdrant-client not installed (optional extra)")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    assert isinstance(get_vector_store(), QdrantVectorStore)


def test_factory_milvus_when_available(monkeypatch, tmp_path):
    if importlib.util.find_spec("pymilvus") is None:
        pytest.skip("pymilvus not installed (optional extra)")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "milvus")
    monkeypatch.setenv("MILVUS_LITE_PATH", str(tmp_path / "ms.db"))
    assert isinstance(get_vector_store(), MilvusVectorStore)


def test_factory_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown vector store backend"):
        get_vector_store("redis")


# ---------- chroma round-trip ----------


def test_chroma_store_round_trip(tmp_path):
    """Chroma upsert + search round-trip. Fast — sqlite, no daemon."""
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed (rag extra)")
    store = ChromaVectorStore(path=str(tmp_path / "cdb"))
    store.reset()  # ensure clean slate
    docs, embs = _make_corpus()
    store.upsert(docs, embs)
    hits = store.search(embs[2], top_k=1)
    assert len(hits) == 1
    assert hits[0].doc.text == "layernorm mean-variance normalisation"
    # Score is similarity (1 - cosine distance), so an exact-vector query
    # should land near 1.0.
    assert hits[0].score > 0.99


def test_chroma_handles_empty_search_before_upsert(tmp_path):
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed (rag extra)")
    store = ChromaVectorStore(path=str(tmp_path / "cdb2"))
    store.reset()
    assert store.search([0.0] * 8, top_k=3) == []


# ---------- milvus (optional) ----------


@pytest.mark.slow
def test_milvus_store_round_trip(tmp_path):
    if importlib.util.find_spec("pymilvus") is None:
        pytest.skip("pymilvus not installed (rag-milvus extra)")
    db = tmp_path / "ms.db"
    store = MilvusVectorStore(db_path=str(db))
    try:
        docs, embs = _make_corpus()
        store.upsert(docs, embs)
        hits = store.search(embs[2], top_k=1)
        assert len(hits) == 1
        assert hits[0].doc.text == "layernorm mean-variance normalisation"
    finally:
        store.close()


# ---------- qdrant (optional) ----------


@pytest.mark.slow
def test_qdrant_store_round_trip(tmp_path):
    if importlib.util.find_spec("qdrant_client") is None:
        pytest.skip("qdrant-client not installed (rag-qdrant extra)")
    store = QdrantVectorStore(path=str(tmp_path / "qd"))
    docs, embs = _make_corpus()
    store.upsert(docs, embs)
    hits = store.search(embs[2], top_k=1)
    assert len(hits) == 1
    assert hits[0].doc.text == "layernorm mean-variance normalisation"
