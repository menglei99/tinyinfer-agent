"""embedding client 的测试。"""

import os

import numpy as np
import pytest

from agent.llm.embedder import (
    DashScopeEmbeddingClient,
    MockEmbeddingClient,
    get_embedding_client,
)


def test_mock_embedder_is_deterministic():
    a = MockEmbeddingClient(dim=128, seed=0).embed(["matmul"])[0]
    b = MockEmbeddingClient(dim=128, seed=0).embed(["matmul"])[0]
    assert a == b


def test_mock_embedder_different_texts_diverge():
    e = MockEmbeddingClient(dim=128, seed=0)
    a, b = e.embed(["matmul fp32 GEMM", "softmax stability max-subtract"])
    # Cosine similarity of two unrelated random unit vectors should be small.
    sim = float(np.dot(a, b))
    assert abs(sim) < 0.3


def test_mock_embedder_returns_unit_vectors():
    e = MockEmbeddingClient(dim=64, seed=42)
    [v] = e.embed(["hello"])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_mock_embedder_returns_correct_shape():
    e = MockEmbeddingClient(dim=256, seed=0)
    out = e.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 256 for v in out)


def test_mock_embedder_handles_empty_list():
    assert MockEmbeddingClient().embed([]) == []


def test_factory_returns_mock_when_provider_unset(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert isinstance(get_embedding_client(), MockEmbeddingClient)


def test_factory_returns_mock_when_dashscope_key_missing(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    # Silent fallback — the embedding pipeline must not break CI when the key
    # is absent. The chat client takes the same approach.
    assert isinstance(get_embedding_client(), MockEmbeddingClient)


def test_factory_picks_dashscope_when_configured(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-fake-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    client = get_embedding_client()
    assert isinstance(client, DashScopeEmbeddingClient)


@pytest.mark.slow
def test_dashscope_embed_real_api():
    """Hits real DashScope. Skipped unless DASHSCOPE_API_KEY is in env."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        pytest.skip("DASHSCOPE_API_KEY not set")
    client = DashScopeEmbeddingClient(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model="text-embedding-v4",
    )
    [v] = client.embed(["matmul fp32 row-major GEMM"])
    assert 256 <= len(v) <= 4096  # DashScope text-embedding-v4 is 1024
    assert any(abs(x) > 1e-6 for x in v)
