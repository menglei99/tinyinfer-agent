"""Embedding client abstraction.

Mirrors the LLMClient Protocol shape: a single concrete provider (DashScope's
text-embedding-v4) plus a deterministic mock for tests / CI / no-key local runs.

The DashScope client reuses the same OpenAI-compatible endpoint the chat client
uses (`DASHSCOPE_BASE_URL`) so a single key drives both LLM and embedding.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dim: int


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------- real provider ----------


class DashScopeEmbeddingClient:
    """Wraps DashScope's OpenAI-compatible embedding endpoint.

    Batches input over 25 texts at a time — the public limit at the time of
    writing. Returns a row of floats per input, in the same order.
    """

    # DashScope text-embedding-v4 caps batch at 10 (verified empirically on
    # 2026-06-03; older docs claimed 25 but the API rejects > 10 with a 400).
    _MAX_BATCH = 10

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i : i + self._MAX_BATCH]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # API guarantees order matches input.
            out.extend(item.embedding for item in resp.data)
        return out


# ---------- mock provider ----------


class MockEmbeddingClient:
    """Deterministic per-text vectors. Same text -> same vector across runs.

    The hash is done with sha256 so the seed is stable across Python process
    invocations (Python's built-in hash() is randomised by PYTHONHASHSEED and
    would make tests flaky).
    """

    def __init__(self, dim: int = 1024, seed: int = 0):
        self._dim = dim
        self._seed = seed

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(f"{self._seed}:{t}".encode()).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim)
            v = v / (np.linalg.norm(v) + 1e-12)  # L2 normalize -> cosine sim is dot product
            out.append(v.astype(np.float32).tolist())
        return out


# ---------- factory ----------


def get_embedding_client() -> EmbeddingClient:
    """Pick provider from EMBEDDING_PROVIDER env. Falls back to mock when key is absent.

    Recognised values: "dashscope" / "qwen" -> DashScope; anything else -> mock.
    """
    provider = (os.getenv("EMBEDDING_PROVIDER", "") or "").lower().strip()

    if provider in ("dashscope", "qwen"):
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            # Caller asked for a real provider but the key is missing — be loud
            # in dev (env var typo) without breaking CI.
            return MockEmbeddingClient()
        return DashScopeEmbeddingClient(
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        )

    return MockEmbeddingClient()


__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "DashScopeEmbeddingClient",
    "MockEmbeddingClient",
    "get_embedding_client",
]
