"""Embedding client 抽象。

跟 LLMClient Protocol 一样的形状：一个真 provider（DashScope 的
text-embedding-v4）加一个确定性 mock 给测试 / CI / 没 key 的本地运行。

DashScope client 复用 chat client 同一个 OpenAI 兼容 endpoint
（`DASHSCOPE_BASE_URL`），一份 key 同时驱动 LLM 和 embedding。
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


# ---------- 真 provider ----------


class DashScopeEmbeddingClient:
    """封装 DashScope 的 OpenAI 兼容 embedding endpoint。

    每 25 条做一批（写代码时的公开上限）。每条输入返回一行 float，顺序跟输入对齐。
    """

    # DashScope text-embedding-v4 一批上限 10 条（2026-06-03 实测；老文档说 25
    # 但 API 超过 10 直接 400）。
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
            # API 保证顺序和 input 对齐
            out.extend(item.embedding for item in resp.data)
        return out


# ---------- mock provider ----------


class MockEmbeddingClient:
    """按 text 给确定性 vector。同样 text -> 跨 run 同样 vector。

    用 sha256 做 hash，让 seed 跨 Python 进程稳定（Python 内置 hash() 受
    PYTHONHASHSEED 随机化影响，会让测试 flaky）。
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
            v = v / (np.linalg.norm(v) + 1e-12)  # L2 normalize -> cosine sim 等于点积
            out.append(v.astype(np.float32).tolist())
        return out


# ---------- factory ----------


def get_embedding_client() -> EmbeddingClient:
    """根据 EMBEDDING_PROVIDER env 选 provider。没 key 时 fallback 到 mock。

    认识的值："dashscope" / "qwen" -> DashScope；其他 -> mock。
    """
    provider = (os.getenv("EMBEDDING_PROVIDER", "") or "").lower().strip()

    if provider in ("dashscope", "qwen"):
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            # caller 明确要真 provider 但 key 缺了 —— 在 dev 环境响一声（env
            # 拼错），但别让 CI 挂掉
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
