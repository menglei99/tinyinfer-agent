"""Hybrid retriever：BM25（词面）+ dense embedding（语义），用 RRF 融合。

为啥两套都要：BM25 抓到 embedding 有时模糊掉的精确 token（op 名、type 名、
error 关键词）；dense embedding 抓 paraphrase。RRF（Reciprocal Rank Fusion）
不用调 calibration threshold 就能把两套混起来。

RRF 公式：
    score(d) = w_bm25 * 1 / (k + rank_bm25(d))
             + (1 - w_bm25) * 1 / (k + rank_dense(d))

默认值：w_bm25 = 0.4，k = 60。原 RRF 论文给的值，作为起点效果不错。
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

from agent.llm.embedder import EmbeddingClient, MockEmbeddingClient
from agent.rag.cache import cached_embed
from agent.rag.corpus import load_default_corpus
from agent.rag.store import Document, ScoredDoc, VectorStore, get_vector_store


def _tokenize(text: str) -> list[str]:
    """空白 + 字母数字分词，小写化。

    够用于纯英文 corpus。要支持中文要换成 jieba，列在 DEFERRED_VALIDATION.md
    里作为未来工作。
    """
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)]


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        corpus: list[Document],
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        *,
        use_cache: bool = True,
    ) -> None:
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        self._store = store
        self._embedder = embedder
        self._corpus = list(corpus)
        self._bm25_weight = float(bm25_weight)
        self._rrf_k = int(rrf_k)
        self._use_cache = bool(use_cache)

        self._bm25 = self._build_bm25()
        self._index_corpus()

    # ---------- setup ----------

    def _build_bm25(self):
        """懒 import 让 base 安装不依赖 rank-bm25。"""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "HybridRetriever 需要 rank-bm25，请装 `pip install -e .[rag]`。"
            ) from exc
        tokenised = [_tokenize(d.text) for d in self._corpus]
        # rank-bm25 在空 corpus 上会 raise；这里挡一下，让 caller 在测试里能
        # 建空 retriever。
        if not tokenised:
            return None
        return BM25Okapi(tokenised)

    def _index_corpus(self) -> None:
        """把 corpus embed 后 upsert 到向量库。"""
        if not self._corpus:
            return
        texts = [d.text for d in self._corpus]
        if self._use_cache:
            vectors = cached_embed(self._embedder, texts)
        else:
            vectors = self._embedder.embed(texts)
        self._store.upsert(self._corpus, vectors)

    # ---------- retrieval ----------

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredDoc]:
        if not query or not self._corpus:
            return []
        # 从两个子系统各拉一个更大的 candidate 池，让 RRF 拿到 rank 信息
        # （任一信号给高分的 doc 都覆盖）。4*top_k 是常用启发；上限 = corpus 大小。
        pool = min(max(top_k * 4, 10), len(self._corpus))

        bm25_ranks = self._bm25_ranks(query, pool)
        dense_ranks = self._dense_ranks(query, pool)

        # 融合：任一 ranking 里出现的 doc 都从每个信号拿一个分数（没出现的
        # 当 rank 无穷大）。
        all_ids = set(bm25_ranks) | set(dense_ranks)
        wb, wd = self._bm25_weight, 1.0 - self._bm25_weight
        k = self._rrf_k
        big_rank = pool + k + 1  # 没命中比任何命中都差

        fused: dict[str, float] = {}
        for doc_id in all_ids:
            rb = bm25_ranks.get(doc_id, big_rank)
            rd = dense_ranks.get(doc_id, big_rank)
            fused[doc_id] = wb / (k + rb) + wd / (k + rd)

        # 按 fused score 取 top-k，并列时按 doc_id 稳定排序（让测试好写）
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        by_id = {d.doc_id: d for d in self._corpus}
        return [ScoredDoc(doc=by_id[doc_id], score=score) for doc_id, score in ordered]

    def _bm25_ranks(self, query: str, pool: int) -> dict[str, int]:
        if self._bm25 is None:
            return {}
        scores = self._bm25.get_scores(_tokenize(query))
        # 倒序 argsort，取 top `pool`
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
        return {self._corpus[i].doc_id: rank + 1 for rank, i in enumerate(idx)}

    def _dense_ranks(self, query: str, pool: int) -> dict[str, int]:
        if self._use_cache:
            [qvec] = cached_embed(self._embedder, [query])
        else:
            [qvec] = self._embedder.embed([query])
        hits = self._store.search(qvec, top_k=pool)
        return {h.doc.doc_id: rank + 1 for rank, h in enumerate(hits)}

    # ---------- 自省 ----------

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)


def build_default_retriever(
    *,
    embedder: Optional[EmbeddingClient] = None,
    store: Optional[VectorStore] = None,
    corpus: Optional[list[Document]] = None,
    use_cache: bool = True,
) -> HybridRetriever:
    """graph wiring + CLI 共用的单一入口。

    所有 arg 都可选。默认值：
      - embedder: get_embedding_client()（EMBEDDING_PROVIDER 不设时用 mock）
      - store:    get_vector_store()（默认 chroma；可通过 VECTOR_STORE_BACKEND
                                       env 切到 numpy/qdrant/milvus）
      - corpus:   load_default_corpus()
    """
    embedder = embedder or MockEmbeddingClient()
    store = store or get_vector_store()
    corpus = corpus if corpus is not None else load_default_corpus()
    return HybridRetriever(store=store, embedder=embedder, corpus=corpus, use_cache=use_cache)


__all__ = ["HybridRetriever", "build_default_retriever"]
