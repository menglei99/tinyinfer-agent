"""Hybrid retriever: BM25 (lexical) + dense embedding (semantic), fused with RRF.

Why both: BM25 catches exact tokens (op names, type names, error keywords) that
embeddings sometimes blur; dense embeddings catch paraphrase. Reciprocal rank
fusion blends them without tuning a calibration threshold.

RRF formula:
    score(d) = w_bm25 * 1 / (k + rank_bm25(d))
             + (1 - w_bm25) * 1 / (k + rank_dense(d))

Defaults: w_bm25 = 0.4, k = 60. These are the values from the original RRF
paper and work well as a starting point.
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
    """Whitespace + alphanumeric tokenization. Lowercased.

    Good enough for an English-only corpus. Switching to jieba for Chinese is
    a future extra; flagged in DEFERRED_VALIDATION.md.
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
        """Lazy import keeps the base install free of rank-bm25."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "HybridRetriever needs rank-bm25. Install with `pip install -e .[rag]`."
            ) from exc
        tokenised = [_tokenize(d.text) for d in self._corpus]
        # rank-bm25 raises on an empty corpus; guard so callers can build an
        # empty retriever for tests.
        if not tokenised:
            return None
        return BM25Okapi(tokenised)

    def _index_corpus(self) -> None:
        """Embed and upsert the corpus into the vector store."""
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
        # Pull a wider candidate pool from each subsystem so RRF has rank info
        # for any doc that either signal scored highly. 4*top_k is a common
        # heuristic; capped at corpus size.
        pool = min(max(top_k * 4, 10), len(self._corpus))

        bm25_ranks = self._bm25_ranks(query, pool)
        dense_ranks = self._dense_ranks(query, pool)

        # Fuse: every doc that appears in either ranking gets a score from each
        # signal (rank infinity for misses).
        all_ids = set(bm25_ranks) | set(dense_ranks)
        wb, wd = self._bm25_weight, 1.0 - self._bm25_weight
        k = self._rrf_k
        big_rank = pool + k + 1  # treat misses as worse than any hit

        fused: dict[str, float] = {}
        for doc_id in all_ids:
            rb = bm25_ranks.get(doc_id, big_rank)
            rd = dense_ranks.get(doc_id, big_rank)
            fused[doc_id] = wb / (k + rb) + wd / (k + rd)

        # Top-k by fused score, ties broken by doc_id (stable for tests).
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        by_id = {d.doc_id: d for d in self._corpus}
        return [ScoredDoc(doc=by_id[doc_id], score=score) for doc_id, score in ordered]

    def _bm25_ranks(self, query: str, pool: int) -> dict[str, int]:
        if self._bm25 is None:
            return {}
        scores = self._bm25.get_scores(_tokenize(query))
        # Argsort descending, take top `pool`.
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
        return {self._corpus[i].doc_id: rank + 1 for rank, i in enumerate(idx)}

    def _dense_ranks(self, query: str, pool: int) -> dict[str, int]:
        if self._use_cache:
            [qvec] = cached_embed(self._embedder, [query])
        else:
            [qvec] = self._embedder.embed([query])
        hits = self._store.search(qvec, top_k=pool)
        return {h.doc.doc_id: rank + 1 for rank, h in enumerate(hits)}

    # ---------- introspection ----------

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
    """Single entry point for graph wiring + CLI.

    All args optional. Defaults:
      - embedder: get_embedding_client() (mock unless EMBEDDING_PROVIDER is set)
      - store:    get_vector_store()     (chroma by default; numpy/qdrant/milvus
                                          via VECTOR_STORE_BACKEND env var)
      - corpus:   load_default_corpus()
    """
    embedder = embedder or MockEmbeddingClient()
    store = store or get_vector_store()
    corpus = corpus if corpus is not None else load_default_corpus()
    return HybridRetriever(store=store, embedder=embedder, corpus=corpus, use_cache=use_cache)


__all__ = ["HybridRetriever", "build_default_retriever"]
