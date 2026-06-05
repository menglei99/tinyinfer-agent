"""Vector store abstraction for the RAG retriever.

Backends:
    NumpyVectorStore   -> in-memory; cosine similarity via matrix multiply.
                          Default in tests + no-deps fallback.
    ChromaVectorStore  -> embedded persistent store via `chromadb`. Default for
                          live runs (sqlite-backed; works on Windows out of the
                          box).
    QdrantVectorStore  -> embedded local-mode `qdrant-client`. Optional. Rich
                          metadata filter syntax; same client also speaks to a
                          Qdrant server.
    MilvusVectorStore  -> wraps pymilvus.MilvusClient against milvus-lite.
                          Optional; kept for users who already invested in the
                          Milvus stack.

All four expose the same `upsert` / `search` shape so the rest of the
pipeline doesn't care which one is wired in.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


# ---------- data classes ----------


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredDoc:
    doc: Document
    score: float

    def to_dict(self) -> dict:
        return {"doc": asdict(self.doc), "score": float(self.score)}


# ---------- protocol ----------


class VectorStore(Protocol):
    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None: ...
    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredDoc]: ...


# ---------- numpy backend ----------


class NumpyVectorStore:
    """In-memory cosine-similarity store.

    Embeddings are L2-normalized at upsert time so search reduces to a single
    matmul + topk.

    Dim is inferred from the first upsert — keeps the call site free of an
    `embedding_dim` argument and makes swapping to a different embedder a
    one-line change.
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, dim), unit-norm rows
        self._dim: Optional[int] = None

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        if not docs:
            return

        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid divide-by-zero for the (very unlikely) all-zero vector.
        arr = arr / np.maximum(norms, 1e-12)

        if self._dim is None:
            self._dim = arr.shape[1]
            self._matrix = arr
        else:
            if arr.shape[1] != self._dim:
                raise ValueError(
                    f"embedding dim mismatch: have {self._dim}, got {arr.shape[1]}"
                )
            assert self._matrix is not None
            self._matrix = np.vstack([self._matrix, arr])

        self._docs.extend(docs)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredDoc]:
        if self._matrix is None or not self._docs:
            return []
        if filter_expr:
            # Numpy backend ignores filters for now — flagged in DEFERRED_VALIDATION.
            pass

        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._matrix @ q  # cosine similarity since both sides unit-norm.

        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [ScoredDoc(doc=self._docs[int(i)], score=float(scores[int(i)])) for i in idx]

    @property
    def size(self) -> int:
        return len(self._docs)


# ---------- chroma backend (default for live runs) ----------


class ChromaVectorStore:
    """Embedded persistent Chroma store.

    Why Chroma as the default: it installs cleanly on Windows (no native gRPC
    fight), is sqlite-backed (no separate daemon), and is the most common
    "real" vector DB in LangChain demos — meaning the patterns here transfer
    to most other RAG tutorials.

    All embeddings supplied by the caller — we never hand text to Chroma's
    built-in embedder; the same embedder powers the dense path of the
    HybridRetriever already.
    """

    _COLLECTION = "tinyinfer-rag"

    def __init__(self, path: str = ".chroma_db", collection_name: Optional[str] = None) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires chromadb. Install with `pip install -e .[rag]`."
            ) from exc

        self._chromadb = chromadb
        self._path = str(Path(path).resolve())
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection_name = collection_name or self._COLLECTION
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            # cosine to match the NumpyVectorStore semantics.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        if not docs:
            return
        self._collection.upsert(
            ids=[d.doc_id for d in docs],
            embeddings=embeddings,
            documents=[d.text for d in docs],
            metadatas=[(d.metadata or {"_": ""}) for d in docs],
        )

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredDoc]:
        kwargs = {
            "query_embeddings": [list(query_vec)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_expr:
            # Chroma `where` expects a dict; if the caller still passes a string
            # we leave it on the floor (no-op) rather than raise.
            if isinstance(filter_expr, dict):
                kwargs["where"] = filter_expr

        res = self._collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs_text = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[ScoredDoc] = []
        for doc_id, text, meta, dist in zip(ids, docs_text, metas, dists):
            # Chroma returns cosine *distance* (1 - similarity). Flip so larger
            # = better, matching the numpy/milvus convention used elsewhere.
            score = float(1.0 - dist)
            out.append(
                ScoredDoc(
                    doc=Document(doc_id=str(doc_id), text=text or "", metadata=dict(meta or {})),
                    score=score,
                )
            )
        return out

    def reset(self) -> None:
        """Drop the collection. Useful for tests."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )


# ---------- qdrant backend (optional) ----------


class QdrantVectorStore:
    """Embedded Qdrant via `qdrant-client` local mode.

    Qdrant gives the richest filter syntax of the optional backends. The same
    client also talks to a Qdrant server (`url="http://host:6333"`); we only
    use local mode here to stay Windows-friendly.
    """

    _COLLECTION = "tinyinfer_rag"

    def __init__(
        self,
        path: str = ".qdrant_db",
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as _m  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "QdrantVectorStore requires qdrant-client. "
                "Install with `pip install -e .[rag-qdrant]`."
            ) from exc

        self._models = _m
        self._collection = collection_name or self._COLLECTION
        if url:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=str(Path(path).resolve()))
        self._dim: Optional[int] = None

    def _ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            self._dim = dim
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._models.VectorParams(
                size=dim, distance=self._models.Distance.COSINE
            ),
        )
        self._dim = dim

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        if not docs:
            return
        dim = len(embeddings[0])
        self._ensure_collection(dim)

        points = [
            self._models.PointStruct(
                # Qdrant point ids must be int or uuid; hash strings to ints.
                id=abs(hash(d.doc_id)) & ((1 << 63) - 1),
                vector=vec,
                payload={"doc_id": d.doc_id, "text": d.text, **(d.metadata or {})},
            )
            for d, vec in zip(docs, embeddings)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredDoc]:
        if not self._client.collection_exists(self._collection):
            return []
        # qdrant-client 1.10+ deprecated `search` in favour of `query_points`.
        resp = self._client.query_points(
            collection_name=self._collection,
            query=list(query_vec),
            limit=top_k,
            with_payload=True,
        )
        hits = getattr(resp, "points", resp)
        out: list[ScoredDoc] = []
        for h in hits:
            payload = dict(getattr(h, "payload", None) or {})
            doc_id = str(payload.pop("doc_id", getattr(h, "id", "")))
            text = str(payload.pop("text", ""))
            out.append(
                ScoredDoc(
                    doc=Document(doc_id=doc_id, text=text, metadata=payload),
                    score=float(getattr(h, "score", 0.0)),
                )
            )
        return out


# ---------- milvus backend (optional, legacy) ----------


class MilvusVectorStore:
    """Wraps pymilvus.MilvusClient against a sqlite-backed milvus-lite db.

    Kept for users who already use the Milvus stack. Not the default — Chroma
    is gentler to install on Windows and has wider LangChain ecosystem support.
    """

    _COLLECTION = "tinyinfer_rag"

    def __init__(self, db_path: str = ".milvus_lite.db") -> None:
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MilvusVectorStore requires pymilvus + milvus-lite. "
                "Install with `pip install -e .[rag-milvus]`."
            ) from exc

        self._db_path = str(Path(db_path).resolve())
        self._client = MilvusClient(self._db_path)
        self._dim: Optional[int] = None

    def _ensure_collection(self, dim: int) -> None:
        if self._client.has_collection(self._COLLECTION):
            self._dim = dim
            return
        self._client.create_collection(
            collection_name=self._COLLECTION,
            dimension=dim,
            primary_field_name="id",
            id_type="string",
            max_length=128,
            vector_field_name="vector",
        )
        self._dim = dim

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        if not docs:
            return

        dim = len(embeddings[0])
        self._ensure_collection(dim)
        if self._dim is not None and dim != self._dim:
            raise ValueError(
                f"embedding dim mismatch: collection has {self._dim}, got {dim}"
            )

        rows = []
        for d, vec in zip(docs, embeddings):
            rows.append(
                {
                    "id": d.doc_id,
                    "vector": vec,
                    "text": d.text,
                    **{f"meta_{k}": v for k, v in (d.metadata or {}).items()},
                }
            )
        self._client.upsert(collection_name=self._COLLECTION, data=rows)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredDoc]:
        if not self._client.has_collection(self._COLLECTION):
            return []
        kwargs: dict = {
            "collection_name": self._COLLECTION,
            "data": [query_vec],
            "limit": top_k,
            "output_fields": ["text"],
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        hits = self._client.search(**kwargs)
        if not hits:
            return []

        out: list[ScoredDoc] = []
        for hit in hits[0]:
            entity = hit.get("entity", {}) or {}
            text = entity.get("text", "")
            meta = {k[5:]: v for k, v in entity.items() if k.startswith("meta_")}
            doc_id = str(hit.get("id", ""))
            score = float(hit.get("distance", 0.0))
            out.append(ScoredDoc(doc=Document(doc_id=doc_id, text=text, metadata=meta), score=score))
        return out

    def close(self) -> None:
        self._client.close()


# ---------- factory ----------


_BACKEND_ALIASES = {
    "": "chroma",
    "default": "chroma",
    "chromadb": "chroma",
}


def get_vector_store(backend: Optional[str] = None, **kwargs) -> VectorStore:
    """Build a VectorStore.

    Backend names: "numpy", "chroma" (default), "qdrant", "milvus".
    Reads VECTOR_STORE_BACKEND env var when `backend` is None.
    """
    backend = (backend or os.getenv("VECTOR_STORE_BACKEND", "") or "").lower()
    backend = _BACKEND_ALIASES.get(backend, backend) or "chroma"

    if backend == "numpy":
        return NumpyVectorStore()
    if backend == "chroma":
        path = kwargs.pop("path", os.getenv("CHROMA_PATH", ".chroma_db"))
        return ChromaVectorStore(path=path, **kwargs)
    if backend == "qdrant":
        path = kwargs.pop("path", os.getenv("QDRANT_PATH", ".qdrant_db"))
        url = kwargs.pop("url", os.getenv("QDRANT_URL"))
        return QdrantVectorStore(path=path, url=url, **kwargs)
    if backend == "milvus":
        db_path = kwargs.pop("db_path", os.getenv("MILVUS_LITE_PATH", ".milvus_lite.db"))
        return MilvusVectorStore(db_path=db_path, **kwargs)
    raise ValueError(f"unknown vector store backend: {backend}")


__all__ = [
    "Document",
    "ScoredDoc",
    "VectorStore",
    "NumpyVectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "MilvusVectorStore",
    "get_vector_store",
]
