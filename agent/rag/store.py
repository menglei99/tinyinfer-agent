"""RAG retriever 用的向量库抽象。

后端：
    NumpyVectorStore   -> 内存式；用矩阵乘做 cosine similarity。
                          测试默认 + 没依赖时的 fallback。
    ChromaVectorStore  -> 通过 `chromadb` 的嵌入式持久化 store。运行时默认
                          （sqlite 后端；Windows 上开箱即用）。
    QdrantVectorStore  -> 嵌入式本地 `qdrant-client`。可选。filter 语法最丰富，
                          同一个 client 也能连 Qdrant server。
    MilvusVectorStore  -> 用 pymilvus.MilvusClient 包 milvus-lite。可选；保留
                          给已经投入 Milvus 栈的用户。

四个都暴露同一套 `upsert` / `search` 形状，pipeline 其余部分不关心底下接的是哪个。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


# ---------- data class ----------


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


# ---------- numpy 后端 ----------


class NumpyVectorStore:
    """内存版 cosine-similarity store。

    embedding 在 upsert 时就 L2-normalize，search 退化成一次 matmul + topk。

    维度从第一次 upsert 推断 —— 调用方不需要传 `embedding_dim` 参数，换 embedder
    时改一行就行。
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, dim)，行做了 unit-norm
        self._dim: Optional[int] = None

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        if not docs:
            return

        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # 防止全零向量除以 0（虽然非常不可能）
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
            # numpy 后端目前忽略 filter —— 列在 DEFERRED_VALIDATION 里
            pass

        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._matrix @ q  # 双方都做了 unit-norm，点积就是 cosine sim

        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [ScoredDoc(doc=self._docs[int(i)], score=float(scores[int(i)])) for i in idx]

    @property
    def size(self) -> int:
        return len(self._docs)


# ---------- chroma 后端（运行时默认）----------


class ChromaVectorStore:
    """嵌入式持久化 Chroma store。

    为啥默认选 Chroma：Windows 安装干净（不用跟原生 gRPC 死磕）、sqlite 后端
    （不用单独 daemon）、是 LangChain demo 里最常见的"真"向量库 —— 这里的模式
    搬到大多数其他 RAG 教程上都能用。

    embedding 全部由 caller 提供 —— 不让 Chroma 去做内置 embed；同一个
    embedder 已经在驱动 HybridRetriever 的 dense 路径。
    """

    _COLLECTION = "tinyinfer-rag"

    def __init__(self, path: str = ".chroma_db", collection_name: Optional[str] = None) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore 需要 chromadb，请装 `pip install -e .[rag]`。"
            ) from exc

        self._chromadb = chromadb
        self._path = str(Path(path).resolve())
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection_name = collection_name or self._COLLECTION
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            # 用 cosine 跟 NumpyVectorStore 语义对齐
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
            # Chroma 的 `where` 期望 dict；caller 还传字符串就 silently no-op，
            # 不报错。
            if isinstance(filter_expr, dict):
                kwargs["where"] = filter_expr

        res = self._collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs_text = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[ScoredDoc] = []
        for doc_id, text, meta, dist in zip(ids, docs_text, metas, dists):
            # Chroma 返回的是 cosine *distance*（1 - similarity）。翻一下让"大 = 好"
            # 跟 numpy/milvus 后端的约定对齐。
            score = float(1.0 - dist)
            out.append(
                ScoredDoc(
                    doc=Document(doc_id=str(doc_id), text=text or "", metadata=dict(meta or {})),
                    score=score,
                )
            )
        return out

    def reset(self) -> None:
        """drop 掉 collection。给测试用。"""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )


# ---------- qdrant 后端（可选）----------


class QdrantVectorStore:
    """通过 `qdrant-client` local 模式跑嵌入式 Qdrant。

    Qdrant 在可选后端里 filter 语法最丰富。同一个 client 也能连 Qdrant server
    （`url="http://host:6333"`）；这里只用 local 模式，保 Windows 友好。
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
                "QdrantVectorStore 需要 qdrant-client。"
                "请装 `pip install -e .[rag-qdrant]`。"
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
                # Qdrant 的 point id 必须是 int 或 uuid，把字符串 hash 成 int
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
        # qdrant-client 1.10+ 弃用了 `search`，换 `query_points`
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


# ---------- milvus 后端（可选，遗留）----------


class MilvusVectorStore:
    """用 pymilvus.MilvusClient 包 sqlite 后端的 milvus-lite db。

    给已经在用 Milvus 栈的用户保留。不是默认 —— Chroma 在 Windows 上更好装，
    且 LangChain 生态支持更广。
    """

    _COLLECTION = "tinyinfer_rag"

    def __init__(self, db_path: str = ".milvus_lite.db") -> None:
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MilvusVectorStore 需要 pymilvus + milvus-lite。"
                "请装 `pip install -e .[rag-milvus]`。"
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
    """构造一个 VectorStore。

    后端名："numpy"、"chroma"（默认）、"qdrant"、"milvus"。
    `backend` 不传时读 VECTOR_STORE_BACKEND env。
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
