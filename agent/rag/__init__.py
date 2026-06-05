"""RAG retrieval primitives.

Public surface kept tight: import what graph nodes need, hide what they don't.
"""

from agent.rag.cache import cached_embed
from agent.rag.corpus import load_default_corpus
from agent.rag.retriever import HybridRetriever, build_default_retriever
from agent.rag.store import (
    ChromaVectorStore,
    Document,
    MilvusVectorStore,
    NumpyVectorStore,
    QdrantVectorStore,
    ScoredDoc,
    VectorStore,
    get_vector_store,
)

__all__ = [
    "Document",
    "ScoredDoc",
    "VectorStore",
    "NumpyVectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "MilvusVectorStore",
    "get_vector_store",
    "HybridRetriever",
    "build_default_retriever",
    "load_default_corpus",
    "cached_embed",
]
