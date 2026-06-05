"""On-disk embedding cache.

Embeddings cost an API call. The seed corpus doesn't change between runs, so
caching by sha256(model + text) keeps subsequent runs (and tests) free.

Cache layout: one JSON file per text under cache_dir/. File name is the digest;
contents are `{"text": ..., "model": ..., "vector": [...]}`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence


def _key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()


def _model_name(embedder) -> str:
    """Best-effort: use the embedder's `_model` attr if present, else its class name.

    The mock embedder doesn't have a model name; using the class is enough to
    keep mock vectors out of a real-API cache.
    """
    return getattr(embedder, "_model", embedder.__class__.__name__)


def cached_embed(
    embedder,
    texts: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
) -> list[list[float]]:
    """Embed `texts`, hitting the cache when possible. Misses go to embedder.embed."""
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(".embedding_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = _model_name(embedder)
    out: list[Optional[list[float]]] = [None] * len(texts)
    misses_idx: list[int] = []
    misses_text: list[str] = []

    for i, t in enumerate(texts):
        path = cache_dir / f"{_key(t, model)}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                out[i] = list(cached["vector"])
                continue
            except (OSError, json.JSONDecodeError, KeyError):
                # Treat corrupt cache entries as misses; they'll be overwritten.
                pass
        misses_idx.append(i)
        misses_text.append(t)

    if misses_text:
        fresh = embedder.embed(misses_text)
        if len(fresh) != len(misses_text):
            raise RuntimeError(
                f"embedder returned {len(fresh)} vectors for {len(misses_text)} inputs"
            )
        for idx, text, vec in zip(misses_idx, misses_text, fresh):
            out[idx] = list(vec)
            path = cache_dir / f"{_key(text, model)}.json"
            path.write_text(
                json.dumps({"text": text, "model": model, "vector": list(vec)}),
                encoding="utf-8",
            )

    return [v for v in out if v is not None]


__all__ = ["cached_embed"]
