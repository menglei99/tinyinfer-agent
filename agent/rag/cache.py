"""embedding 的磁盘缓存。

embedding 一次要打一次 API。seed corpus 跨 run 不变，所以按 sha256(model + text)
缓存能让后续 run 和测试不花钱。

cache 布局：每个 text 一个 JSON 文件，放在 cache_dir/ 下。文件名 = digest；
内容 = `{"text": ..., "model": ..., "vector": [...]}`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence


def _key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()


def _model_name(embedder) -> str:
    """尽力取一个：embedder 有 `_model` attr 就用，否则用类名。

    Mock embedder 没 model 名；用类名足够把 mock vector 跟真 API 缓存隔离开。
    """
    return getattr(embedder, "_model", embedder.__class__.__name__)


def cached_embed(
    embedder,
    texts: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
) -> list[list[float]]:
    """embed `texts`，能命中 cache 就命中；miss 的走 embedder.embed。"""
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
                # 缓存损坏当 miss 处理；后续会被覆盖
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
