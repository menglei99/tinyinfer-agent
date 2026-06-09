"""JSONL trace writer。

每个 event 一行。schema 故意做成扁平、稳定的，让 trace 文件能 diff、能回放、
不用专门 viewer 就能看。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, trace_path: str | Path | None = None):
        if trace_path is None:
            trace_dir = Path(os.getenv("TRACE_DIR", "traces"))
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_id = uuid.uuid4().hex[:12]
            trace_path = trace_dir / f"run-{int(time.time())}-{self.trace_id}.jsonl"
        else:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_id = trace_path.stem

        self.path = trace_path
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)  # 行缓冲

    def write(self, event_type: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "trace_id": self.trace_id,
            "event": event_type,
            **fields,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.write("error", error_type=str(exc_type.__name__), message=str(exc))
        self.close()


def _json_default(obj: Any) -> Any:
    # Pydantic model、enum、Path
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)
