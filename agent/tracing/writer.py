"""JSONL trace writer.

Each event is one line. Schema is intentionally flat and stable so trace
files can be diffed, replayed, and inspected without a custom viewer.
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
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered

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
    # Pydantic models, enums, paths.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)
