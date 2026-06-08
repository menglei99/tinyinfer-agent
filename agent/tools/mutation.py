"""Apply unified-diff mutations against an in-repo source tree and roll back.

Used by the fault-injection benchmark to flip a clean checkout into a "buggy"
state, build, run the agent's generated tests, then revert. Implemented as a
context manager so a half-finished run never leaves the source tree poisoned.

We intentionally implement just enough of the unified-diff format to handle our
benchmark seeds (single-file, single-hunk, no rename/copy/binary). If a seed
grows to multi-hunk or multi-file later it should keep working — the parser
handles those — but we don't try to be `patch(1)`.
"""

from __future__ import annotations

import io
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Hunk:
    old_start: int  # 1-based line number in the OLD file where this hunk begins
    old_lines: list[str]  # lines as they appear before the patch (no leading marker)
    new_lines: list[str]  # lines as they should appear after the patch


@dataclass
class FilePatch:
    path: str  # repo-relative path (the "b/..." side of the diff)
    hunks: list[Hunk]


_HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    """Parse a unified diff into FilePatch objects.

    Recognises the standard `diff --git`, `--- a/...`, `+++ b/...`, and `@@`
    headers; ignores `index` and other metadata lines. A `+++ /dev/null`
    target raises — we don't model file deletion (no seed needs it).
    """
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    current_hunk: Hunk | None = None

    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            if target == "/dev/null":
                raise ValueError("file deletion not supported in benchmark mutations")
            current = FilePatch(path=target, hunks=[])
            patches.append(current)
            current_hunk = None
            continue

        if raw.startswith("--- ") or raw.startswith("diff ") or raw.startswith("index "):
            continue

        m = _HUNK_HEADER.match(raw)
        if m:
            if current is None:
                raise ValueError(f"hunk header before any file header: {raw!r}")
            current_hunk = Hunk(
                old_start=int(m.group(1)),
                old_lines=[],
                new_lines=[],
            )
            current.hunks.append(current_hunk)
            continue

        if current_hunk is None:
            # Pre-hunk garbage (file mode etc.) — skip.
            continue

        if not raw:
            # Empty body line counts as both a context line in old and new.
            current_hunk.old_lines.append("")
            current_hunk.new_lines.append("")
            continue

        marker, body = raw[0], raw[1:]
        if marker == " ":
            current_hunk.old_lines.append(body)
            current_hunk.new_lines.append(body)
        elif marker == "-":
            current_hunk.old_lines.append(body)
        elif marker == "+":
            current_hunk.new_lines.append(body)
        elif marker == "\\":
            # "\ No newline at end of file" — we always preserve original trailing
            # newline behaviour, so skip the marker.
            continue
        else:
            # Unknown marker — be loud rather than silently misapplying.
            raise ValueError(f"unrecognised diff line: {raw!r}")

    return patches


def _apply_hunk_to_lines(lines: list[str], hunk: Hunk) -> list[str]:
    """Return a new list of file lines with `hunk` applied.

    Strict match first; on miss we scan a small window around `old_start`
    looking for a position where every old-line equals the source. This
    handles seeds whose `old_start` is off by a few lines because the file
    grew an include or a blank line since the diff was authored — same
    behaviour as `patch(1) --fuzz=N` with N=2 lines.

    A match must be unambiguous: if zero or multiple positions in the window
    fit, we raise so a silently-misapplied patch can't poison the source.
    """
    fuzz = 5  # lines on either side of the nominal old_start
    target_lines = hunk.old_lines

    candidates: list[int] = []
    nominal = hunk.old_start - 1  # 0-based
    for delta in range(-fuzz, fuzz + 1):
        start_idx = nominal + delta
        end_idx = start_idx + len(target_lines)
        if start_idx < 0 or end_idx > len(lines):
            continue
        if lines[start_idx:end_idx] == target_lines:
            candidates.append(start_idx)

    if not candidates:
        # Be loud: dump the first offending line so the user can fix the seed.
        first_off = ""
        if 0 <= nominal < len(lines):
            actual = lines[nominal]
            expected = target_lines[0] if target_lines else ""
            first_off = (
                f"\n  expected: {expected!r}\n  actual:   {actual!r}"
            )
        raise ValueError(
            f"hunk context mismatch at line {hunk.old_start}: "
            f"no match within ±{fuzz} lines.{first_off}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"hunk at line {hunk.old_start} is ambiguous: "
            f"matches at line(s) {[c + 1 for c in candidates]}"
        )

    start_idx = candidates[0]
    end_idx = start_idx + len(target_lines)
    return lines[:start_idx] + list(hunk.new_lines) + lines[end_idx:]


def _split_keep_endings(text: str) -> tuple[list[str], str]:
    """Split file text into (lines_without_endings, trailing_terminator).

    Tracks the trailing newline separately so the rejoin preserves "does
    the file end in a newline?" — many C++ source files do, and tampering
    with that bit is a common silent regression in patch tools.
    """
    if text.endswith("\r\n"):
        trailing = "\r\n"
        body = text[:-2]
    elif text.endswith("\n"):
        trailing = "\n"
        body = text[:-1]
    else:
        trailing = ""
        body = text
    # splitlines() drops all line terminators; the trailing one is handled above.
    lines = body.splitlines()
    return lines, trailing


def _join_lines(lines: list[str], trailing: str) -> str:
    return "\n".join(lines) + trailing


@contextmanager
def apply_diff(
    diff_text: str, project_root: Path | str = ".", *, encoding: str = "utf-8"
) -> Iterator[list[Path]]:
    """Apply `diff_text` against files under `project_root`; revert on exit.

    Yields the list of mutated file paths so the caller can log them. Original
    file contents are kept in memory; on exit (success OR exception) every
    touched file is rewritten to its pre-mutation state.
    """
    root = Path(project_root)
    patches = parse_unified_diff(diff_text)

    backups: dict[Path, bytes] = {}
    touched: list[Path] = []

    try:
        for patch in patches:
            target = root / patch.path
            if not target.exists():
                raise FileNotFoundError(f"patch target missing: {target}")

            original_bytes = target.read_bytes()
            backups[target] = original_bytes

            text = original_bytes.decode(encoding)
            lines, trailing = _split_keep_endings(text)
            # Apply hunks bottom-up so earlier line-number indices don't shift.
            for hunk in sorted(patch.hunks, key=lambda h: h.old_start, reverse=True):
                lines = _apply_hunk_to_lines(lines, hunk)

            new_text = _join_lines(lines, trailing)
            target.write_text(new_text, encoding=encoding, newline="")
            touched.append(target)

        yield touched
    finally:
        # Restore byte-for-byte to avoid any encoding/EOL drift.
        for path, original in backups.items():
            try:
                path.write_bytes(original)
            except OSError as exc:
                # Surface restoration failures loudly — a poisoned source tree
                # is the worst outcome. We re-raise the original problem after
                # writing what we can; without this the user might never notice.
                _bail_loudly = io.StringIO()
                _bail_loudly.write(
                    f"FATAL: failed to restore {path}: {exc}. "
                    f"Original content is in the backups dict."
                )
                raise RuntimeError(_bail_loudly.getvalue()) from exc


__all__ = ["FilePatch", "Hunk", "apply_diff", "parse_unified_diff"]
