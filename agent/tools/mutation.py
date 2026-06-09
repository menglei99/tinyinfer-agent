"""把 unified-diff mutation 应用到仓库内的源码树上，然后回滚。

被 fault-injection benchmark 用来把 clean checkout 翻成 "buggy" 状态，build，
跑 agent 生成的测试，然后 revert。封装成 context manager，保证半途出错也
不会把源码树留在被污染状态。

故意只实现 unified-diff 的最小子集，够 benchmark 的 seed 用就行（单文件、
多 hunk、不处理 rename/copy/binary）。如果以后 seed 长成多文件，parser 也能
处理；只是别指望我们做成 `patch(1)`。
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
    old_start: int  # OLD 文件里 hunk 起始行号（1-based）
    old_lines: list[str]  # patch 之前的行（不含 +/- 前缀）
    new_lines: list[str]  # patch 之后期望的行


@dataclass
class FilePatch:
    path: str  # 仓库相对路径（diff 里 "b/..." 那一侧）
    hunks: list[Hunk]


_HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    """把 unified diff 解析成 FilePatch 列表。

    认识标准的 `diff --git`、`--- a/...`、`+++ b/...`、`@@` 这几种 header；
    忽略 `index` 和其他 metadata 行。`+++ /dev/null` 直接 raise —— 我们不
    建模文件删除（benchmark 没这种需求）。
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
            # hunk 之前的杂项（file mode 等）—— 跳过
            continue

        if not raw:
            # 空 body 行同时算作 old 和 new 的 context 行
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
            # "\ No newline at end of file" —— 我们始终保留原文件的尾换行
            # 行为，所以这个 marker 直接跳过
            continue
        else:
            # 未知 marker —— 大声报错，别静默错应用
            raise ValueError(f"unrecognised diff line: {raw!r}")

    return patches


def _apply_hunk_to_lines(lines: list[str], hunk: Hunk) -> list[str]:
    """返回应用了 `hunk` 后的新文件行列表。

    先严格匹配；不中时在 `old_start` 附近一个小窗口里扫，找一个 old-line
    全部对得上源码的位置。这处理 seed 的 `old_start` 因为文件后来加了 include
    或空行而偏移几行的情况 —— 等价于 `patch(1) --fuzz=N`（N=2 行）。

    匹配必须**唯一**：窗口里 0 个或多个都对得上时，raise，避免静默错应用
    把源码污染。
    """
    fuzz = 5  # nominal old_start 两侧各扫几行
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
        # 大声报错：把第一个对不上的行 dump 出来让用户能修 seed
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
    """把文件文本拆成 (lines_without_endings, trailing_terminator)。

    单独跟踪尾换行，rejoin 时保留"文件是否以换行结尾"—— 很多 C++ 源文件
    是这样，而 patch 工具偷偷改了这个 bit 是常见的静默 regression。
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
    # splitlines() 会丢掉所有 line terminator；上面单独处理了 trailing
    lines = body.splitlines()
    return lines, trailing


def _join_lines(lines: list[str], trailing: str) -> str:
    return "\n".join(lines) + trailing


@contextmanager
def apply_diff(
    diff_text: str, project_root: Path | str = ".", *, encoding: str = "utf-8"
) -> Iterator[list[Path]]:
    """把 `diff_text` 应用到 `project_root` 下的文件，退出时 revert。

    yield 被改动的文件路径列表给 caller log。原始文件内容存到内存里；退出
    （正常或异常）时所有 touched 文件都被覆写回 pre-mutation 状态。
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
            # 从底到顶应用 hunk，前面 hunk 的行号偏移不影响后面
            for hunk in sorted(patch.hunks, key=lambda h: h.old_start, reverse=True):
                lines = _apply_hunk_to_lines(lines, hunk)

            new_text = _join_lines(lines, trailing)
            target.write_text(new_text, encoding=encoding, newline="")
            touched.append(target)

        yield touched
    finally:
        # 字节对字节恢复，避免 encoding/EOL 漂移
        for path, original in backups.items():
            try:
                path.write_bytes(original)
            except OSError as exc:
                # 恢复失败要大声报错 —— 被污染的源码树是最糟的结果。
                # 没这个 raise 用户可能永远发现不了。
                _bail_loudly = io.StringIO()
                _bail_loudly.write(
                    f"FATAL: failed to restore {path}: {exc}. "
                    f"Original content is in the backups dict."
                )
                raise RuntimeError(_bail_loudly.getvalue()) from exc


__all__ = ["FilePatch", "Hunk", "apply_diff", "parse_unified_diff"]
