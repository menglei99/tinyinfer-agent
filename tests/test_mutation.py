"""Tests for the in-process unified-diff applier used by the fault-injection benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.mutation import (
    apply_diff,
    parse_unified_diff,
)


_ONE_LINE_DIFF = """\
diff --git a/src/foo.cpp b/src/foo.cpp
index 1111111..2222222 100644
--- a/src/foo.cpp
+++ b/src/foo.cpp
@@ -2,3 +2,3 @@
 line1
-line2_old
+line2_new
 line3
"""


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")


# ---------- parsing ----------


def test_parse_basic_single_hunk():
    patches = parse_unified_diff(_ONE_LINE_DIFF)
    assert len(patches) == 1
    p = patches[0]
    assert p.path == "src/foo.cpp"
    assert len(p.hunks) == 1
    h = p.hunks[0]
    assert h.old_start == 2
    assert h.old_lines == ["line1", "line2_old", "line3"]
    assert h.new_lines == ["line1", "line2_new", "line3"]


def test_parse_rejects_file_deletion():
    txt = (
        "diff --git a/x b/x\n"
        "--- a/x\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-bye\n"
    )
    with pytest.raises(ValueError, match="file deletion"):
        parse_unified_diff(txt)


# ---------- apply + revert ----------


def test_apply_then_revert(tmp_path: Path):
    _write_tree(tmp_path, {"src/foo.cpp": "line0\nline1\nline2_old\nline3\nline4\n"})

    with apply_diff(_ONE_LINE_DIFF, tmp_path) as touched:
        assert touched == [tmp_path / "src/foo.cpp"]
        body = (tmp_path / "src/foo.cpp").read_text(encoding="utf-8")
        assert "line2_new" in body
        assert "line2_old" not in body

    # Context manager exit must restore the file byte-for-byte.
    final = (tmp_path / "src/foo.cpp").read_text(encoding="utf-8")
    assert final == "line0\nline1\nline2_old\nline3\nline4\n"


def test_revert_runs_on_exception(tmp_path: Path):
    _write_tree(tmp_path, {"src/foo.cpp": "line0\nline1\nline2_old\nline3\nline4\n"})

    with pytest.raises(RuntimeError, match="boom"):
        with apply_diff(_ONE_LINE_DIFF, tmp_path):
            raise RuntimeError("boom")

    # Even with an exception inside the with-block, the source must be restored.
    final = (tmp_path / "src/foo.cpp").read_text(encoding="utf-8")
    assert final == "line0\nline1\nline2_old\nline3\nline4\n"


def test_missing_target_file_raises(tmp_path: Path):
    # Source dir doesn't contain the patch target.
    with pytest.raises(FileNotFoundError, match="patch target missing"):
        with apply_diff(_ONE_LINE_DIFF, tmp_path):
            pass


def test_context_mismatch_raises(tmp_path: Path):
    """If the context lines around the hunk don't match the source, we bail —
    silently misapplying a patch is the worst possible outcome."""
    _write_tree(tmp_path, {"src/foo.cpp": "different\nlines\nentirely\nnow\nok\n"})
    with pytest.raises(ValueError, match="hunk context mismatch"):
        with apply_diff(_ONE_LINE_DIFF, tmp_path):
            pass

    # And the file is restored to its pre-attempt state.
    assert (tmp_path / "src/foo.cpp").read_text(encoding="utf-8") == (
        "different\nlines\nentirely\nnow\nok\n"
    )


def test_preserves_trailing_newline_presence(tmp_path: Path):
    """File ending without a final newline must stay that way after revert."""
    src_no_eol = "line0\nline1\nline2_old\nline3\nline4"  # no trailing \n
    _write_tree(tmp_path, {"src/foo.cpp": src_no_eol})

    with apply_diff(_ONE_LINE_DIFF, tmp_path):
        body = (tmp_path / "src/foo.cpp").read_text(encoding="utf-8")
        # After mutation: still no trailing newline.
        assert not body.endswith("\n")
        assert "line2_new" in body

    assert (tmp_path / "src/foo.cpp").read_text(encoding="utf-8") == src_no_eol


def test_apply_against_real_benchmark_seed(tmp_path: Path):
    """Seed 01 must apply cleanly against the real tinyinfer/src/matmul.cpp."""
    repo_src = Path("tinyinfer/src/matmul.cpp").read_text(encoding="utf-8")
    _write_tree(tmp_path, {"tinyinfer/src/matmul.cpp": repo_src})

    seed = Path("benchmark/seeds/01_matmul_offbyone_inner.diff").read_text(encoding="utf-8")
    with apply_diff(seed, tmp_path):
        body = (tmp_path / "tinyinfer/src/matmul.cpp").read_text(encoding="utf-8")
        assert "p + 1 < k" in body

    # Original restored.
    assert (tmp_path / "tinyinfer/src/matmul.cpp").read_text(encoding="utf-8") == repo_src
