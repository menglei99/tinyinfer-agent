"""Compare ablation runs of the fault-injection benchmark.

Reads N JSON reports (one per configuration) and prints a markdown table of
the headline metrics — accuracy, average tests-per-seed, critic pass rate,
average missing dimensions, RAG context use.

Usage:
    python -m benchmark.compare results/mock_norag.json results/qwen_norag.json \
        results/qwen_rag.json results/qwen_sc3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Optional


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(report: dict) -> dict:
    results = report.get("results", []) or []
    n = len(results)

    def avg(key: str) -> Optional[float]:
        vals = [r.get(key) for r in results if r.get(key) is not None]
        return mean(vals) if vals else None

    def rate(key: str) -> Optional[float]:
        vals = [bool(r.get(key)) for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "total": report.get("total", n),
        "caught": report.get("caught", 0),
        "accuracy": report.get("accuracy", 0.0),
        "avg_tests_per_seed": avg("tests_generated"),
        "critic_pass_rate": rate("critic_passed"),
        "avg_critic_iterations": avg("critic_iterations"),
        "avg_missing_dims": avg("critic_missing_count"),
        "avg_context_docs": avg("context_docs_used"),
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ablation runs")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--md", type=Path, default=None, help="Also write markdown output here")
    args = parser.parse_args(argv)

    summaries: dict[str, dict] = {}
    for p in args.reports:
        if not p.is_file():
            print(f"skip: {p} (not found)")
            continue
        summaries[p.stem] = _summary(_load(p))

    if not summaries:
        print("No reports loaded.")
        return 1

    cols = [
        ("Config", lambda name, s: name),
        ("Accuracy", lambda name, s: f"{s['caught']}/{s['total']} ({s['accuracy']:.0%})"),
        ("Avg tests/seed", lambda name, s: _fmt(s["avg_tests_per_seed"])),
        ("Critic pass rate", lambda name, s: _fmt(s["critic_pass_rate"])),
        ("Avg critic iters", lambda name, s: _fmt(s["avg_critic_iterations"])),
        ("Avg missing dims", lambda name, s: _fmt(s["avg_missing_dims"])),
        ("Avg RAG docs", lambda name, s: _fmt(s["avg_context_docs"])),
    ]

    lines: list[str] = []
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines.append(header)
    lines.append(sep)
    for name, s in summaries.items():
        lines.append("| " + " | ".join(c[1](name, s) for c in cols) + " |")

    md = "\n".join(lines)
    print(md)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n", encoding="utf-8")
        print(f"\nWrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
