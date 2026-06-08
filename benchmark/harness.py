"""Fault-injection benchmark for tinyinfer-agent.

Layout:
    benchmark/seeds/<NN>_<op>_<fault>.diff       — known-bad commits
    benchmark/harness.py                          — runs the agent on each
                                                    seed and computes accuracy

Two oracles available, selected by `FAULT_ORACLE` (or the constructor arg):

- ``heuristic`` (default, fast, cmake-free): a generated test "catches" the
  fault when its name OR rationale lexically overlaps the fault dimension.
  Saturates at 100% on the current seed set — see devlog 05.
- ``real`` (slow, requires docker + cmake): apply the mutation to a clean
  checkout, compile, run only the agent's generated tests; "caught" means at
  least one generated test fails post-mutation. The fault dimension keywords
  are no longer consulted.

`_fault_caught_heuristic` and `_fault_caught_real` are split apart so a single
benchmark run can use either without crosstalk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Manifest: maps seed file name -> expected (target_op, dimension_keywords).
# A seed is considered "caught" when the generated test set contains at least
# one test whose name OR rationale (lower-cased) contains ALL of the keyword
# alternatives in any group. Groups are alternatives ("any of these suffices").
_SEED_MANIFEST: dict[str, dict] = {
    "01_matmul_offbyone_inner.diff": {
        "op": "matmul_fp32",
        "dim_groups": [("k1",), ("aligned",), ("long_reduction",)],
        "fault": "off-by-one in inner k loop",
    },
    "02_matmul_swapped_indices.diff": {
        "op": "matmul_fp32",
        "dim_groups": [("rect",), ("thin",), ("aligned",)],
        "fault": "swapped right-operand indices -> transposed B",
    },
    "03_matmul_uninit_accumulator.diff": {
        "op": "matmul_fp32",
        "dim_groups": [("square",), ("rect",), ("aligned",)],
        "fault": "accumulator reads stale c[] (no zero-init)",
    },
    "04_matmul_fp16_accumulator.diff": {
        "op": "matmul_fp32",
        "dim_groups": [("long_reduction",), ("k128",), ("stability",)],
        "fault": "downcast accumulator loses precision on long k",
    },
    "05_softmax_missing_max_subtract.diff": {
        "op": "softmax_fp32",
        "dim_groups": [("stability", "overflow"), ("stability", "large")],
        "fault": "missing max-subtract -> overflow on large logits",
    },
    "06_softmax_wrong_normalize.diff": {
        "op": "softmax_fp32",
        "dim_groups": [("basic",), ("batch",), ("aligned",)],
        "fault": "normalises by n instead of sum(exp)",
    },
    "07_softmax_exp_before_subtract.diff": {
        "op": "softmax_fp32",
        "dim_groups": [("stability", "overflow"), ("stability", "large")],
        "fault": "exp evaluated before max subtraction",
    },
    "08_layernorm_no_eps.diff": {
        "op": "layernorm_fp32",
        "dim_groups": [("near_constant",), ("stability",)],
        "fault": "eps dropped from sqrt(var + eps)",
    },
    "09_layernorm_var_n_minus_one.diff": {
        "op": "layernorm_fp32",
        "dim_groups": [("thin",), ("n2",), ("basic",)],
        "fault": "Bessel's correction (n-1) instead of n",
    },
    "10_layernorm_int_mean.diff": {
        "op": "layernorm_fp32",
        "dim_groups": [("basic",), ("batch",), ("stability",)],
        "fault": "integer-truncated mean",
    },
}


@dataclass
class HarnessResult:
    seed_name: str
    op_detected: Optional[str]
    tests_generated: int
    skills_used: list[str]
    fault_likely_caught: bool
    detail: str = ""
    # Richer signals — used by the ablation comparison even when the binary
    # `fault_likely_caught` saturates. The structural/LLM critic outcome is
    # where RAG actually shows up; the test count is where self-consistency
    # and LLM shape planning shows up.
    critic_passed: Optional[bool] = None
    critic_iterations: int = 0
    critic_missing_count: int = 0
    context_docs_used: int = 0
    # Real-oracle bookkeeping. "heuristic" results leave these at defaults.
    oracle_mode: str = "heuristic"
    build_status: Optional[str] = None  # "ok" | "build_failed" | "configure_failed" | "skipped"
    failed_test_names: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    total: int
    caught: int
    results: list[HarnessResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.caught / self.total) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "caught": self.caught,
            "accuracy": self.accuracy,
            "results": [
                {
                    "seed": r.seed_name,
                    "op": r.op_detected,
                    "tests_generated": r.tests_generated,
                    "skills_used": r.skills_used,
                    "fault_likely_caught": r.fault_likely_caught,
                    "detail": r.detail,
                    "critic_passed": r.critic_passed,
                    "critic_iterations": r.critic_iterations,
                    "critic_missing_count": r.critic_missing_count,
                    "context_docs_used": r.context_docs_used,
                    "oracle_mode": r.oracle_mode,
                    "build_status": r.build_status,
                    "failed_test_names": r.failed_test_names,
                }
                for r in self.results
            ],
        }


def _fault_caught_heuristic(generated_tests, expected: dict) -> tuple[bool, str]:
    """Did the agent produce a test that lexically targets the fault dimension?

    Match logic: for each `dim_group` (tuple of keyword alternatives), any test
    whose lower-cased name OR rationale contains at least one of the keywords
    counts as a hit for that group. A seed is "caught" if AT LEAST ONE group
    has a hit — we treat groups as OR-of-ORs.
    """
    op = expected["op"]
    haystack = []
    for t in generated_tests:
        if t.op_name != op:
            continue
        haystack.append(t.test_name.lower())
        if t.rationale:
            haystack.append(t.rationale.lower())

    if not haystack:
        return False, f"no tests targeted {op}"

    for group in expected["dim_groups"]:
        for kw in group:
            if any(kw in h for h in haystack):
                return True, f"hit on keyword '{kw}'"
    return False, f"no test name/rationale matched any of {expected['dim_groups']}"


# ---------- real oracle (build + ctest) ----------


def _parse_failed_test_names(ctest_stdout: str) -> list[str]:
    """Pull `*** Failed` line names out of ctest output.

    ctest line format: '<n>/<m> Test #<i>: <name> ........***Failed   <t> sec'
    """
    failed: list[str] = []
    for line in ctest_stdout.splitlines():
        if "***Failed" in line or "***Exception" in line:
            # Take the text between "Test #N: " and the dots.
            marker = "Test #"
            idx = line.find(marker)
            if idx < 0:
                continue
            after = line[idx + len(marker):]
            # name follows colon-space
            if ":" not in after:
                continue
            tail = after.split(":", 1)[1].strip()
            # name continues until run of dots or whitespace
            stop = 0
            for ch in tail:
                if ch in (".", " ", "\t"):
                    break
                stop += 1
            name = tail[:stop]
            if name:
                failed.append(name)
    return failed


def _real_oracle_test_filter(op: str) -> str:
    """ctest -R regex: include all agent-generated executables for this op.

    The build emits e.g. `test_matmul_fp32_generated`,
    `test_matmul_fp32_perf_generated`, `test_matmul_fp32_memory_generated`.
    All of them share the `<op>` stem.
    """
    return rf"test_{op}.*_generated"


def _fault_caught_real(
    expected: dict,
    seed_path: Path,
    *,
    project_dir: Path,
    build_dir: Path,
) -> tuple[bool, str, str, list[str]]:
    """Apply mutation, rebuild, run only the agent's generated tests.

    Returns (caught, detail, build_status, failed_names).

    "caught" semantics: build succeeded AND at least one generated test failed
    on the mutated source. A build failure is reported separately so it
    doesn't quietly become a false negative.
    """
    from agent.tools import cmake_driver, mutation

    op = expected["op"]
    seed_diff = seed_path.read_text(encoding="utf-8")

    # The patch path inside each seed is `tinyinfer/src/*.cpp`, anchored at
    # the repo root. project_dir is the repo root for this oracle.
    with mutation.apply_diff(seed_diff, project_dir):
        b = cmake_driver.build(build_dir)
        if not b.ok:
            tail = (b.stderr or b.stdout)[-400:]
            return (False, f"build_failed: {tail}", "build_failed", [])

        r = cmake_driver.ctest(build_dir, test_filter=_real_oracle_test_filter(op))

    # After the with-block the source tree is restored. ctest's verdict still
    # reflects what happened on the mutated source.
    failed = _parse_failed_test_names(r.stdout)
    if r.ok:
        return (False, "all generated tests passed on mutated source", "ok", [])
    if not failed:
        # ctest returned non-zero but we couldn't parse a failed name — surface
        # the tail so the user can debug.
        tail = r.stdout[-400:] if r.stdout else r.stderr[-400:]
        return (True, f"ctest non-zero; tail: {tail}", "ok", [])
    return (True, f"failed tests: {', '.join(failed)}", "ok", failed)


class FaultInjectionHarness:
    def __init__(
        self,
        seeds_dir: Path,
        *,
        llm=None,
        retriever=None,
        oracle: Optional[str] = None,
        project_dir: Optional[Path] = None,
        build_dir: Optional[Path] = None,
    ):
        self.seeds_dir = Path(seeds_dir)
        self.llm = llm
        self.retriever = retriever
        # Lazy graph build so a misconfigured run still gets a clear error.
        self._graph = None

        mode = (oracle or os.getenv("FAULT_ORACLE", "heuristic") or "heuristic").lower()
        if mode not in ("heuristic", "real"):
            raise ValueError(f"unknown oracle mode {mode!r}; expected 'heuristic' or 'real'")
        self.oracle_mode = mode

        # Repo root used as both the cmake source root AND the mutation patch
        # root (seeds use `tinyinfer/src/...` paths). Default: cwd.
        self.project_root = Path(project_dir) if project_dir else Path.cwd()
        self.build_dir = (
            Path(build_dir)
            if build_dir
            else Path(os.getenv("TINYINFER_BUILD_DIR", "tinyinfer/build-docker"))
        )
        self._real_prepared = False

    def _ensure_graph(self):
        if self._graph is None:
            from agent.graph import build_graph

            self._graph = build_graph(llm=self.llm, retriever=self.retriever)
        return self._graph

    def _prepare_real_oracle(self) -> Optional[str]:
        """Configure + build once on a clean source tree, before any mutation.

        Subsequent per-seed builds are incremental — only the file the
        mutation touched gets recompiled. Returns an error string if the
        oracle isn't usable on this host (caller can fall back to heuristic).
        """
        if self._real_prepared:
            return None

        from agent.tools import cmake_driver

        if not cmake_driver.cmake_available():
            return (
                "real oracle requested but cmake/docker not available "
                "(set TINYINFER_DOCKER_IMAGE or install cmake)"
            )

        cfg = cmake_driver.configure(
            self.project_root / "tinyinfer", self.build_dir
        )
        if not cfg.ok:
            return f"configure_failed: {(cfg.stderr or cfg.stdout)[-400:]}"

        b = cmake_driver.build(self.build_dir)
        if not b.ok:
            return f"clean_build_failed: {(b.stderr or b.stdout)[-400:]}"

        self._real_prepared = True
        return None

    def run_one(self, seed_path: Path) -> HarnessResult:
        try:
            return self._run_one_inner(seed_path)
        except Exception as exc:
            # Network blips during long ablation runs must not kill the whole
            # benchmark — one transient failure shouldn't void 9 good results.
            return HarnessResult(
                seed_name=seed_path.name,
                op_detected=None,
                tests_generated=0,
                skills_used=[],
                fault_likely_caught=False,
                detail=f"run failed: {type(exc).__name__}: {exc}",
                critic_passed=None,
                oracle_mode=self.oracle_mode,
            )

    def _run_one_inner(self, seed_path: Path) -> HarnessResult:
        graph = self._ensure_graph()
        diff = seed_path.read_text(encoding="utf-8")
        initial = {
            "diff": diff,
            "diff_path": str(seed_path),
            "trace_id": f"bench-{seed_path.stem}",
            "critic_iterations": 0,
        }
        # Run pipeline, accumulate state.
        final: dict = {}
        for chunk in graph.stream(initial, stream_mode="updates"):
            for _name, update in chunk.items():
                for k, v in update.items():
                    if isinstance(v, list) and isinstance(final.get(k), list):
                        final[k] = final[k] + v
                    else:
                        final[k] = v

        ops = final.get("changed_ops", []) or []
        tests = final.get("generated_tests", []) or []
        skills = sorted({(t.skill.value if hasattr(t.skill, "value") else str(t.skill)) for t in tests})

        critic = final.get("critic_verdict")
        critic_passed = bool(critic.passed) if critic is not None else None
        critic_iters = int(final.get("critic_iterations") or 0)
        critic_missing = len(critic.missing_dimensions) if critic is not None else 0
        context_docs = len((final.get("retrieved_docs") or []))

        expected = _SEED_MANIFEST.get(seed_path.name)
        if expected is None:
            return HarnessResult(
                seed_name=seed_path.name,
                op_detected=ops[0].name if ops else None,
                tests_generated=len(tests),
                skills_used=skills,
                fault_likely_caught=False,
                detail="no manifest entry",
                critic_passed=critic_passed,
                critic_iterations=critic_iters,
                critic_missing_count=critic_missing,
                context_docs_used=context_docs,
                oracle_mode=self.oracle_mode,
            )

        if self.oracle_mode == "real":
            prep_err = self._prepare_real_oracle()
            if prep_err is not None:
                # Real oracle unavailable on this host — fall back to heuristic
                # so the run still produces a result, but record the reason.
                caught, detail = _fault_caught_heuristic(tests, expected)
                return HarnessResult(
                    seed_name=seed_path.name,
                    op_detected=ops[0].name if ops else None,
                    tests_generated=len(tests),
                    skills_used=skills,
                    fault_likely_caught=caught,
                    detail=f"{detail} (real oracle unavailable: {prep_err})",
                    critic_passed=critic_passed,
                    critic_iterations=critic_iters,
                    critic_missing_count=critic_missing,
                    context_docs_used=context_docs,
                    oracle_mode="real_fallback_heuristic",
                    build_status="skipped",
                )

            caught, detail, build_status, failed_names = _fault_caught_real(
                expected, seed_path,
                project_dir=self.project_root,
                build_dir=self.build_dir,
            )
            return HarnessResult(
                seed_name=seed_path.name,
                op_detected=ops[0].name if ops else None,
                tests_generated=len(tests),
                skills_used=skills,
                fault_likely_caught=caught,
                detail=detail,
                critic_passed=critic_passed,
                critic_iterations=critic_iters,
                critic_missing_count=critic_missing,
                context_docs_used=context_docs,
                oracle_mode="real",
                build_status=build_status,
                failed_test_names=failed_names,
            )

        caught, detail = _fault_caught_heuristic(tests, expected)
        return HarnessResult(
            seed_name=seed_path.name,
            op_detected=ops[0].name if ops else None,
            tests_generated=len(tests),
            skills_used=skills,
            fault_likely_caught=caught,
            detail=detail,
            critic_passed=critic_passed,
            critic_iterations=critic_iters,
            critic_missing_count=critic_missing,
            context_docs_used=context_docs,
            oracle_mode="heuristic",
        )

    def run_all(self) -> BenchmarkReport:
        seeds = sorted(self.seeds_dir.glob("*.diff"))
        results = [self.run_one(p) for p in seeds]
        caught = sum(1 for r in results if r.fault_likely_caught)
        return BenchmarkReport(total=len(results), caught=caught, results=results)


# ---------- CLI entry point ----------


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="tinyinfer-agent fault-injection benchmark")
    parser.add_argument("--seeds", type=Path, default=Path("benchmark/seeds"))
    parser.add_argument("--mock", action="store_true", help="Force mock LLM regardless of env")
    parser.add_argument("--no-rag", action="store_true", help="Skip retriever construction")
    parser.add_argument(
        "--oracle",
        choices=["heuristic", "real"],
        default=None,
        help="Override FAULT_ORACLE env. 'real' applies the mutation, builds, and runs ctest.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write the report to this path")
    args = parser.parse_args(argv)

    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"

    from agent.llm import get_llm_client

    llm = get_llm_client()
    retriever = None
    if not args.no_rag:
        try:
            from agent.llm import get_embedding_client
            from agent.rag import build_default_retriever

            retriever = build_default_retriever(embedder=get_embedding_client())
        except ImportError:
            retriever = None

    harness = FaultInjectionHarness(
        args.seeds, llm=llm, retriever=retriever, oracle=args.oracle,
    )
    report = harness.run_all()

    print(f"Accuracy: {report.caught}/{report.total} = {report.accuracy:.1%}  (oracle={harness.oracle_mode})")
    for r in report.results:
        flag = "OK " if r.fault_likely_caught else "X  "
        bs = f" build={r.build_status}" if r.build_status else ""
        print(f"  {flag} {r.seed_name}: op={r.op_detected} tests={r.tests_generated}{bs} ({r.detail})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "FaultInjectionHarness",
    "BenchmarkReport",
    "HarnessResult",
]
