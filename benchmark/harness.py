"""tinyinfer-agent 的 fault-injection benchmark。

布局：
    benchmark/seeds/<NN>_<op>_<fault>.diff       — 已知 bad commit
    benchmark/harness.py                          — 跑 agent 过每个 seed，算 accuracy

两种 oracle 可选，用 `FAULT_ORACLE` env 或构造器 arg 切：

- ``heuristic``（默认，快，无需 cmake）：generated test 名字 OR rationale 词面命中
  fault 维度关键词就算"caught"。当前 seed 集上饱和到 100% —— 见 devlog 05。
- ``real``（慢，需要 docker + cmake）：把 mutation 应用到 clean checkout，build，
  只跑 agent 生成的测试；"caught" = 至少一个 generated test 在 mutated source 上
  fail。不再看 fault 维度关键词。

`_fault_caught_heuristic` 和 `_fault_caught_real` 拆开，所以同一次 benchmark
run 用其中一个不会和另一个串台。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# manifest：seed 文件名 -> 期望 (target_op, 维度关键词)。
# 一个 seed 算"caught"的条件：生成的测试集中至少有一个 test 的名字 OR rationale
# （转小写后）能命中任一 group 里全部关键词。group 之间是 OR 关系。
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
    # 更细的信号 —— 二值 `fault_likely_caught` 饱和时 ablation 还能靠这些
    # 区分配置。structural/LLM critic 的判决是 RAG 真正体现价值的地方；
    # 测试数量是 self-consistency 和 LLM shape planning 体现的地方。
    critic_passed: Optional[bool] = None
    critic_iterations: int = 0
    critic_missing_count: int = 0
    context_docs_used: int = 0
    # Real-oracle 记账字段。"heuristic" 结果这些都用默认值。
    oracle_mode: str = "heuristic"
    build_status: Optional[str] = None  # "ok" | "build_failed" | "configure_failed" | "skipped"
    failed_test_names: list[str] = field(default_factory=list)
    # Oracle-feedback reflexion 记账。attempts=1 是常规路径；attempts=2 表示
    # 我们带着 oracle-miss lesson 又跑了一遍 generator。
    oracle_reflexion_used: bool = False
    attempts: int = 1


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
                    "oracle_reflexion_used": r.oracle_reflexion_used,
                    "attempts": r.attempts,
                }
                for r in self.results
            ],
        }


def _fault_caught_heuristic(generated_tests, expected: dict) -> tuple[bool, str]:
    """agent 生成的测试有没有词面上瞄准 fault 维度？

    匹配逻辑：每个 `dim_group`（关键词候选 tuple），任一 test 的小写化名字 OR
    rationale 含有任一关键词就算这个 group 命中。一个 seed 算 "caught" 当且
    仅当**至少一个 group** 命中——group 之间是 OR 关系。
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


# ---------- real oracle（build + ctest）----------


def _parse_failed_test_names(ctest_stdout: str) -> list[str]:
    """从 ctest 输出里抽出 `*** Failed` 行的测试名。

    ctest 行格式：'<n>/<m> Test #<i>: <name> ........***Failed   <t> sec'
    """
    failed: list[str] = []
    for line in ctest_stdout.splitlines():
        if "***Failed" in line or "***Exception" in line:
            # 取 "Test #N: " 和 dots 之间的内容
            marker = "Test #"
            idx = line.find(marker)
            if idx < 0:
                continue
            after = line[idx + len(marker):]
            # 名字跟在 colon-space 后
            if ":" not in after:
                continue
            tail = after.split(":", 1)[1].strip()
            # 名字一直到遇到 dots 或空白为止
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
    """ctest -R 正则：只跑这个 op 的 agent-generated executable。

    build 出来的可执行名形如 `test_matmul_fp32_generated`、
    `test_matmul_fp32_perf_generated`、`test_matmul_fp32_memory_generated`，
    都带 `<op>` 这个 stem。
    """
    return rf"test_{op}.*_generated"


# Oracle 反馈 lesson。**故意不**把 mutation diff 的具体内容回灌给 generator——那
# 等于给 LLM 抄答案。lesson 只告诉它："你这一轮的测试在 mutated source 上全过
# 了（漏了真实 bug），请重新考虑你没 vary 过哪些维度"。剩下的让 LLM 自己想。
ORACLE_MISS_LESSON = (
    "Oracle 反馈：你这一轮生成的测试在 mutated source 上全部通过了，"
    "也就是说有一个真实 bug 被漏过了。仅靠 input shape 的多样性没能触发它。"
    "请重新考虑除 shape 之外的维度：output buffer 初值（预填非零 vs 零初始化）、"
    "输入/输出 alias、非默认 stride、NaN / Inf / denormal 输入、"
    "以及 API 合约本身（哪些参数必须被写入，哪些只读）。"
    "在下一轮 shape plan 里至少加一个 case 覆盖以上任一维度。"
)


def _fault_caught_real(
    expected: dict,
    seed_path: Path,
    *,
    project_dir: Path,
    build_dir: Path,
) -> tuple[bool, str, str, list[str]]:
    """应用 mutation，重新 build，只跑 agent 生成的测试。

    返回 (caught, detail, build_status, failed_names)。

    "caught" 语义：build 成功 AND 至少一个 generated test 在 mutated source
    上失败。Build 失败单独标记，不让它被无声当成 false negative。
    """
    from agent.tools import cmake_driver, mutation

    op = expected["op"]
    seed_diff = seed_path.read_text(encoding="utf-8")

    # seed 里的 patch path 是 `tinyinfer/src/*.cpp`，以 repo root 为锚。
    # project_dir 就是 oracle 用的 repo root。
    with mutation.apply_diff(seed_diff, project_dir):
        b = cmake_driver.build(build_dir)
        if not b.ok:
            tail = (b.stderr or b.stdout)[-400:]
            return (False, f"build_failed: {tail}", "build_failed", [])

        r = cmake_driver.ctest(build_dir, test_filter=_real_oracle_test_filter(op))

    # with 块退出后 src 树已经恢复。ctest 的判决依然反映 mutated source 上的结果。
    failed = _parse_failed_test_names(r.stdout)
    if r.ok:
        return (False, "all generated tests passed on mutated source", "ok", [])
    if not failed:
        # ctest 返回非零但没解析出失败名 —— 把 tail 打出来让用户能 debug
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
        oracle_reflexion: Optional[bool] = None,
    ):
        self.seeds_dir = Path(seeds_dir)
        self.llm = llm
        self.retriever = retriever
        # graph 懒构建：配错时仍能看到明确的报错，而不是 import 时就崩。
        self._graph = None

        mode = (oracle or os.getenv("FAULT_ORACLE", "heuristic") or "heuristic").lower()
        if mode not in ("heuristic", "real"):
            raise ValueError(f"unknown oracle mode {mode!r}; expected 'heuristic' or 'real'")
        self.oracle_mode = mode

        # Oracle 反馈式 reflexion：real oracle 报 missed 时，带一条通用
        # miss-lesson 再跑一次 generator。默认 off（一次跑会变 2x wall time
        # + 2x LLM 成本）。只在 real-oracle mode 下有意义；heuristic mode 下
        # 这个开关被静默忽略。
        if oracle_reflexion is None:
            env_val = (os.getenv("ORACLE_REFLEXION") or "off").lower().strip()
            oracle_reflexion = env_val in ("1", "true", "yes", "on")
        self.oracle_reflexion = bool(oracle_reflexion)

        # Repo root 同时作为 cmake 的 source root 和 mutation 应用的 root
        # （seeds 里的 path 是 `tinyinfer/src/...`）。默认就是 cwd。
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
        """跑 mutation 之前，在干净源码上做一次性 configure + build。

        后续每个 seed 的 build 都是增量 —— 只有 mutation 改过的文件会被重编。
        返回错误字符串说明 oracle 在本机不可用（caller 可以 fallback 回
        heuristic）。
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
            # 长 ablation 跑里偶发的网络故障不能让整个 benchmark 挂掉 ——
            # 一次 transient failure 不该作废其他 9 个 good results。
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

    def _run_graph(self, seed_path: Path, *, lessons: Optional[list[str]] = None) -> dict:
        """跑一遍完整 pipeline，把最终聚合的 state dict 返回。

        抽出来是为了让 oracle-feedback reflexion 能用不同的 `lessons` 调两次。
        AgentState 的 list reducer 在 critic loop 里会跨迭代 append；这里我们
        每次 attempt 都喂一个全新的 initial state。
        """
        graph = self._ensure_graph()
        initial: dict = {
            "diff": seed_path.read_text(encoding="utf-8"),
            "diff_path": str(seed_path),
            "trace_id": f"bench-{seed_path.stem}",
            "critic_iterations": 0,
        }
        if lessons:
            initial["reflexion_lessons"] = list(lessons)

        final: dict = {}
        for chunk in graph.stream(initial, stream_mode="updates"):
            for _name, update in chunk.items():
                for k, v in update.items():
                    if isinstance(v, list) and isinstance(final.get(k), list):
                        final[k] = final[k] + v
                    else:
                        final[k] = v
        return final

    def _run_one_inner(self, seed_path: Path) -> HarnessResult:
        final = self._run_graph(seed_path)

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
                # Real oracle 在本机不可用 —— fallback 回 heuristic 让 run
                # 仍能产出结果，但 detail 里说清楚原因。
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

            # Oracle-feedback reflexion：第一轮 agent 的测试在 mutated source
            # 上全过（= 漏抓 fault）且 build 成功，且用户开了开关，就带着一条
            # oracle-miss lesson 再跑一次 graph + oracle。lesson 是通用的——
            # 不告诉 generator mutation 内容是什么，只告诉它"漏了"，并提示
            # 该往哪些维度方向找（output buffer init / aliasing / stride / NaN）。
            attempts = 1
            reflexion_used = False
            if (
                self.oracle_reflexion
                and not caught
                and build_status == "ok"
            ):
                reflexion_used = True
                attempts = 2
                # 用全新 state 重跑 pipeline，带上 oracle-miss lesson。
                # install_tests_node 按 (op, suffix) 命名落盘，所以新一轮
                # 写的 .cpp 会覆盖前一轮。
                final = self._run_graph(seed_path, lessons=[ORACLE_MISS_LESSON])
                tests = final.get("generated_tests", []) or []
                skills = sorted({
                    (t.skill.value if hasattr(t.skill, "value") else str(t.skill))
                    for t in tests
                })
                critic = final.get("critic_verdict")
                critic_passed = bool(critic.passed) if critic is not None else None
                critic_iters = int(final.get("critic_iterations") or 0)
                critic_missing = len(critic.missing_dimensions) if critic is not None else 0
                context_docs = len((final.get("retrieved_docs") or []))

                caught, detail, build_status, failed_names = _fault_caught_real(
                    expected, seed_path,
                    project_dir=self.project_root,
                    build_dir=self.build_dir,
                )
                detail = (
                    f"[oracle-reflexion attempt 2] {detail}"
                    if caught else
                    f"[oracle-reflexion attempt 2, still missed] {detail}"
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
                oracle_reflexion_used=reflexion_used,
                attempts=attempts,
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
    parser.add_argument(
        "--oracle-reflexion",
        action="store_true",
        default=None,
        help=(
            "If the real oracle reports missed, re-run the generator once with a "
            "generic oracle-miss lesson and re-evaluate. Doubles wall time on missed seeds."
        ),
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
        oracle_reflexion=args.oracle_reflexion,
    )
    report = harness.run_all()

    suffix = " +oracle-reflexion" if harness.oracle_reflexion else ""
    print(
        f"Accuracy: {report.caught}/{report.total} = {report.accuracy:.1%}  "
        f"(oracle={harness.oracle_mode}{suffix})"
    )
    for r in report.results:
        flag = "OK " if r.fault_likely_caught else "X  "
        bs = f" build={r.build_status}" if r.build_status else ""
        attempts = f" attempts={r.attempts}" if r.attempts > 1 else ""
        print(f"  {flag} {r.seed_name}: op={r.op_detected} tests={r.tests_generated}{bs}{attempts} ({r.detail})")

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
