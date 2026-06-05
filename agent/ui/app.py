"""Streamlit UI for tinyinfer-agent.

Run it (not import it):

    pip install -e ".[ui]"
    streamlit run agent/ui/app.py

Three panels:
  1. Input  — paste a diff or upload a .diff/.patch file, pick LLM provider + toggles
  2. Tests  — generated C++ test files, one expander per (op, skill)
  3. Critic — verdict badge + coverage report + retrieved RAG context

The app constructs the same LangGraph pipeline the CLI uses, so behaviour is
identical to `python -m agent.cli analyze`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Guard against accidental `import agent.ui.app` — Streamlit apps must be run
# via `streamlit run`, which executes this module as __main__.
if __name__ != "__main__" and not os.getenv("STREAMLIT_RUNTIME_ENV") and "streamlit" not in os.getenv("_", ""):
    # Soft guard: don't hard-raise (pytest may import the module for coverage),
    # but the heavy UI code below is gated on _running_under_streamlit().
    pass


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _collect_final_state(graph, initial: dict) -> dict:
    final: dict = {}
    for chunk in graph.stream(initial, stream_mode="updates"):
        for _name, update in chunk.items():
            for k, v in update.items():
                if isinstance(v, list) and isinstance(final.get(k), list):
                    final[k] = final[k] + v
                else:
                    final[k] = v
    return final


def main() -> None:
    import streamlit as st

    from agent.graph import build_graph
    from agent.llm import get_embedding_client, get_llm_client

    st.set_page_config(page_title="tinyinfer-agent", layout="wide")
    st.title("tinyinfer-agent — regression test generator")
    st.caption("Paste a C++ operator diff; the agent generates numerical / perf / memory regression tests.")

    # ---------- sidebar controls ----------
    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox(
            "LLM provider",
            ["mock", "qwen", "deepseek", "openai", "glm", "anthropic"],
            index=0,
            help="mock needs no API key. Others read keys from .env.",
        )
        use_rag = st.checkbox("Enable RAG retrieval", value=True)
        sc_n = st.slider("Self-consistency N (0 = off)", 0, 5, 0)
        st.markdown("---")
        st.caption("Execution (cmake/ctest) is intentionally not wired into the UI.")

    # ---------- input panel ----------
    st.subheader("1. Input diff")
    uploaded = st.file_uploader("Upload a .diff / .patch", type=["diff", "patch", "txt"])
    sample_dir = Path("demo/diffs")
    sample_names = [p.name for p in sorted(sample_dir.glob("*.diff"))] if sample_dir.is_dir() else []
    sample_choice = st.selectbox("…or pick a sample diff", ["(none)"] + sample_names)

    diff_text = ""
    if uploaded is not None:
        diff_text = uploaded.read().decode("utf-8", errors="replace")
    elif sample_choice != "(none)":
        diff_text = (sample_dir / sample_choice).read_text(encoding="utf-8")
    diff_text = st.text_area("Diff content", value=diff_text, height=240)

    run = st.button("Generate tests", type="primary", disabled=not diff_text.strip())

    if not run:
        return

    # ---------- run pipeline ----------
    os.environ["LLM_PROVIDER"] = provider
    os.environ["SELF_CONSISTENCY_N"] = str(sc_n)

    retriever = None
    if use_rag:
        try:
            retriever = __import__(
                "agent.rag", fromlist=["build_default_retriever"]
            ).build_default_retriever(embedder=get_embedding_client())
        except ImportError as exc:
            st.warning(f"RAG disabled: {exc}")

    with st.status("Running agent pipeline…", expanded=False) as status:
        graph = build_graph(llm=get_llm_client(), retriever=retriever)
        initial = {
            "diff": diff_text,
            "diff_path": "<ui>",
            "trace_id": "ui",
            "critic_iterations": 0,
            "execute_requested": False,
        }
        final = _collect_final_state(graph, initial)
        status.update(label="Done", state="complete")

    # ---------- output panels ----------
    ops = final.get("changed_ops", []) or []
    tests = final.get("generated_tests", []) or []
    critic = final.get("critic_verdict")
    docs = final.get("retrieved_docs", []) or []

    col_tests, col_critic = st.columns([3, 2])

    with col_tests:
        st.subheader("2. Generated tests")
        st.write(f"Detected ops: {', '.join(o.name for o in ops) or '(none)'}")
        # Group by (op, file_suffix) so each generated .cpp shows once.
        seen: dict[str, str] = {}
        for t in tests:
            suffix = ""
            if isinstance(t.inputs, dict) and isinstance(t.inputs.get("file_suffix"), str):
                suffix = f"_{t.inputs['file_suffix']}"
            seen.setdefault(f"test_{t.op_name}{suffix}_generated.cpp", t.cpp_source or "")
        if not seen:
            st.info("No tests generated.")
        for fname, src in seen.items():
            with st.expander(fname):
                st.code(src, language="cpp")

    with col_critic:
        st.subheader("3. Critic")
        if critic is not None:
            if critic.passed:
                st.success("Critic PASS")
            else:
                st.error("Critic FAIL")
            if critic.missing_dimensions:
                st.write("Missing dimensions:")
                st.write(critic.missing_dimensions)
            if critic.coverage_report:
                st.write("Coverage report:")
                st.json(critic.coverage_report)
        else:
            st.info("No critic verdict.")

        if docs:
            st.subheader("RAG context")
            for d in docs[:5]:
                doc = d.get("doc", {})
                st.markdown(f"**{doc.get('doc_id','')}** (score={d.get('score',0):.3f})")
                st.caption(doc.get("text", ""))


if _running_under_streamlit():
    main()
elif __name__ == "__main__":
    # Allow `python agent/ui/app.py` to print a helpful hint.
    print("This is a Streamlit app. Run it with:\n    streamlit run agent/ui/app.py")
