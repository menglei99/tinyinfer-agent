# 待新机器验证清单

本项目的这些条目**代码已经写好**，但**没在本机端到端验证过**。
计划是等新开发机到了在新机上一次性跑完。每条都写明跑什么、看什么。

## 1. C++ build + ctest 走 MCP

**代码路径**：`agent.graph.nodes.execute_tests_node` → `agent.mcp.client.call_tools` → `agent.mcp.server` → `cmake configure / build`、`ctest`。

**当前状态**：返回一个 SKIPPED 的 `ExecutionResult`，因为本机 PATH 上没 `cmake`。

**验证方法**：装好 cmake + 一个 C++17 编译器（MSVC、g++ 或 clang++），然后：

```bash
python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff --mock --execute
ctest --test-dir tinyinfer/build --output-on-failure
```

期望：`Execution (via MCP toolchain): PASS`。生成的 `.cpp` 文件应该能编译，
baseline 的 matmul / softmax / layernorm 测试应该全部通过。

## 2. 真实 LLM 的 ablation 数字（已部分完成）

**代码路径**：`agent.llm.get_llm_client`，`LLM_PROVIDER=qwen` + `DASHSCOPE_API_KEY`。
`agent.skills.numerical` 读 `SELF_CONSISTENCY_N`。

**当前状态**：本机已经用 Qwen 跑过 5 种配置的对比（mock_norag / qwen_norag / qwen_rag / qwen_sc3 / qwen_rag_reflexion），数据落在 `results/ablation.md`。详见 devlog 05、06。

剩余开销大的实验：

- 每个配置重跑 3-5 次取平均，把 temperature 噪声压下去
- 把 fault-injection seeds 从 10 扩到 30，让 accuracy 列脱离饱和

**复跑方法**：

```bash
LLM_PROVIDER=qwen EMBEDDING_PROVIDER=dashscope \
    python -m benchmark.harness --seeds benchmark/seeds --json results/qwen_rag.json

python -m benchmark.compare results/*.json --md results/ablation.md
```

## 3. Server 模式向量库（Docker）

**代码路径**：`docker-compose.yml`（Qdrant + 可选 Milvus profile）+ `agent.rag.store.QdrantVectorStore` / `MilvusVectorStore`。

**当前状态**：本机默认走 Chroma（嵌入式），已经验证可跑。Server 模式的 compose
栈代码已写，但没 `up` 过。

**验证方法**：

```bash
# Qdrant server（最简单，单容器）
docker compose up -d qdrant
VECTOR_STORE_BACKEND=qdrant QDRANT_URL=http://localhost:6333 \
    python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff

# Milvus standalone（重，etcd + minio + milvus 三件套）
docker compose --profile milvus up -d
VECTOR_STORE_BACKEND=milvus \
    python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff
```

**风险**：`MilvusVectorStore.__init__` 用的是 milvus-lite 的构造器（sqlite path）。
要连 standalone server，构造器需要加一个分支：如果 `MILVUS_HOST` / `MILVUS_URI`
有值就传 `uri=f"http://{host}:{port}"`，否则继续用 sqlite db_path。

## 4. 向量库 metadata filter 语法

**代码路径**：`agent.rag.store.*VectorStore.search` 上的 `filter_expr` 参数。

**当前状态**：numpy 和 chroma 后端忽略它；qdrant / milvus 路径转发它但目前
没人传值。

**风险**：Chroma 用 dict 形式的 `where={...}`，Qdrant 用 `Filter` model
对象，Milvus 用表达式字符串。第一次用 metadata 过滤时需要写一层薄的适配器，
或者把 caller 钉到具体某个后端。

## 5. LangSmith 自动 trace（已在本机验证）

**代码路径**：环境变量即开即用 —— LangGraph 在三个 env vars 设好时自动 instrument。
`agent/llm/client.py` 里的 `_maybe_wrap_for_langsmith` 把 OpenAI 客户端包成
`langsmith.wrappers.wrap_openai`，这样 LLM 调用也作为正式 LLM span 出现，
不只是 chain span。

**当前状态**：在本机已经验证 trace 真的进 dashboard 了（含 LangGraph chain
tree + Qwen `ChatOpenAI` LLM span 含 prompt/completion）。

**新机器复现**：

```bash
pip install langsmith
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=tinyinfer-agent
python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff
```

CLI banner 应该打印 `LangSmith: enabled (project=tinyinfer-agent)`。
访问 LangSmith dashboard，每次 pipeline 调用会有一条 run，里面是完整的
节点树和 LLM 调用 prompt/completion。

## 6. Streamlit UI 交互验收

**代码路径**：`agent/ui/app.py`。

**当前状态**：在本机用 Windows + Python 3.13 启动正常。模块 import 正常。
完整点选交互未做。

**验证方法**：`streamlit run agent/ui/app.py`，然后：

1. 从 sample dropdown 选 `sample_matmul_tiled.diff`。点 "Generate tests"。
   期望看到三个测试文件 expander（numerical / perf / memory）和一条 critic verdict。
2. 关掉 "Enable RAG retrieval"。重跑。期望右栏 "RAG context" 部分消失。
3. 把 self-consistency N 调到 3，provider 切到 `qwen`，重跑。
   期望在 trace 里看到 3 个 LLM 调用（TRACE_DIR=traces）。

## 7. 中文语料（未来）

种子 corpus 目前刻意全 ASCII，因为 `rank-bm25` 是按空白分词的，
混入中文会静默掉重要的检索 token。要支持中文算子文档或注释：

- 加一个 `[rag-zh]` extra，依赖 `jieba`
- 给 `HybridRetriever.__init__` 加一个 `tokenize: Callable[[str], list[str]]` 参数
- 默认仍然空白分词；中文字符（按 Unicode range 检测）走 `jieba.lcut`

不阻塞当前主线，只是未来工作。
