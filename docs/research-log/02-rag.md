# Devlog 02 —— RAG 检索（BM25 + dense + RRF）

**目标**：让测试生成器和 critic 站在已有上下文（算子语义、源码注释、
历史 fault-injection 总结）的肩膀上，不要每次都从零重新发明 shape 覆盖。

## 为什么用混合检索

纯 BM25 抓得住开发者打的精确字面（`matmul`、`simd`、`eps`、op 名），
但抓不住同义复述（"loop reorder for cache locality" vs "tile blocking"）。
纯 dense retrieval 反过来：泛化好，但在罕见技术 token 上失分。
我们用 **Reciprocal Rank Fusion** 把两者揉起来：

```
score(d) = w_bm25 * 1/(k + rank_bm25(d))  +  (1 - w_bm25) * 1/(k + rank_dense(d))
```

`w_bm25 = 0.4`、`k = 60`。是原始 RRF 论文给的值，作为起点很稳。
偏向 dense（60% 权重）是有意的 —— BM25 单独跑容易把"含罕见 token
但其实不相关"的文档抬太高。

## 后端选择：Chroma 默认，Qdrant / Milvus / numpy 可替

向量库被抽象在 `VectorStore` Protocol 后面，有四个后端：

- `NumpyVectorStore` —— 内存里的 cosine similarity（一次 matmul）。
  无外部依赖。默认用于测试，也是无依赖兜底。Dim **在第一次 upsert 时推断**，
  不在构造时声明。
- `ChromaVectorStore` —— 通过 `chromadb` 的嵌入式持久化。**默认的生产用后端**。
  sqlite 后端，无 daemon，Windows 上一装就跑通。LangChain 生态主流。
- `QdrantVectorStore` —— `qdrant-client` 的 local 模式（文件路径）或 server 模式
  （`QDRANT_URL`）。通过 `[rag-qdrant]` extra 可选。filter 语法最丰富；
  长大之后同一份 client 还能直接接 server。
- `MilvusVectorStore` —— `pymilvus` 接 milvus-lite。通过 `[rag-milvus]` extra 可选。
  给已经投入 Milvus 栈的用户保留。

我们考虑过 Milvus 当默认，但在 Windows 上它的安装路径会拖进 pyarrow + grpcio
（约 50MB 原生 wheel），常常被杀毒软件文件锁卡住 pip。Chroma 是单个 ~25MB
的 wheel，直接 work。

`search()` 上的 `filter_expr` 参数在每个后端都预留了。numpy 和 chroma
今天忽略它（chroma 的 filter 是 dict 不是字符串 —— 真接 metadata 过滤的时候
调用点要加一层薄适配）。dense-search-by-vector 这条路抽象是统一的；
metadata 过滤要 follow-up。

## Embedding 选择：DashScope text-embedding-v4

1024 维，OpenAI 兼容 API，跟 Qwen LLM 共用一个 key。在开发机上端到端验过。
`DASHSCOPE_API_KEY` 为空时工厂回退到 `MockEmbeddingClient` —— 这对 CI
很重要，无 secret 的 fork PR 仍能保持绿。

mock embedder 是**确定性的**（sha256(text) → seed → 单位范数高斯），
否则断言检索顺序的测试会 flaky。

## Embedding 磁盘缓存

一个小磁盘缓存（`.embedding_cache/<sha256>.json`），按 `(model, text)` key 存。
种子语料不至于每次跑测试都再付 API 钱。缓存 key 里有 embedder 的 `_model`
属性，所以 mock 向量和真实向量不会撞 key。

## 语料策划

三个来源，纯 ASCII：

1. **手写算子语义**（20 条）—— 每个算子、它的 edge shape、它的数值陷阱
   的简洁描述。比贴 ONNX spec 强 —— spec 大部分是套话。
2. **源码注释** 从 `tinyinfer/src/*.cpp` 里 scrape —— 抓住代码"为什么这么写"，
   精度极高。
3. **Fault-injection rationale**（10 条，每条对应一个 benchmark 种子）—— 短句
   描述故障是什么、什么测试能抓到。

纯 ASCII 是因为 rank-bm25 按空白分词，混入中文会静默掉 BM25 召回率。
换成中文分词器（jieba）作为后续 extra 留给未来。

## 接线

`retrieve_context` 节点夹在 `route_skill` 和 `generate_tests` 之间。
它从 changed-op 名 + summary + diff 头 400 字符构造 query，
调 `retriever.retrieve(query, top_k=5)`，把命中写成纯 dict 存到
`state["retrieved_docs"]`。是 dict 不是 `ScoredDoc` 对象，
因为 LangGraph state 要保持 JSON 可序列化给 trace writer。

critic 把 top-3 检索文档拼到它的 LLM prompt 里，作为一个
"Relevant context from corpus" 块。结构 critic 不变 —— 我们想让它保持
确定性。

## 已知 work 的部分

- 默认 retriever + mock embedder + Chroma store，端到端完整跑通。
  87 个 pytest case 全绿。
- 四个后端（numpy / chroma / qdrant local / milvus-lite）在 Windows
  开发 venv 上 round-trip insert + search 都通过。
- DashScope `text-embedding-v4` 对真实 diff 返回 1024 维向量
  （用 Qwen + 真实 embedding 端到端验过）。

## 留给新机器的事

- 跨后端的真实 metadata 过滤（Chroma 用 dict where、Qdrant 用 Filter
  model、Milvus 用表达式字符串 —— 第一次用要写一层薄适配）。
- Server 模式向量库（docker-compose 里有 Qdrant + Milvus 栈，写好没起）。
- 如果 corpus 引入中文字符串，BM25 路径要换中文分词器。
- LangSmith dashboard 可观察性（环境变量自动 trace，没设 key）。
