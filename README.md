# pi —— 从零手写的 Agent 运行时

不依赖 LangChain / 任何 Agent SDK，用标准库 + 少量必要包从零实现：
`urllib` 手写 SSE、ThreadPoolExecutor 工具并发、会话树（MySQL 持久化）、
tiktoken 上下文压缩，外加一套带评测闭环的代码库 RAG。

---

## 快速开始

```bash
uv sync

# ① 构建代码索引（需要 Ollama 已启动且已拉取 bge-m3）
uv run python rag/build_index.py

# ② 装本地 cross-encoder（重排档用，不需要 llama.cpp / torch）
uv add onnxruntime tokenizers

# ③ 四档检索评测（要 Ollama；③④ 两档还要 rerank；生成档要 DeepSeek key）
uv run python eval/run.py                          # 出三维度对比表
uv run python eval/run.py --md eval/result.md      # 刷新结果表到文件

# ④ 真机 REPL（需要 DeepSeek key + MySQL，见 .env）
uv run python main.py
```

> 外部依赖两档：**RAG** 需要 Ollama（`ollama serve` + `ollama pull bge-m3`）；
> **REPL** 额外需要 `.env` 里的 DeepSeek key 和一个可用的 MySQL。
>
> 重排后端默认 onnx（`onnxruntime` 进程内跑），也可用 `RAG_RERANK_BACKEND=llama-server`
> 起 `llama-server --rerank`；都没有自动降级成纯 RRF。

真机 REPL 里的指令：

| 会话内（动 lane 光标） | 跨会话（动文件） |
|---|---|
| `/branch` 开分支 · `/main` 回主线 · `/back` 光标退一步 | `/new <名>` · `/open <名>` · `/fork <名>` · `/list` |

---

## 架构

```
main.py            REPL 入口 + Steering（后台线程读 stdin）
   ↓
agent/             Agent 运行时
   loop.py         双层循环：外层管会话生命周期，内层跑工具轮次
   protocol.py     数据契约 + stream_deepseek（手写 urllib + SSE）
   context.py      tiktoken 计数 → 超限 → 摘要压缩
   storage.py      会话树持久化：parentId 成链 + lane 光标（MySQL）
tools/             files.py: read_file / write_file（路径沙箱）
                   search.py: search_code（包装 rag/ 检索链路）
rag/               AST 切片 → BM25 + 向量 → RRF → cross-encoder 重排
                   build_index.py · index/ 索引数据 · models/ 模型权重
eval/              metrics.py · adversarial.py · run.py · result.md
                   data/cases.jsonl · cache/gen_cache.jsonl
```

数据流：Steering 收输入 → `run_loop` → `streamAssistantResponse`（从树读历史 → 转协议 →
压缩 → 调 LLM → 流式吐字）→ 有 `tool_calls` 就 `executeToolCalls` → 结果 append 回树 →
循环到模型 stop → 阻塞等下一句。

---

## 技术点

1. **双层循环 + steering** — 内层跑工具轮次、外层管会话；deferred 暂存防止用户中途插话把 `tool_call` 和 `tool_result` 拆到两条分支。
2. **手写 SSE + 永不抛错** — 工具调用按 `index` 归位拼接；未知 `finish_reason` 映射成 `error`；任何异常都转成 error 事件 yield 出去。
3. **工具并行 + 保序 + 截断保护** — 提交顺序即收集顺序；批次里有 `sequential` 工具就整批串行；`length` 截断时拒绝执行半截 JSON。
4. **会话树** — `parentId` 成链 + lane 当光标，LLM 读的永远是 `tree.for_path()`，回溯 / 开分支 / fork 只是移动指针。
5. **上下文压缩** — 超限时摘要压缩只改"给模型看的视图"，原始消息一条不丢。
6. **代码库 RAG** — AST 语法级切片、代码专用分词、BM25 + 向量双路召回、RRF 按排名融合、cross-encoder 重排（onnx / llama-server 两后端，都没有则降级）。
7. **评测闭环** — 四档递进（切片 → +混合检索 → +重排 → +路由）× 两指标（召回率 Recall@20 + 对抗幻觉率），结果表见下。
8. **路径沙箱** — `commonpath` 校验，拦住提示注入导致的越界读写。

---

## 评测结果

索引 208 chunk · 可解答集 39 条 · 对抗集 18 条（域外 10 / 近义 8）。召回池 20、交付 5。生成与判官均用 deepseek-chat（T=0）。

### 检索质量

| 策略 | 档位 | Recall@20 | MRR@5 | nDCG@5 | p50_ms | p95_ms | errors |
|---|---|---|---|---|---|---|---|
| lexical | 单路基线 | 0.462 | 0.343 | 0.366 | 0.036 | 0.621 | 0 |
| vector | 单路基线 | 0.974 | 0.660 | 0.714 | 4.827 | 6.747 | 0 |
| hybrid | ② 混合 | 0.974 | 0.565 | 0.610 | 5.341 | 7.702 | 0 |
| hybrid_routed | ④ 无重排 | 0.974 | 0.677 | 0.727 | 5.213 | 7.101 | 0 |
| hybrid+rerank_noroute | ③ 重排 | 0.974 | 0.628 | 0.694 | 475.072 | 548.490 | 0 |
| hybrid+rerank+route | ④ 路由 | 0.974 | 0.628 | 0.694 | 471.445 | 553.650 | 0 |

> Recall@20 量召回池有没有 gold；MRR@5 / nDCG@5 量交付给模型的前 5 条排得好不好。
> p50/p95 只量检索本身（query 向量已预计算，不含 embedding 往返），只能横向比。

### 上下文质量

| 策略 | 档位 | Context Recall | Context Precision |
|---|---|---|---|
| lexical | 单路基线 | 0.410 | 0.345 |
| vector | 单路基线 | 0.833 | 0.662 |
| hybrid | ② 混合 | 0.705 | 0.569 |
| hybrid_routed | ④ 无重排 | 0.833 | 0.682 |
| hybrid+rerank_noroute | ③ 重排 | 0.846 | 0.629 |
| hybrid+rerank+route | ④ 路由 | 0.846 | 0.629 |

> 量交付给模型的那 5 条：相关 ⟺ 与 gold 标注行区间重叠。

### 生成可靠性

| 档位 | Faithfulness↑ | Hallucination↓ | 域外 | 近义 | 可解答 | 作答率 | 平均声明数 |
|---|---|---|---|---|---|---|---|
| ① 无上下文 | — | 0.000 | 0.000 | 0.000 | — | 0.000 | — |
| ② 混合 | 0.996 | 0.010 | 0.000 | 0.000 | 0.031 | 0.821 | 6.094 |
| ③ 重排 | 1.000 | 0.042 | 0.000 | 0.125 | 0.000 | 0.872 | 6.412 |
| ④ 路由 | 1.000 | 0.042 | 0.000 | 0.125 | 0.000 | 0.872 | 6.412 |

> Faithfulness = 能被上下文支持的论断占比（只在真正作答的样本上取均值）；Hallucination = 域外 / 近义 / 可解答三项的宏平均。
> ① 无上下文是不给检索结果的基线，prompt 要求"上下文不足就拒答"，所以通常全拒答（作答率 0）。

评测集：`eval/data/cases.jsonl`（39 条可解答）、`eval/adversarial.py`（18 条对抗）、`eval/cache/`（生成缓存）。
