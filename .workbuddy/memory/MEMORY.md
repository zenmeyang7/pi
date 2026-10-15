# pi 项目长期记忆

## 技术决策

- **reranker 推理后端**：用 DirectML（onnxruntime-directml），不用 onnxruntime-gpu。后者 wheel 不带 cuBLAS/cuDNN，需额外装 CUDA Toolkit 13 + cuDNN 9，成本过高。DirectML 零额外配置，RTX 4050 上 DmlExecutionProvider 直接生效。provider 由 `RAG_RERANK_PROVIDER` 控制（auto 按 cuda→dml→cpu 顺序尝试），batch 由 `RAG_RERANK_BATCH` 控制（默认 16）。
- **省探活**：.env 设 `RAG_RERANK_BACKEND=onnx`，跳过 auto 模式对 localhost:8080 的 5s llama-server 探活超时。
- **装包镜像**：uv add 默认走 PyPI 在国内会卡死，统一用 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

## 性能数据

- rerank benchmark（2026-10-14）：DirectML，20 docs/query，5 次 p50=376ms（vs CPU 1-2s/query）。
