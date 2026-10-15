import os

# ---------- embedding：Ollama ----------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "16"))
EMBED_TIMEOUT = 120

# ---------- rerank：cross-encoder，两个后端 ----------
RERANK_URL = os.getenv("RAG_RERANK_URL", "http://localhost:8080/v1/rerank")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED") == "1"
RERANK_TIMEOUT = 60
# 后端：auto（先 llama-server 后 onnx）/ onnx / llama-server / none。
# onnx 是本地进程内推理，不用起服务也不用 torch——只要 onnxruntime + tokenizers
# 和一份 ONNX 模型。bge-reranker-base（278M）从魔搭下：
#   https://www.modelscope.cn/models/BAAI/bge-reranker-base
RERANK_BACKEND = os.getenv("RAG_RERANK_BACKEND", "auto")
RERANK_ONNX_DIR = os.getenv("RAG_RERANK_ONNX_DIR", "rag/models/bge-reranker-base")
# ONNX 推理设备：auto（按 cuda→dml→cpu 顺序尝试，第一个可用胜出）/ cuda / dml / cpu。
# 有 NVIDIA GPU 时装 onnxruntime-gpu，auto 自动走 CUDAExecutionProvider；
# 装不上的环境（无 GPU / 只装了 CPU 版 onnxruntime）自动 fallback 到 CPU，不会报错。
RERANK_PROVIDER = os.getenv("RAG_RERANK_PROVIDER", "auto")
# 重排批大小：GPU 上 16 吞吐更高（单批摊薄 kernel launch 开销）；
# 纯 CPU 跑可降到 8（内存吃紧时），可按机器调。
RERANK_BATCH = int(os.getenv("RAG_RERANK_BATCH", "16"))

# ---------- 切片 ----------
CHUNK_MAX_LINES = 60        # 单个 chunk 行数上限
CHUNK_OVERLAP_LINES = 8     # 长节点/长文件按窗口切时的重叠行数

# ---------- 检索 ----------
RECALL_K = 20               # 每路召回的候选数（rerank 的输入池）
FINAL_K = 5                 # 最终返回给模型的条数
RRF_K = 60                  # RRF 平滑常数，论文默认 60
# 融合的权重设置
RRF_WEIGHTS = [0.3, 1.0]
# 词法路路由：query 含稀有标识符时才启用 BM25（详见 store.has_distinctive_code_token）。
# 关掉即退化为"无条件双路召回"，用于消融对比。
ROUTE_LEXICAL = True

INDEX_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml",".pdf",".docs",".c",".c++"}
INDEX_IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".vscode", ".idea", "dist", "build", "index", "sessions",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".workbuddy",
    "tests",
    "models", ".ollama",
}
# 按文件名排除，不分目录：
# - adversarial.py = 对抗题正文，进语料会让对抗题变漏检题（检索直接命中它）
# - result.md = 只有数字、没有代码的评测报告，进语料会把数字当检索素材污染召回
INDEX_IGNORE_FILES = {"adversarial.py", "result.md"}
# 大文件闸门：超过这个字节数的不进索引。代码文件不会有这么大，
# 卡这道闸是为了让"忘了加 ignore 目录"不至于把整个索引搞废。
INDEX_MAX_FILE_BYTES = 300_000
