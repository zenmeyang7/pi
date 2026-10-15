"""重排层：cross-encoder rerank，两个后端任选。

**为什么不能拿 embedding 模型凑**：reranker 要把 (query, doc) 拼成一条序列
做联合编码；bge-m3 这类 bi-encoder 是各自编码完算余弦，拿它的向量重排等于
把召回时算过的分数再算一遍，没有增量。

onnx 后端（默认，零服务）：onnxruntime + tokenizers 在进程内跑一次 BERT 前向，
模型放 rag/models/bge-reranker-base/。
llama-server 后端：llama-server --rerank -m bge-reranker-v2-m3.gguf --pooling rank --port 8080

两个都拿不到就退回"保持召回原序"，检索链路照样跑得通。
"""
import json
import os
import urllib.error
import urllib.request

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Reranker:
    name = "base"

    def rerank(self, query, docs, top_k):
        raise NotImplementedError


class NullReranker(Reranker):
    """降级实现：不做任何重排，直接截断。用于消融实验的 baseline。"""
    name = "none"

    def rerank(self, query, docs, top_k):
        return list(range(min(top_k, len(docs))))


class LlamaServerReranker(Reranker):
    name = "cross-encoder"

    def rerank(self, query, docs, top_k):
        if not docs:
            return []
        body = {"model": config.RERANK_MODEL, "query": query, "documents": docs}
        req = urllib.request.Request(
            config.RERANK_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=config.RERANK_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            raise RuntimeError(f"rerank 服务调用失败：{e}") from e

        results = data.get("results") or []
        if not results:
            raise RuntimeError(f"rerank 返回空结果：{str(data)[:200]}")
        # 显式按分数降序，不依赖服务端是否已排序
        ranked = sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        return [r["index"] for r in ranked[:top_k]]


def _resolve_providers(spec):
    """把配置里的设备描述符翻译成 onnxruntime 的 providers 列表。

    onnxruntime 按列表顺序尝试，第一个能用的胜出——所以 auto 模式把
    CUDA 放最前、CPU 兜底，装了 onnxruntime-gpu 自动走 GPU，没装自动 fallback。
    指定值（cuda/dml/cpu）则只放那一个 + CPU 兜底，避免"想强制 CPU 却被
    auto 抢走 GPU"的歧义。
    """
    s = (spec or "auto").lower()
    if s == "auto":
        return ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    if s == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if s == "dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if s == "cpu":
        return ["CPUExecutionProvider"]
    # 不认识的值兜底走 CPU，并打一条 warn 让人知道配置没生效
    import warnings
    warnings.warn(f"未知 RAG_RERANK_PROVIDER={spec!r}，回退 CPU")
    return ["CPUExecutionProvider"]


class OnnxCrossEncoderReranker(Reranker):
    """本地 cross-encoder：onnxruntime 跑一次 BERT 前向，不依赖任何外部服务。

    name 故意和 llama-server 版**同名**：两者都是真的 cross-encoder，
    上层只关心"这次有没有真的重排"，不关心模型是哪家的。
    """
    name = "cross-encoder"
    MAX_LEN = 512          # bge-reranker 的上限，超了 tokenizer 自己截
    # 送给 cross-encoder 的文本**按字符截断**：query 是自然语言，chunk 是代码，
    # 一个 chunk 后半段的大段实现细节对"哪段更相关"几乎没有信息量，
    # 却会稀释开头的符号名和注释。实测（bge-reranker-base + 本仓库 24 条 query）：
    # 全文 0.750 / 截 400 字符 0.833 / 截 200 字符 0.792 —— 截太狠又会丢掉
    # 真正相关的段落，所以取中间那档。
    MAX_CHARS = 400

    def __init__(self, model_dir=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = model_dir or config.RERANK_ONNX_DIR
        if not os.path.isabs(d):
            d = os.path.join(ROOT, d)
        self.dir = d

        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.MAX_LEN)
        self.tok.enable_padding()
        # providers 按优先级尝试：auto 时 cuda→dml→cpu，列表里第一个可用胜出。
        # 这样 GPU 环境自动加速、纯 CPU 环境自动 fallback，不用改代码。
        providers = _resolve_providers(config.RERANK_PROVIDER)
        self.sess = ort.InferenceSession(os.path.join(d, "model.onnx"),
                                         providers=providers)
        # 实际生效的 provider（onnxruntime 落到列表里第一个能用的）：
        # 暴露出来方便排查"以为用 GPU 实际 fallback 到了 CPU"
        self.active_provider = self.sess.get_providers()[0]
        # 批大小：GPU 上默认 16（config.RERANK_BATCH），纯 CPU 可降到 8
        self.batch = config.RERANK_BATCH
        # 不同导出版本带的输入不一样（有的没有 token_type_ids），按实际有的喂
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def rerank(self, query, docs, top_k):
        if not docs:
            return []
        # 按长度分桶再分批：批内长度差得越多，padding 出来的无效 token 越多，
        # 而 attention 是 O(n²)——短的跟长的凑一批等于替长的买单。
        idxs = sorted(range(len(docs)),
                      key=lambda i: min(len(docs[i]), self.MAX_CHARS))
        scores = [0.0] * len(docs)
        for i in range(0, len(idxs), self.batch):
            batch = idxs[i:i + self.batch]
            enc = self.tok.encode_batch(
                [(query, docs[j][:self.MAX_CHARS]) for j in batch])
            feed = {"input_ids": [e.ids for e in enc],
                    "attention_mask": [e.attention_mask for e in enc]}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = [e.type_ids for e in enc]
            out = self.sess.run(None, feed)[0]
            # (B, 1) 取第 0 列；(B,) 直接用
            for j, o in zip(batch, out):
                scores[j] = float(o[0] if len(o.shape) and o.shape[-1] == 1 else o)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        return order[:top_k]


def _probe():
    """探活：故意发一个极小的 rerank 请求，确认服务真的在。"""
    body = {"model": config.RERANK_MODEL, "query": "ping", "documents": ["pong"]}
    req = urllib.request.Request(
        config.RERANK_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        json.loads(resp.read())


def get_reranker(warn=print):
    """按配置挑一个真 cross-encoder；都拿不到才降级。

    auto 的顺序是"先服务后本地"：llama-server 版吞吐高（常驻显存），
    本地 onnx 版零依赖。顺序反过来也行，只是本地版一旦装好就总是命中，
    服务版永远轮不上。
    """
    if not config.RERANK_ENABLED:
        return NullReranker()

    backend = config.RERANK_BACKEND
    tried = []

    if backend in ("auto", "llama-server"):
        try:
            _probe()
            return LlamaServerReranker()
        except Exception as e:
            tried.append(f"llama-server：{e}")
            if backend == "llama-server":
                warn(f"[rag] rerank 服务不可用，已降级为纯 RRF 融合（{e}）")
                return NullReranker()

    if backend in ("auto", "onnx"):
        try:
            return OnnxCrossEncoderReranker()
        except Exception as e:            # 没装包 / 没下模型，都走这里
            tried.append(f"onnx：{e}")

    warn("[rag] 没有可用的 cross-encoder，已降级为纯 RRF 融合：\n      "
         + "\n      ".join(tried)
         + "\n      修法二选一：uv add onnxruntime tokenizers 后跑 "
           "rag/download_reranker.py；\n      或起 llama-server --rerank "
           "-m bge-reranker-v2-m3.gguf --pooling rank --port 8080")
    return NullReranker()
