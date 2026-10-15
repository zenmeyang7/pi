"""embedding 层：调本地 Ollama 的 /api/embed。

只用标准库 urllib，和项目里 StreamFn 的风格保持一致，不引额外依赖。
"""
import json
import urllib.error
import urllib.request

from . import config


class EmbedError(RuntimeError):
    pass


def _post(url, body, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise EmbedError(
            f"连不上 {url}（{e.reason}）。请确认 Ollama 已启动：`ollama serve`"
        ) from e
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise EmbedError(
            f"Ollama 返回 {e.code}：{detail}\n"
            f"模型 {config.EMBED_MODEL} 可能没拉取：`ollama pull {config.EMBED_MODEL}`"
        ) from e


def _embed_batch(batch):
    data = _post(f"{config.OLLAMA_BASE}/api/embed",
                 {"model": config.EMBED_MODEL, "input": batch},
                 config.EMBED_TIMEOUT)
    vecs = data.get("embeddings")
    # 顺序必须和输入严格一致，否则向量和文本会错位——这是最隐蔽的一类 bug
    if not vecs or len(vecs) != len(batch):
        raise EmbedError(f"embedding 数量不匹配：{len(vecs or [])} != {len(batch)}")
    return vecs


def embed(texts):
    """str -> 单条向量；list[str] -> list[向量]。自动分批。"""
    if isinstance(texts, str):
        return _embed_batch([texts])[0]
    out = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        out.extend(_embed_batch(texts[i:i + config.EMBED_BATCH]))
    return out


def health_check():
    """起服务前验一下，比跑到一半报错友好。"""
    try:
        req = urllib.request.Request(f"{config.OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read())
    except Exception as e:
        return False, f"Ollama 未就绪：{e}"
    names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
    if config.EMBED_MODEL.split(":")[0] not in names:
        return False, f"未找到模型 {config.EMBED_MODEL}，请先 `ollama pull {config.EMBED_MODEL}`"
    return True, f"Ollama OK，模型 {config.EMBED_MODEL} 可用"


if __name__ == "__main__":
    ok, msg = health_check()
    print(("OK  " if ok else "FAIL") + msg)
    if ok:
        v = embed("def rerank(query, docs): ...")
        print(f"向量维度 {len(v)}，前 3 维 {[round(x, 4) for x in v[:3]]}")
