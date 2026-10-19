"""生成可靠性：真调 LLM 生成答案，再让判官判它编没编。

前两个维度（检索质量 / 上下文质量）量的是"喂给模型的东西对不对"，但**对的东西喂进去，
模型也不一定老实**。这一维量的是最后一公里：拿到上下文之后有没有照实说。

    检索 top-k -> 拼上下文 -> 生成（agent 用的同一个模型）-> 判官
                                |                          |
                                +-- 不忠实就拒答？          +-- 论断级：有几条论断能被上下文支持
                                                           +-- 回答级：有没有断言一个答案
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 传输层参数：与 agent/protocol.py 保持同一套语义（那边是流式的，这边是一次性请求，
# 因为判官要 JSON、生成要 temperature=0，而 protocol 的 body 里根本没有 temperature）。
# 本地 4B 模型一次生成（含思考链）比云端慢一个量级，60s 会误判成超时重试。
HTTP_TIMEOUT = int(os.getenv("LLM_TIMEOUT") or 180)
MAX_RETRIES = 2
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
RETRY_BASE_DELAY = 0.5

# 改了任何 prompt / 判据，这个版本号 +1 —— 它参与缓存 key，等于一次性作废旧缓存。
# 不作废的后果：新 rubric 下跑出来的报告，一半数字来自旧答案，没人看得出来。
PROMPT_VERSION = 1

# 生成端和判官都用 LLM_MODEL（默认 deepseek-flash）——它必须和 REPL 里那个是同一个：
# 换了模型名，缓存 key 变了，旧缓存自动作废，不会拿旧答案当新成绩。
DEFAULT_MODEL = os.getenv("LLM_MODEL") or "deepseek-flash"
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "gen_cache.jsonl")

# 固定拒答串：生成端被要求"答不了就原样输出这一行"，检测端子串匹配即可。
REFUSAL = "根据提供的上下文无法回答该问题。"


class ChatError(RuntimeError):
    """一次调用彻底失败（重试过、或错误本身不值得重试）。绝不静默转成空字符串。"""


# ==================== 传输 ====================

def chat(model, messages, *, api_key=None, base_url=None, temperature=0.0,
         json_object=False, timeout=HTTP_TIMEOUT, max_retries=MAX_RETRIES,
         chat_fn=None):
    """一次性对话调用，返回回答正文。

    chat_fn 是留给测试的注入点（假函数就不会联网），和 Retriever 注入 embed_fn 一个路子。
    不能复用 agent/protocol.py 的 stream_deepseek：它只走 SSE，且 body 里没有 temperature，
    拿不到 T=0；而 agent/ 这一层是冻结的，不该为了评测去改生产链路。
    """
    if chat_fn is not None:
        return chat_fn(model, messages, json_object=json_object)

    api_key = api_key or os.getenv("LLM_API_KEY") or ""
    base_url = (base_url or os.getenv("LLM_BASE_URL")
                or "https://api.deepseek.com").rstrip("/")
    if not api_key or not base_url:
        raise ChatError("缺少 LLM_API_KEY / LLM_BASE_URL（先 load_dotenv）")

    # 评测要可复现：thinking 开着时 DeepSeek 会忽略 temperature，且带推理开销
    body = {"model": model, "messages": messages,
            "temperature": temperature, "stream": False,
            "thinking": {"type": "disabled"}}
    if json_object:
        # 判官走 JSON mode：省掉"从一段散文里抠 JSON"的脆弱环节。
        # 代价是模型不再输出推理过程，但判据本身足够机械，不需要它思考。
        body["response_format"] = {"type": "json_object"}
    # ensure_ascii=False + 显式 utf-8：这台机器默认按 GBK 编解码，中文进去就炸
    # （agent/protocol.py 的 SSE 解析那里记过同一个坑）。
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:200]
            last = ChatError(f"HTTP {e.code}: {detail}")
            if e.code not in RETRY_STATUS:
                raise last          # 401/400 重试一万次也是同样的错
        except Exception as e:      # 超时 / DNS / 连接重置：值得重试
            last = ChatError(f"{type(e).__name__}: {e}")
        if attempt < max_retries:
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last


# ==================== 缓存 ====================

class ResponseCache:
    """磁盘缓存：key -> 回答/判词。一次评测几百次调用，重跑时要能免费且一模一样。

    temperature=0 **不能**保证逐位可复现（MoE 路由、批处理都会让同一 prompt 出不同字），
    所以"可复现"这件事只能由这份缓存来提供：报告里每个数字都能追到一条缓存的原文。
    写入用追加（write-through），跑到一半崩了也不会丢掉已完成的部分。

    ⚠️ 落盘必须串行。生成档默认 8 个线程并发，每个 put 各自 open("a") 后写一整行；
    而一条记录动辄几 KB，TextIOWrapper 会把它拆成多次 write()，两个线程的片段一
    交错，文件里就多出一行半个块 —— 下次加载直接 UnicodeDecodeError，
    整个缓存（几百次付费调用）读不出来。加锁是最小代价的修法。
    """

    def __init__(self, path=DEFAULT_CACHE):
        self.path = path
        self.data = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            # 逐字节读再按行解码：文件里混进过半个块时，只丢那一条，不让整个缓存读不出来
            with open(path, "rb") as fb:
                raw = fb.read()
            for chunk in raw.split(b"\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    rec = json.loads(chunk.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue    # 上次崩在写一半的那一行，跳过即可
                self.data[rec["key"]] = rec["value"]

    @staticmethod
    def key(model, messages):
        blob = json.dumps({"v": PROMPT_VERSION, "model": model, "messages": messages},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def get(self, model, messages):
        k = self.key(model, messages)
        if k in self.data:
            self.hits += 1
            return self.data[k]
        self.misses += 1
        return None

    def put(self, model, messages, value):
        k = self.key(model, messages)
        line = json.dumps({"key": k, "value": value}, ensure_ascii=False) + "\n"
        with self._lock:            # "查重 + 赋值 + 追加"必须是一个整体，见类注释
            if k in self.data:
                return
            self.data[k] = value
            if self.path:
                # 用 bytes 写：绕开 TextIOWrapper 的分块编码，一次 write 就是一个完整行
                with open(self.path, "ab") as f:
                    f.write(line.encode("utf-8"))


def cached_chat(cache, model, messages, **kw):
    """带缓存的 chat：命中就一次网络都不发。"""
    if cache is not None:
        hit = cache.get(model, messages)
        if hit is not None:
            return hit
    out = chat(model, messages, **kw)
    if cache is not None:
        cache.put(model, messages, out)
    return out


# ==================== prompt ====================

GEN_SYSTEM = (
    "你是代码库问答助手。你只能依据【上下文】里给出的代码片段回答问题。\n"
    "规则：\n"
    "1. 只使用【上下文】中的信息，不得用自己的先验知识补充任何没说过的内容。\n"
    "2. 如果【上下文】不足以回答问题，原样输出下面这一行，不要多一个字：\n"
    f"{REFUSAL}\n"
    "3. 提到文件、函数、行号时必须来自【上下文】，不许推测。\n"
)

JUDGE_SYSTEM = (
    "你是严格的评审。给你【上下文】【问题】和【回答】，请把【回答】拆成独立的论断，"
    "逐条判断它能否由【上下文】直接支持。\n"
    "判定规则：\n"
    "- 只依据【上下文】判断，不得使用你自己关于该领域的知识；【上下文】里没写的就算不支持。\n"
    "- 【回答】明确表示无法回答、或指出上下文不足时，论断数记 0。\n"
    "- 复述问题、过渡句、礼貌用语不算论断。\n"
    "只输出 json，不要任何解释："
    '{"n_claims": <论断总数>, "n_supported": <能被上下文支持的论断数>, "reason": "<一句话>"}'
)

ABSTAIN_SYSTEM = (
    "你是严格的评审。给你【上下文】【问题】和【回答】，判断这份回答属于哪一种：\n"
    "- asserted=true：断言了一个具体的答案（哪怕只是一部分）；\n"
    "- asserted=false：指出上下文不足以回答，或没有给出任何具体结论。\n"
    '注意："上下文提到了 X，但没有说明具体是哪一个/怎么做的" 属于 asserted=false。\n'
    "只输出 json，不要任何解释："
    '{"asserted": <true 或 false>, "reason": "<一句话>"}'
)


def build_context(hits):
    """把检索结果拼成给模型看的一段文本，附带 path:行号（grounding 的锚点）。"""
    if not hits:
        return "（没有检索到任何上下文）"
    blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {})
        loc = f"{meta.get('path', '?')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        blocks.append(f"[{i}] {loc}\n{h.get('text', '')}")
    return "\n\n".join(blocks)


def build_gen_messages(query, context_text):
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"【上下文】\n{context_text}\n\n【问题】{query}"}]


def build_judge_messages(query, context_text, answer):
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"【上下文】\n{context_text}\n\n【问题】{query}\n\n【回答】{answer}"}]


def build_abstention_messages(query, context_text, answer):
    return [{"role": "system", "content": ABSTAIN_SYSTEM},
            {"role": "user", "content": f"【上下文】\n{context_text}\n\n【问题】{query}\n\n【回答】{answer}"}]


# ==================== 解析 ====================

def extract_json(text):
    """从模型输出里抠出第一个 JSON 对象；抠不出来返回 None。

    三种现实里都会出现的脏活：```json 围栏、前面先说一段话、结尾带解释。
    用 raw_decode 从第一个 { 开始解析（而不是贪婪正则），第二种和第三种就能一起处理。
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower().startswith("json"):
            s = s[4:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    start = s.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def is_abstention(answer):
    """固定串判拒答：确定性、免费。判官只处理"既没拒答、又不像作答"的灰区。"""
    return REFUSAL in (answer or "")


# ==================== 编排 ====================

def _pmap(fn, items, workers):
    if workers and workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))
    return [fn(i) for i in items]


def _gen_once(query, hits, model, cache, chat_fn, api_key, base_url):
    """生成一条：返回 {"answer", "error"}。错误必须显式带出来，不能当成空回答。"""
    ctx = build_context(hits)
    try:
        ans = cached_chat(cache, model, build_gen_messages(query, ctx),
                          api_key=api_key, base_url=base_url,
                          temperature=0.0, chat_fn=chat_fn)
        return {"query": query, "answer": ans, "context": ctx, "error": None}
    except Exception as e:
        return {"query": query, "answer": "", "context": ctx,
                "error": f"{type(e).__name__}: {e}"}


def _judge_faithful(query, ctx, answer, model, cache, chat_fn, api_key, base_url):
    """论断级判词。解析失败记 error，不记 0 —— 记 0 会把判官的锅算到模型头上。"""
    try:
        raw = cached_chat(cache, model, build_judge_messages(query, ctx, answer),
                          api_key=api_key, base_url=base_url, temperature=0.0,
                          json_object=True, chat_fn=chat_fn)
    except Exception as e:
        return {"n_claims": None, "n_supported": None, "parse_failed": True,
                "error": f"{type(e).__name__}: {e}"}
    obj = extract_json(raw)
    if not obj or "n_claims" not in obj or "n_supported" not in obj:
        return {"n_claims": None, "n_supported": None, "parse_failed": True,
                "error": f"判官输出解析失败: {(raw or '')[:80]}"}
    try:
        n = max(int(obj["n_claims"]), 0)
        m = min(max(int(obj["n_supported"]), 0), n)
    except (TypeError, ValueError):
        return {"n_claims": None, "n_supported": None, "parse_failed": True,
                "error": "判官字段不是整数"}
    return {"n_claims": n, "n_supported": m, "parse_failed": False, "error": None}


def _judge_abstention(query, ctx, answer, model, cache, chat_fn, api_key, base_url):
    """灰区判词：asserted=True 表示"断言了一个答案"（= 该拒答却没拒）。"""
    try:
        raw = cached_chat(cache, model, build_abstention_messages(query, ctx, answer),
                          api_key=api_key, base_url=base_url, temperature=0.0,
                          json_object=True, chat_fn=chat_fn)
    except Exception as e:
        return {"asserted": None, "error": f"{type(e).__name__}: {e}"}
    obj = extract_json(raw)
    if not obj or "asserted" not in obj:
        return {"asserted": None, "error": "判官输出解析失败"}
    return {"asserted": bool(obj["asserted"]), "error": None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_generation(cases, adv, stage_hits, model=DEFAULT_MODEL, *, k=5,
                   chat_fn=None, cache=None, workers=8, limit=None,
                   api_key=None, base_url=None):
    """跑四档的生成 + 判官，返回每档一行指标。

    stage_hits: [(档位标签, 可解答集每条的检索结果, 对抗集每条的检索结果), ...]
                —— 检索结果由调用方（run.py）给，这里不再自己检索一遍。
                没有上下文的档（无检索基线）传 [[], [], ...] 即可。
    limit:      只跑前 N 条（冒烟用），但分母同步缩小，不假装跑全了。
    """
    out = []
    for label, q_hits, a_hits in stage_hits:
        pairs = list(zip(cases, q_hits))[:limit]
        apairs = list(zip(adv, a_hits))[:limit]

        gens = _pmap(lambda p: _gen_once(p[0]["query"], p[1], model, cache,
                                         chat_fn, api_key, base_url),
                     pairs, workers)
        agens = _pmap(lambda p: _gen_once(p[0]["query"], p[1], model, cache,
                                          chat_fn, api_key, base_url),
                      apairs, workers)

        answered = [g for g in gens if not g["error"] and not is_abstention(g["answer"])]
        # 判官只在"真的作答了"的样本上跑：拒答的论断数天然是 0，问了白问（还会诱导
        # 判官给个高分，把"全拒答"洗成满分忠实度）。
        judged = _pmap(lambda g: _judge_faithful(g["query"], g["context"], g["answer"],
                                                 model, cache, chat_fn, api_key, base_url),
                       answered, workers)
        for g, j in zip(answered, judged):
            g.update(j)

        # 对抗集：固定串能判的就不问判官，剩下的（灰区）才问
        grey = [(a, g) for (a, _), g in zip(apairs, agens)
                if not g["error"] and not is_abstention(g["answer"])]
        a_judged = _pmap(lambda t: _judge_abstention(t[0]["query"], t[1]["context"],
                                                     t[1]["answer"], model, cache,
                                                     chat_fn, api_key, base_url),
                         grey, workers)
        for (a, g), j in zip(grey, a_judged):
            g.update(j)
        for a, g in zip([p[0] for p in apairs], agens):
            # 拒答 → asserted=False；生成失败 → None（进不了分母，也不冒充"拒答成功"）
            g.setdefault("asserted", False if not g["error"] else None)

        scored = [g for g in answered if g.get("n_claims")]
        faithfulness = _mean([g["n_supported"] / g["n_claims"] for g in scored])
        claims = _mean([g["n_claims"] for g in scored])
        halluc_answerable = (
            _mean([1.0 if g["n_supported"] < g["n_claims"] else 0.0 for g in scored])
            if scored else None)

        def _rate_kind(kind):
            rows = [g for a, g in zip([p[0] for p in apairs], agens)
                    if a["kind"] == kind and g.get("asserted") is not None]
            return _mean([1.0 if g["asserted"] else 0.0 for g in rows]), len(rows)

        h_ood, n_ood = _rate_kind("out_of_scope")
        h_near, n_near = _rate_kind("near_miss")
        out.append({
            "stage": label,
            # Faithfulness：只对"作答了且有论断"的样本取均值（拒答不算满分）
            "faithfulness": faithfulness,
            # 总分 = 三项宏平均：等权，不被 10:8:39 的样本量比带偏
            "hallucination": _mean([h_ood, h_near, halluc_answerable]),
            "halluc_ood": h_ood,
            "halluc_near": h_near,
            "halluc_answerable": halluc_answerable,
            "answer_rate": len(answered) / len(pairs) if pairs else None,
            "claims": claims,
            "n_ood": n_ood,
            "n_near": n_near,
            # 两类失败分开数：生成挂了是网络/额度问题，判官挂了是 rubric 问题，
            # 混成一个数就没法判断该重跑还是该改 prompt
            "errors": sum(1 for g in gens + agens if g["error"]),
            "parse_failed": sum(1 for g in answered if g.get("parse_failed"))
                            + sum(1 for _, g in grey if g.get("asserted") is None),
            "detail": {"cases": gens, "adv": agens},
        })
    return out
