"""跑检索评测，输出三个维度的三张表。

    uv run python eval/run.py --index rag/index --cases eval/data/cases.jsonl
    RAG_EVAL_GEN=1 uv run python eval/run.py --md eval/result.md

三个维度（口径定义见 eval/metrics.py 与 eval/generation.py）：

    ① 检索质量    Recall@池子 / MRR@交付 / nDCG@交付
    ② 上下文质量  Context Recall / Context Precision
    ③ 生成可靠性  Faithfulness / Hallucination（要真调模型，默认关）

报告只出这三张表。环境信息走 bullet 行 —— 一旦它也是表格，就成了第四张表。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 必须在 import rag.* **之前**：rag.config 是导入时就求值的，
# 晚一步加载，RAG_RERANK_ENABLED / RAG_RERANK_BACKEND 拿到的还是空值，
# 于是 reranker 静默降级成 none，重排档被跳过还不知道为什么。
# agent/config 那边同理：DEEPSEEK_API_KEY 也要在这一步进来，生成档才拿得到。
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from eval.adversarial import ADVERSARIAL_CASES                       # noqa: E402
from eval.generation import (DEFAULT_CACHE, DEFAULT_MODEL, PROMPT_VERSION,  # noqa: E402
                             ResponseCache, run_generation)
from eval.metrics import (STRATEGIES, build_report_tables,  # noqa: E402
                          evaluate_strategy_pool, load_cases, mean_context_metrics,
                          print_table_aligned, render_table)
from rag import config                                               # noqa: E402
from rag.embedder import embed, health_check                         # noqa: E402
from rag.retriever import Retriever                                  # noqa: E402
from rag.reranker import get_reranker                                # noqa: E402
from rag.store import IndexStore, normalize                          # noqa: E402

# 生成档只跑这四档：前三档是累加的组件，① 不给任何上下文 —— 那是"没有 RAG"的
# 诚实基线，也是幻觉率的上界（模型对私有代码库一无所知，只能编）。
STAGE_ROWS = [
    ("① 无上下文", None),
    ("② 混合", "hybrid"),
    ("③ 重排", "hybrid+rerank_noroute"),
    ("④ 路由", "hybrid+rerank+route"),
]

RERANK_UNAVAILABLE_NOTE = "重排服务不可用，本档未评测（降级成纯 RRF 的结果不算重排的成绩）"


def _stub_row(strategy):
    """重排不可用时的占位行：行必须留着，不然读者以为这个策略没测过。"""
    return {"strategy": strategy, "stage": None, "per_case": [],
            "recall": None, "mrr": None, "ndcg": None,
            "p50_ms": None, "p95_ms": None, "errors": None,
            "context_recall": None, "context_precision": None}


def precompute_query_vectors(cases, adv):
    """本批 query 一次性批量 embedding，后面注入给 Retriever 复用。

    不预计算的话，每次 retrieve 都要为**单条** query 发一次 Ollama 请求（约 2s）：
    表一的 p50/p95 整列都会变成同一个 2170ms，检索策略之间的差别（BM25 几十微秒、
    重排几百毫秒）全被这一次网络往返盖住，时间列就白给了。
    """
    qs = [c["query"] for c in cases] + [a["query"] for a in adv]
    return {q: normalize(v) for q, v in zip(qs, embed(qs))}


def make_embed_fn(cache):
    """缓存命中就用批量算好的向量；临时 query（--show）按单条补算一次。"""
    def embed_fn(query):
        if query not in cache:
            cache[query] = normalize(embed([query])[0])
        return cache[query]
    return embed_fn


def build_retrieval_rows(retriever, cases, store, strategies, pool_k, final_k):
    rows = []
    for s in strategies:
        if s.startswith("hybrid+rerank") and retriever.reranker.name != "cross-encoder":
            print(f"[eval] {RERANK_UNAVAILABLE_NOTE}：{s}")
            rows.append(_stub_row(s))
            continue
        r = evaluate_strategy_pool(retriever, cases, s, pool_k, final_k)
        ctx = mean_context_metrics(r["per_case"], cases, store.chunks, final_k)
        rows.append({
            "strategy": s, "stage": r["stage"], "per_case": r["per_case"],
            "recall": r[f"recall@{pool_k}"], "mrr": r[f"mrr@{final_k}"],
            "ndcg": r[f"ndcg@{final_k}"], "p50_ms": r["p50_ms"],
            "p95_ms": r["p95_ms"], "errors": r["errors"],
            "context_recall": ctx["recall"], "context_precision": ctx["precision"],
        })
    return rows


def collect_stage_hits(retriever, cases, adv, rows_by, pool_k, final_k):
    """给生成档准备每档的检索结果（可解答集 + 对抗集各一份，都截到交付条数）。"""
    stage_hits = []
    for label, strategy in STAGE_ROWS:
        if strategy is None:                       # ① 不给上下文
            stage_hits.append((label, [[] for _ in cases], [[] for _ in adv]))
            continue
        row = rows_by.get(strategy)
        if row is None or not row["per_case"]:      # 重排不可用：这一档整行打 —
            stage_hits.append((label, None, None))
            continue
        hits = [h[:final_k] for h in row["per_case"]]
        adv_hits = [retriever.retrieve_by_strategy(a["query"], strategy, k=pool_k)[:final_k]
                    for a in adv]
        stage_hits.append((label, hits, adv_hits))
    return stage_hits


def generation_rows(stage_hits, cases, adv, model, args, rows_by):
    """跑生成档，返回与 STAGE_ROWS 对齐的行（跳过的档给 None）。"""
    import math

    runnable = [(label, q, a) for label, q, a in stage_hits if q is not None]
    if not runnable:
        return [None] * len(STAGE_ROWS)
    cache = ResponseCache(args.cache_path)
    print(f"[eval] 生成+判官：{model}，缓存 {args.cache_path}，并发 {args.workers}"
          f"，prompt 版本 {PROMPT_VERSION}"
          + (f"，冒烟只跑前 {args.gen_limit} 条" if args.gen_limit else ""))
    results = run_generation(cases, adv, runnable, model, k=args.final_k,
                             cache=cache, workers=args.workers,
                             limit=args.gen_limit, api_key=args.api_key,
                             base_url=args.base_url)
    print(f"[eval] 缓存命中 {cache.hits} / 未命中 {cache.misses}"
          f"（命中=这次没发请求；重跑同一份缓存，数字必须完全一致）")
    done = {r["stage"]: r for r in results}
    return [done.get(label) for label, _ in STAGE_ROWS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="rag/index")
    ap.add_argument("--cases", default="eval/data/cases.jsonl")
    ap.add_argument("--pool-k", type=int, default=config.RECALL_K,
                    help="召回池大小（也是评测时的检索 k）")
    ap.add_argument("--final-k", type=int, default=config.FINAL_K,
                    help="交付给模型的条数（上下文/生成两档的口径）")
    ap.add_argument("--md", default=None, help="把三张表写进这个 markdown（覆盖写）")
    ap.add_argument("--show", default=None, help="给这个 query 打印实际检索结果")
    ap.add_argument("--wlex", type=float, default=None, help="词法路融合权重")
    ap.add_argument("--wvec", type=float, default=None, help="向量路融合权重")
    ap.add_argument("--only", default=None, help="只跑这一个策略（权重消融时用，省时间）")
    ap.add_argument("--gen", action="store_true",
                    default=os.getenv("RAG_EVAL_GEN") == "1",
                    help="跑生成可靠性档（真调模型，有 API 成本）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gen-limit", type=int, default=None, help="冒烟：只跑前 N 条")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-path", default=DEFAULT_CACHE)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--json", default=None, help="把原始指标另存为 json")
    args = ap.parse_args()

    # Ollama 没起时必须停在这里。evaluate_strategy_pool 会把检索异常吞掉记进 errors，
    # 一路跑下去会打出一份"全 0 但看着像真结果"的报告 —— 那比报错危险得多。
    ok, msg = health_check()
    if not ok:
        print(f"[eval] embedding 服务不可用，拒绝出一份假报告：{msg}", file=sys.stderr)
        return 2

    store = IndexStore(os.path.abspath(args.index)).load()
    if not store.chunks:
        print("[eval] 索引是空的：先 uv run python rag/build_index.py", file=sys.stderr)
        return 2
    cases = load_cases(args.cases)
    adv = ADVERSARIAL_CASES
    print(f"[eval] 索引 {len(store.chunks)} chunk，可解答 {len(cases)} 条，"
          f"对抗 {len(adv)} 条，召回池 {args.pool_k} / 交付 {args.final_k}")

    reranker = get_reranker()
    print(f"[eval] reranker = {reranker.name}")
    weights = None
    if args.wlex is not None or args.wvec is not None:
        weights = [args.wlex if args.wlex is not None else 1.0,
                   args.wvec if args.wvec is not None else 1.0]
    vec_cache = precompute_query_vectors(cases, adv)
    print(f"[eval] query 向量已批量预计算 {len(vec_cache)} 条"
          f"（p50/p95 因此只量检索本身，不含 embedding 的网络往返）")
    retriever = Retriever(store, reranker, weights=weights,
                          embed_fn=make_embed_fn(vec_cache))

    if args.show:
        from rag.retriever import format_hits
        print(format_hits(retriever.retrieve(args.show, k=args.final_k)))
        return 0

    strategies = [args.only] if args.only else STRATEGIES
    rows = build_retrieval_rows(retriever, cases, store, strategies,
                                args.pool_k, args.final_k)
    rows_by = {r["strategy"]: r for r in rows}

    gen_rows = [None] * len(STAGE_ROWS)
    if args.gen:
        stage_hits = collect_stage_hits(retriever, cases, adv, rows_by,
                                        args.pool_k, args.final_k)
        gen_rows = generation_rows(stage_hits, cases, adv, args.model, args, rows_by)
    else:
        print("[eval] 跳过生成可靠性档（要跑加 --gen 或 RAG_EVAL_GEN=1，"
              "它会真调模型生成答案并判 faithfulness）")

    emit(rows, gen_rows, args, store, cases, adv, reranker)
    return 0


# ==================== 报告：就这三张表 ====================

def emit(rows, gen_rows, args, store, cases, adv, reranker):
    pool_k, final_k = args.pool_k, args.final_k

    tables = build_report_tables(rows, gen_rows, pool_k, final_k,
                                 stage_labels=STAGE_ROWS,
                                 gen_model=args.model if args.gen else None)

    bullets = [
        f"索引 {len(store.chunks)} chunk · 可解答 {len(cases)} 条 · "
        f"对抗 {len(adv)} 条（域外 "
        f"{sum(1 for a in adv if a['kind'] == 'out_of_scope')} / 近义 "
        f"{sum(1 for a in adv if a['kind'] == 'near_miss')}）",
        f"召回池 {pool_k} / 交付 {final_k} · reranker = {reranker.name} · "
        f"gold-chunk 相关判据 = path 相同且行区间重叠",
        "query 向量批量预计算：表一的 p50/p95 只量检索本身（BM25/RRF/重排），"
        "不含 embedding 的 HTTP 往返，只能横向比",
    ]
    if args.gen:
        bullets.append(f"生成与判官：{args.model}（T=0，磁盘缓存 {os.path.basename(args.cache_path)}，"
                       f"prompt 版本 {PROMPT_VERSION}）")
    else:
        bullets.append("生成可靠性档未跑（加 --gen 或 RAG_EVAL_GEN=1）")

    for t in tables:
        print_table_aligned(t["title"], t["cols"], t["rows"], t["footnotes"])
    if args.gen:
        failed = [r for r in gen_rows if r and (r["errors"] or r["parse_failed"])]
        detail = "，".join(f"{r['stage']}: 生成失败 {r['errors']} / 判官失败 {r['parse_failed']}"
                          for r in failed) or "无"
        print(f"\n  · 生成档失败统计：{detail}（失败样本已从对应分母剔除，没有当 0 分算）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"retrieval": rows, "generation": gen_rows},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"[eval] 原始指标已写入 {args.json}")

    if args.md:
        lines = ["# RAG 评测结果（三维度）", ""]
        lines += [f"- {b}" for b in bullets] + [""]
        for t in tables:
            lines += render_table(t["title"], t["cols"], t["rows"], t["footnotes"])
        with open(args.md, "w", encoding="utf-8") as f:   # 覆盖写：报告永远只有三张表
            f.write("\n".join(lines) + "\n")
        print(f"[eval] 三张表已写入 {args.md}")


if __name__ == "__main__":
    raise SystemExit(main())
