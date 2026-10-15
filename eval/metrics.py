"""评估层：没有数字的检索优化都是玄学。

评测集格式（JSONL，一行一例）：
    {"query": "在哪里做向量余弦相似度", "path": "rag/store.py", "lines": [110, 115]}
一个检索结果算命中，需要 path 相同 **且** 行区间有重叠——
只看 path 太松（一个文件几十个 chunk，蒙也能蒙中），看行号才有意义。

报告按三个维度组织，只出三张表（口径细节见各段注释）：

    ① 检索质量    Recall@RECALL_K（召回池）+ MRR/nDCG@FINAL_K（交付集）   —— 找没找到
    ② 上下文质量  Context Recall / Context Precision（喂给模型的那 k 条）  —— 干不干净
    ③ 生成可靠性  Faithfulness / Hallucination（见 eval/generation.py）    —— 编没编

前两个维度在这里算完（纯函数，不需要任何模型）；第三个维度要真调 LLM，
所以单独成模块 —— 没有数字的诚实，和没有数字的玄学一样没用。
"""
import json
import math
import time

from rag import config
from rag.retriever import Retriever

# 后两个是"累加档"：前者 = 混合 + 重排，后者 = 再 + 路由。
# 它们和 hybrid_routed 的区别只是有没有重排，写在一起才能一档只差一个组件。
STRATEGIES = ["lexical", "vector", "hybrid", "hybrid_routed",
              "hybrid+rerank_noroute", "hybrid+rerank+route"]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def is_hit(result, case):
    meta = result["meta"]
    if meta["path"] != case["path"]:
        return False
    gold = case.get("lines")
    if not gold:
        return True
    g0, g1 = gold
    r0, r1 = meta.get("start_line", 0), meta.get("end_line", 0)
    return r0 <= g1 and g0 <= r1      # 区间重叠


def recall_at_k(results, case, k):
    return 1.0 if any(is_hit(r, case) for r in results[:k]) else 0.0


def mrr(results, case, k):
    for rank, r in enumerate(results[:k]):
        if is_hit(r, case):
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(results, case, k):
    """只有一个 gold，所以 IDCG=1，nDCG 退化成 1/log2(rank+1)。"""
    for rank, r in enumerate(results[:k]):
        if is_hit(r, case):
            return 1.0 / math.log2(rank + 2)
    return 0.0


# ==================== 对抗集：幻觉率 ====================
# recall 只回答"该找的找没找到"，回答不了另一半——**不该找的时候有没有瞎找**
# （题目见 eval/adversarial.py，那边被排除出索引）。
#
# 检索层量不到模型最终有没有编，但量得到**幻觉的燃料**：有没有把一段无关的
# 上下文、以超过拒答线的自信度递上去。模型拿着它去编是下游的事，递燃料是
# 检索层能背的责。
#
# 自信度统一用**向量余弦**：四档参与打分的分别是 BM25 分数、余弦、RRF 排名、
# cross-encoder 分数，量纲互不相通，只有余弦四档都算得出来（chunk 向量本来
# 就在索引里），所以"自信度" = chunk 向量 · query 向量。


def vec_index(store):
    """chunk id -> 已归一化向量（build/load 都保证归一化，点积即余弦）。"""
    return {c["id"]: c["vec"] for c in store.chunks}


def score_hits(hits, qvec, vec_by_id, at=None):
    """一次检索结果里每条的自信度（保持返回顺序）。at=None 表示全部。"""
    out = []
    for h in (hits[:at] if at else hits):
        v = vec_by_id.get(h["id"])
        # chunk 被删/索引重建过就可能查不到向量：记 0（= 没自信度），而不是抛错
        out.append(math.sumprod(qvec, v) if v else 0.0)
    return out


def hallucinated(hits, qvec, vec_by_id, tau, at=1):
    """这套结果算不算幻觉：最像答案的那条，有没有越过拒答线 tau。

    at=1：只看 top-1（模型第一眼看到的，最可能被当成事实引用）。
    at=k：top-k 里只要有一条越线就算——模型会综合 k 条作答，这个口径更严。
    """
    s = score_hits(hits, qvec, vec_by_id, at)
    return bool(s) and max(s) >= tau


def hallucination_rate(runs, vec_by_id, tau, at=1):
    """runs: [(hits, qvec), ...] —— 对抗集上被判为幻觉的比例。越低越好。"""
    if not runs:
        return 0.0
    bad = sum(1 for hits, qvec in runs
              if hallucinated(hits, qvec, vec_by_id, tau, at))
    return bad / len(runs)


def percentile(xs, q):
    """线性插值分位数。xs 为空时给 0——调用方会因此跳过这条档。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def calibrate_tau(answerable_scores, q=0.10):
    """拒答线 = 可解答集 top-1 自信度的 q 分位（默认 p10）。

    为什么不写死 0.5 这种常数：换 embedding 模型、换语料，余弦的绝对刻度
    就整体平移，写死的阈值要么全判幻觉、要么全不判。取可解答集的 p10，
    语义是"答得出来的题里，最没底的那道有多没底"——低于这个水平就该拒答，
    而且这个数每次都跟着当前模型和语料自动重新标定。
    """
    return percentile(answerable_scores, q)


def abstention_auc(answerable_scores, adversarial_scores):
    """可分性：P(对抗题的自信度 ≥ 可解答题的自信度)。

    0   = 完美拒答：所有对抗题的自信度都低于所有可解答题；
    0.5 = 系统完全分不清"答得出"和"答不出"，幻觉和正确检索混在一起。
    这是阈值法的补充——幻觉率会随 tau 上下浮动，这个数不会。
    """
    if not answerable_scores or not adversarial_scores:
        return 0.0
    hit = sum(1 for a in adversarial_scores for p in answerable_scores if a >= p)
    return hit / (len(adversarial_scores) * len(answerable_scores))


# ==================== 档位指标（供 corpus_langchain 的四档递进用）====================
# 上面那组是"一个策略一份成绩"，这组是"一档一个口径"。两条新口径的由来：
#
# ① 档（切片）没有检索动作，报 recall 是**拿标注可达率冒充检索成绩**——
#    gold 地址只要落在某个 chunk 上，它就恒等于 1.000，跟检索器好坏无关。
#    那是数据集的健康度，名字该叫 Chunk Coverage。
#
# 拒答能力统一叫 FAR（False Acceptance Rate）：对**本来没有答案**的 query，
# 检索结果的最大余弦越过拒答线 tau 就算一次"错误接受"。它和 hallucination_rate
# 是同一个数的两个名字，所以实现上委派、不复制算术——两份实现迟早算出两个数。


def chunk_coverage(chunks, cases):
    """① 切片档的指标：gold 标注的代码区间有没有被切进至少一个 chunk。

    分母是 query 数（不是 chunk 数）：语义是"这套题里有多少条是**可检索的**"。

    返回 (coverage, uncovered, gold_chunk_counts)：
        coverage          有匹配 chunk 的 query 占比
        uncovered         一条都没落的 case 本身 —— 这属于**标注错了**，
                          不是检索错了；先修数据再谈成绩，否则后面的 recall
                          是在替标注背锅（与 check_set.py 的 ① 门同一件事）
        gold_chunk_counts 每条 case 命中多少个 chunk。>1 说明 gold 跨度大，
                          它在 recall@k 里更容易被吃满，解读逐条结果时要留意
    """
    gold_chunk_counts = [sum(1 for ch in chunks if is_hit(ch, case)) for case in cases]
    uncovered = [case for case, n in zip(cases, gold_chunk_counts) if n == 0]
    return (len(cases) - len(uncovered)) / max(len(cases), 1), uncovered, gold_chunk_counts


def far_from_scores(scores, tau):
    """给定每题"最大自信度"的一列数，算越过拒答线的比例（= FAR）。"""
    if not scores:
        return 0.0
    return sum(1 for v in scores if v >= tau) / len(scores)


def false_acceptance_rate(runs, vec_by_id, tau, at=1):
    """FAR：对抗集里"被系统错误接受"的比例，越低越好。

    与 hallucination_rate 是同一个数的两个名字：那边从检索结果现算余弦，
    这边是对外口径名。委派而不重写，保证两条路径永远给同一个数。
    """
    return hallucination_rate(runs, vec_by_id, tau, at)


def latency_stats(latencies):
    """p50 / p95。p95 是给线上看的：RAG 的延迟长尾很重（重排那档尤其），
    只看中位数会把"每 20 个请求有 1 个慢十倍"整个藏起来。

    用 percentile（线性插值）而不是 sorted[n//2]：n 为偶数时前者才是真正的
    分位数，p95 也只有插值口径说得通。n=61 时两者结果相同。
    """
    return {"p50_ms": percentile(latencies, 0.5),
            "p95_ms": percentile(latencies, 0.95)}


def routing_gain(before, after, cases, k):
    """Query Routing 的净收益：同一批 case 在两个策略下的**配对**比较。

    为什么必须配对：路由是个硬开关（query 里没有稀有标识符就整条关掉词法路），
    所以每条 case 只有三种结果——持平 / 救回（before 漏、after 中）/ 弄丢
    （before 中、after 漏）。只比两个总数的话，"0.567→0.667" 既可能是救回 6 条，
    也可能是救回 8 条同时弄丢 2 条；后者不能说路由有效，前者才敢说。

    返回的 gained/lost 是 case **下标**：文字标签由调用方拼（它手里有 cases，
    而且拼成什么样是报告层的事）。
    """
    n = max(len(cases), 1)
    b = [recall_at_k(r, c, k) for r, c in zip(before, cases)]
    a = [recall_at_k(r, c, k) for r, c in zip(after, cases)]
    return {
        "before": sum(b) / n,
        "after": sum(a) / n,
        "gain": (sum(a) - sum(b)) / n,
        "gained": [i for i, (x, y) in enumerate(zip(b, a)) if y > x],
        "lost": [i for i, (x, y) in enumerate(zip(b, a)) if y < x],
    }


def evaluate_strategy(retriever, cases, strategy, k):
    recalls, mrrs, ndcgs, latencies, errors = [], [], [], [], 0
    per_case = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            results = retriever.retrieve_by_strategy(case["query"], strategy, k=k)
        except Exception:
            errors += 1
            results = []
        per_case.append(results)
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(recall_at_k(results, case, k))
        mrrs.append(mrr(results, case, k))
        ndcgs.append(ndcg_at_k(results, case, k))
    n = max(len(cases), 1)
    return {
        "strategy": strategy,
        f"recall@{k}": sum(recalls) / n,
        f"mrr@{k}": sum(mrrs) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "errors": errors,
        "per_case": per_case,
    }


def run_eval(store, cases, k=None, strategies=None, reranker=None, embed_fn=None,
             weights=None):
    k = k or config.FINAL_K
    strategies = strategies or STRATEGIES
    # 复用同一个 retriever：reranker 只探活一次，BM25 索引只建一次
    retriever = Retriever(store, reranker, embed_fn, weights=weights)
    rows = []
    for s in strategies:
        if s.startswith("hybrid+rerank") and retriever.reranker.name != "cross-encoder":
            # 降级成纯 RRF 的结果不能算成"重排档"的成绩，宁可不出数
            print(f"[eval] 跳过 {s}：rerank 服务不可用（降级中）")
            continue
        rows.append(evaluate_strategy(retriever, cases, s, k))
    return rows


def print_table(rows, k=None):
    k = k or config.FINAL_K
    if not rows:
        print("没有可对比的结果。")
        return
    cols = ["strategy", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", "p50_ms", "errors"]
    widths = [max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_markdown(rows, out_path, k=None):
    k = k or config.FINAL_K
    cols = ["strategy", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", "p50_ms", "errors"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"对比表已写入 {out_path}")


CAT_LABELS = [
    ("function_location",   "函数定位"),
    ("class_responsibility", "类职责"),
    ("call_relation",       "调用关系"),
    ("semantic",            "语义"),
    ("vague_description",   "模糊描述"),
]

STAGES = [
    ("① 切片", None),
    ("② +混合检索", "hybrid"),
    ("③ +重排", "hybrid+rerank_noroute"),
    ("④ +路由", "hybrid+rerank+route"),
]


def print_category_table(rows, cases, k=None):
    """从 evaluate_strategy 的 per_case 结果里，按 category 分组算 recall。

    rows 里的 strategy 名和 STAGES 里的一致（hybrid / hybrid+rerank_noroute /
    hybrid+rerank+route）。rerank 不可用时对应策略不在 rows 里，那一列输出 —。
    """
    k = k or config.FINAL_K
    from collections import defaultdict
    cat_idxs = defaultdict(list)
    for i, c in enumerate(cases):
        cat_idxs[c.get("category", "other")].append(i)
    rows_by = {r["strategy"]: r for r in rows}
    if "hybrid+rerank+route" in rows_by:
        rows_by["hybrid+rerank"] = rows_by["hybrid+rerank+route"]

    print(f"\n==== 五方向 × 四档 recall@{k} ====")
    hdr = f"{'方向':<14}" + "".join(f"{lbl:>12}" for lbl, _ in STAGES)
    print(hdr)
    print("-" * len(hdr))
    for cat_key, cat_label in CAT_LABELS:
        idxs = cat_idxs.get(cat_key, [])
        if not idxs:
            continue
        vals = []
        for slabel, s in STAGES:
            if s is None:
                vals.append(1.0)
            elif s in rows_by:
                r = sum(recall_at_k(rows_by[s]["per_case"][i], cases[i], k)
                        for i in idxs) / len(idxs)
                vals.append(r)
            else:
                vals.append(None)
        line = f"{cat_label:<14}" + "".join(
            f"{v:>12.3f}" if v is not None else f"{'—':>12}" for v in vals)
        print(line)
    print("-" * len(hdr))
    all_vals = []
    for slabel, s in STAGES:
        if s is None:
            all_vals.append(1.0)
        elif s in rows_by:
            all_vals.append(rows_by[s][f"recall@{k}"])
        else:
            all_vals.append(None)
    line = f"{'全部':<14}" + "".join(
        f"{v:>12.3f}" if v is not None else f"{'—':>12}" for v in all_vals)
    print(line)


def write_category_markdown(rows, cases, out_path, k=None):
    """五方向 × 四档 markdown 表，追加到已有文件末尾。"""
    k = k or config.FINAL_K
    from collections import defaultdict
    cat_idxs = defaultdict(list)
    for i, c in enumerate(cases):
        cat_idxs[c.get("category", "other")].append(i)
    rows_by = {r["strategy"]: r for r in rows}
    if "hybrid+rerank+route" in rows_by:
        rows_by["hybrid+rerank"] = rows_by["hybrid+rerank+route"]

    labels = [lbl for lbl, _ in STAGES]
    lines = ["", "## 五方向 × 四档 recall@{}".format(k), "",
             "| 方向 | " + " | ".join(labels) + " |",
             "|" + "|".join(["---"] * (len(labels) + 1)) + "|"]
    for cat_key, cat_label in CAT_LABELS:
        idxs = cat_idxs.get(cat_key, [])
        if not idxs:
            continue
        vals = []
        for slabel, s in STAGES:
            if s is None:
                vals.append("1.000")
            elif s in rows_by:
                r = sum(recall_at_k(rows_by[s]["per_case"][i], cases[i], k)
                        for i in idxs) / len(idxs)
                vals.append(f"{r:.3f}")
            else:
                vals.append("—")
        lines.append(f"| {cat_label} | " + " | ".join(vals) + " |")
    all_vals = []
    for slabel, s in STAGES:
        if s is None:
            all_vals.append("1.000")
        elif s in rows_by:
            all_vals.append(f"{rows_by[s][f'recall@{k}']:.3f}")
        else:
            all_vals.append("—")
    lines.append(f"| **全部** | " + " | ".join(all_vals) + " |")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"五方向表已追加到 {out_path}")


# ==================== 上下文质量：Context Recall / Context Precision ====================
# RAGAS 的公式，但**相关性判据换成本仓库的 gold**：一条检索结果相关 ⟺ is_hit(chunk, case)。
#
# ⚠️ 两个必须讲明的局限（解读表二前先看这段）：
#   ① 本仓库没有参考答句，所以这里的 Context Recall 是 **gold-chunk 粒度**的召回
#      （gold 标注行区间落在几个 chunk 上），不是 RAGAS 那种"把参考答案拆成论断、数
#      有多少论断能在上下文里找到"的口径。
#   ② 33/39 条只有单个 gold chunk，而**单 gold 二元相关下 Context Precision ≡ 1/rank**，
#      和表一的 MRR 是同一个数。两个指标只有在 gold 跨多个 chunk 的 6 条上才分得开。
#      这是标签决定的，不是实现问题——不写清楚，表一表二看着像两个独立维度。


def gold_chunks(chunks, case):
    """与该 case 标注行区间重叠的全部 chunk（1 到多个）。

    is_hit 认的是 {"meta": ...} 这个形状，而 store 里的 chunk 把 meta 放在 ch["meta"]，
    所以这里包一层 —— 与 tests/test_rag.py 里切片档的算法保持一致。
    """
    return [ch for ch in chunks if is_hit({"meta": ch["meta"]}, case)]


def context_recall(hits, case, chunks, k):
    """召回的上下文覆盖了多少比例的 gold chunk。

    gold 集合为空（标注漂移/该 case 的 gold 不在索引里）→ 记 1.0 不在这里扣分：
    那是**数据**的问题，由 chunk_coverage 和 test_eval_1 负责暴露；在这里记 0
    会让一条标注错误伪装成检索退化。
    """
    gold = gold_chunks(chunks, case)
    if not gold:
        return 1.0
    gold_ids = {ch["id"] for ch in gold}
    got = {h["id"] for h in hits[:k]} & gold_ids
    return len(got) / len(gold_ids)


def context_precision(hits, case, chunks, k):
    """RAGAS/AP@k 口径：Σ_{i=1..k} (P@i · rel_i) / |召回到的相关条数|。

    分母是**召回到的相关条数**而不是 k —— 一条相关的都没召回时记 0.0，不是 1.0：
    "什么都没捞到"不该显得精准（空集上的 precision 是没定义的，取 0 才有意义）。
    """
    gold_ids = {ch["id"] for ch in gold_chunks(chunks, case)}
    rels = [1 if h["id"] in gold_ids else 0 for h in hits[:k]]
    n_rel = sum(rels)
    if not n_rel:
        return 0.0
    acc = 0.0
    for i, rel in enumerate(rels, start=1):
        if rel:
            acc += sum(rels[:i]) / i
    return acc / n_rel


def mean_context_metrics(hits_by_case, cases, chunks, k):
    """逐条算再取均值——先平均分母再算会把多 gold 的题权重放大。"""
    n = max(len(cases), 1)
    return {
        "recall": sum(context_recall(h, c, chunks, k)
                      for h, c in zip(hits_by_case, cases)) / n,
        "precision": sum(context_precision(h, c, chunks, k)
                         for h, c in zip(hits_by_case, cases)) / n,
    }


# ==================== 两阶段检索口径 ====================
# 检索链路是两段的：召回池（RECALL_K=20）→ 重排/截断 → 交付 top-k（FINAL_K=5）。
# 只用 FINAL_K 报 recall 说不清问题出在哪一段：池子里就没有（召回该背的锅）还是
# 池子里有但没被选出来（重排/截断的锅）。所以一次检索、两套口径分开报。
#
# 为什么能只跑一次：候选池 pool = fused[:recall_k] 与传进来的 k 无关
# （rag/retriever.py 的 retrieve），重排也是给整个池打分后才截断，所以
# **k=pool_k 的结果的前 final_k 条 ≡ k=final_k 的结果**。

# 策略 -> 档位标签：三张表共用同一套行标签，行与行才对得上
STRATEGY_STAGE = {
    "lexical": "单路基线",
    "vector": "单路基线",
    "hybrid": "② 混合",
    "hybrid_routed": "④ 无重排",
    "hybrid+rerank_noroute": "③ 重排",
    "hybrid+rerank+route": "④ 路由",
}


def evaluate_strategy_pool(retriever, cases, strategy, pool_k, final_k):
    """检索一次 pool_k，派生 recall@pool_k（召回池）与 mrr/nDCG@final_k（交付集）。"""
    latencies, errors, per_case = [], 0, []
    for case in cases:
        t0 = time.perf_counter()
        try:
            results = retriever.retrieve_by_strategy(case["query"], strategy, k=pool_k)
        except Exception:
            # 与 evaluate_strategy 同口径：错误只计数，异常不往上抛（一条 query 炸掉
            # 不该让整张表没有输出），但 errors 列必须留着——否则 Ollama 挂了会打出
            # 一份"全 0 但看着像真结果"的报告。
            errors += 1
            results = []
        per_case.append(results)
        latencies.append((time.perf_counter() - t0) * 1000)
    n = max(len(cases), 1)
    ls = latency_stats(latencies)
    return {
        "strategy": strategy,
        "stage": STRATEGY_STAGE.get(strategy, "—"),
        f"recall@{pool_k}": sum(recall_at_k(r, c, pool_k)
                                for r, c in zip(per_case, cases)) / n,
        f"mrr@{final_k}": sum(mrr(r, c, final_k)
                              for r, c in zip(per_case, cases)) / n,
        f"ndcg@{final_k}": sum(ndcg_at_k(r, c, final_k)
                               for r, c in zip(per_case, cases)) / n,
        "p50_ms": ls["p50_ms"],
        "p95_ms": ls["p95_ms"],
        "errors": errors,
        "per_case": per_case,
    }


# ==================== 报告层：三张表 ====================
# 三个维度各一张表，没有第四张。列定义集中在 emit_report 的调用处（eval/run.py），
# 这里只管排版——终端要对齐（CJK 占两列），markdown 要能被报告文件直接吃下。


def _w(s):
    """终端显示宽度：中日韩字符占两列。

    f-string 的 {:<14} 按**字符数**补空格，中文一字符两列宽，表就歪了。
    """
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s, width, right=False):
    gap = " " * max(width - _w(s), 0)
    return gap + s if right else s + gap


def _cell(v):
    """None 打 —（不是 0）：重排服务没起和"这一档考了 0 分"是两件事。"""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_table(title, cols, rows, footnotes=()):
    """cols: [(表头, 取值键)]；rows: [dict]。返回 markdown 行。"""
    lines = [f"## {title}", "",
             "| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(key)) for _, key in cols) + " |")
    if footnotes:
        lines += [""] + [f"> {f}" for f in footnotes]
    lines.append("")
    return lines


def print_table_aligned(title, cols, rows, footnotes=()):
    heads = [h for h, _ in cols]
    body = [[_cell(r.get(key)) for _, key in cols] for r in rows]
    widths = []
    for j, h in enumerate(heads):
        cells = [_w(row[j]) for row in body] or [0]
        widths.append(max(_w(h), *cells))
    print(f"\n==== {title} ====")
    print(" | ".join(_pad(h, w) for h, w in zip(heads, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(" | ".join(_pad(c, w) for c, w in zip(row, widths)))
    for f in footnotes:
        print(f"  · {f}")


# 三张表的列定义只此一份：run.py 的终端/markdown 输出和 tests 里的打印都取这里，
# 免得两处各写一套列，改了一处忘了另一处（评测报告最忌讳"数字对不上"）。
_GEN_METRIC_KEYS = ("faithfulness", "hallucination", "halluc_ood", "halluc_near",
                    "halluc_answerable", "answer_rate", "claims")


def build_report_tables(rows, gen_rows, pool_k, final_k, stage_labels=(), gen_model=None):
    gen_rows = list(gen_rows or [])
    out = []
    for i, (label, _) in enumerate(stage_labels or []):
        r = gen_rows[i] if i < len(gen_rows) else None
        if r is None:
            r = {"stage": label}
        elif gen_model is None:                 # 生成档没跑：列留着，值打 —
            r = dict(r)
            for k in _GEN_METRIC_KEYS:
                r[k] = None
        out.append(r)
    gen_rows = out

    t1 = ("一、检索质量",
          [("策略", "strategy"), ("档位", "stage"),
           (f"Recall@{pool_k}", "recall"), (f"MRR@{final_k}", "mrr"),
           (f"nDCG@{final_k}", "ndcg"), ("p50_ms", "p50_ms"),
           ("p95_ms", "p95_ms"), ("errors", "errors")],
          rows,
          [f"Recall@{pool_k} 量的是**召回池**（RECALL_K={pool_k}）里有没有 gold；",
           f"MRR@{final_k} / nDCG@{final_k} 量的是模型实际看到的前 {final_k} 条排得好不好 —— "
           "两段分开看，才知道问题出在召回还是重排。",
           "p50/p95 只量**检索本身**：query 向量已批量预计算，不含 embedding 的 HTTP 往返"
           "（一次约 2s，会把所有行压成同一个数，时间列就白给了）—— 只能横向比，不是线上端到端延迟。",
           "⚠️ 池子口径，与旧报告的 recall@5 不是同一个数，**不可直接比**。",
           "errors = 检索抛异常的条数（异常记 0 分但保留在分母里）；不为 0 时先修服务再看指标。"])
    t2 = ("二、上下文质量",
          [("策略", "strategy"), ("档位", "stage"),
           ("Context Recall", "context_recall"),
           ("Context Precision", "context_precision")],
          rows,
          [f"量的是交付给模型的那 {final_k} 条：一条结果相关 ⟺ 与 gold 标注行区间重叠。",
           "⚠️ gold-chunk 粒度，不是 RAGAS 的「论断」粒度 —— 本仓库没有参考答句。",
           "⚠️ 单 gold 二元相关下 Context Precision ≡ 1/rank ≡ MRR；本仓库 33/39 条只有单个 gold，"
           "两个指标只在 gold 跨多个 chunk 的那 6 条上分得开。",
           "Context Precision 的分母是「召回到的相关条数」：一条相关的都没召回记 0.000（不是 1.000）。"])
    t3 = ("三、生成可靠性",
          [("档位", "stage"), ("Faithfulness↑", "faithfulness"),
           ("Hallucination↓", "hallucination"), ("域外", "halluc_ood"),
           ("近义", "halluc_near"), ("可解答", "halluc_answerable"),
           ("作答率", "answer_rate"), ("平均声明数", "claims")],
          gen_rows,
          ["Faithfulness = 能被上下文支持的论断 / 总论断，只在**真正作答**的样本上取均值；"
           "拒答记 null 且不进分母 —— 否则「全拒答」会被洗成满分。",
           "Hallucination = 下面三项的**宏平均**（等权，不被各项的样本量比带偏）。",
           "域外 = 语料零命中的对抗题里没拒答的比例。**太好拒，只作下限**，别当整体安全度看。",
           "近义 = 「语料里有很像的素材、但答案不存在」的对抗题里没拒答的比例 —— 幻觉主产区。",
           "可解答 = 作答了的答案里含未支撑断言的占比（回答级，不是论断级）。",
           "作答率必须和 Faithfulness 一起看：作答率掉下去时 Faithfulness 会虚假变好。",
           "① 无上下文 = 不给任何检索结果直接问模型，是没有 RAG 的基线。本 prompt 要求「上下文不足就拒答」，"
           "所以这一档通常是**全拒答**（作答率 0）：它幻觉率低是拒答的结果，**读不出「没有 RAG 也安全」**。",
           "判官与生成用的是同一个模型（self-preference 偏差），数字适合横向比较各档，"
           "不宜当作绝对忠实度；论断级计数（而不是打分）是为了把这种偏袒的影响降到能看见的程度。",
           f"生成与判官：{gen_model or '未跑（加 --gen / RAG_EVAL_GEN=1）'}"])

    return [{"title": t[0], "cols": t[1], "rows": t[2], "footnotes": t[3]}
            for t in (t1, t2, t3)]
