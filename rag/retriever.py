"""检索层：把两路召回、融合、重排串成一条链路。

    query
      |
      +-- ① 词法召回  BM25（workspace 里的精确/符号匹配） ──┐
      |                                                     +-- ③ RRF 融合
      +-- ② 向量召回  embedding 语义相似 ───────────────────┘        |
                                                                    v
                                                        ④ cross-encoder 重排
                                                                    |
                                                                    v
                                                              top-k + 来源标注

**为什么两路都要**：词法路对符号名/报错文本这类"字面命中"最准，
向量路对"我记得有个处理日志压缩的东西"这类描述最准，两者错得不一样，
融合后能互相补位。重排则是在融合结果上做一次精细的相关性判断。
"""
from . import config
from .embedder import embed
from .reranker import get_reranker
from .store import rrf


class Retriever:
    """持有索引和 reranker，避免每次调用都重建 BM25 索引 / 重新探活。

    embed_fn 做成可注入的：评估和单测时传假向量就能脱离 Ollama 跑全链路。
    """

    def __init__(self, store, reranker=None, embed_fn=None, weights=None,
                 route_lexical=None):
        self.store = store
        self.reranker = reranker if reranker is not None else get_reranker()
        self.embed_fn = embed_fn or embed
        self.weights = list(weights) if weights else list(config.RRF_WEIGHTS)
        # 是否对词法路做"稀有标识符"路由（默认按 config，可显式关掉做消融）
        self.route_lexical = config.ROUTE_LEXICAL if route_lexical is None else route_lexical
        self.stats = {}

    def retrieve(self, query, k=None, use_lexical=True, use_vector=True,
                 use_rerank=True, recall_k=None, route_lexical=None):
        k = k or config.FINAL_K
        recall_k = recall_k or config.RECALL_K
        if not self.store.chunks:
            return []
        
        qv = self.embed_fn(query) if use_vector else None

        # 路由：query 里没有稀有标识符时，词法路只会贡献噪声，直接不召回
        do_route = self.route_lexical if route_lexical is None else route_lexical
        routed_out = False
        if use_lexical and do_route and not self.store.has_distinctive_code_token(query):
            use_lexical = False
            routed_out = True

        lexical = self.store.lexical_search(query, recall_k) if use_lexical else []
        vector = self.store.vector_search_by_vec(qv, recall_k) if use_vector else []

        if use_lexical and use_vector:
            fused = rrf([lexical, vector], weights=self.weights)
        else:
            fused = lexical or vector

        # 重排的输入池：融合后的前 recall_k 个
        pool = fused[:recall_k]
        # 整块重建而非就地改：一次 retrieve 的 stats 不该残留上一次的键。
        # routed_out 必须在这里带上——它是在重建**之前**算出来的。
        self.stats = {
            "lexical_hits": len(lexical),
            "vector_hits": len(vector),
            "fused": len(fused),
            "pool": len(pool),
            # 词法路是"被路由关掉"还是"参与了但零命中"——两者表现都是 0 条，
            # 不记下来调试时根本分不清
            "lexical_routed_out": routed_out,
            # 记录**实际**用没用上重排，而不是配置里开没开——
            # 降级发生时评估表才不会把纯 RRF 的结果算成重排的效果
            "reranked": bool(use_rerank and pool and self.reranker.name == "cross-encoder"),
        }

        if use_rerank and pool:
            docs = [self.store.chunks[i]["text"] for i in pool]
            try:
                order = self.reranker.rerank(query, docs, k)
                picked = [pool[i] for i in order]
            except Exception as e:      # 重排失败不能让整个检索失败
                self.stats["rerank_error"] = str(e)
                picked = pool[:k]
        else:
            picked = pool[:k]

        return [
            {"id": self.store.chunks[i]["id"],
             "score": None,
             "text": self.store.chunks[i]["text"],
             "meta": self.store.chunks[i]["meta"]}
            for i in picked
        ]

    def retrieve_by_strategy(self, query, strategy, k=None):
        """给评估用：同一个 query 跑不同策略，保证对比公平。

        命名里的 `noroute` / `+route` 是为了把**四档做成严格累加**的：
        后者 = 前者 + 正好一个组件，档与档之间的差就是那个组件的净贡献。
        """
        if strategy == "lexical":
            return self.retrieve(query, k=k, use_lexical=True, use_vector=False, use_rerank=False)
        if strategy == "vector":
            return self.retrieve(query, k=k, use_lexical=False, use_vector=True, use_rerank=False)
        if strategy == "hybrid":
            # 无条件双路：词法路永远参与，用于和路由版对比
            return self.retrieve(query, k=k, use_lexical=True, use_vector=True,
                                 use_rerank=False, route_lexical=False)
        if strategy == "hybrid_routed":
            # 路由版：只有 query 含稀有标识符时才让词法路发言
            return self.retrieve(query, k=k, use_lexical=True, use_vector=True,
                                 use_rerank=False, route_lexical=True)
        if strategy == "hybrid+rerank_noroute":
            # ③ = ② + 重排（关掉路由，保证这一档只多了一个变量）
            return self.retrieve(query, k=k, use_lexical=True, use_vector=True,
                                 use_rerank=True, route_lexical=False)
        if strategy == "hybrid+rerank+route":
            # ④ = ③ + 路由
            return self.retrieve(query, k=k, use_lexical=True, use_vector=True,
                                 use_rerank=True, route_lexical=True)
        if strategy == "hybrid+rerank":
            # 旧名，等价于 ④（保留以免跑过的对比表和脚本失效）
            return self.retrieve(query, k=k, use_lexical=True, use_vector=True,
                                 use_rerank=True, route_lexical=True)
        raise ValueError(f"未知策略：{strategy}")


def retrieve(query, store, k=None, reranker=None, embed_fn=None):
    """无状态便捷入口（每次新建 Retriever，reranker 会重新探活，慎用循环）。"""
    return Retriever(store, reranker, embed_fn).retrieve(query, k=k)


def format_hits(hits, max_chars=2400):
    """把检索结果格式化成给模型看的样子。

    每条都带 `路径:起止行`，模型拿到后可以直接 read_file 精读，
    也便于它在回答里引用来源。
    """
    if not hits:
        return "没有检索到相关代码。可以换个说法再搜，或直接用 read_file 查看具体文件。"
    parts = []
    for h in hits:
        m = h["meta"]
        loc = f"{m['path']}:{m.get('start_line', '?')}-{m.get('end_line', '?')}"
        body = h["text"]
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...（已截断）"
        parts.append(f"### {loc}\n```\n{body}\n```")
    return "\n\n".join(parts)
