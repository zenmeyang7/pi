import heapq
import json
import math
import os
import re

from . import config
from .chunker import chunk_file, iter_files, rel_path
from .embedder import embed

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text):
    """代码检索专用分词：ASCII 按 camel/snake 拆开，中文按 uni + bi-gram 切。"""
    out = []
    for tok in _TOKEN_RE.findall(text):
        parts = _CAMEL_RE.sub(r"\1 \2", tok.replace("_", " ")).split()
        out.extend(p.lower() for p in parts if p)
        out.append(tok.lower())
    for run in _CJK_RE.findall(text):
        out.extend(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


class BM25:
    """带倒排索引的 BM25。

    倒排（word -> [(doc_id, tf)]）把打分范围从"全部文档"缩到"至少含一个
    查询词的文档"。打分结果和暴力扫描**完全一致**，
    但查询复杂度从 O(N·L) 降到 O(命中文档数·L)，索引构建时多花一次遍历。
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.lens = [len(t) for t in self.tokens]
        self.avg = sum(self.lens) / max(len(self.lens), 1)

        df = {}
        # postings: word -> [(doc_id, 该文档内的词频)]
        # df:       word -> 出现在几个文档里（算 IDF 用）
        self.postings = {}
        for i, t in enumerate(self.tokens):
            tf = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            for w, f in tf.items():
                self.postings.setdefault(w, []).append((i, f))
                df[w] = df.get(w, 0) + 1
        n = len(self.tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        out = [0.0] * len(self.lens)
        q_terms = tokenize(query)              # 只分词一次，别放进内层循环
        norm = self.k1 * (1 - self.b)          # 与文档长度无关的部分，提出循环
        for w in q_terms:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings.get(w, ()):
                denom = f + norm + self.b * self.k1 * self.lens[i] / self.avg
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def normalize(v):
    """L2 归一化。向量存进索引前先归一化，检索时点积 = 余弦，
    省掉每次查询对每个 chunk 重算两次平方和。"""
    n = math.sqrt(math.sumprod(v, v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b):
    """余弦相似度。math.sumprod 是 3.12+ 的 C 实现，比逐位乘加快一个数量级；
    若向量已归一化（索引构建时保证），等价于点积。"""
    na = math.sqrt(math.sumprod(a, a))
    nb = math.sqrt(math.sumprod(b, b))
    return math.sumprod(a, b) / (na * nb + 1e-9)


def _top_k_by_dot(qv, chunks, k):
    """归一化向量上的点积 top-k。qv 假定已归一化。"""
    scored = ((math.sumprod(qv, c["vec"]), i) for i, c in enumerate(chunks))
    return [i for _, i in heapq.nlargest(k, scored)]


class IndexStore:
    def __init__(self, index_dir):
        self.dir = index_dir
        self.chunks_path = os.path.join(index_dir, "chunks.jsonl")
        self.manifest_path = os.path.join(index_dir, "manifest.json")
        self.chunks = []
        self._bm25 = None

    def build(self, root, force=False, verbose=print):
        os.makedirs(self.dir, exist_ok=True)
        changed = self._changed_files(root, force)
        reused = [] if force else self._carry_over(root, changed)

        fresh_chunks = []
        for rel in changed:
            path = os.path.join(root, rel)
            fresh_chunks.extend(chunk_file(path, root))

        verbose(f"[rag] {len(changed)} 个文件需重建，{len(reused)} 个 chunk 复用")

        if fresh_chunks:
            verbose(f"[rag] embedding {len(fresh_chunks)} 个 chunk ...")
            vecs = embed([c["text"] for c in fresh_chunks])
            for c, v in zip(fresh_chunks, vecs):
                # 归一化后再落盘：检索时点积即余弦，省去每查询的全量范数计算
                c["vec"] = normalize(v)

        self.chunks = reused + fresh_chunks
        self._bm25 = None
        self._save(root, changed)
        verbose(f"[rag] 索引完成：{len(self.chunks)} 个 chunk -> {self.chunks_path}")

    def _manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _changed_files(self, root, force):
        old = {} if force else self._manifest().get("files", {})
        changed = []
        for path in iter_files(root):
            rel = rel_path(path, root)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if force or old.get(rel) != mtime:
                changed.append(rel)
        return sorted(changed)

    def _carry_over(self, root, changed):
        """复用未改动文件的旧 chunk（含旧向量），省掉重复 embedding。

        注意过滤掉**已被删除**的文件的 chunk：删除的文件不会出现在 changed
        里（它已经不在工作区了），不过滤的话旧 chunk 会永远残留在索引中。
        """
        changed_set = set(changed)
        if not os.path.exists(self.chunks_path):
            return []
        # 注意：changed_set 为空**不能**提前返回——那正是"零改动重建"的场景，
        # 提前返回会把整个索引清空（本条由 test_index_drops_chunks_of_deleted_files 锁定）
        alive = {rel_path(p, root) for p in iter_files(root)}   # 当前实际存在的文件
        out = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c["meta"]["path"] in changed_set:
                    continue
                if c["meta"]["path"] not in alive:
                    continue            # 文件已删除：丢弃，而不是带进新索引
                c["vec"] = normalize(c["vec"])   # 旧索引向量统一归一化
                out.append(c)
        return out

    def _save(self, root, changed):
        mtimes = self._manifest().get("files", {})
        for rel in changed:
            try:
                mtimes[rel] = os.path.getmtime(os.path.join(root, rel))
            except OSError:
                pass
        alive = {rel_path(p, root) for p in iter_files(root)}
        mtimes = {k: v for k, v in mtimes.items() if k in alive}

        tmp = self.chunks_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(tmp, self.chunks_path)      # 原子替换，避免中途崩坏索引
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"model": config.EMBED_MODEL, "files": mtimes}, f,
                      ensure_ascii=False, indent=2)

    def load(self):
        self.chunks = []
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
        # 兼容旧索引（向量未归一化）：load 时统一归一化，幂等操作
        for c in self.chunks:
            c["vec"] = normalize(c["vec"])
        self._bm25 = None
        return self

    def _bm25_index(self):
        if self._bm25 is None:
            self._bm25 = BM25([c["text"] for c in self.chunks])
        return self._bm25

    def has_distinctive_code_token(self, query, max_df_ratio=0.05, min_len=3):
        """query 里有没有"稀有的 ASCII 标识符"——BM25 的主场。

        为什么需要这个判断：强 embedding（bge-m3）对代码语义匹配已经很准，
        纯中文语义查询走向量路即可；而词法路在这类查询上匹配到的往往是
        注释里的中文 bigram，属于噪声，反而稀释融合结果（实测：等权融合
        recall@5 0.708 < 纯向量 0.792）。

        词法路真正无可替代的是**罕见符号名**：`executeToolCallsParallel`、
        `finish_reason` 这类词在整个语料里只出现少数几次，BM25 靠 IDF
        一击必中。所以这里只在 query 含这类词时才让词法路发言。
        """
        if not self.chunks:
            return False
        bm = self._bm25_index()
        n = max(len(self.chunks), 1)
        for w in set(tokenize(query)):
            if len(w) < min_len or not any(c.isascii() and c.isalpha() for c in w):
                continue
            postings = bm.postings.get(w)
            if postings and len(postings) <= max(2, int(n * max_df_ratio)):
                return True          
        return False

    def lexical_search(self, query, k):
        """路 1：词法召回——精确、便宜，负责符号名/关键字这类查询。"""
        if not self.chunks:
            return []
        scores = self._bm25_index().scores(query)
        # nlargest 是 O(N log k)，比全排序 O(N log N) 便宜；分数为 0 的不进结果
        scored = ((s, i) for i, s in enumerate(scores) if s > 0)
        return [i for _, i in heapq.nlargest(k, scored)]

    def vector_search(self, query, k):
        """路 2：语义召回——负责"意思对但字不对"的查询。"""
        if not self.chunks:
            return []
        return self.vector_search_by_vec(normalize(embed(query)), k)

    def vector_search_by_vec(self, qv, k):
        """已经算好 query 向量时用这个，避免评估时重复 embedding。
        传入前需归一化（build/load 已保证 chunk 向量是单位向量）。"""
        if not self.chunks:
            return []
        return _top_k_by_dot(qv, self.chunks, k)


def rrf(rank_lists, k=None, weights=None):
    """Reciprocal Rank Fusion：把多路召回按**排名**融合。

    用排名而不是分数，是因为 BM25 的分数和余弦相似度量纲完全不同，
    没法直接相加；而排名是无量纲的。这是混合检索里最省事的融合方式。

    weights：每一路的权重（默认等权 1:1）。**为什么要可调**——
    等权融合假设两路一样可靠，但实际不是：对"中文语义描述"类查询，
    词法路匹配到的往往是注释里的中文 bigram，属于噪声；这类噪声
    只要排进词法路 top-1，就能拿到 1/(k+1) 的分，足以挤掉向量路
    rank-2 的正确结果（1/(k+2)）。给弱路降权是标准解法（weighted RRF）。
    """
    k = k or config.RRF_K
    if weights is None:
        weights = [1.0] * len(rank_lists)
    scores = {}
    for w, ranked in zip(weights, rank_lists):
        for rank,idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
