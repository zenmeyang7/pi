"""代码检索工具：把 rag/ 的检索链路包装成一个 AgentTool。

和 tools/files.py 分开的理由是依赖形态不同：read_file / write_file 是纯本地
IO，随时可用；search_code 要先有一份构建好的向量索引，缺索引时它得能降级
成一句可操作的提示，而不是直接把会话炸掉。
"""
from agent.protocol import AgentTool

TOP_K = 5

_retriever = None


def init_retriever(index_dir="rag/index"):
    """在 main 里调一次：加载已构建好的索引。"""
    global _retriever
    from rag.retriever import Retriever
    from rag.store import IndexStore

    store = IndexStore(index_dir).load()
    if not store.chunks:
        print(f"[tools] 索引为空（{index_dir}），search_code 暂不可用。"
              f"先跑：uv run python rag/build_index.py")
        return None
    _retriever = Retriever(store)
    print(f"[tools] 代码索引就绪：{len(store.chunks)} 个 chunk，"
          f"reranker={_retriever.reranker.name}")
    return _retriever


def search_code():
    """混合检索：BM25 词法 + 向量语义 -> RRF 融合 -> cross-encoder 重排。"""

    def execute(args):
        if _retriever is None:
            return "代码索引未加载，请先运行 rag/build_index.py 构建索引。"
        query = args.get("query", "")
        if not query.strip():
            return "query 不能为空。"
        # 代码 chunk 很长，塞太多反而稀释注意力
        try:
            k = min(int(args.get("k", TOP_K)), 10)
        except (TypeError, ValueError):
            k = TOP_K    # 模型偶尔把 k 填成非数字，按默认值走而不是报错
        if k < 1:
            k = 1
        from rag.retriever import format_hits

        return format_hits(_retriever.retrieve(query, k=k))

    return AgentTool(
        name="search_code",
        execute=execute,
        execution_mode="parallel",
        definition={
            "type": "function",
            "function": {
                "name": "search_code",
                "description": (
                    "在整个代码库里检索相关代码片段，返回 路径:起止行 和代码内容。"
                    "需要定位某个功能的实现在哪里、找与某个概念相关的代码，"
                    "或者不清楚该读哪个文件时，先用这个工具，"
                    "再根据返回的行号用 read_file 精读。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用自然语言描述你要找的东西，例如「工具并行执行是怎么实现的」",
                        },
                        "k": {
                            "type": "integer",
                            "description": "返回条数，默认 5，上限 10",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    )
