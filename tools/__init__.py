"""Agent 的工具层：本地文件读写 + 代码库语义检索。

分两个文件是因为依赖形态不同（详见各文件的 docstring）。
装配顺序有要求：`search_code` 的工具对象本身不依赖索引，但它的 execute
在**调用时**读模块级 `_retriever`，所以 main 必须先 `init_retriever()`。
"""
from .files import read_file, safe_path, write_file
from .search import init_retriever, search_code

__all__ = ["read_file", "write_file", "safe_path", "search_code",
           "init_retriever", "get_tools"]


def get_tools():
    """给 AgentContext 用的工具列表。调用前请先 init_retriever()。"""
    return [read_file(), write_file(), search_code()]
