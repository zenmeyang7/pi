"""代码 RAG：切片 -> embedding -> 混合召回 -> cross-encoder 重排 -> 评估。"""

from .retriever import retrieve, format_hits
from .store import IndexStore

__all__ = ["retrieve", "format_hits", "IndexStore"]
