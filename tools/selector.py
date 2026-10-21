import math

from agent.protocol import AgentTool
from rag.embedder import embed
from rag.store import BM25


def _tool_text(tool: AgentTool) -> str:
    fn = tool.definition.get("function", {})
    params = (fn.get("parameters") or {}).get("properties") or {}
    parts = [str(fn.get("name") or tool.name), str(fn.get("description") or "")]
    for name, schema in params.items():
        parts.append(name)
        if isinstance(schema, dict) and schema.get("description"):
            parts.append(str(schema["description"]))
    return "\n".join(part for part in parts if part)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class ToolSelector:
    def __init__(self, top_k: int = 3, always_keep=("read_file",), embed_fn=None):
        self.top_k = top_k
        self.always_keep = set(always_keep)
        self._embed = embed_fn or embed
        self._texts: dict[str, str] = {}
        self._vecs: dict[str, list[float]] = {}

    @staticmethod
    def _query(history) -> str:
        for message in reversed(history):
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else ""
        return ""

    @staticmethod
    def _used_tools(history) -> set[str]:
        names = set()
        for message in reversed(history):
            if message.get("role") == "user":
                break
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    names.add(name)
        return names

    def _text_for(self, tool: AgentTool) -> str:
        text = _tool_text(tool)
        if self._texts.get(tool.name) != text:
            self._texts[tool.name] = text
            self._vecs.pop(tool.name, None)
        return text

    def _embed_scores(self, query, tools, texts):
        missing = [tool.name for tool in tools if tool.name not in self._vecs]
        if missing:
            by_name = {tool.name: text for tool, text in zip(tools, texts)}
            vectors = self._embed([by_name[name] for name in missing])
            for name, vector in zip(missing, vectors):
                self._vecs[name] = vector

        query_vec = self._embed(query)
        return [_cosine(query_vec, self._vecs[tool.name]) for tool in tools]

    def select(self, history, tools):
        if not tools:
            return tools

        query = self._query(history)
        if not query.strip():
            return tools

        texts = [self._text_for(tool) for tool in tools]

        try:
            scores = self._embed_scores(query, tools, texts)
        except Exception:
            scores = BM25(texts).scores(query)
            if max(scores, default=0.0) <= 0:
                return tools

        ranked = sorted(zip(tools, scores), key=lambda item: item[1], reverse=True)
        keep = set(self.always_keep) | self._used_tools(history)
        names = {tool.name for tool, _ in ranked[:self.top_k]} | keep
        return [tool for tool in tools if tool.name in names]
