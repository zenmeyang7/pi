# type.py
from typing import List, Dict, Callable, Optional, Iterable
import json
import urllib.request
from jsonl_tree import JsonlTree
# 一条消息（wire format）：{"role","content","tool_calls"?,"tool_call_id"?,"stop_reason"?}
# 定义消息的格式 "str":object
AgentMessage = Dict[str, object]
Message = Dict[str, object]
AssistantMessage = Dict[str, object]
ToolResultMessage = Dict[str, object]
# 事件也是 dict：{"type": ..., ...}
# 定义事件的格式 "type":pbject
AgentEvent = Dict[str, object]
ToolCall = Dict[str, object]  # OpenAI 的 tool call 块

AgentEventSink = Callable[[AgentEvent], None]
StreamFn = Callable[[str, Dict, Dict], Iterable[AgentEvent]]


# ================= AgentTool =================
class AgentTool:
    """可执行工具：definition 给 LLM 看，execute 给我们执行。"""

    def __init__(self, name: str, execute, definition: Optional[Dict] = None, execution_mode: str = "parallel"):
        self.name = name
        self.execute = execute  # fn(tool_call_id: str, args: dict) -> str
        self.execution_mode = execution_mode
        self.definition = definition or {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }


# ================= AgentContext =================
class AgentContext:
    def __init__(self, system_prompt: str = "", messages:Optional[List]=None, tools: Optional[List] = None,tree = None):
        self.system_prompt = system_prompt
        self.messages = messages if messages is not None else []
        self.tools = tools if tools is not None else []
        self.tree = tree
    def append_message(self,m):
        if self.tree is not None:
            self.tree.append(message=m)
        self.messages.append(m)

# ================= convert_to_llm =================
def convert_to_llm_default(messages: List[AgentMessage]) -> List[Message]:
    """AgentMessage -> Message：只留 LLM 协议认识的三张牌，并剥掉多余键。"""
    out: List[Message] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant", "tool"):
            continue  # custom / notification 只给 UI，不进 LLM
        msg: Dict[str, object] = {"role": role, "content": m.get("content", "")}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id")
        out.append(msg)
    return out


# ================= AgentLoopconfig =================
class AgentLoopconfig:
    def __init__(
        self,
        model: str = "",
        get_steering_messages: Optional[Callable[[], List]] = None,
        wait_for_steering: Optional[Callable[[], List]] = None,
        convert_to_llm=None,
        tool_execution: str = "sequential",
    ):
        self.model = model
        self.get_steering_messages = get_steering_messages or (lambda: [])
        self.wait_for_steering = wait_for_steering or (lambda: [])
        self.convert_to_llm = convert_to_llm or convert_to_llm_default
        self.tool_execution = tool_execution


# ================= 工具函数 =================
def map_stop_reason(finish_reason: str) -> str:
    # 未知的 finish_reason 当作 error 而非 stop：宁可让上层报错，不要误判成正常结束
    return {"stop": "stop", "tool_calls": "toolUse", "length": "length"}.get(finish_reason, "error")


# ================= StreamFn（同步生成器，永不抛错）=================
def StreamFn(model: str, llm_context: Dict, option: Dict):
    """调 LLM，yield 一个 done/error 事件。任何异常都转成 error 事件，不抛出。"""
    system = llm_context["system_prompt"]
    messages = llm_context["messages"]
    tools = llm_context["tools"]
    api_key = option.get("api_key")
    base_url = option.get("base_url")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": [t.definition for t in tools],
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    full_content = ""
    tool_calls = []
    stop_reason = None
    try:
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choice = chunk["choices"][0]
                delta = choice.get("delta",{})
                piece = delta.get("content", "")
                if piece:
                    full_content += piece
                    yield {
                        "type": "delta",
                        "content":piece,
                    }
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index",len(tool_calls))
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id":"","type":"function","function":{
                        "name":"","arguments":""}})
                    slot = tool_calls[idx]
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function",{})
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    stop_reason = map_stop_reason(choice["finish_reason"])
            reason = stop_reason or "stop"
            yield{
                "type": "done",
                "reason": reason,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                    "tool_calls": tool_calls,
                    "stop_reason": reason,
                },
            }
    except Exception as e:
        yield {
            "type": "error",
            "reason": "error",
            "message": {
                "role": "assistant",
                "content": full_content,
                "tool_calls": tool_calls,
                "stop_reason": "error",
                "error_message": str(e),
            },
        }
