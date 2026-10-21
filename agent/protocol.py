"""协议层：agent 里流动的数据长什么样，以及怎么跟大模型说话。"""
from typing import List, Dict, Callable, Optional, Iterable
import json
import time
import urllib.request
import urllib.error

# 网络层参数
HTTP_TIMEOUT = 60        
MAX_RETRIES = 2           
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}   
RETRY_BASE_DELAY = 0.5     
AgentMessage = Dict[str, object]
Message = Dict[str, object]
AssistantMessage = Dict[str, object]
ToolResultMessage = Dict[str, object]
AgentEvent = Dict[str, object]
ToolCall = Dict[str, object]  

AgentEventSink = Callable[[AgentEvent], None]
StreamFn = Callable[[str, Dict, Dict], Iterable[AgentEvent]]


class AgentTool:
    """可执行工具：definition 给 LLM 看，execute 给我们执行。"""

    def __init__(self, name: str, execute, definition: Optional[Dict] = None, execution_mode: str = "parallel"):
        self.name = name
        self.execute = execute  # fn(args: dict) -> str（str 也接受，工具层统一转错误消息）
        self.execution_mode = execution_mode
        self.definition = definition or {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }


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


class AgentLoopconfig:
    def __init__(
        self,
        model: str = "",
        get_steering_messages: Optional[Callable[[], List]] = None,
        wait_for_steering: Optional[Callable[[], List]] = None,
        convert_to_llm=None,
        tool_execution: str = "sequential",
        max_turns: int = 25,
        before_tool_call: Optional[Callable[[str, dict], Optional[dict]]] = None,
        after_tool_call: Optional[Callable[[str, object, bool], Optional[dict]]] = None,
        tool_selector: Optional[Callable[[List, List], List]] = None
    ):
        self.model = model
        self.get_steering_messages = get_steering_messages or (lambda: [])
        self.wait_for_steering = wait_for_steering or (lambda: [])
        self.convert_to_llm = convert_to_llm or convert_to_llm_default
        self.tool_execution = tool_execution
        # 一个"任务"最多跑多少轮工具循环。0 或 None 表示不限（不推荐）。
        self.max_turns = max_turns
        self.before_tool_call = before_tool_call
        self.after_tool_call  = after_tool_call
        self.tool_selector = tool_selector


def map_stop_reason(finish_reason: str) -> str:
    # 未知的 finish_reason 当作 error 而非 stop：宁可让上层报错，不要误判成正常结束
    return {"stop": "stop", "tool_calls": "toolUse", "length": "length"}.get(finish_reason, "error")


def stream_deepseek(model: str, llm_context: Dict, option: Dict):
    """调 LLM，yield delta / done / error 事件。任何异常都转成 error 事件，不抛出。

    分成两层：
      - `_stream_once`：真的发一次请求、逐块解析 SSE；
      - `stream_deepseek`：在外面包一层重试。

    """
    attempts = int(option.get("max_retries", MAX_RETRIES)) + 1
    for attempt in range(attempts):
        emitted_delta = False
        failure = None
        for event in _stream_once(model, llm_context, option):
            if event["type"] == "error":
                failure = event
                break
            if event["type"] == "delta":
                emitted_delta = True
            yield event
        if failure is None:
            return
        if emitted_delta or not failure.get("retryable") or attempt == attempts - 1:
            yield failure               # 吐过字 / 不值得重试 / 次数用尽 → 老实报错
            return
        time.sleep(RETRY_BASE_DELAY * (2 ** attempt))


def _stream_once(model: str, llm_context: Dict, option: Dict):
    """发一次请求：手写 urllib + SSE 逐行解析。异常全部转成 error 事件。"""
    system = llm_context["system_prompt"]
    messages = llm_context["messages"]
    tools = llm_context["tools"]
    api_key = option.get("api_key")
    base_url = option.get("base_url")
    timeout = option.get("timeout", HTTP_TIMEOUT)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    if tools:
        body["tools"] = [t.definition for t in tools]
    # 采样参数：不传就用服务端默认值，摘要这种要求稳定的调用由上层显式指定
    for key in ("temperature", "max_tokens", "thinking"):
        if option.get(key) is not None:
            body[key] = option[key]
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    full_content = ""
    tool_calls = []
    stop_reason = None
    try:
        # timeout 必须显式传（见上面 HTTP_TIMEOUT）：默认无超时，网络一挂就永远卡住
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                # SSE 流是 UTF-8：不显式指定的话 Windows 会用 GBK 解码，
                # 中文回答直接 UnicodeDecodeError
                line = raw.decode("utf-8", errors="replace").strip()
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
    except urllib.error.HTTPError as e:
        # 4xx 里只有 408/409/429 值得重试；400/401/403 重试一万次也是同样的错
        yield _error_event(full_content, tool_calls, e,
                           retryable=e.code in RETRY_STATUS, status=e.code)
    except Exception as e:
        # 网络类错误（超时 / DNS / 连接重置）一律可重试
        yield _error_event(full_content, tool_calls, e, retryable=True, status=None)


def _error_event(full_content: str, tool_calls: list, exc: Exception,
                 retryable: bool, status: Optional[int]) -> AgentEvent:
    """把异常包成统一的 error 事件：形状固定，上层只认 type/error_message。"""
    return {
        "type": "error",
        "reason": "error",
        "retryable": retryable,
        "status": status,
        "message": {
            "role": "assistant",
            "content": full_content,
            "tool_calls": tool_calls,
            "stop_reason": "error",
            "error_message": str(exc),
        },
    }
