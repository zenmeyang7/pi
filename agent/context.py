import json
import re
from dataclasses import dataclass

from tokenizers import Tokenizer

from agent.llm import model as llm_model
from agent.protocol import stream_deepseek

# 模型最大上下文：GET /models 返回 deepseek-flash context_window=1048576
MAX_TOKENS = 1048576
OUTPUT_TOKENS = 2000
SAFE_TOKENS = round(MAX_TOKENS * 0.05)
CONTEXT_BUDGET = int((MAX_TOKENS - OUTPUT_TOKENS - SAFE_TOKENS) * 0.9)

RECENT_KEEP = 5
SUMMARY_MAX_TOKENS = 1000
MAX_COMPACT_ROUNDS = 3

# 压缩级别触发阈值（占可用上下文的比例）
L1_RATIO = 0.60
L2_RATIO = 0.75
L3_RATIO = 0.85
L4_RATIO = 0.92

HAND_OFF_SUMMARY = """
    你在执行上下文压缩。请为另一个将接续该任务的 LLM 创建一份交接摘要。
    必须逐字保留用户给出的所有标识符、暗号、数字、路径和约束条件。
    包括：
    当前进展和已做出的关键决策
    重要上下文、约束条件或用户偏好
    仍需完成的事项（清晰的下一步）
    继续推进所需的关键数据、示例或参考资料
    只输出摘要本身，不要输出分析过程或客套话。保持简洁、结构化。
"""

# token 计数必须用目标模型的 tokenizer，否则阈值会整体偏移
tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")

def count_tokens(text) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False).ids)


def count_message_tokens(messages) -> int:
    """消息实际占用的 token。

    tool_calls.arguments 也要算：write_file 的参数可能装着上千 token 的内容。
    """
    total = 0
    for m in messages:
        total += count_tokens(m.get("role", ""))
        total += count_tokens(m.get("content", ""))
        total += count_tokens(m.get("tool_call_id", ""))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += count_tokens(fn.get("name", ""))
            total += count_tokens(fn.get("arguments", ""))
    return total


def count_context_tokens(system_prompt: str, messages: list, tools: list) -> int:
    total = count_tokens(system_prompt) + count_message_tokens(messages)
    for t in tools:
        total += count_tokens(json.dumps(t.definition, ensure_ascii=False))
    return total


L1_TOOL_MIN_TOKENS = 50
L1_TOOL_MARK = "[旧工具结果已省略：{tokens} tokens，需要时重新调用工具]"


def _is_adjacent_duplicate(cleaned, message, content) -> bool:
    """相邻且完全相同的 user/assistant 消息只留一条。"""
    return bool(cleaned
                and message.get("role") in ("user", "assistant")
                and not message.get("tool_calls")
                and content
                and cleaned[-1].get("role") == message.get("role")
                and cleaned[-1].get("content") == message["content"])


def clean_redundant_messages(messages, keep_recent=RECENT_KEEP):
    """L1：删空消息和相邻重复消息；旧工具结果换成占位符。

    带 tool_calls 的空 assistant 不能删，它要和 tool 结果配对。
    """
    cleaned = []
    cutoff = len(messages) - keep_recent

    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            content = content.strip()
            if not content and role in ("user", "assistant") and not m.get("tool_calls"):
                continue
            m = {**m, "content": content}

        if role == "tool" and i < cutoff and isinstance(content, str):
            n = count_tokens(content)
            if n > L1_TOOL_MIN_TOKENS:
                m["content"] = L1_TOOL_MARK.format(tokens=n)

        if not _is_adjacent_duplicate(cleaned, m, content):
            cleaned.append(m)

    return cleaned


TRUNCATION_MARK = "\n…[tool result truncated]…\n"
# 只删成对的完整协议块：裸删标签会改坏"正在讨论 <tool_call> 语法"的正文
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _normalize_assistant_text(text: str) -> str:
    """去掉正文里重复的工具调用协议块和首尾空白。"""
    return _TOOL_CALL_BLOCK.sub("", text).strip()


def _compact_arguments(arguments):
    """JSON 参数压成紧凑形式。

    解析失败说明参数可能已被截断，原样保留：改写坏数据比不改更危险。
    """
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments.strip()
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _truncate_head_tail(text: str, max_tokens: int) -> str:
    """保留头尾：工具输出的开头是结构，结尾通常是结论或报错。"""
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    return tok.decode(ids[:head]) + TRUNCATION_MARK + tok.decode(ids[-tail:])


def _compact_tool_call(tool_call: dict) -> dict:
    """复制到 function.arguments 这一层，避免改到原消息。"""
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return tool_call.copy()
    if "arguments" in function:
        function = {**function, "arguments": _compact_arguments(function["arguments"])}
    return {**tool_call, "function": function}


def _compact_old_message(message: dict, max_tools_token: int) -> dict:
    cleaned = message.copy()
    role = message.get("role")
    content = message.get("content")

    if isinstance(content, str):
        cleaned["content"] = (_normalize_assistant_text(content)
                              if role == "assistant" else content.strip())
        if role == "tool" and count_tokens(cleaned["content"]) > max_tools_token:
            cleaned["content"] = _truncate_head_tail(cleaned["content"], max_tools_token)

    tool_calls = message.get("tool_calls")
    if tool_calls:
        cleaned["tool_calls"] = [_compact_tool_call(tc) for tc in tool_calls]

    return cleaned


def compact_tools_result(messages, max_tools_token, keep_recent=RECENT_KEEP):
    """L2：压缩旧消息里的工具痕迹。

    先做无损规范化（去协议块、去空白、JSON 压紧凑），再对仍超长的工具结果
    做头尾截断。顺序反过来会把清理后本来放得下的内容也切掉。
    """
    if max_tools_token <= 0:
        return messages

    cutoff = len(messages) - keep_recent
    return [_compact_old_message(m, max_tools_token) if i < cutoff else m
            for i, m in enumerate(messages)]


def add_summary(base_system_prompt: str, summary: str) -> str:
    if not summary:
        return base_system_prompt
    return f"{base_system_prompt}\n历史摘要:{summary}"


def _summarize(model, llm_context, api_key, base_url) -> str:
    """复用 SSE 通道做摘要；关掉 thinking，避免推理 token 吃掉输出预算。"""
    option = {"api_key": api_key, "base_url": base_url,
              "temperature": 0, "max_tokens": SUMMARY_MAX_TOKENS,
              "thinking": {"type": "disabled"}}
    summary = ""
    for event in stream_deepseek(model, llm_context, option):
        if event["type"] in ("done", "error"):
            summary = (event.get("message") or {}).get("content") or ""
    return summary.strip()


def compact_old_history(messages, system_prompt, api_key, base_url, recent_keep):
    """L3/L4：把旧消息总结成摘要，返回 (summary, 保留的近端消息)。

    摘要为空或没变短就退回 ("", 原 messages)：宁可不压，也不要丢掉历史
    还换回一段垃圾。
    """
    if len(messages) <= recent_keep:
        return "", messages

    old_messages = messages[:-recent_keep]
    recent_messages = messages[-recent_keep:]
    # 摘要请求不带工具：定义只烧 token，还会干扰生成
    llm_context = {"messages": old_messages, "system_prompt": system_prompt, "tools": []}
    summary = _summarize(llm_model(), llm_context, api_key, base_url)

    if not summary or count_tokens(summary) >= count_message_tokens(old_messages):
        return "", messages
    return summary, recent_messages


def _handoff_prompt(previous_summary=""):
    """已有旧摘要时必须要求合并，否则多轮压缩会丢掉更早的历史。"""
    if not previous_summary:
        return HAND_OFF_SUMMARY
    return (HAND_OFF_SUMMARY
            + "\n\n以下是更早之前的摘要，必须合并进新摘要，不得丢失：\n"
            + previous_summary)


def context_pressure(system_prompt, messages, tools, api_key, base_url,
                     previous_summary=""):
    """按上下文占用比例逐级压缩，返回 (system_prompt, messages, summary)。

    summary 非空表示做了 L3/L4，调用方应写成 checkpoint；L1/L2 返回 ""。
    previous_summary 是当前分支上一次压缩的摘要，计数和再摘要时都要带上。
    """
    base_system_prompt = system_prompt
    system_prompt = add_summary(base_system_prompt, previous_summary)

    available_tokens = MAX_TOKENS - OUTPUT_TOKENS - SAFE_TOKENS
    used = count_context_tokens(system_prompt, messages, tools)
    rank = used / available_tokens
    print(f"[rank]:{rank}")
    print(f"清理前的token数量:{used}")

    if rank <= L1_RATIO:
        print("不压缩")
        return system_prompt, messages, ""

    if rank <= L2_RATIO:
        print("L1轻量清理")
        messages = clean_redundant_messages(messages)
        print(f"清理后的数量{count_context_tokens(system_prompt, messages, tools)}")
        return system_prompt, messages, ""

    if rank <= L3_RATIO:
        print("L2压缩工具结果")
        messages = compact_tools_result(messages, max_tools_token=2000)
        print(f"清理后的数量{count_context_tokens(system_prompt, messages, tools)}")
        return system_prompt, messages, ""

    if rank <= L4_RATIO:
        print("L3压缩历史会话")
        summary, messages = compact_old_history(
            messages=messages,
            system_prompt=_handoff_prompt(previous_summary),
            api_key=api_key,
            base_url=base_url,
            recent_keep=RECENT_KEEP,
        )
        if summary:
            system_prompt = add_summary(base_system_prompt, summary)
        print(f"清理后的数量{count_context_tokens(system_prompt, messages, tools)}")
        return system_prompt, messages, summary

    print("L4：强制压缩")
    summary = ""
    # 有上限：摘要如果压不下去（空摘要 / 比原文长），必须能停下来
    for _ in range(MAX_COMPACT_ROUNDS):
        new_summary, compacted = compact_old_history(
            messages=messages,
            system_prompt=_handoff_prompt(summary or previous_summary),
            api_key=api_key,
            base_url=base_url,
            recent_keep=2,
        )
        if not new_summary:
            print("摘要未生效，停止压缩")
            break

        summary = new_summary
        messages = compacted
        system_prompt = add_summary(base_system_prompt, summary)

        # 重新统计，不能假设压缩后一定回到预算内
        if count_context_tokens(system_prompt, messages, tools) > CONTEXT_BUDGET:
            messages = compact_tools_result(messages, max_tools_token=1000)
        if count_context_tokens(system_prompt, messages, tools) <= CONTEXT_BUDGET:
            break
    else:
        print("达到压缩轮次上限，仍未回到预算内")

    return system_prompt, messages, summary
