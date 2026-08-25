# 上下文管理
# 写token计数器
import tiktoken
import json
_enc = tiktoken.get_encoding("cl100k_base")
def count_token(text:str)->int:
    if not text:
        return 0
    return len(_enc.encode(text))
def count_message_tokens(messages):
    total = 0
    for m in messages:
        total += count_token(m.get("role",""))
        total += count_token(m.get("content",""))
        total += count_token(m.get("tool_call_id",""))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function",{})
            total += count_token(fn.get("name",""))
            total += count_token(fn.get("arguments",""))
    return total
def count_context_tokens(system_prompt,messages,tools):
    total = count_token(system_prompt)
    total += count_message_tokens(messages)
    for t in tools:
        total += count_token(json.dumps(t.definition,ensure_ascii = False))
    return total
if __name__ == "__main__":
    # 模拟一条 user 消息 + 一个 add 工具定义，看总数是否合理增长
    msgs = [{"role": "user", "content": "计算1+2和3*4"}]
    tool_def = {"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}
    print(count_context_tokens("你是计算器", msgs, [{"definition": tool_def}]))
