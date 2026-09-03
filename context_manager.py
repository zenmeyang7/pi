# 上下文管理
# 写token计数器
import tiktoken
import json
from type import StreamFn
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
def summarize(messages,api_key,base_url)->str:
    text = ""
    if not messages:
        print("没有历史消息")
        return ""
    for m in messages:
        text+=f"{m.get("role","")}:{m.get("content","")}\n"
    model = "deepseek-chat"
    system_prompt = "你是摘要助手"
    user = "请用一下段对话总结以下对话的重点:\n\n" + text
    llm_context = {
            "system_prompt": system_prompt,
            "messages": [{"role":"user","content":user}],
            "tools": [],
        }
    options = {
            "api_key": api_key,
            "base_url": base_url,
        }
    final_message = ""
    for event in StreamFn(model,llm_context,options):
        if event["type"] in ["done","error"]:
            final_message = event["message"]["content"]
    return final_message

RECENT_KEEP = 5
MAX_TOKENS = 8000
def compact_context(system_prompt,messages,tools,max_tokens,api_key,base_url):
    if count_context_tokens(system_prompt,messages,tools)<max_tokens:
        return (system_prompt,messages)
    if len(messages)<RECENT_KEEP:
        return (system_prompt,messages)
    old = messages[:-RECENT_KEEP]
    recent = messages[-RECENT_KEEP:]
    summary = summarize(old,api_key,base_url)
    new_system = system_prompt+"\n\n[旧对话摘要]\n" +summary
    return (new_system,recent)

if __name__ == "__main__":
    # 模拟一条 user 消息 + 一个 add 工具定义，看总数是否合理增长
    msgs = [{"role": "user", "content": "计算1+2和3*4"}]
    tool_def = {"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}
    print(count_context_tokens("你是计算器", msgs, [{"definition": tool_def}]))
