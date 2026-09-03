# agent的执行流程 —— 同步版，消息一律 dict（OpenAI wire format）
import json
import os
from type import AgentContext, AgentEventSink, AgentLoopconfig, StreamFn
from context_manager import count_context_tokens,compact_context,MAX_TOKENS,count_token,count_message_tokens
def run_loop(context: AgentContext, newmessages: list, config: AgentLoopconfig,
             emit: AgentEventSink, streamfunction: StreamFn):
    currentContext = context
    pendingMessages = config.get_steering_messages() or []

    # 外层循环：agent 该停了但还有 steering/follow-up 进来 → 继续
    while True:
        has_more_tool_calls = True

        # 内层循环：工具调用轮次
        while has_more_tool_calls or pendingMessages:
            # 先去看用户有没有额外的输入
            if pendingMessages:
                # 内部的存储结构现在是一个树，就需要把消息 append 到树里，append_message内部就会判断树是不是仍然存在，如果不存在就直接append到messages里
                for m in pendingMessages:
                    emit({"type": "message_start", "message": m})
                    emit({"type": "message_end", "message": m})
                    currentContext.append_message(m)
                    newmessages.append(m)
                pendingMessages = []

            # 调用 LLM，得到一个 assistant 消息 dict
            message = streamAssistantResponse(currentContext,config, emit, streamfunction)
            # 得到一条新消息
            
            newmessages.append(message)

            # error / aborted：统一收尾，把错误消息带回去
            if message.get("stop_reason") in ("error", "aborted"):
                emit({"type": "turn_end", "message": message, "toolResults": []})
                emit({"type": "agent_end", "messages": newmessages})
                return newmessages
            # 查看返回的消息里面有无工具的调用
            tool_calls = message.get("tool_calls") or []
            tool_results = []
            has_more_tool_calls = False   # 默认这轮结束
            if tool_calls:
                # 处理工具调用，如果是 length 截断保护 → 不执行，直接报错；否则执行工具调用 超出token限制
                if message.get("stop_reason") == "length":
                    # 截断保护：不执行，全部报错，模型下一轮自己重发
                    for tc in tool_calls:
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Tool call {tc.get('function', {}).get('name', '?')} "
                                       "was not executed: output hit token limit, "
                                       "arguments may be truncated. Re-issue with complete arguments.",
                        })
                else:
                    # 正常执行：整个工具批次跑一遍，返回 tool_result 消息列表 处理的是大模型返回的最新消息
                    executed = executeToolCalls(currentContext, message, config, emit)
                    tool_results = executed["messages"]
                has_more_tool_calls = True   # 有工具调用 → 至少还要一轮

            # 工具结果推回历史（下一轮 LLM 能看到）
            for tr in tool_results:
                currentContext.append_message(tr)
            newmessages.extend(tool_results)

            emit({"type": "turn_end", "message": message, "toolResults": tool_results})

            # 每轮结束后轮询 steering（用户可能打字了）
            pendingMessages = config.get_steering_messages() or []

        # 内层退出：agent 暂时没有更多动作了。
        # 交互模式（REPL）：阻塞等用户下一条输入，来了就继续外层循环；否则结束。
        pendingMessages = config.wait_for_steering() or []
        if not pendingMessages:
            break

    emit({"type": "agent_end", "messages": newmessages})
    return newmessages


def run_agent_loop(prompts: list, context: AgentContext, config: AgentLoopconfig,
                   emit: AgentEventSink, streamFn: StreamFn) -> list:
    newMessages = list(prompts)
    currentContext = context
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    # 这段是看有没有程序里面写的
    for prompt in prompts:
        emit({"type": "message_start", "message": prompt})
        emit({"type": "message_end", "message": prompt})
        currentContext.append_message(prompt)
    run_loop(currentContext, newMessages, config, emit, streamFn)
    return newMessages


def run_agent_loop_continue():
    # agent 发现错误需要重试：context 里已有 user/toolResult，不追加新消息直接跑
    raise NotImplementedError("复制 run_agent_loop 但跳过 prompts 注入；且最后一条必须是 user/toolResult")


def streamAssistantResponse(context: AgentContext, config: AgentLoopconfig,
                            emit: AgentEventSink, streamFuction: StreamFn):
    # AgentMessage -> Message（只留 user/assistant/tool 并剥掉多余键）
    tree = context.tree # 存储的历史消息
    history = tree.for_path() if tree else context.messages
    history = config.convert_to_llm(history)
    options = {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL"),
        }
    # 压缩上下文，用最大token和最近消息数限制
    system_prompt,llmessages = compact_context(context.system_prompt,history,context.tools,MAX_TOKENS,options["api_key"],options["base_url"])
    llm_context = {
        "system_prompt": system_prompt,
        "messages": llmessages,
        "tools": context.tools,
    }
    # 计算本轮调用消耗的token数
    tokens = count_context_tokens(system_prompt,llmessages,context.tools)
    # 消费 StreamFn 的事件流，在 done/error 处摘出裸消息（去掉 message 壳）
    finalmessage = None
    # 找到大模型返回的最终一条消息
    for event in streamFuction(config.model, llm_context, options):
        if event["type"] in ("done", "error"):
            finalmessage = event["message"]
        elif event["type"] == "delta":
            emit(event)
    # 最终消息进 context（下一轮可见），并透传 start/end
    context.append_message(finalmessage)
    emit({"type": "message_start", "message": finalmessage})
    emit({"type": "message_end", "message": finalmessage})
    return finalmessage


# 工具的执行，分发到顺序 / 并行两条路径
def executeToolCalls(currentContext: AgentContext, assistant_message: dict,
                     config: AgentLoopconfig, emit: AgentEventSink):
    # 需要调用的工具
    tool_calls = assistant_message.get("tool_calls") or []
    # 是否串行
    has_sequential = False 
    for tc in tool_calls:
        # 找到当前需要调用的工具
        name = tc.get("function", {}).get("name")
        # 判断我注册的工具里面有没有
        target_tool = next((t for t in currentContext.tools if t.name == name), None)
        # 如果有，并且是串行模式，就标记为串行
        if target_tool and target_tool.execution_mode == "sequential":
            has_sequential = True
            break
    # 如果有任何一个工具是串行模式，就走顺序执行，否则走并行执行
    if config.tool_execution == "sequential" or has_sequential:
        return executeToolCallsSequential(currentContext, tool_calls, config, emit)
    return executeToolCallsParallel(currentContext, tool_calls, config, emit)


# 顺序：一个一个调用工具
def executeToolCallsSequential(currentContext: AgentContext, tool_calls: list,
                               config: AgentLoopconfig, emit: AgentEventSink):
    messages = []
    for tool_call in tool_calls:
        # 找到需要调用的工具的id和name，发出开始事件
        tool_call_id = tool_call["id"]
        name = tool_call.get("function", {}).get("name")
        emit({"type": "tool_execution_start", "toolCallId": tool_call_id,
              "toolName": name, "args": tool_call.get("function", {}).get("arguments")})

        # 准备（找工具 + 解析参数）→ 执行（捕获异常）
        preparation = prepareToolCall(currentContext, tool_call)
        if preparation["kind"] == "immediate":
            content, is_error = preparation["result"], preparation["isError"]
        else:# 工具已经准备好了
            executed = executePreparedToolCall(preparation, emit)
            content, is_error = executed["result"], executed["isError"]
        # content 是工具的返回结果，is_error 是是否出错
        emit({"type": "tool_execution_end", "toolCallId": tool_call_id,
              "toolName": name, "result": content, "isError": is_error})
        messages.append(createToolResultMessage(tool_call_id, name, content, is_error))
    return {"messages": messages, "terminate": False}


# 并行：目前退化成顺序（真正的并发 = concurrent.futures，是后续话题）
def executeToolCallsParallel(currentContext: AgentContext, tool_calls: list,
                             config: AgentLoopconfig, emit: AgentEventSink):
    return executeToolCallsSequential(currentContext, tool_calls, config, emit)


# 准备工具 + 校验参数
def prepareToolCall(currentContext: AgentContext, tool_call: dict) -> dict:
    # 从currentContext的tools里面检查是不是有对应的工具
    name = tool_call.get("function", {}).get("name")
    tool = next((t for t in currentContext.tools if t.name == name), None)
    if not tool:
        return {"kind": "immediate", "result": f"Tool {name} not found", "isError": True}
    try:
        args = json.loads(tool_call.get("function", {}).get("arguments") or "{}")
    except Exception as e:
        return {"kind": "immediate", "result": f"Invalid arguments: {e}", "isError": True}
    return {"kind": "prepared", "tool": tool, "args": args, "tool_call_id": tool_call["id"]}


# 真正的调用工具：错误转成文本，永不抛
def executePreparedToolCall(prepared: dict, emit: AgentEventSink) -> dict:
    try:
        result = prepared["tool"].execute(prepared["tool_call_id"], prepared["args"])
        return {"result": result, "isError": False}
    except Exception as e:
        return {"result": f"ERROR: {e}", "isError": True}


# 允许改写结果（afterToolCall 钩子）：最小版跳过，留扩展点
def finalizeExecutedToolCall():
    pass


# 包装成一条"工具结果消息"（wire format）
def createToolResultMessage(tool_call_id: str, tool_name: str, content: str, is_error: bool) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id,"name":tool_name,
            "content": content, "isError": is_error}
