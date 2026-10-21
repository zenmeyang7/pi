"""Agent 循环：外层管会话生命周期，内层跑工具轮次。

退出靠两个哨位：没有待处理的用户消息、且上一轮没有工具调用 —— 模型不会对
assistant 消息再做回复，所以没人再问时必须停下，否则它会自己续写下去。

工具执行分串行 / 并行两条路径；上下文超限时先压缩再送进模型。
"""
import json
from agent.protocol import AgentContext, AgentEventSink, AgentLoopconfig, StreamFn
from agent.context import context_pressure
from agent.llm import options as llm_options, model as llm_model
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List
TOOL_WORKERS = 2
DEFAULT_MAX_TURNS = 25


def run_loop(context: AgentContext, newmessages: list, config: AgentLoopconfig,
             emit: AgentEventSink, streamfunction: StreamFn):
    currentContext = context
    pendingMessages = config.get_steering_messages() or []
    max_turns = getattr(config, "max_turns", None) or DEFAULT_MAX_TURNS
    turns = 0

    # 外层：agent 该停了但又有 steering / follow-up 进来 → 继续
    while True:
        has_more_tool_calls = True

        # 内层：工具调用轮次
        while has_more_tool_calls or pendingMessages:
            if pendingMessages:
                # 落库到会话树；没有树时退回 context.messages
                for m in pendingMessages:
                    emit({"type": "message_start", "message": m})
                    emit({"type": "message_end", "message": m})
                    currentContext.append_message(m)
                    newmessages.append(m)
                pendingMessages = []

            # 协议守卫：历史最后一条若已是 assistant（agent 已答完，没人再问），
            history = currentContext.tree.for_path() if currentContext.tree else currentContext.messages
            if not pendingMessages and history and history[-1]["role"] == "assistant":
                break

            # 轮次上限：模型若陷入"反复调同一个工具"会一直转下去，
            if turns >= max_turns:
                emit({"type": "turn_limit_exceeded", "turns": turns, "maxTurns": max_turns})
                break

            message = streamAssistantResponse(currentContext,config,history,emit, streamfunction)
            turns += 1
            newmessages.append(message)

            # error / aborted：统一收尾，把错误消息带回去
            if message.get("stop_reason") in ("error", "aborted"):
                emit({"type": "turn_end", "message": message, "toolResults": []})
                emit({"type": "agent_end", "messages": newmessages})
                return newmessages

            tool_calls = message.get("tool_calls") or []
            tool_results = []
            has_more_tool_calls = False
            if tool_calls:
                if message.get("stop_reason") == "length":
                    # 截断保护：参数可能已被截断，一律不执行、全部报错，
                    # 让模型下一轮带上完整参数重发 —— 半截参数写文件比不写更糟
                    for tc in tool_calls:
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Tool call {tc.get('function', {}).get('name', '?')} "
                                       "was not executed: output hit token limit, "
                                       "arguments may be truncated. Re-issue with complete arguments.",
                        })
                else:
                    executed = executeToolCalls(currentContext, message, config, emit)
                    tool_results = executed["messages"]
                has_more_tool_calls = True   # 有工具调用 → 至少还要一轮

            # 工具结果推回历史（下一轮 LLM 能看到）
            for tr in tool_results:
                currentContext.append_message(tr)
            newmessages.extend(tool_results)

            emit({"type": "turn_end", "message": message, "toolResults": tool_results})

            # 每轮结束后轮询 steering（用户可能已经打字了）
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
    for prompt in prompts:
        emit({"type": "message_start", "message": prompt})
        emit({"type": "message_end", "message": prompt})
        currentContext.append_message(prompt)
    run_loop(currentContext, newMessages, config, emit, streamFn)
    return newMessages


def run_agent_loop_continue(context: AgentContext, config: AgentLoopconfig,
                            emit: AgentEventSink, streamFn: StreamFn) -> list:
    """错误重试 / 续跑入口：不再注入新的 prompt，直接用 context 里已有的历史接着跑。

    和 run_agent_loop 的唯一区别就是「跳过 prompts 注入」。
    前置条件：历史最后一条不能是 assistant —— 否则协议守卫会立刻跳过模型调用，
    跑了个寂寞。这是调用方误用（不是运行时异常），所以直接抛。
    """
    history = context.tree.for_path() if context.tree else context.messages
    if history and history[-1].get("role") == "assistant":
        raise ValueError(
            "run_agent_loop_continue 要求历史尾部是 user/tool；"
            "当前是 assistant，模型没有新输入可回复（协议守卫会直接跳过）。"
        )
    newMessages = []
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    return run_loop(context, newMessages, config, emit, streamFn)


def streamAssistantResponse(context: AgentContext, config: AgentLoopconfig,history:List,
                            emit: AgentEventSink, streamFuction: StreamFn):
    raw_history = history          # 保留 id，压缩写回时要定位 keep_from
    # AgentMessage -> Message（只留 user/assistant/tool 并剥掉多余键）
    history = config.convert_to_llm(history)
    tools = context.tools
    if config.tool_selector is not None:
        tools = config.tool_selector(history, context.tools)
    options = llm_options()
    model = config.model or llm_model()
    # 超限时压缩：保留最近若干条，把更早的交给模型总结成一段摘要。
    # 上一轮的摘要从树里读，压缩结果再写回成 checkpoint。
    previous_summary = context.tree.latest_summary() if context.tree else ""
    system_prompt,llmessages,summary = context_pressure(
        context.system_prompt,history,tools,
        options["api_key"],options["base_url"],
        previous_summary=previous_summary)
    if summary and context.tree:
        # 压缩后的消息是原历史的尾部，用第一条的 id 标记"从这里往后保留"
        keep_from = raw_history[-len(llmessages)]["id"] if llmessages else None
        context.tree.append_compaction(summary, keep_from)
    llm_context = {
        "system_prompt": system_prompt,
        "messages": llmessages,
        "tools": tools,
    }
    # 消费 StreamFn 的事件流，在 done / error 处摘出裸消息（去掉 message 壳）
    finalmessage = None
    for event in streamFuction(model, llm_context, options):
        if event["type"] in ("done", "error"):
            finalmessage = event["message"]
        elif event["type"] == "delta":
            emit(event)
    # 最终消息进 context（下一轮可见），并透传 start/end
    context.append_message(finalmessage)
    emit({"type": "message_start", "message": finalmessage})
    emit({"type": "message_end", "message": finalmessage})
    return finalmessage


def executeToolCalls(currentContext: AgentContext, assistant_message: dict,
                     config: AgentLoopconfig, emit: AgentEventSink):
    tool_calls = assistant_message.get("tool_calls") or []
    # 只要这批里有一个工具声明了 sequential，整批降级为串行：
    # 写操作一旦和读操作并行，"写文件再读回来"就会读到旧内容
    has_sequential = False
    for tc in tool_calls:
        name = tc.get("function", {}).get("name")
        target_tool = next((t for t in currentContext.tools if t.name == name), None)
        if target_tool and target_tool.execution_mode == "sequential":
            has_sequential = True
            break
    if config.tool_execution == "sequential" or has_sequential:
        return executeToolCallsSequential(currentContext, tool_calls, config, emit)
    return executeToolCallsParallel(currentContext, tool_calls, config, emit)


def executeToolCallsSequential(currentContext: AgentContext, tool_calls: list,
                               config: AgentLoopconfig, emit: AgentEventSink):
    messages = []
    for tool_call in tool_calls:
        tool_call_id = tool_call["id"]
        name = tool_call.get("function", {}).get("name")
        emit({"type": "tool_execution_start", "toolCallId": tool_call_id,
              "toolName": name, "args": tool_call.get("function", {}).get("arguments")})

        preparation = prepareToolCall(currentContext, tool_call,config)
        if preparation["kind"] == "immediate":
            executed = {"result": preparation["result"], "isError": preparation["isError"]}
        else:
            executed = executePreparedToolCall(preparation, emit)
        # immediate（工具不存在 / 参数非法 / 被 before 拦下）也要过钩子：
        # 审计不能漏掉"尝试了但没执行成功"的调用
        executed = finalizeExecutedToolCall(name,executed,config)
        content, is_error = executed["result"], executed["isError"]
        emit({"type": "tool_execution_end", "toolCallId": tool_call_id,
              "toolName": name, "result": content, "isError": is_error})
        messages.append(createToolResultMessage(tool_call_id, name, content, is_error))
    return {"messages": messages, "terminate": False}


def executeToolCallsParallel(currentContext: AgentContext, tool_calls, config, emit):
    messages = []
    # results 严格按「提交顺序」保存：慢的先跑完也不会抢到前面去。
    # 每一项统一是 (tool_call_id, name, payload)，payload 要么是 Future，
    # 要么是已经确定的 (content, is_error) 二元组 —— 统一形状才不会在解包时炸。
    results = []
    with ThreadPoolExecutor(max_workers=TOOL_WORKERS) as executor:
        for tc in tool_calls:
            name = tc.get("function", {}).get("name")        # ← 不是 tc["name"]！
            args = tc.get("function", {}).get("arguments")
            emit({"type": "tool_execution_start", "toolCallId": tc["id"],
                  "toolName": name, "args": args})
            preparation = prepareToolCall(currentContext, tc,config)
            if preparation["kind"] == "immediate":
                # 工具不存在 / 参数非法 / 被 before_tool_call 拦下：不用提交，结果已定
                results.append((tc["id"], name,
                                (preparation["result"], preparation["isError"])))
            else:
                fut = executor.submit(executePreparedToolCall, preparation, emit)
                results.append((tc["id"], name, fut))
        for tid, name, payload in results:                    # 按原序收
            if isinstance(payload, Future):
                executed = payload.result()
            else:
                executed = {"result": payload[0], "isError": payload[1]}
            # 钩子统一在主线程的收集阶段跑：Future 结果和 immediate 结果都覆盖
            executed = finalizeExecutedToolCall(name, executed, config)
            content, is_error = executed["result"], executed["isError"]
            emit({"type": "tool_execution_end", "toolCallId": tid,
                  "toolName": name, "result": content, "isError": is_error})
            messages.append(createToolResultMessage(tid, name, content, is_error))
    return {"messages": messages, "terminate": False}

def prepareToolCall(currentContext: AgentContext, tool_call: dict,config:AgentLoopconfig) -> dict:
    name = tool_call.get("function", {}).get("name")
    tool = next((t for t in currentContext.tools if t.name == name), None)
    if not tool:
        return {"kind": "immediate", "result": f"Tool {name} not found", "isError": True}
    raw_args = tool_call.get("function", {}).get("arguments")
    try:
        args = json.loads(raw_args or "{}")
    except Exception as e:
        return {"kind": "immediate", "result": f"Invalid arguments: {e}", "isError": True}
    if not isinstance(args, dict):
        return {"kind": "immediate", "result": f"Invalid arguments: expected object, got {type(args).__name__}",
                "isError": True}
    if config.before_tool_call:
        res = config.before_tool_call(name,args)
        if res and res.get("block"):
            return {"kind":"immediate","result":res["reason"],"isError":True}
    return {"kind": "prepared", "tool": tool, "args": args, "tool_call_id": tool_call["id"]}


# 工具抛的异常一律转成文本结果：工具出错不该让整个 agent 循环崩掉
def executePreparedToolCall(prepared: dict, emit: AgentEventSink) -> dict:
    try:
        result = prepared["tool"].execute(prepared["args"])
        return {"result": result, "isError": False}
    except Exception as e:
        return {"result": f"ERROR: {e}", "isError": True}


# after_tool_call 钩子：允许改写工具结果
def finalizeExecutedToolCall(name,outcome,config):
    if not config.after_tool_call:
        return outcome
    res = config.after_tool_call(name,outcome["result"],outcome["isError"])
    if not res:
        return outcome
    return {"result":res.get("result",outcome["result"]),"isError":res.get("isError",outcome["isError"])}


# 包装成一条"工具结果消息"（wire format）
def createToolResultMessage(tool_call_id: str, tool_name: str, content: str, is_error: bool) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id,"name":tool_name,
            "content": content, "isError": is_error}
