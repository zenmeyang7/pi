"""Agent 运行时：把「调模型 → 有工具就执行 → 结果喂回去」这条循环自己写出来。

    protocol.py  数据契约 + StreamFn（手写 urllib + SSE）
    loop.py      双层循环：外层管会话生命周期，内层跑工具轮次
    context.py   tiktoken 计数 + 超限压缩（只改给模型看的视图）
    storage.py   会话树的持久化（parentId 成链 + lane 光标）
"""
from .loop import run_agent_loop, run_agent_loop_continue
from .protocol import (AgentContext, AgentEvent, AgentEventSink, AgentLoopconfig,
                       AgentMessage, AgentTool, StreamFn, stream_deepseek)

__all__ = ["run_agent_loop", "run_agent_loop_continue",
           "AgentContext", "AgentEvent", "AgentEventSink", "AgentLoopconfig",
           "AgentMessage", "AgentTool", "StreamFn", "stream_deepseek"]
