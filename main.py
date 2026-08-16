# main.py —— 驱动 agent_loop 的 REPL 演示
# 真实 DeepSeek 模型：agent 算完会停下等你输入下一条，输入 quit 结束。
import sys
import queue
import threading
from dotenv import load_dotenv
from type import StreamFn
from jsonl_tree import JsonlTree
# Windows 控制台默认 GBK，这里强制 UTF-8 避免中文乱码
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env")

from type import AgentContext, AgentLoopconfig, AgentTool
from agent_loop import run_agent_loop

# ==================== 1. 两个假工具 ====================
def make_add_tool():
    def execute(tool_call_id: str, args: dict) -> str:
        a, b = args["a"], args["b"]
        return f"{a} + {b} = {a + b}"
    return AgentTool(
        name="add",
        execute=execute,
        execution_mode="parallel",       # add 是并行工具
        definition={
            "type": "function",
            "function": {
                "name": "add",
                "description": "两个数相加",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    )

def make_multiply_tool():
    def execute(tool_call_id: str, args: dict) -> str:
        a, b = args["a"], args["b"]
        return f"{a} * {b} = {a * b}"
    return AgentTool(
        name="multiply",
        execute=execute,
        execution_mode="sequential",     # multiply 是顺序工具 -> 强制整批走顺序路径
        definition={
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "两个数相乘",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    )

# ==================== 3. steering：读取用户终端输入 ====================
class Steering():
    """后台线程读 stdin：__call__ 是"轮询"（agent 干活时非阻塞检查），
    wait 是"阻塞等待"（agent 答完后停下来等你下一条输入）。"""
    def __init__(self):
        self.input_queue = queue.Queue()
        threading.Thread(target=self.input_thread, daemon=True).start()

    def input_thread(self):
        while True:
            txt = input()
            if txt.strip():                       # 空行直接忽略，不放进队列
                self.input_queue.put(txt.strip())

    def __call__(self):                           # 轮询：非阻塞
        out = []
        while not self.input_queue.empty():
            out.append({"role": "user", "content": self.input_queue.get()})
        return out

    def wait(self):                               # 阻塞：等一条输入
        line = self.input_queue.get()             # Queue.get() 原生阻塞，不空转
        if line.lower() in ("quit", "exit", "q"):
            return []                             # 用户说退出 -> 结束 REPL
        return [{"role": "user", "content": line}]


# ==================== 4. 事件打印器 ====================
def make_printer():
    def emit(event):
        t = event["type"]
        if t == "agent_start":
            print("\n  [agent_start] agent 开始")
        elif t == "turn_start":
            print("  [turn_start]")
        elif t == "message_start":
            m = event["message"]
            print(f"  [message_start] {m['role']:>9}: {m.get('content','')}")
        elif t == "message_end":
            print("  [message_end  ]")
        elif t == "tool_execution_start":
            print(f"  [tool_exec_start] {event['toolName']}({event['args']})")
        elif t == "tool_execution_end":
            status = "OK" if not event["isError"] else "ERROR"
            print(f"  [tool_exec_end  ] {event['toolName']} -> {status}: {event['result']}")
        elif t == "turn_end":
            n = len(event["toolResults"])
            print(f"  [turn_end     ] 工具结果 {n} 条 -> 推回 context")
        elif t == "agent_end":
            n = len(event["messages"])
            print(f"  [agent_end    ] 结束，共 {n} 条消息\n")
    return emit

# ==================== 5. 跑起来 ====================
def run_scenario(title, prompts, config, context):
    print(f"===== {title} =====")
    transcript = run_agent_loop(
        prompts, context, config,
        make_printer(),
        StreamFn,
    )
    print(f"----- 完整转录（{len(transcript)} 条消息）-----")
    for m in transcript:
        extra = ""
        if m.get("tool_calls"):
            extra = " tool_calls=" + ",".join(tc["function"]["name"] for tc in m["tool_calls"])
        if m.get("stop_reason"):
            extra += f" stop_reason={m['stop_reason']}"
        print(f"  {m['role']:>9}: {m.get('content','')}{extra}")
    print()



# 主程序
if __name__ == "__main__":
    tools = [make_add_tool(), make_multiply_tool()]
    session = JsonlTree("test_session.jsonl")
    if session.exists():
        session.load()
        prompts = []
        ctx_a = AgentContext(system_prompt="你是计算器，能用工具就用工具。", tools=tools,tree=session)
        print("已恢复上次对话")
    else:
        session.create()
        prompts = [{"role":"user","content":"计算1+2和3*4"}]
        ctx_a = AgentContext(system_prompt="你是计算器，能用工具就用工具。", tools=tools,tree=session)
    # ---- 场景 A：REPL —— agent 答完会停下等你输入，quit 结束 ----
    class ConfigA(AgentLoopconfig):
        def __init__(self):
            self.steering = Steering()
            super().__init__(
                model="deepseek-chat",
                get_steering_messages=self.steering,      # 干活时：非阻塞轮询有没有新输入
                wait_for_steering=self.steering.wait,     # 答完后：阻塞等你的下一条输入
                tool_execution="parallel",
            )

    run_scenario(
        "场景 A：REPL —— 启动后直接输入；agent 答完停下等你；输入 quit 结束",
        prompts=prompts,
        config=ConfigA(),
        context=ctx_a
    )
