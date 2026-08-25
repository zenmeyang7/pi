# main.py —— 驱动 agent_loop 的 REPL 演示
# 真实 DeepSeek 模型：agent 算完会停下等你输入下一条，输入 quit 结束。
import sys
import queue
import threading
from dotenv import load_dotenv
from type import StreamFn
from jsonl_tree import JsonlTree,SessionRepo
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
    def __init__(self,context,repo):
        self.context = context
        self.repo =repo
        self.deferred = [] 
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
            line=self.input_queue.get()
            if self._classify(line) == "msg":
                out.append({"role":"user","content":line})
            else:
                self.deferred.append(line)
        return out

    # 先处理掉第一轮中未处理的部分，然后去处理追加的部分
    def wait(self):                                       
        while self.deferred:
            line = self.deferred.pop(0)
            status = self._classify(line)
            if status == "quit":
                return []
            if status == "command":
                self._run_command(line)
                continue
            return [{"role":"user","content":line}]
        while True:
            line = self.input_queue.get()
            status = self._classify(line)
            if status == "quit":
                return []
            if status == "command":
                self._run_command(line)
                continue
            return [{"role":"user","content":line}]

    # 对终端输入的语言进行分类处理
    def _classify(self,line):
        if line.lower() in ("quit","exit","q"):
            return "quit"
        elif line.startswith("/"):
            return "command"
        else:
            return "msg" 

    def _run_command(self,line):
        # 没有会话树
        cmd = line.strip().lower()
        if cmd == "/branch":
            name=self.context.tree.branch()
            print(f"已经换到了新的{name}分支上")
        elif cmd == "/main":
            self.context.tree.switch("main")
            print("已经切回到主线\“main\"分支上")
        elif cmd == "/list":
            sessions = self.repo.list()
            if not sessions:
                print("还没有创建对话")
                return
            for s in sorted(sessions,key=lambda s:s["createdAt"],reverse=True):
                print(f"{s["name"]}")
        elif cmd.startswith("/new"):
            name = cmd[5:].strip()
            if not name:
                print("/new 使用错误，用法:/new<会话名>")
                return
            try:
                new_tree=self.repo.create_tree(name)
            except FileExistsError as e:
                print(e)
                return
            self.context.tree = new_tree
            print("已经切换到新的对话")
        elif cmd.startswith("/open"):
            name = cmd[6:].strip()
            if not name:
                print("/open用法错误,用法:/open<会话名>")
            new_tree = self.repo.open(name)
            self.context.tree = new_tree
            print("会话已经打开")
        elif cmd.startswith("/fork"):
            # 新开一个对话可以保留原来的历史记录 有点像github里面的fork
            name = cmd[6:].strip()
            if not name:
                print("/fork 指令缺少名字")
                return
            try:
                new_tree = self.repo.fork(name,self.context.tree)
            except FileExistsError as e:
                print(e)
                return
            self.context.tree = new_tree
        elif cmd =="/back":
            if self.context.tree is None:
                print("当前没有会话，先new或者是/open")
                return
            res = self.context.tree.back()
            if res is None:
                print("已经回退到头")
            else:
                print("已经回溯一步")
        else:
            print(f"未知命令{line}")


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
    repo = SessionRepo("sessions")
    session = repo.list()
    if not session:
        tree = repo.create_tree("default")
        prompts = [{"role":"user","content":"计算1+2和3*4"}]
        print("首次运行，默认对话为default")
    else:
        session.sort(key = lambda s:s["createdAt"],reverse=True)
        newset = session[0]["name"]
        tree = repo.open(newset)
        prompts = []
    ctx_a = AgentContext(system_prompt="你是计算机，能用工具就用工具",tools=tools,tree=tree)
    # ---- 场景 A：REPL —— agent 答完会停下等你输入，quit 结束 ----
    class ConfigA(AgentLoopconfig):
        def __init__(self):
            self.steering = Steering(ctx_a,repo=repo)
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
        context=ctx_a,
    )
