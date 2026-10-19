# main.py —— 驱动 agent_loop 的 REPL 演示
# 模型由 .env 的 LLM_MODEL / LLM_BASE_URL 决定（默认 deepseek-flash）。
import sys
import queue
import threading
from dotenv import load_dotenv
from agent.loop import run_agent_loop
from agent.protocol import AgentContext, AgentLoopconfig, AgentTool, stream_deepseek
from agent.llm import model as llm_model
from agent.storage import SqlRepo, get_conn, init_schema
from tools import get_tools
from tools.search import init_retriever

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env")

class Steering():
    def __init__(self,context,repo):
        self.context = context
        self.repo =repo
        self.deferred = [] 
        self.input_queue = queue.Queue()
        threading.Thread(target=self.input_thread, daemon=True).start()

    def input_thread(self):
        while True:
            try:
                txt = input()
            except (EOFError, KeyboardInterrupt):
                self.input_queue.put("quit")
                return
            if txt.strip():                       
                self.input_queue.put(txt.strip())

    def __call__(self):                           
        out = []
        while not self.input_queue.empty():
            line=self.input_queue.get()
            if self._classify(line) == "msg":
                out.append({"role":"user","content":line})
            else:
                self.deferred.append(line)
        return out

    # 先消化上一轮暂存的 deferred，再去阻塞等新输入
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

    def _classify(self,line):
        if line.lower() in ("quit","exit","q"):
            return "quit"
        elif line.startswith("/"):
            return "command"
        else:
            return "msg" 

    def _run_command(self,line):
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
        elif cmd.startswith("/open"):
            name = cmd[6:].strip()
            if not name:
                print("/open用法错误,用法:/open<会话名>")
            new_tree = self.repo.open(name)
            self.context.tree = new_tree
        elif cmd.startswith("/fork"):
            # fork：把源会话的历史复制过来
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


def make_printer():
    def emit(event):
        t = event["type"]
        if t == "agent_start":
            print("\n  [agent_start] agent 开始")
        elif t == "turn_start":
            print("  [turn_start]")
        elif t == "message_start":
            print("  [message_start]")
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
        elif t == "delta":
            print(event["content"], end="", flush=True)
    return emit

def after_tool_call(name, result, is_error):
    status = "ERROR" if is_error else "OK"
    preview = str(result).replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    print(f"  [after_tool_call] {name} -> {status}: {preview}", file=sys.stderr)
    return {"result": result, "isError": is_error}


def run_scenario(title, prompts, config, context):
    print(f"===== {title} =====")
    transcript = run_agent_loop(
        prompts, context, config,
        make_printer(),
        stream_deepseek,
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



if __name__ == "__main__":
    init_retriever("rag/index")
    tool_list = get_tools()
    conn = get_conn()
    init_schema(conn)          # 首次运行先建表
    repo = SqlRepo(conn)
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
    ctx_a = AgentContext(system_prompt="你是计算机，能用工具就用工具",tools=tool_list,tree=tree)
    class ConfigA(AgentLoopconfig):
        def __init__(self):
            self.steering = Steering(ctx_a,repo=repo)
            super().__init__(
                model=llm_model(),
                get_steering_messages=self.steering,      # 干活时：非阻塞轮询有没有新输入
                wait_for_steering=self.steering.wait,     # 答完后：阻塞等你的下一条输入
                tool_execution="parallel",         # 无依赖的工具可并行；整批里有 sequential 工具时会自动降级
                after_tool_call=after_tool_call,   # 每次工具执行后审计（含未执行成功的调用）
            )

    run_scenario(
        "场景 A：REPL —— 启动后直接输入；agent 答完停下等你；输入 quit 结束",
        prompts=prompts,
        config=ConfigA(),
        context=ctx_a,
    )
    conn.close()
