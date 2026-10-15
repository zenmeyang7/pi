import os
from agent.protocol import AgentTool


def safe_path(path):
    root = os.path.abspath(os.getcwd())
    # 先 normpath 掉 ../ 之类的花招，再判 commonpath —— 顺序反了就会被绕过
    target = os.path.abspath(os.path.normpath(path))
    try:
        ans = os.path.commonpath([target,root]) == root
    except ValueError:
        ans = False
    return target if ans else None

def read_file():
    def execute(args):
        path = args.get("path")
        limit = args.get("limit")
        safe=safe_path(path=path)
        if  safe is None:
            return "该文件不在当前文件夹下，没有权限执行"
        # limit 由模型填，偶尔填成字符串（"20"），切片会 TypeError；这里收敛一次
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                return f"limit 必须是整数，收到 {limit!r}"
        with open(safe,"r",encoding="utf-8") as f:
            text = f.readlines()
        return "".join(text[:limit])
    return AgentTool(
        name="read_file",
        execute=execute,
        execution_mode="parallel",
        definition={
            "type":"function",
            "function":{
                "name":"read_file",
                "description":"读取指定的文件夹下的内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                      "path":{"type":"string"},
                      "limit":{"type":"integer"}
                       },
                    "required": ["path"],
                },
            }
        }
    )

def write_file():
    def execute(args):
        path = args.get("path")
        text = args.get("text")
        safe = safe_path(path=path)
        if safe is None:
            return "该文件不在当前文件夹下，没有权限执行"
        # 父目录可能还不存在，直接 open 会 FileNotFoundError
        parent = os.path.dirname(safe)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(safe,"w",encoding="utf-8") as f:
            f.write(text)
        return "已经成功写入"
    return AgentTool(
        name="write_file",
        execute=execute,
        # 写操作有副作用：和 read_file 同批并行时，"写文件再读回来"会读到旧内容，
        # 甚至文件还没创建。标 sequential 后整批降级为串行（见 executeToolCalls）。
        execution_mode="sequential",
        definition={
            "type":"function",
            "function":{
                "name":"write_file",
                "description":"写入内容到文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                      "path":{"type":"string"},
                      "text":{"type":"string"}
                       },
                    "required": ["path","text"],
                },
            }
        }
    )
