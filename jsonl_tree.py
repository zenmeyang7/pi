import threading
import os
import json
import time
import uuid
class JsonlTree:
    def __init__(self,path):
        self.path = path
        # 用id找到对应的消息
        self.entries = {}
        # 查询分叉树
        self.lanes = {}
        # 我需要往哪一个分支上面写
        self.active = "main"
        # 对应会话的编号
        self.session_id = None
        # 会话创建的时间
        self.created_at = None
        # 一个锁对象，保证同一时刻只有一个人写如文件
        self._lock = threading.Lock()

    # 判断文件是否为空
    def exists(self):
        if os.path.exists(self.path) and os.path.getsize(self.path)>0:
            return True
        return False

    # 创建一个新的会话
    def create(self):
        print("创建一个新的对话中")
        self.session_id = uuid.uuid4().hex
        self.created_at = int(time.time()*1000)
        self.lanes["main"]=None
        print(f"新对话的id是{self.session_id},新对话创建的时间是{self.created_at}")
        header = {"kind":"header","id":self.session_id,"createdAt":self.created_at}
        with open(self.path,"w",encoding="UTF-8",newline="\n")as f:
            f.write(json.dumps(header,ensure_ascii=False)+"\n")
        return self.session_id

    # 写入文件当成日记
    def _write(self,obj):
        print("正在将操作写入文件")
        with self._lock:
            with open(self.path,"a",encoding="UTF-8",newline="\n")  as f:
                f.write(json.dumps(obj,ensure_ascii=False)+"\n")

    # 在文件中追加消息
    def append(self,message,kind="message",lane = None):
        print("正在存入追加的新消息")
        lane = lane or self.active
        print(f"当前在{lane}分支上")
        entry = {
            "id":uuid.uuid4().hex,
            "message":message,
            "lane":lane,
            "parentId":self.lanes.get(lane),
            "type":kind,
            "timestamp":int(time.time()*1000)
        }
        self._write({"kind":"entry","entry":entry})
        self.entries[entry["id"]] = entry
        self.lanes[lane] = entry["id"]
        self.active = lane
        return entry

    # 建立会话内不同的分支
    def branch(self,start_id=None):
        print("正在该节点去建立你所需要的分支")
        leaf = start_id if start_id is not None else self.lanes.get(self.active)
        name = f"branch-{uuid.uuid4().hex[:8]}"
        self._write({"kind":"lane","lane":name,"leafId":leaf,"timestamp":int(time.time()*1000)})
        self.lanes[name] = leaf
        self.active = name
        return name

    # 转变分支
    def switch(self,lane):
        print("正在转变到你想要的分支")
        self._write({"kind":"lane","lane":lane,"leafId":self.lanes[lane],"timestamp":int(time.time()*1000)})
        self.active = lane

    # 遍历路径 将消息从旧到新返回给大模型
    def for_path(self,lane=None):
        print("正在找到所有的历史消息并返回给大模型")
        lane = lane or self.active
        # 存放每一个节点对象
        out = []
        cur = self.lanes.get(lane)
        while cur is not None and self.entries.get(cur) is not None:
            out.append(self.entries[cur])
            # 先通过id找到节点对象，然后通过父节点找到上一条记录
            cur = self.entries[cur]["parentId"]
        out.reverse()
        return [e["message"] for e in out]

    # 加载文件
    def load(self):
        print("加载所有的操作")
        if not os.path.exists(self.path):
            return
        # 一个哨兵，记录最后一行的车道是谁
        last_lane = "main"
        for line in open(self.path,"r",encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m["kind"] == "header":
                self.session_id = m.get("id")
                self.created_at = m.get("createdAt")
            elif m["kind"]=="entry":
                if m["entry"]["parentId"] is not None and m["entry"]["parentId"] not in self.entries:
                    continue
                self.entries[m["entry"]["id"]] = m["entry"]
                self.lanes[m["entry"]["lane"]] = m["entry"]["id"]
                last_lane = m["entry"]["lane"]
            elif m["kind"] == "lane":
                last_lane = m["lane"]
        self.lanes.setdefault("main",None)
        self.active = last_lane
