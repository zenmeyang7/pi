"""会话树的持久化：一个 session = 一棵树。

entries 存节点（消息存在 message 列），parent_id 存边；lanes 存每条分支的
叶子光标，所以换个 lane 就是换一条历史路径，回溯靠沿 parent_id 往回走。
"""
import pymysql
from dotenv import load_dotenv
import os
import uuid
import time
import json
load_dotenv()

def get_conn():
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    db = os.getenv("MYSQL_DB")
    conn = pymysql.connect(host=host,user=user,password=password,database=db)
    return conn

def init_schema(conn):
    sql_1 = """
        CREATE TABLE IF NOT EXISTS sessions(
            session_id CHAR(32) PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            created_at BIGINT NOT NULL
        );
    """
    sql_2 = """
        CREATE TABLE IF NOT EXISTS entries(
        id CHAR(32) PRIMARY KEY,
        session_id CHAR(32),
        lane VARCHAR(32) NOT NULL DEFAULT 'main',
        kind VARCHAR(20) NOT NULL DEFAULT 'message',
        parent_id CHAR(32) NULL,
        message JSON NOT NULL,
        ts BIGINT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id));
    """
    sql_3 = """
        CREATE TABLE IF NOT EXISTS lanes ( 
        session_id CHAR(32) NOT NULL,
        lane VARCHAR(32) NOT NULL DEFAULT 'main',
        leaf_id CHAR(32) NULL,
        PRIMARY KEY(session_id, lane)
        )
    """
    with conn.cursor() as cur:
        cur.execute(sql_1)
        cur.execute(sql_2)
        cur.execute(sql_3)
    conn.commit()

class tree():
    def __init__(self, conn, session_id=None):
        self.conn = conn
        self.session_id = session_id
        self.active = "main"
    def create(self,name):
        self.session_id = uuid.uuid4().hex
        created_at = int(time.time()*1000)
        with self.conn.cursor() as cur:
            text1 = """
                INSERT INTO sessions(session_id,name,created_at)
                VALUES(%s,%s,%s)
            """
            text2 = """
                INSERT INTO lanes(session_id,lane,leaf_id)
                VALUES(%s,'main',NULL)
                """
            cur.execute(text1,(self.session_id,name,created_at))
            cur.execute(text2,(self.session_id,))
            self.conn.commit()
        return self.session_id
    def append(self, message, kind="message", lane=None):
        lane = lane or self.active
        entry_id = uuid.uuid4().hex
        ts = int(time.time() * 1000)
        with self.conn.cursor() as cur:
            cur.execute("SELECT leaf_id FROM lanes WHERE session_id=%s AND lane=%s",
                        (self.session_id, lane))
            row = cur.fetchone()
            parent_id = row[0] if row else None
            cur.execute("INSERT INTO entries(id, session_id, lane, kind, parent_id, message, ts) "
                        "VALUES(%s, %s, %s, %s, %s, %s, %s)",
                        (entry_id, self.session_id, lane, kind, parent_id,
                         json.dumps(message, ensure_ascii=False), ts))
            cur.execute("UPDATE lanes SET leaf_id=%s WHERE session_id=%s AND lane=%s",
                        (entry_id, self.session_id, lane))
        self.conn.commit()
        self.active = lane
        return entry_id
    def for_path(self, lane=None):
        lane = lane or self.active
        out = []
        with self.conn.cursor() as cur:
            cur.execute("SELECT id,message FROM entries WHERE session_id=%s AND lane=%s ORDER BY ts, id",
                        (self.session_id, lane))
            for row in cur.fetchall():
                m = json.loads(row[1])
                m["id"] = row[0]
                out.append(m)
        return out

    # 当前叶子：append 时当 parent，back 时往回退
    def current_leaf(self, lane):
        with self.conn.cursor() as cur:
            cur.execute("SELECT leaf_id FROM lanes WHERE session_id=%s AND lane=%s",
                        (self.session_id, lane))
            row = cur.fetchone()
            return row[0] if row else None

    def branch(self, start_id=None):
        leaf = start_id if start_id is not None else self.current_leaf(self.active)
        name = f"branch-{uuid.uuid4().hex[:8]}"
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO lanes(session_id, lane, leaf_id) VALUES(%s, %s, %s)",
                        (self.session_id, name, leaf))
        self.conn.commit()
        self.active = name
        return name

    def switch(self, lane):
        self.active = lane

    def back(self, lane=None):
        lane = lane or self.active
        leaf = self.current_leaf(lane)
        if leaf is None:
            return None
        with self.conn.cursor() as cur:
            cur.execute("SELECT parent_id FROM entries WHERE id=%s", (leaf,))
            row = cur.fetchone()
            parent = row[0] if row else None
        if parent is None:
            return None
        with self.conn.cursor() as cur:
            cur.execute("UPDATE lanes SET leaf_id=%s WHERE session_id=%s AND lane=%s",
                        (parent, self.session_id, lane))
        self.conn.commit()
        self.active = lane
        return parent

    def load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT lane FROM entries WHERE session_id=%s ORDER BY ts DESC, id DESC LIMIT 1",
                        (self.session_id,))
            row = cur.fetchone()
        self.active = row[0] if row else "main"


class SqlRepo:
    def __init__(self, conn):
        self.conn = conn
    def list(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT name, created_at FROM sessions")
            return [{"name": r[0], "createdAt": r[1]} for r in cur.fetchall()]

    def create_tree(self, name):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE name=%s", (name,))
            if cur.fetchone():
                raise FileExistsError(f"会话{name}已经存在")
        t = tree(self.conn)
        t.create(name)
        return t
    
    def open(self, name):
        with self.conn.cursor() as cur:
            cur.execute("SELECT session_id FROM sessions WHERE name=%s", (name,))
            row = cur.fetchone()
        if row is None:
            t = tree(self.conn)
            t.create(name)
        else:
            t = tree(self.conn, session_id=row[0])
            t.load()
        return t

    # fork：复制源会话的当前路径到新会话
    def fork(self, name, source_tree):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE name=%s", (name,))
            if cur.fetchone():
                raise FileExistsError("文件已经存在")
        new_tree = tree(self.conn)
        new_tree.create(name)
        for m in source_tree.for_path():
            new_tree.append(message=m)
        return new_tree