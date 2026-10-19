import ast
import os

from . import config


def iter_files(root):
    """遍历工作区，产出待索引的文件绝对路径。

    三道过滤：目录黑名单 + 文件名黑名单 + 文件大小。大文件那道是**兜底**——
    模型权重、分词器 json、数据集这类东西一旦被当成源码索引，
    一个文件就能切出上万个 chunk（索引 162 → 19403，真踩过）。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.INDEX_IGNORE_DIRS]
        for name in filenames:
            if name in config.INDEX_IGNORE_FILES:
                continue
            if os.path.splitext(name)[1].lower() not in config.INDEX_EXTENSIONS:
                continue
            p = os.path.join(dirpath, name)
            try:
                if os.path.getsize(p) > config.INDEX_MAX_FILE_BYTES:
                    continue
            except OSError:           
                continue
            yield p


def rel_path(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def make_chunk(rel, text, **meta):
    """id 由 path + 行号派生：重建索引幂等，评估时也能直接当 gold 引用。"""
    meta = {"path": rel, **meta}
    chunk_id = f"{rel}:{meta.get('start_line', 0)}-{meta.get('end_line', 0)}"
    return {"id": chunk_id, "text": text, "meta": meta}


def _slice_python(src, rel):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []          # 语法错的文件退回行窗口，不让单个坏文件中断整个索引
    lines = src.splitlines()
    if not lines:
        return []
    out = []

    # ① 文件头（module docstring + import 区）单独成块。
    body = tree.body
    i, header_end = 0, 0
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        header_end = body[0].end_lineno or 0
        i = 1
    while i < len(body) and isinstance(body[i], (ast.Import, ast.ImportFrom)):
        header_end = body[i].end_lineno or header_end
        i += 1
    if header_end:
        out.append(make_chunk(
            rel, f"{rel}::<module>\n" + "\n".join(lines[:header_end]),
            kind="header", symbol="<module>",
            start_line=1, end_line=min(header_end, len(lines)),
        ))

    # ② 顶层函数 / 类
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.extend(_slice_node(lines, node, rel, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            out.extend(_slice_class(lines, node, rel))
    return out


def _decorator_start(lines, node):
    """把紧贴在上面的装饰器 / 注释一起带上，它们常含关键信息。"""
    start = max(0, (node.lineno or 1) - 1)
    while start > 0 and lines[start - 1].lstrip().startswith(("@", "#")):
        start -= 1
    return start


def _slice_node(lines, node, rel, kind, symbol):
    start = _decorator_start(lines, node)
    end = node.end_lineno or node.lineno
    span = end - start

    if span <= config.CHUNK_MAX_LINES:
        text = f"{rel}::{symbol}\n" + "\n".join(lines[start:end])
        return [make_chunk(rel, text, kind=kind, symbol=symbol,
                           start_line=start + 1, end_line=end)]

    # 超长节点：按行窗口切，带 overlap，避免把一段逻辑拦腰截断
    out, step, i, part = [], config.CHUNK_MAX_LINES - config.CHUNK_OVERLAP_LINES, start, 0
    while i < end:
        j = min(i + config.CHUNK_MAX_LINES, end)
        text = f"{rel}::{symbol}（第 {part + 1} 段）\n" + "\n".join(lines[i:j])
        out.append(make_chunk(rel, text, kind=kind, symbol=symbol,
                              start_line=i + 1, end_line=j, part=part))
        part += 1
        i += step
    return out


def _slice_class(lines, node, rel):
    out = []
    # 类头：class 行 + docstring + 类属性 = "这个类是什么"的概览
    head_end = node.lineno
    if node.body:
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            head_end = first.end_lineno or head_end
    if head_end > node.lineno:
        text = f"{rel}::class {node.name}\n" + "\n".join(lines[node.lineno - 1:head_end])
        out.append(make_chunk(rel, text, kind="class", symbol=node.name,
                              start_line=node.lineno, end_line=head_end))

    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.extend(_slice_node(lines, sub, rel, "method",
                                   f"{node.name}.{sub.name}"))
    return out



def _slice_lines(src, rel):
    lines = src.splitlines()
    if not lines:
        return []
    out, step, i, part = [], config.CHUNK_MAX_LINES - config.CHUNK_OVERLAP_LINES, 0, 0
    while i < len(lines):
        j = min(i + config.CHUNK_MAX_LINES, len(lines))
        text = f"{rel}（第 {part + 1} 段）\n" + "\n".join(lines[i:j])
        out.append(make_chunk(rel, text, kind="text", symbol=rel,
                              start_line=i + 1, end_line=j, part=part))
        part += 1
        if j >= len(lines):
            break
        i += step
    return out



def chunk_file(path, root):
    """把一个文件切成一串 chunk；读不了就返回空列表，不抛异常。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    rel = rel_path(path, root)
    if path.lower().endswith(".py"):
        chunks = _slice_python(src, rel)
        if chunks:
            return chunks
    return _slice_lines(src, rel)
