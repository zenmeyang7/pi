import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import health_check          # noqa: E402
from rag.store import IndexStore               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="要索引的工作区根目录")
    ap.add_argument("--out", default="rag/index", help="索引输出目录")
    ap.add_argument("--force", action="store_true", help="忽略 mtime 缓存，全量重建")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)

    ok, msg = health_check()
    print(("[rag] OK  " if ok else "[rag] FAIL ") + msg)
    if not ok:
        return 1

    store = IndexStore(out)
    store.build(root, force=args.force)

    # 简单自检：索引完立刻抽查几个 chunk，确认向量和文本没串位
    for c in store.chunks[:2]:
        print(f"  {c['id']}  dim={len(c['vec'])}  {c['text'].splitlines()[0][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
