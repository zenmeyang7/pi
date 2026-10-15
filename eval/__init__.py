"""检索评测：度量口径 + 评测集 + 命令行入口。

    metrics.py        recall/mrr/ndcg + 对抗题幻觉率/拒答线标定
    adversarial.py    对抗集（"这个项目里根本没有答案"的 query）
    data/cases.jsonl  可解答集（39 条，五方向各 7-8 条，带 gold 路径与行区间）
    generation.py     生成评测（Faithfulness / Hallucination，调真 LLM）
    run.py            命令行入口，出三维度对比表
    result.md         最近一次真机评测的结果表
    cache/            生成缓存（gen_cache.jsonl，磁盘缓存省重复调 LLM）

⚠️ adversarial.py 和 result.md 被 rag/config.py 按文件名排除出索引：
   对抗题的主题词一旦进语料，检索就能直接命中它，幻觉率会假性变差。
"""
