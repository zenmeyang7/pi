from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

# 构造测试数据
data = {
    "question": [
        "compact_context 函数怎么压缩超限的上下文？"
    ],
    "answer": [
        "GIL是全局解释器锁，限制同一时刻只有一个线程执行Python字节码。"
    ],
    "contexts": [
        ["GIL全局解释器锁，保证同一时刻一个线程执行字节码。"]
    ],
    "ground_truth": [
        "GIL（全局解释器锁），CPython的锁，同一时刻仅有一个线程执行字节码。"
    ],
    "ground_truth_contexts": [
        ["GIL全局解释器锁，保证同一时刻一个线程执行字节码。"]
    ]
}
ds = Dataset.from_dict(data)

llm = LangchainLLMWrapper(ChatOllama(model="qwen2:7b"))

result = evaluate(
    ds,
    metrics=[context_precision, context_recall,faithfulness,answer_relevancy],
    llm=llm
)
