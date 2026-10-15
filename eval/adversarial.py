"""对抗集：语料里根本没有答案的问题。

recall 只回答"该找的找没找到"，回答不了另一半——**不该找的时候有没有瞎找**。
分两类，难度差很多，分开报才看得出问题在哪：

    out_of_scope  问这个项目压根没碰过的东西。词法路零命中，向量路也只能捞到
                  远房亲戚 —— 这类**太好拒绝**，全对说明不了什么；错了则说明
                  拒答线根本没设。
    near_miss     语料里**有**很像的素材，答案却依然不存在：项目没有向量库，
                  但 rag/store.py 里实实在在写着向量检索。这才是幻觉的主产区
                  ——检索确实捞到了"向量检索"的代码，模型顺着就能编出个名字。

must_not 只给 out_of_scope：这些词在语料里必须 0 命中，否则这条题就是出错了
的题（语料里真有 Redis，那就不是幻觉，是漏检）。由
test_adv_set_is_really_unanswerable 守着——它不需要 Ollama，纯文本扫。

⚠️ 本文件被 rag/config.py 的 INDEX_IGNORE_FILES 排除出索引：
   对抗题的主题词一旦进语料，检索就能直接命中它，幻觉率会假性变差。
   同理，别在 README / 源码注释里写下这些主题词。
"""

ADVERSARIAL_CASES = [
    # ---- out_of_scope：语料里一个字都没有 ----
    # 注意 must_not 的选词：得像"redis"这样在语料里一个字都没有。
    # "缓存" 就不行——rag/build_index.py 的注释里真有这个词。
    {"query": "项目里 Redis 缓存的淘汰策略是怎么配的", "kind": "out_of_scope",
     "must_not": ["redis", "淘汰"]},
    {"query": "K8s 的 Deployment 副本数和滚动更新在哪配", "kind": "out_of_scope",
     "must_not": ["k8s", "kubernetes", "副本", "滚动更新"]},
    {"query": "gRPC 的双向流是在哪个模块实现的", "kind": "out_of_scope",
     "must_not": ["grpc", "双向流"]},
    {"query": "Kafka 消费者组的位移是怎么提交的", "kind": "out_of_scope",
     "must_not": ["kafka", "消费者", "位移"]},
    {"query": "JWT 鉴权中间件怎么校验 token 过期", "kind": "out_of_scope",
     "must_not": ["jwt", "鉴权", "中间件"]},
    {"query": "Prometheus 的监控指标在哪些地方打点", "kind": "out_of_scope",
     "must_not": ["prometheus", "打点", "监控"]},
    {"query": "前端 React 的虚拟列表是怎么实现的", "kind": "out_of_scope",
     "must_not": ["react", "虚拟列表"]},
    {"query": "Celery 的定时异步任务在哪配置", "kind": "out_of_scope",
     "must_not": ["celery", "定时", "异步"]},
    {"query": "分布式锁 Redlock 怎么保证互斥", "kind": "out_of_scope",
     "must_not": ["redlock", "分布式锁", "互斥"]},
    # 同理不能用"注册"（agent/loop.py 里真的有工具注册），只锁"密码/加密/登录"
    {"query": "用户注册登录的密码是用什么算法加密的", "kind": "out_of_scope",
     "must_not": ["密码", "加密", "登录"]},
    # ---- near_miss：素材很像，答案没有 ----
    {"query": "向量检索用的是哪个向量数据库", "kind": "near_miss"},
    {"query": "MySQL 分库分表的路由规则是怎么写的", "kind": "near_miss"},
    {"query": "工具调用失败之后的重试退避策略是怎么做的", "kind": "near_miss"},
    {"query": "上下文压缩用的摘要模型是哪一个", "kind": "near_miss"},
    {"query": "多租户之间的数据是怎么隔离的", "kind": "near_miss"},
    {"query": "接口的限流和熔断降级是在哪里做的", "kind": "near_miss"},
    {"query": "会话历史是怎么增量同步到云端的", "kind": "near_miss"},
    {"query": "Elasticsearch 的倒排索引是怎么建的", "kind": "near_miss"},
]
