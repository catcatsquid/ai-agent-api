"""验证 seed 向量库：命中率、相似度、域外判定"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_tool import KnowledgeBase, RELEVANCE_THRESHOLD

STORE = Path(__file__).resolve().parent / "seed" / "vector_store"
kb = KnowledgeBase(persist_dir=str(STORE))

print("分块数:", kb.count())
print("阈值  :", RELEVANCE_THRESHOLD)

CASES = [
    ("死锁的四个必要条件是什么", True),
    ("死锁和饥饿有什么区别", True),
    ("银行家算法属于死锁避免还是检测", True),
    ("SJF 调度为什么会导致饥饿", True),
    ("信号量 P 操作 S 为负数代表什么", True),
    ("TCP 为什么需要三次握手", True),
    ("Transformer 的注意力机制为什么比 RNN 快", False),
    ("大模型的 temperature 和 top_p 有什么区别", False),
]

print("\n%-42s %-8s %-8s %s" % ("query", "top1_sim", "top1_src", "判定"))
for q, should_hit in CASES:
    hits = kb.search(q, top_k=3)
    top = hits[0] if hits else {}
    # 单查询路径下 score 就是余弦相似度；多查询/融合路径才有独立 sim 字段
    sim = top.get("sim") if top.get("sim") is not None else top.get("score")
    src = (top.get("source") or "-")[:16]
    inside = (sim is not None and sim >= RELEVANCE_THRESHOLD)
    verdict = "域内" if inside else "域外"
    flag = "" if inside == should_hit else "  ← 与预期不符"
    print("%-42s %-8s %-8s %s%s" % (q[:40], f"{sim:.3f}" if sim is not None else "-", src, verdict, flag))
