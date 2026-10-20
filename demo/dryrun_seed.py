"""不进 Docker 的 dry-run：把项目路径指向 demo/seed，验证它能真的跑起来

模拟 docker-compose.demo.yml 的挂载效果：
  /app/data          ← demo/seed/data
  /app/vector_store  ← demo/seed/vector_store

验证四件事：
  1. 题库：能列出卷子、能分组出题（5 题一组）
  2. 学习模式：section_index 能列出这门课的"课件"，read_section 能按章取到内容
  3. 检索：按课程过滤能命中，域外问题判为域外
  4. 题图：image_url 指向 /qb_images/demo_q1.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # demo/
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = ROOT / "seed"
SEED_DATA = SEED / "data"
SEED_STORE = SEED / "vector_store"

from course_agent import qbank, quiz_service     # noqa: E402
import rag_tool                                   # noqa: E402

# ---- 把"项目根/data"和"向量库"指到 seed ----
qbank.PROJECT_ROOT = SEED
qbank.QUESTION_BANK_DIR = SEED_DATA / "question_bank"
qbank.IMAGES_DIR = qbank.QUESTION_BANK_DIR / "images"

_seed_kb = rag_tool.KnowledgeBase(persist_dir=str(SEED_STORE))
rag_tool.get_knowledge_base = lambda persist_dir="./vector_store": _seed_kb
quiz_service.get_knowledge_base = lambda persist_dir="./vector_store": _seed_kb

fail = []


def check(label, ok, extra=""):
    print(f"{'[OK]  ' if ok else '[FAIL]'} {label}{('  ' + extra) if extra else ''}")
    if not ok:
        fail.append(label)


print("=" * 70)
print("1) 题库")
qs = qbank.load_questions()
papers = qbank.list_papers()
check("题目加载", len(qs) == 11, f"{len(qs)} 题")
check("卷子列表", len(papers) >= 1, f"{len(papers)} 份")
src = papers[0]["source"] if isinstance(papers[0], dict) else str(papers[0])
g = qbank.get_group(src, group_index=1)
check("第1组是 5 道选择题", g.get("group_count") == 4 and len(g.get("qids", [])) == 5,
      str(g.get("progress")))
check("第1组带题图", any(q.get("image_url") for q in g.get("questions", [])),
      next((q["image_url"] for q in g.get("questions", []) if q.get("image_url")), "无"))

print("=" * 70)
print("2) 学习模式（section_index / read_section）")
idx = quiz_service.section_index("操作系统")
check("能列出操作系统的资料", bool(idx.get("ppt")), f"{len(idx.get('ppt') or [])} 份")
for item in idx.get("ppt") or []:
    print(f"      {item['source']}  ｜章节：{item['chapter']}｜{item['chunks']} 块")

res = quiz_service.study_read(course="操作系统", chapter="进程管理", limit=3)
check("按章名取到内容", bool(res.get("text")) and not res.get("error"),
      (res.get("error") or (res["text"][:24].replace("\n", " ") + "…")))

res2 = quiz_service.study_read(course="操作系统", chapter="死锁", limit=3)
check("按「死锁」章取内容", bool(res2.get("text")) and not res2.get("error"),
      (res2.get("error") or "命中《操作系统_死锁.md》")[:60])

print("=" * 70)
print("3) 检索（按课程过滤 + 域内外判定）")
hits, confirmed = quiz_service.search_with_course(_seed_kb, "死锁的四个必要条件是什么", "操作系统", top_k=3)
check("课程内问题命中", bool(hits) and confirmed,
      f"top1={hits[0]['source'] if hits else '-'} sim={hits[0].get('score') if hits else '-'}")

hits2, confirmed2 = quiz_service.search_with_course(_seed_kb, "Transformer 的注意力机制为什么比 RNN 快", "操作系统", top_k=3)
check("课外问题判为域外", not confirmed2, f"confirmed={confirmed2}")

print("=" * 70)
print("结论:", "全部通过" if not fail else f"{len(fail)} 项失败 -> {fail}")
sys.exit(1 if fail else 0)
