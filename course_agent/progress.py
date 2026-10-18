"""全局学习进度（**跨会话**）—— 学到哪、刷到哪、哪套卷子刷到一半

## 为什么需要它

书签（学到哪）和刷题游标（出到第几组）原来只存在**单条会话**的 state 里（checkpoint），
于是新开一个窗口就"忘了"，Agent 每次都得重新问"你之前学到哪了"。
把这个账本提成全局的，新窗口开场就能接着上次给建议。

## 为什么不放 data/

`data/` 是课程资料目录：会被 `/kb` 扫到、被"测试不许污染 data/"的守卫盯着。
所以落 `runtime/progress.json`（与 checkpoints 同级，已在 .gitignore）。

## 写入必须是**确定性**的

只从工具返回里推导（`record_study_position` / `get_paper_questions` /
`grade_answers`），**不要求模型多做任何动作** —— 与 `nodes._extract_*_session`
同一条纪律：模型记不住的事交给代码。

## 评测不许污染它

评测打的是真实 HTTP，出题/批改会顺手写进度 —— 于是你会看到"今天刷了 30 套卷子"。
所以 `eval-*` / `probe-*` 前缀的 thread 一律**只读不写**（见 `is_ephemeral`）。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("agent")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 测试用环境变量指到临时文件（绝不碰真实进度）
PROGRESS_PATH = Path(
    __import__("os").environ.get("PROGRESS_FILE")
    or (PROJECT_ROOT / "runtime" / "progress.json"))

_LOCK = threading.Lock()
_VERSION = 1
# 评测/探针造的会话不许写进度（否则用户的"学习进度"被评测刷满）
_EPHEMERAL_PREFIXES = ("eval-", "probe-")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_ephemeral(thread_id) -> bool:
    return bool(thread_id) and str(thread_id).startswith(_EPHEMERAL_PREFIXES)


# ============================================================
# 读写
# ============================================================

def _migrate(data: dict) -> dict:
    """把账本升级到当前结构（只升不降）

    · 旧文件缺字段 → 补齐（version 从 0 升到 1）
    · 文件版本**比程序新** → 只警告，原样返回（强行改会被下一次保存抹掉新字段）
    """
    try:
        version = int(data.get("version") or 0)
    except (TypeError, ValueError):
        version = 0

    if version > _VERSION:
        logger.warning("进度文件版本 %s 高于本程序支持的 %s —— 按旧逻辑读，"
                       "新字段会被忽略（不会丢，但看不见）", version, _VERSION)
        return data

    if version < 1:
        data.setdefault("user", "local")
        data.setdefault("courses", {})
        data.setdefault("papers", {})
        # 逐课程/逐卷补齐子结构（旧文件可能只有顶层键）
        for bucket in (data.get("courses") or {}).values():
            if isinstance(bucket, dict):
                bucket.setdefault("sections", {})
                bucket.setdefault("last", "")
        for rec in (data.get("papers") or {}).values():
            if isinstance(rec, dict):
                rec.setdefault("groups", {})
        logger.info("进度账本从版本 %s 升级到 %s", version, _VERSION)

    data["version"] = _VERSION
    return data


def load() -> dict:
    """读进度（文件不存在/损坏都返回空账本，**绝不抛异常**）"""
    if not PROGRESS_PATH.exists():
        return {"version": _VERSION, "user": "local", "courses": {}, "papers": {}}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        logger.warning("进度文件读不出来，当作空账本（原文件保留）: %s", e)
        return {"version": _VERSION, "user": "local", "courses": {}, "papers": {}}
    if not isinstance(data, dict):
        return {"version": _VERSION, "user": "local", "courses": {}, "papers": {}}
    data = _migrate(data)
    data.setdefault("version", _VERSION)
    data.setdefault("user", "local")
    data.setdefault("courses", {})
    data.setdefault("papers", {})
    return data


def _save(data: dict) -> None:
    data["updated_at"] = _now()
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    # 先写临时文件再替换：中途崩不会留下半个 JSON（读-改-写被并发打断也会损坏）
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_PATH)


def _mutate(fn, thread_id=None):
    """持锁读-改-写。ephemeral 线程直接返回 None（不写、不报错）。"""
    if is_ephemeral(thread_id):
        return None
    with _LOCK:
        data = load()
        changed = fn(data)
        if changed:
            _save(data)
        return data


def reset() -> None:
    """清空进度（用户手动重置；测试也用它隔离）"""
    with _LOCK:
        if PROGRESS_PATH.exists():
            PROGRESS_PATH.unlink()


# ============================================================
# 写入：两件事 —— 学到哪、刷到哪
# ============================================================

def record_study(pos: dict, thread_id=None) -> None:
    """记一条学习位置（来自 `record_study_position` 的 recorded 字段）

    pos 形如 {"mode","course","chapter","section","source","status"}；
    section 是模型自己起的节标题（刻意不做成状态机）。

    两条容错（都是实测会发生的）：
      · 模型会传 `chapter=""` 把上一轮记住的章抹掉 → 空值**不覆盖**已有值
      · 章名空但课件名已知时，挂到**同一份课件已有的那条**记录上，不许新开一条
    """
    if not isinstance(pos, dict) or not str(pos.get("course") or "").strip():
        return
    course = str(pos["course"]).strip()
    source = str(pos.get("source") or "").strip()
    chapter = str(pos.get("chapter") or "").strip()

    def _apply(data):
        bucket = data["courses"].setdefault(course, {"sections": {}, "last": ""})
        sections = bucket["sections"]
        # 决定挂到哪条记录上
        if chapter:
            key = "|".join([source or "-", chapter])
        else:
            key = next((k for k, v in sections.items()
                        if (v.get("source") or "") == source), None)
            if key is None:
                key = "|".join([source or "-", "-"])
        old = sections.get(key) or {}
        # 空值不覆盖已有值（模型会把上一轮的章/标题写成空串）
        sections[key] = {
            "source": source or old.get("source", ""),
            "chapter": chapter or old.get("chapter", ""),
            "title": str(pos.get("section") or "").strip() or old.get("title", ""),
            "status": str(pos.get("status") or "").strip() or old.get("status", "讲解中"),
            "mode": str(pos.get("mode") or "").strip() or old.get("mode", ""),
            "first_at": old.get("first_at") or _now(),
            "last_at": _now(),
            "visits": int(old.get("visits") or 0) + 1,
        }
        if isinstance(old.get("last_offset"), int):
            sections[key]["last_offset"] = old["last_offset"]
        bucket["last"] = key
        bucket["last_at"] = _now()
        return True

    _mutate(_apply, thread_id)


def record_read(course: str, source: str, offset, thread_id=None) -> None:
    """记"读到哪一块"（来自 `read_section` 的返回）—— 供下一轮续读

    实测模型记不住返回值里那句"下次用 offset=15 继续"，所以交给状态。
    """
    course = str(course or "").strip()
    if not course:
        return
    try:
        off = int(offset)
    except (TypeError, ValueError):
        return

    def _apply(data):
        bucket = data["courses"].setdefault(course, {"sections": {}, "last": ""})
        sections = bucket["sections"]
        src = str(source or "").strip()
        # 同一个文件已经有一条带章名的记录就挂到它上面（否则会多出一条重复的节）
        key = next((k for k, v in sections.items() if v.get("source") == src), None)
        if key is None:
            key = "|".join([src or "-", "-"])
        sec = sections.setdefault(key, {
            "source": src, "chapter": "", "title": "",
            "status": "讲解中", "mode": "", "first_at": _now(), "visits": 0})
        if int(sec.get("last_offset") or -1) == off:
            return False                      # 没变就不写文件
        sec["last_offset"] = off
        sec["last_at"] = _now()
        sec.setdefault("visits", 0)
        bucket["last"] = key
        bucket["last_at"] = _now()
        return True

    _mutate(_apply, thread_id)


def record_quiz(payload: dict, thread_id=None) -> None:
    """记刷题进度（来自 `get_paper_questions` 的返回）

    **"刷到一半"的定义（用户拍板）**：记录写到第几组了 ——
    `next_group_index is not None` 就是没刷完，下一组是它；`None` 表示这份卷子刷完了。
    """
    if not isinstance(payload, dict) or not payload.get("paper"):
        return
    paper = str(payload["paper"]).strip()
    gi = payload.get("group_index")
    nxt = payload.get("next_group_index")

    def _apply(data):
        rec = data["papers"].setdefault(paper, {"groups": {}})
        rec["type"] = payload.get("type") or rec.get("type", "")
        rec["cursor"] = {
            "group_index": gi,
            "next_group_index": nxt,
            "qids": list(payload.get("qids") or []),
            "at": _now(),
        }
        rec["finished"] = nxt is None
        groups = rec.setdefault("groups", {})
        if gi is not None:
            g = groups.setdefault(str(gi), {})
            g.setdefault("status", "已出题")
            g["total"] = len(payload.get("qids") or []) or g.get("total")
            g["at"] = _now()
        return True

    _mutate(_apply, thread_id)


def record_grade(paper: str, payload: dict, thread_id=None) -> None:
    """记批改结果（来自 `grade_answers` 的返回）

    两件事：
      · 把当前那一组标成"已批改"并记分数
      · **维护错题集合**（A5）：这批做错的 qid 加进来，这批做对的从里面移出
        —— 决定 D4：「做对就移出、做错保留」。"重做错题"直接读 `wrong_qids`。

    `grade_answers` 的 `results[]` 每条都带 qid / correct（我查过
    `qbank.grade_answer` 的实现），所以**不需要改任何工具**。
    `correct is None`（需人工/模型评分但没判出来）**不算错题** —— 不能凭猜。
    """
    if not paper or not isinstance(payload, dict) or "total" not in payload:
        return

    results = payload.get("results") or []
    wrong_now = {str(r.get("qid")) for r in results
                 if r.get("qid") and r.get("correct") is False}
    right_now = {str(r.get("qid")) for r in results
                 if r.get("qid") and r.get("correct") is True}

    def _apply(data):
        rec = data["papers"].setdefault(str(paper).strip(), {"groups": {}})
        cursor = rec.get("cursor") or {}
        gi = cursor.get("group_index")
        score = f"{payload.get('correct', 0)}/{payload.get('total', 0)}"
        rec["last_grade"] = {
            "correct": payload.get("correct"),
            "total": payload.get("total"),
            "wrong": payload.get("wrong"),
            "accuracy": payload.get("accuracy"),
            "at": _now(),
        }
        # 错题集合：做对的移出、做错的加进来（见 D4）
        old_wrong = {str(q) for q in (rec.get("wrong_qids") or [])}
        merged = (old_wrong - right_now) | wrong_now
        if merged:
            rec["wrong_qids"] = sorted(merged)
        else:
            rec.pop("wrong_qids", None)
        if gi is not None:
            g = rec.setdefault("groups", {}).setdefault(str(gi), {})
            g["status"] = "已批改"
            g["score"] = score
            g["wrong"] = len(wrong_now)
            g["at"] = _now()
        return True

    _mutate(_apply, thread_id)


# ============================================================
# 读：给提示词用的紧凑文本 + 给学生的建议
# ============================================================

def _section_line(sec: dict) -> str:
    name = sec.get("title") or sec.get("chapter") or sec.get("source") or "（未命名）"
    src = sec.get("source") or ""
    tail = f" · {src}" if src and src not in name else ""
    off = sec.get("last_offset")
    off_txt = f"，读到第 {off} 块之后" if isinstance(off, int) else ""
    return f"{name}{tail}（{sec.get('status') or '讲解中'}{off_txt}）"


def _paper_line(rec: dict) -> str:
    cur = rec.get("cursor") or {}
    gi, nxt = cur.get("group_index"), cur.get("next_group_index")
    last = rec.get("last_grade") or {}
    score = (f"，上次 {last.get('correct')}/{last.get('total')}"
             if last.get("total") else "")
    # 错题（A5/D4）：做对就移出，所以这里的数量就是"还没弄懂的"
    wrong = len(rec.get("wrong_qids") or [])
    wrong_txt = f"，错题 {wrong} 道待重做" if wrong else ""
    if nxt is not None:
        return f"做到第 {gi} 组，下一组是第 {nxt} 组{score}{wrong_txt}"
    if cur:
        return f"已刷完{score}{wrong_txt}"
    return f"还没开始{score}{wrong_txt}"


def summary(max_lines: int = 8) -> str:
    """一段紧凑的进度文本（注入提示词用；空账本返回 ""）

    刻意限行数：每次请求都注入，不能无节制地长。
    """
    data = load()
    lines = []
    for course, bucket in (data.get("courses") or {}).items():
        last_key = bucket.get("last")
        sec = (bucket.get("sections") or {}).get(last_key) if last_key else None
        if not sec:
            continue
        done = sum(1 for s in (bucket.get("sections") or {}).values()
                   if (s.get("status") or "") == "已通过")
        lines.append(f"- {course}：上次在 {_section_line(sec)}，已通过 {done} 节")
    for paper, rec in (data.get("papers") or {}).items():
        lines.append(f"- 卷子《{paper}》：{_paper_line(rec)}")
    if not lines:
        return ""
    return "\n".join(lines[:max_lines])


def has_any() -> bool:
    """有没有任何进度记录（决定开场要不要走"展示进度 + 给建议"那条路）"""
    try:
        data = load()
        return bool(data.get("courses") or data.get("papers"))
    except Exception:                                   # noqa: BLE001
        return False


def brief() -> str:
    """给学生的「进度 + 四种模式建议」——**代码生成、可照抄**

    为什么要代码生成：这句话每次开场都要说，而"措辞 + 别漏掉某个建议"交给模型
    会时好时坏（这个项目在 `group_payload` / `closing_block` 上已经这么做过）。
    模型只需要按进度挑一条、或原样转述。
    """
    data = load()
    courses = data.get("courses") or {}
    papers = data.get("papers") or {}
    if not courses and not papers:
        return ""

    lines = ["【已知进度】（直接用，**不要再问「你之前学到哪了」**）"]
    lines.append(summary() or "- （暂无）")

    opts = []
    # ① 继续学：挑最近活跃的那门课
    active = [(c.get("last_at") or "", c) for c in courses.values() if c.get("last")]
    last_sec, last_course = None, ""
    if active:
        active.sort(reverse=True)
        bucket = active[0][1]
        last_course = next((c for c, b in courses.items() if b is bucket), "")
        sec = (bucket.get("sections") or {}).get(bucket.get("last"))
        if sec:
            last_sec = sec
            opts.append(f"接着学：从 {_section_line(sec)} 继续（学习模式）")
    # ② 自拟练习（**始终给**：用户明确要的四选一里必须有它）
    # 带上**当前章节**当 topic —— 否则"出几道题练手"要靠模型自己从进度里猜
    topic = ""
    if last_sec:
        topic = last_sec.get("chapter") or last_sec.get("title") or ""
    if topic and last_course:
        opts.append(f"自拟练习：围绕「{last_course} · {topic}」出 3~5 道题（AI 出题）")
    else:
        opts.append("自拟练习：围绕最近学的知识点出 3~5 道题（AI 出题）")
    # ③ 整套卷子：优先接上"刷到一半"的那份
    half = [(p, r) for p, r in papers.items() if not r.get("finished")]
    if half:
        p, r = half[0]
        nxt = (r.get("cursor") or {}).get("next_group_index")
        where = f"第 {nxt} 组" if nxt is not None else "新的组"
        opts.append(f"刷题：把《{p}》的{where}做完（刷题模式）")
    else:
        opts.append("刷题：从题库里挑一套卷子做（刷题模式）")
    # ④ 复习
    if courses:
        total = sum(len(c.get("sections") or {}) for c in courses.values())
        opts.append(f"复习：把已学过的 {total} 节快速过一遍（复习模式）")

    lines.append("")
    lines.append("【这一轮只能做一件事：把上面的进度说给学生，并让他在下面选一条】")
    lines.append("⚠️ 学生还没选之前，**不要开始讲内容**（不要说\"我们来看这一节\"）、"
                 "不要调 read_section / list_course_sections。")
    for i, o in enumerate(opts, 1):
        lines.append(f"{i}. {o}")
    return "\n".join(lines)


def snapshot() -> dict:
    """GET /progress 的返回（原样给前端/调试用）"""
    return load()


# ============================================================
# 四选一的兜底：模型爱漏掉其中一条
# ============================================================

_MODE_KEYS = {
    "接着学": ("接着学", "继续学"),
    "自拟练习": ("自拟练习", "自拟", "AI 出题", "出几道题", "出 3"),
    "刷题": ("刷题", "做题", "卷子"),
    "复习": ("复习",),
}


def missing_modes(answer: str) -> list[str]:
    """回答里漏掉哪几种模式（用于兜底补全）"""
    text = str(answer or "")
    return [m for m, keys in _MODE_KEYS.items() if not any(k in text for k in keys)]


def ensure_modes(answer: str) -> tuple[str, list[str]]:
    """保证开场回答里**四种模式都在**，漏的用代码补在后面

    为什么要兜底（实测）：提示里写清"四条全部列出"，模型仍然只说了三条
    （把「自拟练习」漏了）。这与 `ensure_closing` 是同一类问题：措辞模型化、
    但"不能漏项"这种可靠动作交给代码。
    """
    missing = missing_modes(answer)
    if not missing:
        return answer, []
    text = brief()
    lines = [ln for ln in text.splitlines() if ln[:2].strip().rstrip(".").isdigit()]
    add = []
    for name in missing:
        hit = next((ln for ln in lines if name in ln or (name == "刷题" and "刷题" in ln)), None)
        if hit:
            # 去掉原编号，免得后面出现"2. ... 1. ..."这种怪编号
            add.append(re.sub(r"^\d+\.\s*", "", hit.strip()))
    if not add:
        return answer, []
    out = str(answer or "").rstrip() + "\n\n另外还可以：\n" + "\n".join(add)
    return out, add
