"""题库访问层 —— 全项目唯一的题库读写入口

设计目标（解决原 tutor_agent.py 里内联实现的几个致命问题）:
1. 合并加载：question_bank.json 并不是全部题库（有两份试卷只存在于
   questions_*.json 里，从未被合并进去）。这里统一扫描所有题库文件并按
   (source, qid) 去重，保证新增试卷自动生效，不再依赖某个生成脚本。
2. 题号唯一正确：原实现用 `re.search(r"(\\d+)", qid)` 取题号，遇到
   "2026OS_Q1" / "2019网模拟_F1" 这类 qid 会取到年份（2026/2019），
   导致按题号批改永远匹配不上。这里改为一律使用结构化字段 q_num / display_num。
3. 分组由代码保证：选择题/填空题/判断题 5 题一组，综合题 1 题一组。
   分组不再靠 prompt 里写死规则去"请求"模型遵守。
4. 不泄题：出题接口只返回题号/题型/知识点/图片，绝不返回 text 与 answer。
5. 批改依据分级：选择题看 answer，判断题做归一化，填空题/综合题结合
   answer + reference_points + sub_questions，并在依据不可靠时明确标注，
   而不是拿一个错位的 'C' 去和学生答案比对。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 项目根目录（不受启动时 CWD 影响 —— 原实现用相对路径，
# 从别的目录启动 uvicorn 就会静默读不到题库）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTION_BANK_DIR = PROJECT_ROOT / "data" / "question_bank"
IMAGES_DIR = QUESTION_BANK_DIR / "images"

# 题型顺序（按考试卷面习惯）
TYPE_ORDER = ["选择题", "填空题", "判断题", "综合题", "整页综合"]

# 每组的题量：这几类 5 题一组，其余（综合题/整页综合）1 题一组
GROUP_OF_FIVE = ("选择题", "填空题", "判断题")


def group_size(qtype: str) -> int:
    """该题型每组题量"""
    return 5 if qtype in GROUP_OF_FIVE else 1


def _type_rank(qtype: str) -> int:
    try:
        return TYPE_ORDER.index(qtype)
    except ValueError:
        return len(TYPE_ORDER)


# ============================================================
# 加载
# ============================================================

_CACHE: dict = {"key": None, "questions": []}


def _bank_files() -> list[Path]:
    """所有题库文件：单卷文件 + 合并文件"""
    if not QUESTION_BANK_DIR.exists():
        return []
    files = sorted(QUESTION_BANK_DIR.glob("questions_*.json"))
    combined = QUESTION_BANK_DIR / "question_bank.json"
    if combined.exists():
        files.append(combined)
    return files


def _cache_key(files: list[Path]):
    return tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)


def load_questions(force: bool = False) -> list[dict]:
    """加载全部题目（带 mtime 缓存，同一进程内不重复解析）"""
    files = _bank_files()
    key = _cache_key(files)
    if not force and _CACHE["key"] == key and _CACHE["questions"]:
        return _CACHE["questions"]

    merged: dict[tuple, dict] = {}
    for f in files:
        try:
            with open(f, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
        except Exception:
            continue  # 单个文件损坏不影响其余题库
        if not isinstance(data, list):
            continue
        for q in data:
            if not isinstance(q, dict) or not q.get("qid"):
                continue
            # 用 (source, qid) 去重：qid 在不同试卷间会重复（如 'S1'）
            merged.setdefault((q.get("source", ""), q["qid"]), q)

    questions = list(merged.values())
    _CACHE["key"] = key
    _CACHE["questions"] = questions
    return questions


def _qno(q: dict) -> int:
    """题号：优先结构化字段，绝不从 qid 里猜年份

    注意源数据里 q_num 可能是浮点（真实存在 q_num=4.1 这种脏数据），
    所以不能只用 isinstance(v, int)。
    """
    for field in ("q_num", "display_num"):
        v = q.get(field)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
    return 999


def _display_no(q: dict) -> int:
    v = q.get("display_num")
    if isinstance(v, bool):
        v = None
    if isinstance(v, (int, float)):
        return int(v)
    return _qno(q)


def section_numbers(questions: list[dict]) -> dict[str, int]:
    """给一组题编号：按题型分段，各段内从 1 开始

    为什么不直接用题库的 display_num / q_num：
      真实数据不可靠 —— 同一份卷子里 display_num=1 会同时出现在选择题、
      填空题、判断题、综合题上（4 道不同的题共用一个号），
      2019 年那份期末卷甚至出现 q_num=4.1、选择题 display_num 从 13 开始。
      用"题型内第几题"编号，既符合卷面直觉，也永远唯一。
    """
    numbers: dict[str, int] = {}
    counters: dict[str, int] = {}
    for t in sorted({q.get("type", "") for q in questions}, key=_type_rank):
        n = 0
        for q in questions:
            if q.get("type") != t:
                continue
            n += 1
            numbers[q["qid"]] = n
    return numbers


def _sort_key(q: dict):
    return (_type_rank(q.get("type", "")), _qno(q))


# ============================================================
# 试卷
# ============================================================

_YEAR = re.compile(r"(19|20)\d{2}")


def _years(text: str) -> set[str]:
    """提取文本里的 4 位年份（用于判断是不是同一场考试）"""
    return {m.group(0) for m in _YEAR.finditer(str(text or ""))}


def _norm_name(s: str) -> str:
    """试卷名归一化：去空白/全半角括号/大小写，便于模糊匹配"""
    s = str(s or "")
    for a, b in (("（", "("), ("）", ")"), ("【", "["), ("】", "]"),
                 ("－", "-"), ("—", "-"), ("　", "")):
        s = s.replace(a, b)
    return re.sub(r"[\s()\[\]\-_·、,，。.]", "", s).lower()


def _is_subsequence(needle: str, haystack: str) -> bool:
    """needle 是否是 haystack 的子序列（顺序保留、可不连续）

    学生和模型更习惯写简称，如「2019网期末」对应
    「2019秋本部计算机网络期末考试回忆版」—— 字符集重合度算不出来，
    但"按顺序出现"能准确识别。
    """
    if not needle:
        return False
    it = iter(haystack)
    return all(ch in it for ch in needle)


def resolve_paper(name: str, course: str | None = None) -> tuple[dict | None, list[str]]:
    """把用户/模型给的试卷名解析成真实 source

    course 只是"同名歧义时的倾向"，绝不能拿来当硬过滤 ——
    否则学生上一题在做计网、下一句换成操作系统，就会被筛掉。
    （这个坑真实存在过：会话里记着 course=计算机网络，查操作系统卷子直接查不到。）

    Returns:
        (命中的试卷信息, 候选名列表)。命中不到时第一项为 None，第二项是候选试卷名。
    """
    papers = list_papers()
    all_names = [p["source"] for p in papers]
    if not papers:
        return None, []

    def prefer_hint(cands: list[dict]) -> dict:
        """同名/同相似度时优先返回本课程的卷子"""
        if course and len(cands) > 1:
            same = [p for p in cands if course in p["course"]]
            if same:
                return same[0]
        return cands[0]

    # 1. 精确匹配（全库范围，不受课程提示影响）
    for p in papers:
        if p["source"] == name:
            return p, []

    target = _norm_name(name)
    if not target:
        return None, all_names

    # 2. 归一化后相等
    norm_hits = [p for p in papers if _norm_name(p["source"]) == target]
    if norm_hits:
        return prefer_hint(norm_hits), []

    # 3. 互相包含（学生/模型少写"（回忆版）"之类的后缀）
    #    年份不同的一律不算命中：模型很爱编出"2026春计算机网络模拟题"这种不存在的卷名，
    #    如果任它模糊命中最像的那份，就会把学生带到他没选的卷子上。
    tgt_years = _years(name)

    def year_ok(p: dict) -> bool:
        return not tgt_years or not _years(p["source"]) or (_years(p["source"]) & tgt_years)

    contains = [p for p in papers
                if year_ok(p) and (target in _norm_name(p["source"]) or _norm_name(p["source"]) in target)]
    if len(contains) == 1:
        return contains[0], []
    if contains:
        return None, [p["source"] for p in contains]

    # 4. 简称：把输入当作 source 的子序列（顺序保留）
    #    「2019网期末」→「2019秋本部计算机网络期末考试回忆版」
    #    年份不同的一律不算，避免把学生带到别的年份的卷子上
    pool = [p for p in papers if year_ok(p)]
    sub_hits = [p for p in pool
                if len(target) >= 3 and _is_subsequence(target, _norm_name(p["source"]))]
    if len(sub_hits) == 1:
        return sub_hits[0], []
    if sub_hits:
        return None, [p["source"] for p in sub_hits]

    # 5. 逐字重合度兜底（同样要求年份对得上，且相似度要足够高）
    scored = []
    for p in pool:
        src = _norm_name(p["source"])
        overlap = len(set(target) & set(src))
        if overlap:
            scored.append((overlap / max(len(set(src)), 1), p))
    if scored:
        scored.sort(key=lambda x: -x[0])
        best_ratio, best = scored[0]
        clearly_best = len(scored) == 1 or best_ratio > scored[1][0] + 0.05
        if clearly_best and best_ratio >= 0.6:
            return best, []
        return None, [p["source"] for _, p in scored[:3]]

    if tgt_years:
        return None, [p["source"] for p in papers if year_ok(p)] or all_names
    return None, all_names


def list_papers(course: str | None = None) -> list[dict]:
    """列出试卷（按课程分组信息附带在每条记录里）

    每条记录附带 by_type / total / auto_gradable 等统计，方便模型直接口述，
    不需要模型自己数题。
    """
    questions = load_questions()
    buckets: dict[str, dict] = {}
    for q in questions:
        src = q.get("source", "")
        if not src:
            continue
        if course and course not in (q.get("course") or ""):
            continue
        item = buckets.setdefault(src, {
            "source": src,
            "course": q.get("course", ""),
            "total": 0,
            "by_type": {},
            "gradable": 0,
        })
        item["total"] += 1
        t = q.get("type", "")
        item["by_type"][t] = item["by_type"].get(t, 0) + 1
        if grading_basis(q)["usable"]:
            item["gradable"] += 1

    papers = list(buckets.values())
    for p in papers:
        p["by_type"] = {t: p["by_type"][t] for t in sorted(p["by_type"], key=_type_rank)}
        p["auto_gradable"] = f"{p['gradable']}/{p['total']}"
    papers.sort(key=lambda p: (p["course"], p["source"]))
    return papers


def paper_questions(source: str) -> list[dict]:
    """某份试卷的全部题目（按题型 + 题号排序）"""
    qs = [q for q in load_questions() if q.get("source") == source]
    qs.sort(key=_sort_key)
    return qs


def paper_outline(source: str) -> dict:
    """试卷结构概览：每个题型有几组、每组几题"""
    qs = paper_questions(source)
    numbers = section_numbers(qs)
    sections = []
    for t in sorted({q.get("type", "") for q in qs}, key=_type_rank):
        group = [q for q in qs if q.get("type") == t]
        size = group_size(t)
        sections.append({
            "type": t,
            "count": len(group),
            "group_size": size,
            "group_count": (len(group) + size - 1) // size,
            "numbers": [str(numbers[q["qid"]]) for q in group],
        })
    return {"source": source, "total": len(qs), "sections": sections}


# ============================================================
# 出题（分组由代码决定，不依赖 prompt）
# ============================================================

def image_url(q: dict) -> str | None:
    """题目图片 URL（中文/括号文件名需要 URL 编码，否则部分客户端取不到图）"""
    img = q.get("image")
    if not img:
        return None
    from urllib.parse import quote

    return f"/qb_images/{quote(str(img))}"


def _public_item(q: dict, no: int) -> dict:
    """出题给学生时对外暴露的字段 —— 刻意不含 text / answer / reference_points"""
    return {
        "no": no,
        "type": q.get("type", ""),
        "topic": q.get("topic", ""),
        "image_url": image_url(q),
    }


def get_group(source: str, qtype: str | None = None,
              group_index: int = 0, size: int | None = None) -> dict:
    """取一组题

    Args:
        source: 试卷名
        qtype: 题型；为空时按试卷题型顺序逐段推进（配合 group_index 全局取组）
        group_index: 组序号（从 1 开始）
        size: 覆盖默认组大小（一般不用传）

    Returns:
        {"source","type","group_index","group_count","questions":[...],
         "has_more","next":{...} 或 None, "grading_hint": "..."}
    """
    qs = paper_questions(source)
    if not qs:
        return {"error": f"找不到试卷: {source}", "available_papers": [p["source"] for p in list_papers()]}

    types = sorted({q.get("type", "") for q in qs}, key=_type_rank)

    if qtype and qtype not in types:
        return {
            "error": f"《{source}》没有「{qtype}」题型",
            "sections": paper_outline(source)["sections"],
        }

    # 把整份试卷按题型切成有序的"组列表"，这样全局 group_index 可以连续推进
    groups: list[tuple[str, list[dict]]] = []
    for t in types:
        bucket = [q for q in qs if q.get("type") == t]
        step = size or group_size(t)
        for i in range(0, len(bucket), step):
            groups.append((t, bucket[i:i + step]))

    if qtype:
        groups = [(t, g) for (t, g) in groups if t == qtype]
    if not groups:
        return {"error": f"《{source}》没有可出的题"}

    idx = max(1, int(group_index or 1))
    if idx > len(groups):
        return {
            "source": source,
            "finished": True,
            "total_groups": len(groups),
            "message": "这份卷子的所有分组都已经出完了。",
        }

    t, members = groups[idx - 1]
    numbers = section_numbers(qs)
    nxt = None
    if idx < len(groups):
        nt, ng = groups[idx]
        nxt = {"group_index": idx + 1, "type": nt, "count": len(ng)}

    return {
        "source": source,
        "type": t,
        "group_index": idx,
        "group_count": len(groups),
        "questions": [_public_item(q, numbers[q["qid"]]) for q in members],
        "qids": [q["qid"] for q in members],
        # 题号（题型内序号）跟 qid 的对应关系，批改时按它定位
        "numbers": {numbers[q["qid"]]: q["qid"] for q in members},
        # 交错排列的题型之间进度不好用"第X组"表达，给出人话进度
        "progress": f"第 {idx}/{len(groups)} 组（{t} {len(members)} 题）",
        "has_more": idx < len(groups),
        "next": nxt,
        "grading_hint": (
            "学生提交答案后调用 grade_answers，把上面 qids 原样传入 qids 参数，"
            "并把学生原话原样传入 user_answer_text。"
        ),
    }


# ============================================================
# 批改
# ============================================================

_LETTERS = re.compile(r"[A-Ha-h]")
# 选项字母只取到 H：真实试卷最多见过 A-E，取到 H 有余量；
# 不能放到 A-Z —— 那样 "the answer is B" 这种英文句子会被当成一串选项字母。
_CJK = re.compile(r"[\u4e00-\u9fff]")
# 纯选项作答的形状：只允许选项字母、数字、空白和常见分隔符。
# 用来把 "ABCDE" / "D B D A C" / "1A2B" 和 "the answer is B" 区分开。
_POSITIONAL_OK = re.compile(r"^[\sA-Ha-h0-9,，、.．:：;；/|()\[\]（）-]+$")
# 更宽的"作答样"形状（允许任意字母），用于兜底前判断"这明显是在作答但没解析出来"
_ANSWER_ONLY = re.compile(r"^[\sA-Za-z0-9,，、.．:：;；/|()\[\]（）-]+$")
_JUDGE_TRUE = {"对", "正确", "√", "✓", "t", "true", "yes", "是", "y"}
_JUDGE_FALSE = {"错", "错误", "×", "✗", "x", "f", "false", "no", "否", "n"}


def normalize_judge(text) -> bool | None:
    """判断题答案归一化为 True/False，无法识别返回 None"""
    s = str(text or "").strip().strip("。.，,、）)】]").lower()
    if s in _JUDGE_TRUE:
        return True
    if s in _JUDGE_FALSE:
        return False
    return None


def normalize_choice(text) -> str:
    """单题选择题答案归一化为有序去重大写字母串（支持多选，如 'AC'）"""
    return "".join(sorted(set(_LETTERS.findall(str(text or "").upper()))))


def letters_in_order(text) -> str:
    """按出现顺序提取字母（不去重、不排序）

    用于"一道题一个字母"的连写作答（如 'ABCDA' 对应 5 道题）。
    绝不能复用 normalize_choice —— 它会把 'ABCDA' 去重成 'ABCD'。
    """
    return "".join(_LETTERS.findall(str(text or ""))).upper()


# 选项字母前常见的提示词："选C" / "答案是C" / "答：C"
_CHOICE_CUE = re.compile(r"(?:选|择|答案|答|是|为|应选)\s*[:：]?\s*([A-Ha-h]+)")


def extract_choice(chunk: str, single: bool = True) -> str:
    """从一段作答文本里提取选项字母

    纯字母/带分隔符（"A"、"A、C"）直接提取；
    夹在中文解释里（"我选C，因为A和B都不对"）只认紧跟在"选/答案/是"后面的字母，
    避免把解释里提到的其他选项也当成作答（原实现的实际 bug）。
    """
    chunk = str(chunk or "").strip()
    if not chunk:
        return ""
    if not _CJK.search(chunk):
        letters = normalize_choice(chunk)
        return letters[:1] if single else letters

    m = _CHOICE_CUE.search(chunk)
    if m:
        letters = normalize_choice(m.group(1))
        return letters[:1] if single else letters

    # 中文语境但没有提示词：只接受紧随题号标记后的孤立字母（如 "1.A 因为..."）
    m = re.match(r"^[A-Ha-h](?![A-Za-z])", chunk)
    if m:
        return m.group(0).upper()
    return ""


def resolve_question_numbers(questions: list[dict], numbers: dict[str, int],
                             nos, restrict_to=None) -> tuple[list[str], dict]:
    """把"题号"解析成 qid 列表

    题号只在**题型内**唯一：选择题第 1 题和综合题第 1 题都是「1」。
    所以：
      1. 优先在 restrict_to（通常传"当前这一组"）里找 —— 学生说"第3题"
         指的就是刚展示给他的那组里的第 3 题；
      2. 全卷范围内只有该题号唯一对应一道题时才采信；
      3. 一个题号对应多道题时**不猜**，放进 ambiguous 让调用方去要 qids。
    （和 display_num 那次是同一类坑：绝不能静默挑一个题型。）

    Returns:
        (qids, ambiguous)  ambiguous: {题号: [候选 qid, ...]}
    """
    by_qid = {q["qid"]: q for q in questions}
    restrict = list(restrict_to or [])

    in_scope: dict[int, str] = {}
    for qid in restrict:
        n = numbers.get(qid)
        if n is not None and qid in by_qid:
            in_scope.setdefault(n, qid)

    full: dict[int, list[str]] = {}
    for qid, n in numbers.items():
        if qid in by_qid:
            full.setdefault(n, []).append(qid)

    out: list[str] = []
    ambiguous: dict[int, list[str]] = {}
    for raw in (nos or []):
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n in in_scope:
            out.append(in_scope[n])
            continue
        cands = full.get(n) or []
        if len(cands) == 1:
            out.append(cands[0])
        elif len(cands) > 1:
            ambiguous[n] = cands
    return list(dict.fromkeys(out)), ambiguous


def grading_basis(q: dict) -> dict:
    """这道题的批改依据

    usable=False 表示依据不足或明显不可靠（原实现会拿错位的单字母答案硬批，
    这里改为显式标注，让上层如实告诉学生"这题需要人工核对/给参考答案"）。
    """
    qtype = q.get("type", "")
    answer = (q.get("answer") or "")
    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False)
    answer = str(answer).strip()

    refs = [str(x).strip() for x in (q.get("reference_points") or []) if str(x).strip()]

    sub_answers = []
    for s in (q.get("sub_questions") or []):
        if isinstance(s, dict) and s.get("answer") not in (None, ""):
            a = s["answer"]
            if isinstance(a, (dict, list)):
                a = json.dumps(a, ensure_ascii=False)
            sub_answers.append(f"{s.get('label', '')} {a}".strip())

    # 填空题的标准答案是单个 A-D 字母 = 抽取错位（真题里 20 道如此），不可用
    misassigned = qtype in ("填空题",) and bool(re.fullmatch(r"[A-Ha-h]", answer))

    usable = bool(answer or refs or sub_answers)
    return {
        "answer": answer,
        "reference_points": refs,
        "sub_answers": sub_answers,
        "letter_answer": normalize_choice(answer) if re.fullmatch(r"[A-Ha-h]+", answer or "") else "",
        "misassigned": misassigned,
        "usable": usable,
    }


_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_to_int(s: str) -> int | None:
    """中文数字转整数（支持 1~99 的常见写法，如 一/十/十一/二十）"""
    s = (s or "").strip()
    if not s:
        return None
    if s == "十":
        return 10
    if "十" in s:
        a, _, b = s.partition("十")
        tens = _CN_DIGITS.get(a, 1) if a else 1
        ones = _CN_DIGITS.get(b, 0) if b else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(s)


# 题号标记。三种写法都要认，否则很自然的作答会被判成"没识别出来"：
#   1. 第1题 / 第一题 / 第 1 题
#   2. 1. / 1、 / 1) / 1：
#   3. D2.  —— 选项字母紧跟下一个题号（"1.D2.B3.D4.A5.C" 这种连写）
# 原来只有第 2 种，且数字前加了 (?<![0-9A-Za-z]) 反查，
# 结果 "D2." 因为前面是字母 D 而匹配不上，整串被当成一道题的答案。
_NUM_MARK = re.compile(
    r"第\s*(\d{1,3}|[一二三四五六七八九十]{1,3})\s*[题、.．)）:：]?\s*"
    r"|(?<![0-9A-Za-z])(\d{1,3})\s*[、.．)）:：]\s*"
    r"|(?<=[A-Ha-h])(\d{1,3})\s*[、.．)）:：]\s*"
)


def _numbered_split(text: str, expected: set[int]) -> list[tuple[int, str]]:
    """按 "1.xxx 2.yyy" / "第一题A" 切分，只认出现在预期题号集合里的数字"""
    marks = []
    for m in _NUM_MARK.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3)
        num = _cn_to_int(raw) if raw and not raw.isdigit() else (int(raw) if raw else None)
        if num is not None and num in expected:
            marks.append((m.start(), m.end(), num))
    marks.sort()
    # 去掉重叠（同一处可能被多个分支匹配到）
    dedup = []
    for mk in marks:
        if dedup and mk[0] < dedup[-1][1]:
            continue
        dedup.append(mk)
    out = []
    for i, (_, end, num) in enumerate(dedup):
        stop = dedup[i + 1][0] if i + 1 < len(dedup) else len(text)
        out.append((num, text[end:stop].strip()))
    return out


# "全都选A" / "全选A" / "一律填B" —— 明确表示整组同一个选项
_ALL_SAME = re.compile(r"(?:全都|全部|全|都|一律|统统|通通|每个都)\s*(?:选|是|填|答|为)?\s*([A-Ha-h])")


def parse_submission(user_text: str, questions: list[dict],
                     numbers: dict[str, int] | None = None) -> tuple[dict, list[dict]]:
    """把学生的一段话解析成 {qid: 答案}

    Args:
        user_text: 学生原话
        questions: 本次批改范围内的题目（通常就是刚出的那一组）
        numbers: qid -> 展示给学生的题号。不传则按列表顺序 1..N 编号
                 （组内顺序即展示顺序，正是学生看到的号）。

    Returns:
        (answer_map, notes)  notes 记录解析过程中的告警，便于如实反馈给学生

    这里刻意不再用题库的 display_num / q_num 做定位：
    同一份卷子里 display_num=1 会同时属于选择题和综合题，
    按它查表会把「1.D」判到综合题上（真实踩过）。
    """
    text = str(user_text or "").strip()
    notes: list[dict] = []
    if not text or not questions:
        return {}, notes

    # 单题：整段就是答案。这时才敢做"整段解析"，但选择题仍要避免把
    # 解释里提到的其它选项算作答（"我选C，因为A和B都不对" -> C）
    if len(questions) == 1:
        q = questions[0]
        qtype = q.get("type")
        if qtype == "选择题":
            letters = extract_choice(text, single=False)
            if letters:
                return {q["qid"]: letters}, notes
        elif qtype == "判断题":
            j = normalize_judge(text)
            if j is not None:
                return {q["qid"]: "对" if j else "错"}, notes
        return {q["qid"]: text}, notes

    if numbers is None:
        numbers = {q["qid"]: i + 1 for i, q in enumerate(questions)}
    by_no = {numbers[q["qid"]]: q for q in questions if q["qid"] in numbers}

    def locate(num: int):
        """题号 → 题目

        先按"题型内序号"查（第 6 题就是第 6 题）；
        查不到时退回"组内第几题" —— 学生看着 6~10 号题，却常写成「1.A 2.B」，
        这两种都要认。少了这个兜底，第 2 组往后的填空/判断题答案会整组解析失败。
        """
        if num in by_no:
            return by_no[num]
        if 1 <= num <= len(questions):
            return questions[num - 1]
        return None

    expected = set(by_no) | set(range(1, len(questions) + 1))
    answer_map: dict[str, str] = {}

    # ① 带题号
    numbered = _numbered_split(text, expected)
    for num, chunk in numbered:
        q = locate(num)
        if not q or not chunk:
            continue
        if q.get("type") == "选择题":
            # 纯字母可以多选（"1.AC"）；夹在解释里只认一个字母
            letters = extract_choice(chunk, single=bool(_CJK.search(chunk)))
            if not letters:
                notes.append({"no": num, "warning": "没从这段里认出选项字母", "raw": chunk[:60]})
                continue
            answer_map[q["qid"]] = letters
        elif q.get("type") == "判断题":
            j = normalize_judge(chunk)
            if j is None:
                notes.append({"no": num, "warning": "没认出对/错", "raw": chunk[:60]})
                continue
            answer_map[q["qid"]] = "对" if j else "错"
        else:
            # 非选择题收到"单个选项字母"或"光秃秃的对/错"说明定位错了
            # （学生答的是选择题/判断题），跳过而不是当成填空/综合题的作答 ——
            # 否则一份选择题或判断题答案会被拿去批改别的题型，
            # 还会把那些题的参考答案整段泄给学生
            if re.fullmatch(r"[A-Ha-h]{1,2}", chunk):
                notes.append({"no": num, "warning": "这题不是选择题，但收到的是选项字母，已跳过",
                              "raw": chunk[:20]})
                continue
            if q.get("type") != "判断题" and normalize_judge(chunk) is not None:
                notes.append({"no": num, "warning": "这题不是判断题，但收到的是对/错，已跳过",
                              "raw": chunk[:20]})
                continue
            answer_map[q["qid"]] = chunk

    if answer_map:
        return answer_map, notes

    # 认出了题号但一条都没解析成功（比如给填空题交了选项字母）：
    # 说明确实是在按题号作答，只是内容对不上，这里必须如实返回空，
    # 不能退到"整段塞给第 1 题"的兜底 —— 那会把整句话当成某一道题的答案。
    if numbered:
        return {}, notes

    # ② "全都选A" / "我选AAAAA" —— 整组同一个选项
    #    只在两种情况下采信，避免把"学生只答了前 3 题"误判成"整组都选 A"：
    #      a) 有明确的"全部/都"之类措辞
    #      b) 重复的字母数 ≥ 题目数（他给够了那么多字母，且全一样）
    if all(q.get("type") == "选择题" for q in questions):
        cue = _ALL_SAME.search(text)
        letters_all = letters_in_order(text)
        same_run = bool(letters_all) and len(set(letters_all)) == 1
        if cue:
            letter = normalize_choice(cue.group(1))
            if letter:
                return {q["qid"]: letter for q in questions}, notes
        if same_run and len(letters_all) >= len(questions):
            return {q["qid"]: letters_all[0] for q in questions}, notes

    # ③ 整份纯字母作答（如 "ABCDE"）—— 只有全程无中文、且整段只由
    #    选项字母/数字/分隔符组成时才敢按位解析：
    #      · "因为A和B都不对" 含中文 → 排除（原实现的 bug）
    #      · "the answer is B" 含英文单词 → 排除（放宽到 A-H 后必须挡住的）
    if (not _CJK.search(text) and _POSITIONAL_OK.match(text)
            and all(q.get("type") == "选择题" for q in questions)):
        letters = letters_in_order(text)
        if letters and len(letters) <= len(questions):
            for i, ch in enumerate(letters):
                answer_map[questions[i]["qid"]] = ch
            # 字母数少于题目数时说清楚是"只答了前几题"，不要让学生以为整组都批了
            if len(letters) < len(questions):
                notes.append({"warning": f"按顺序对应前 {len(letters)} 题",
                              "raw": text[:40]})
            return answer_map, notes
        if letters:
            # 字母比题目多，说明对应关系确定不了 —— 如实返回空，
            # 不能退到"整段塞给第 1 题"（那会把 "ABCDEFGHIJ" 当成某道题的答案）
            notes.append({"warning": f"认出 {len(letters)} 个选项字母，多于题目数"
                                     f"（{len(questions)}），无法确定对应关系",
                          "raw": text[:40]})
            return {}, notes

    # ④ 判断题连写（如 "对错对错对"）
    if all(q.get("type") == "判断题" for q in questions):
        tokens = re.findall(r"对|错|正确|错误|√|×|✓|✗", text)
        if len(tokens) >= len(questions):
            for q, tok in zip(questions, tokens):
                j = normalize_judge(tok)
                if j is not None:
                    answer_map[q["qid"]] = "对" if j else "错"
            return answer_map, notes

    # ⑤ 兜底之前：整段是个"作答样"的短标记（只有字母/数字/分隔符、且没有空格，
    #    例如 "ABCDEFGHIJ"、"ZZZZZ"）却没解析出对应关系 —— 如实返回空，
    #    让上层请学生换个写法。绝不能把这一串当成某一道题的答案，
    #    那等于凭空造出一个批改结果。
    if _ANSWER_ONLY.match(text) and not re.search(r"\s", text.strip()):
        notes.append({"warning": "这段看起来是作答，但没能确定它对应哪几题",
                      "raw": text[:60]})
        return {}, notes

    # ⑥ 兜底：整段交给第一题（并提示解析不确定）
    notes.append({"warning": "没能确定这段话对应哪几题，已按整组第 1 题处理", "raw": text[:80]})
    answer_map[questions[0]["qid"]] = text
    return answer_map, notes


def parse_against_paper(user_text: str, questions: list[dict]) -> tuple[dict, list[dict]]:
    """没有"当前这一组"时，把整卷作为批改范围来解析

    整卷里题号在不同题型间会重复（选择题第 1 题、综合题第 1 题），
    所以要按题型分段各试一次，取"解析出的题数最多"的那一段 ——
    配合"非选择题不接受单个选项字母"的规则，
    一份选择题答案就只会落到选择题那一段上。
    """
    sections: dict[str, list[dict]] = {}
    for q in questions:
        sections.setdefault(q.get("type", ""), []).append(q)

    best: tuple[int, dict, list[dict]] = (0, {}, [])
    for _t, qs in sections.items():
        numbers = section_numbers(questions)
        amap, notes = parse_submission(user_text, qs, numbers)
        if len(amap) > best[0]:
            best = (len(amap), amap, notes)
    return best[1], best[2]


def grade_answer(q: dict, user_answer: str, no: int | None = None) -> dict:
    """批改单题（不做 LLM 调用；需要 LLM 的题型返回 needs_llm=True）"""
    qtype = q.get("type", "")
    basis = grading_basis(q)
    # 展示给学生看的"正确答案"：优先正式答案；answer 错位时退回参考要点/小问答案，
    # 避免把 'C' 这种抽取噪声当成标准答案念出来
    shown = basis["answer"]
    if basis["misassigned"] or not shown:
        shown = "；".join(basis["sub_answers"]) or "；".join(basis["reference_points"][:3]) or "（题库缺少参考答案）"
    out = {
        "no": no if no is not None else _display_no(q),
        "qid": q["qid"],
        "type": qtype,
        "topic": q.get("topic", ""),
        "user_answer": user_answer,
        "correct_answer": shown,
    }

    if qtype == "选择题":
        if not basis["letter_answer"]:
            out.update(correct=None, needs_manual=True,
                       explanation="这题的选项答案缺失或不是字母，无法自动判定。")
        else:
            ok = normalize_choice(user_answer) == basis["letter_answer"]
            out.update(correct=ok, explanation="回答正确。" if ok else f"正确答案是 {basis['answer']}。")
    elif qtype == "判断题":
        u = normalize_judge(user_answer)
        c = normalize_judge(basis["answer"])
        if c is None:
            out.update(correct=None, needs_manual=True, explanation="这题的标准答案无法识别。")
        elif u is None:
            out.update(correct=None, needs_manual=True,
                       explanation=f"没认出你的对/错表述。标准答案是「{basis['answer']}」。")
        else:
            out.update(correct=u == c,
                       explanation="回答正确。" if u == c else f"正确答案是「{basis['answer']}」。")
    # 填空题 / 综合题 / 整页综合 -> 交给 LLM 判，依据在 basis 里
    elif not basis["usable"]:
        out.update(correct=None, needs_manual=True,
                   explanation="这题的参考答案缺失，无法自动批改。")
    else:
        out["needs_llm"] = True
        if basis["misassigned"]:
            out["basis_warning"] = "题库里这道填空题的 answer 字段抽取错位（是个选项字母），已改用参考要点批改。"

    # 答错或判不了时，把参考要点一并带出来：
    # 只回一句"正确答案是 B"学生学不到东西，而模型手里没有题面/答案依据就没法讲，
    # 所以批改结果必须自带可讲解的素材。
    if out.get("correct") is not True:
        points = basis["reference_points"] or basis["sub_answers"]
        if points:
            out["reference_points"] = [p[:240] for p in points[:6]]
    return out


def build_llm_basis(q: dict, no: int | None = None) -> dict:
    """给 LLM 阅卷用的题干与参考答案（仅在批改阶段使用，不出现在出题接口里）"""
    basis = grading_basis(q)
    subs = []
    for s in (q.get("sub_questions") or []):
        if isinstance(s, dict):
            a = s.get("answer")
            if isinstance(a, (dict, list)):
                a = json.dumps(a, ensure_ascii=False)
            subs.append(f"{s.get('label', '')} {a}".strip() if a else str(s.get("label", "")))
    return {
        "no": no if no is not None else _display_no(q),
        "qid": q["qid"],
        "type": q.get("type", ""),
        "text": q.get("text", ""),
        # 抽取错位的单字母答案绝不能当标准答案喂给阅卷模型，直接置空
        "answer": "" if basis["misassigned"] else basis["answer"],
        "reference_points": basis["reference_points"],
        "sub_questions": subs,
        "answer_unreliable": basis["misassigned"],
    }
