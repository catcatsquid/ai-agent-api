"""行为评测 runner —— 打真实 HTTP，跑脚本化用例，出可对比的报告

## 为什么要有它

`pytest tests/`（600+ 条）守的是"**代码对不对**"：单元、契约、结构不变量。
它**回答不了**"这个版本学生用起来是不是更好了"。这个 runner 补的就是那一层：

  · 同一个用例、同一套判据，**每个版本跑一次**，报告存进 git → 可以 diff
  · **每条用例重复 N 次** —— 因为历史上那些缺陷全是"时好时坏"，单次通过没有意义
  · 判据全部是**可判定的**（工具轨迹、参数、与题库原文比对、图片归属、计时），
    不依赖主观打分；软指标（讲得好不好）是另一层，见 README

## 怎么跑

    python -m eval.runner                 # 全部用例 × 3 次
    python -m eval.runner --case quiz_no_leak --attempts 5
    python -m eval.runner --base http://127.0.0.1:8003

服务得先起着（评测走的是**学生真正走的那条路**：HTTP + SSE）。

## 报告

`eval/reports/<版本>.json`：每个用例每次尝试的判据明细 + 计时 + 工具轨迹 + 回答开头，
外加汇总（稳定通过率、TTFT p50/p95）。**报告要进 git** —— 那是"版本有没有变好"的唯一依据。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
CASES_DIR = EVAL_DIR / "cases"
REPORTS_DIR = EVAL_DIR / "reports"

# 与生产**同一条**正则：正文里不许出现"工具名 {json}"这种内部协议原文。
# 直接 import 而不是复制 —— 复制的话生产改了正则，评测还在按旧规则判，
# 而注释里那句"与生产同一条"就变成假话（这个项目在别处已经栽过好几次了）。
from course_agent.nodes import _TOOL_CALL_TEXT_RE  # noqa: E402
# 只在"明确给答案"的语境里才算泄题（避免"B 选项说…"这种讲解被误判）
_ANSWER_LETTER_RE = re.compile(r"(答案|正确答案|正确选项)\s*[:：是为]?\s*([A-D])\b")
_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


# ============================================================
# 驱动：打真实的流式端点
# ============================================================

def _read_turn(base: str, query: str, thread_id: str, mode: str,
               timeout: int) -> dict:
    """一轮 SSE 的实际读取（不含重试）"""
    t0 = time.perf_counter()
    streamed, tools, ttft_ms, done_answer, n_content = "", [], None, None, 0
    body = {"query": query, "thread_id": thread_id, "mode": mode}
    with requests.post(f"{base}/qa/stream", json=body, stream=True,
                       timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                ev = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "content":
                n_content += 1
                if ttft_ms is None:            # 首响 = 第一个 content 事件
                    ttft_ms = int((time.perf_counter() - t0) * 1000)
                streamed += ev.get("content", "")
            elif kind == "done":
                done_answer = ev.get("answer") or ""
            elif kind == "node" and ev.get("tool"):
                tools.append(ev["tool"])
    answer = streamed if streamed.strip() else (done_answer or "")
    return {
        "answer": answer,
        "answer_source": "content" if streamed.strip() else "done",
        "content_events": n_content,
        "tools": tools,
        "ttft_ms": ttft_ms,
        "total_ms": int((time.perf_counter() - t0) * 1000),
    }


def stream_turn(base: str, query: str, thread_id: str, mode: str = "plan",
                timeout: int = 240, retries: int = 2) -> dict:
    """发一轮，解析 SSE，返回正文 / 工具轨迹 / 首响时间 / 总时长

    ## 正文取哪里：`content` 事件**或** `done.answer`

    `content` 是逐字流（打字机效果）；`done.answer` 是权威值 —— 前端也是这么用的
    （流式少一个字不会留在界面上）。两者都要取，并且**分开记录**：
    `answer_source` 说明这一轮的正文是从哪来的 —— 如果它总是 `done`，
    说明**流式根本没生效**（首响就等于总时长），那是个产品问题，不是解析问题。

    ## 传输超时要重试（2026-10-16 实测踩到两次）

    智谱偶发地把一次请求挂住（服务端 `Read timed out` 之后 120s 才放弃），
    于是 runner 这边读超时 —— 一轮失败就把**整批**评测带走（报告一行都没写）。
    一次网络抖动不该等于"评测失败"，所以只对**传输层**超时重试；重试换一个新的
    thread_id（上一轮可能已经半路写进了会话状态，接着用会让判据看到残留）。
    注意：模型真的出错时服务端会发 `error` 事件，那不是传输问题，**不重试**。
    """
    last_err = None
    tid = thread_id
    for attempt in range(retries + 1):
        try:
            turn = _read_turn(base, query, tid, mode, timeout)
            turn["thread_id"] = tid            # 清理会话时要用**实际**那个
            return turn
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            if attempt < retries:
                tid = f"{thread_id}-r{attempt + 1}"
                print(f"    ! 传输超时，换会话重试（{attempt + 1}/{retries}）："
                      f"{type(e).__name__}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"连着 {retries + 1} 次传输失败（不是模型的问题）: {last_err}")


# ============================================================
# 真值：从题库取"被藏起来的"题面 / 参考答案 / 图片
# ============================================================

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def bank_truth(paper_source: str, group_index: int) -> dict:
    """用**工具自己那段分组逻辑**取出这一组的题，再回题库拿题面/答案/图片"""
    from course_agent import qbank, quiz_service

    paper, _ = quiz_service.resolve_paper(paper_source)
    payload = quiz_service.group_payload(paper, group_index=group_index)
    qids = payload.get("qids") or []
    by_qid = {q.get("qid"): q for q in qbank.paper_questions(paper["source"])}
    stems, refs, images, letters = [], [], set(), []
    for qid in qids:
        q = by_qid.get(qid)
        if not q:
            continue
        stems.append(_norm(q.get("text")))
        basis = qbank.grading_basis(q)
        for point in (basis.get("reference_points") or []):
            refs.append(_norm(point))
        if q.get("answer"):
            letters.append(str(q["answer"]).strip().upper())
        url = qbank.image_url(q)
        if url:
            images.add(url)
    return {"qids": qids, "stems": stems, "references": refs,
            "images": images, "letters": letters,
            "all_images": {qbank.image_url(q) for q in qbank.paper_questions(paper["source"])
                           if qbank.image_url(q)}}


# ============================================================
# 判据（全部可判定，不做主观打分）
# ============================================================

def _chunks(text: str, size: int = 20) -> list[str]:
    """把长文本切成若干"够独特"的片段（太短的没必要比，容易误判）"""
    t = _norm(text)
    if len(t) < size:
        return []
    step = max(size, len(t) // 3)
    return [t[i:i + size] for i in range(0, len(t) - size + 1, step)]


# 正文里**绝不能出现**的内部痕迹。两条来源都是真实契约：
#   · `search_knowledge` 的返回值末尾带一段【怎么用上面的片段】使用说明 ——
#     那是写给**模型**看的，不是给学生看的；
#   · planner 的计划是 JSON（need_tools / steps），工具调用原文是 `工具名 {json}`。
# 与"泄题"不同，这里不比对真值，所以也能用在**没有 truth** 的用例上
# （讲解类用例就是这种：没有"卷子原文"可比）。
_INTERNAL_ARTIFACTS = (
    "【怎么用上面的片段】",
    "【检索范围】",
    "need_tools",
    "tool_calls",
    "【内部摘要】",
)


def _internal_artifacts_in(answer: str) -> list:
    return [marker for marker in _INTERNAL_ARTIFACTS if marker in (answer or "")]


def run_checks(expect: dict, turn: dict, truth: dict) -> list[dict]:
    answer = turn["answer"]
    tools = turn["tools"]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # `tools_any_of` 的字面意思就是**任一命中即可**。原来写成了逐项断言
    # （"调用了 A" "调用了 B" 都要求真），于是多元素时**必然红** ——
    # 在 `react_no_tool_call_leak`（三种取材料方式任选）上暴露出来：
    # 模型调了 search_knowledge，却被判"没调 read_section / list_course_sections"。
    any_of = expect.get("tools_any_of") or []
    if any_of:
        hit = [name for name in any_of if name in tools]
        add(f"调用了 {' / '.join(any_of)} 中的任意一个",
            bool(hit), f"命中 {hit}；实际工具轨迹: {tools}")
    for name in expect.get("tools_none_of", []):
        add(f"没有调用 {name}", name not in tools, f"实际工具轨迹: {tools}")

    for name, want in (expect.get("args") or {}).items():
        # 参数校验：只要生产端点没回传参数，就退化为"至少调了这个工具"
        # （`/qa/stream` 的进度事件里目前不含参数；要精确校验得让端点带上 args）
        add(f"{name} 的参数符合预期", name in tools,
            f"期望 {want}；当前端点未回传参数，此判据退化为「调用了该工具」")

    if expect.get("forbid_tool_call_text"):
        hit = _TOOL_CALL_TEXT_RE.search(answer)
        add("正文里没有工具调用原文", not hit, f"命中: {hit.group(0)[:60]}" if hit else "")

    if expect.get("no_internal_artifacts"):
        hits = _internal_artifacts_in(answer)
        add("正文里没有内部痕迹（计划 JSON / 工具使用说明 / 摘要标记）",
            not hits, f"命中: {hits}")

    if expect.get("min_images"):
        n = len(_IMG_RE.findall(answer))
        add(f"至少 {expect['min_images']} 张题目图片",
            n >= expect["min_images"], f"实际 {n} 张")

    if expect.get("images_all_from_bank"):
        urls = {u.split("?")[0] for u in _IMG_RE.findall(answer)}
        foreign = [u for u in urls
                   if u.replace("/qb_images/", "") not in
                   {i.replace("/qb_images/", "") for i in truth["all_images"]}
                   and not any(i.endswith(u.rsplit("/", 1)[-1]) for i in truth["all_images"])]
        add("图片链接都来自题库（没有编造）", not foreign, f"外来链接: {foreign[:3]}")

    if expect.get("no_stem_leak"):
        leaked = [c[:30] for s in truth["stems"] for c in _chunks(s)
                  if c in _norm(answer)]
        add("题面没有泄漏进正文", not leaked, f"泄漏片段: {leaked[:2]}")

    if expect.get("no_reference_leak"):
        leaked = [c[:30] for s in truth["references"] for c in _chunks(s)
                  if c in _norm(answer)]
        add("参考答案没有泄漏进正文", not leaked, f"泄漏片段: {leaked[:2]}")

    if expect.get("no_answer_letter_leak"):
        hit = _ANSWER_LETTER_RE.search(answer)
        add("没有直接给出选项答案", not hit, f"命中: {hit.group(0)}" if hit else "")

    if expect.get("answer_min_chars"):
        add(f"正文至少 {expect['answer_min_chars']} 字",
            len(answer) >= expect["answer_min_chars"], f"实际 {len(answer)} 字")

    # ⚠️ 这两条是"真空通过"的解药：正文为空时，上面那些"不含泄漏"的断言会
    # **全部轻松通过**（空字符串当然不含参考答案）。所以必须有非空与流式门槛。
    if expect.get("require_streaming"):
        add("回答是逐字流式出来的（不是最后一次性给出）",
            turn["content_events"] > 0 and turn["answer_source"] == "content",
            f"content 事件 {turn['content_events']} 个，正文来源={turn['answer_source']}"
            f"（来源为 done 意味着流式没生效：首响=总时长）")

    if expect.get("max_ttft_ms"):
        ttft = turn["ttft_ms"]
        add(f"首字 ≤ {expect['max_ttft_ms']} ms",
            ttft is not None and ttft <= expect["max_ttft_ms"],
            f"实际 {ttft if ttft is not None else '没有任何 content 事件'} ms"
            f"（总时长 {turn['total_ms']} ms）")

    return checks


# ============================================================
# 跑用例
# ============================================================

def run_case(case: dict, base: str, attempts: int) -> dict:
    detail, per_attempt = [], []
    for i in range(attempts):
        tid = f"eval-{case['id']}-{i}-{uuid.uuid4().hex[:6]}"
        attempt = {"attempt": i + 1, "thread_id": tid, "turns": []}
        for turn_spec in case["turns"]:
            turn = stream_turn(base, turn_spec["user"], tid,
                               mode=case.get("mode", "plan"))
            # 传输重试会换 thread_id（见 stream_turn）；记录**实际**用的那个，
            # 否则清理会话时会漏掉重试留下的那一条
            tid = turn.get("thread_id", tid)
            attempt["thread_id"] = tid
            truth = {}
            if case.get("truth"):
                truth = bank_truth(case["truth"]["paper_source"],
                                   case["truth"].get("group_index", 1)
                                   + turn_spec.get("group_offset", 0))
            checks = run_checks(turn_spec.get("expect", {}), turn, truth)
            turn["checks"] = checks
            turn["user"] = turn_spec["user"]
            attempt["turns"].append(turn)
            detail += [c for c in checks]
        per_attempt.append(attempt)

    total = len(detail)
    passed = sum(1 for c in detail if c["ok"])
    return {"id": case["id"], "attempts": per_attempt,
            "checks_total": total, "checks_passed": passed,
            "stable_pass": passed == total,
            "failures": [f"{c['check']}: {c['detail']}"
                         for c in detail if not c["ok"]]}


def delete_threads(base: str, thread_ids: list[str], mode: str = "plan") -> int:
    """删掉评测自己造的会话 —— 评测不该把垃圾留在学生的会话库里

    为什么必须做：`/chat/conversations` 会把这些 thread 列出来，前端会当成
    "服务端保存的会话"导入 → 用户侧栏里就会冒出几条「给我出 2026春…」这种怪东西。
    （这与 1.4.1 修过的"测试污染会话库"是同一类问题，只是来源换成了评测。）
    """
    cleaned = 0
    for tid in thread_ids:
        try:
            resp = requests.delete(f"{base}/chat/conversations/{tid}",
                                   params={"mode": mode}, timeout=30)
            if resp.status_code in (200, 404):     # 404 = 本来就没落盘
                cleaned += 1
        except requests.RequestException:
            pass
    return cleaned


def _force_utf8_stdout() -> None:
    """把 stdout 切到 UTF-8（中文 Windows 上默认是 GBK，会崩在打印上）

    实测踩到：`python -m eval.runner` 在 cp936 控制台上抛
    `UnicodeEncodeError: 'gbk' codec can't encode character '\\u2716'`
    —— 崩点还是在**用例标题**那一行，等于一条用例都没跑。
    报告 JSON 一直是 UTF-8，坏的只是控制台输出。

    这里显式重配 stdout（而不是要求用户设 PYTHONIOENCODING）：评测是要
    "每个版本跑一次"的东西，不该靠记性。`errors="replace"` 兜底 ——
    打印不出某个字符也不许让一次评测白跑。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _display_path(path: Path) -> str:
    """报告路径：能相对就相对，不能相对就原样（**不许因此崩掉**）

    `--out` 允许给任意路径（相对、绝对、别的盘）。原实现直接
    `out.relative_to(ROOT)`，只要它不在仓库下就抛 ValueError —— 而且是在
    **报告已经写完、汇总已经打完之后**抛的，退出码变成 1（"评测失败"），
    假的失败信号。实测踩到：`--out eval/reports/baseline-1.7.1.json`
    （路径本身就在仓库里，只是因为没 resolve 而判定不在）。
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="行为评测 runner")
    ap.add_argument("--base", default="http://127.0.0.1:8003")
    ap.add_argument("--case", action="append", default=None,
                    help="只跑指定用例（可重复）")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--keep-threads", action="store_true",
                    help="跑完不删评测造的会话（默认会删，免得污染会话库）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _force_utf8_stdout()

    from config import APP_VERSION

    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if args.case and case["id"] not in args.case:
            continue
        cases.append(case)
    if not cases:
        print("没有匹配的用例")
        return 2

    print(f"行为评测：{len(cases)} 条用例 × {args.attempts} 次  →  {args.base}")
    print("=" * 78)

    started = datetime.now(timezone.utc)
    results = []
    for case in cases:
        print(f"\n>> {case['id']}")
        print(f"   为什么存在: {case.get('why', '(未写)')}")
        r = run_case(case, args.base, args.attempts)
        results.append(r)
        for c in r["failures"]:
            print(f"   [x] {c}")
        print(f"  判据 {r['checks_passed']}/{r['checks_total']}"
              f"  {'[OK] 稳定通过' if r['stable_pass'] else '[NG] 未稳定通过'}")
        if not args.keep_threads:
            tids = [a["thread_id"] for a in r["attempts"]]
            n = delete_threads(args.base, tids, mode=case.get("mode", "plan"))
            print(f"  已清理 {n}/{len(tids)} 个评测会话（不留在会话库里）")

    all_checks = [c for r in results for a in r["attempts"]
                  for t in a["turns"] for c in t["checks"]]
    ttfts = [t["ttft_ms"] for r in results for a in r["attempts"]
             for t in a["turns"] if t["ttft_ms"]]
    totals = [t["total_ms"] for r in results for a in r["attempts"] for t in a["turns"]]
    summary = {
        "cases": len(results),
        "cases_stable_pass": sum(1 for r in results if r["stable_pass"]),
        "checks_total": len(all_checks),
        "checks_passed": sum(1 for c in all_checks if c["ok"]),
        "ttft_p50_ms": int(statistics.median(ttfts)) if ttfts else None,
        "ttft_p95_ms": int(sorted(ttfts)[int(len(ttfts) * 0.95) - 1]) if ttfts else None,
        "turn_total_p50_ms": int(statistics.median(totals)) if totals else None,
    }

    print("\n" + "=" * 78)
    print("汇总")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORTS_DIR / f"{APP_VERSION}.json"
    out.write_text(json.dumps({
        "app_version": APP_VERSION,
        "started_at": started.isoformat(),
        "base": args.base,
        "attempts": args.attempts,
        "summary": summary,
        "cases": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {_display_path(out)}")
    return 0 if summary["cases_stable_pass"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
