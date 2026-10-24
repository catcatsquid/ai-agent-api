"""测量 /qa/stream 的分阶段耗时（不用改生产代码，只给 SSE 事件打到达时间戳）。

服务的流事件顺序固定为：
    start → node(planner) → node(executor_step/tool_step)* → node(synthesizer) → content* → done
其中 node 事件是「节点执行完成即发」，所以相邻事件的到达时间差就等于该阶段耗时。

用法：
    python tools/measure_stages.py                 # 默认 5 次，plan 模式
    python tools/measure_stages.py --n 3 --query "什么是死锁"
    python tools/measure_stages.py --base http://127.0.0.1:8003 --mode react

输出：每次运行的原始分段 + 各段 p50/p95 + 一段可直接粘进文档的 Markdown 表。
"""
import argparse
import json
import statistics
import sys
import time
import uuid

import requests

# 中文 Windows 控制台默认 cp936，直接 print 会 UnicodeEncodeError（评测工具踩过这个坑）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STAGES = ["建连", "规划", "工具", "生成到首字", "流式", "收尾"]


def run_once(base: str, query: str, mode: str, timeout: int) -> dict:
    """跑一轮，返回各阶段耗时（ms）。缺失的阶段为 None。"""
    t0 = time.perf_counter()
    marks = {}          # 事件名 -> 到达时刻
    n_content = 0
    tools = []
    err = None

    body = {"query": query, "thread_id": str(uuid.uuid4()), "mode": mode}
    with requests.post(f"{base}/qa/stream", json=body, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            now = time.perf_counter()
            try:
                ev = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            kind, node = ev.get("type"), ev.get("node")

            if kind == "start":
                marks.setdefault("start", now)
            elif kind == "node":
                if node == "planner":
                    marks["planner"] = now
                elif node in ("executor_step", "tool_step"):
                    marks.setdefault("first_tool", now)
                    marks["last_tool"] = now
                    if ev.get("tool"):
                        tools.append(ev["tool"])
                elif node == "web_search_fallback":
                    marks["last_tool"] = now
                elif node == "synthesizer":
                    marks["synthesizer"] = now
            elif kind == "content":
                n_content += 1
                marks.setdefault("first_content", now)
                marks["last_content"] = now
            elif kind == "done":
                marks["done"] = now
            elif kind == "error":
                err = ev.get("msg")

    def delta(a, b):
        return None if (a not in marks or b not in marks) else int((marks[b] - marks[a]) * 1000)

    has_content = "first_content" in marks
    tools_end = marks.get("last_tool") or marks.get("planner")
    tail_from = marks.get("last_content") or marks.get("synthesizer")

    row = {
        "建连": None if "start" not in marks else int((marks["start"] - t0) * 1000),
        "规划": delta("start", "planner"),
        "工具": delta("planner", "last_tool"),
        "生成到首字": (None if not has_content or tools_end is None
                       else int((marks["first_content"] - tools_end) * 1000)),
        "流式": delta("first_content", "last_content"),
        "收尾": (None if "done" not in marks or tail_from is None
                 else int((marks["done"] - tail_from) * 1000)),
        "首字_总": None if not has_content else int((marks["first_content"] - t0) * 1000),
        "整轮_总": int((marks.get("done", time.perf_counter()) - t0) * 1000),
        "content事件": n_content,
        "工具列表": tools,
        "error": err,
    }
    return row


def p(values, q=50):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    if q == 50:
        return int(statistics.median(vals))
    idx = min(len(vals) - 1, int(len(vals) * q) - 1)
    return int(vals[max(idx, 0)])


def fmt(v):
    return "-" if v is None else f"{v} ms"


def main():
    ap = argparse.ArgumentParser(description="测量 /qa/stream 的分阶段耗时")
    ap.add_argument("--base", default="http://127.0.0.1:8003")
    ap.add_argument("--query", default="什么是死锁")
    ap.add_argument("--mode", default="plan", choices=["plan", "react"])
    ap.add_argument("--n", type=int, default=5, help="重复次数（同一问题、每轮新 thread_id）")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    print(f"目标 {args.base}/qa/stream  mode={args.mode}  n={args.n}")
    print(f"问题：{args.query}\n")

    rows = []
    for i in range(1, args.n + 1):
        try:
            r = run_once(args.base, args.query, args.mode, args.timeout)
        except Exception as e:
            print(f"第 {i} 次失败：{type(e).__name__}: {e}")
            continue
        rows.append(r)
        stage_txt = "  ".join(f"{k}={fmt(r[k])}" for k in STAGES)
        print(f"#{i}  {stage_txt}")
        print(f"     首字 {fmt(r['首字_总'])} · 整轮 {fmt(r['整轮_总'])} · "
              f"content {r['content事件']} · 工具 {r['工具列表'] or '无'}"
              + (f" · ERROR {r['error']}" if r["error"] else ""))

    if not rows:
        print("\n没有成功样本，检查服务是否在跑、端口是否正确。")
        return

    print("\n== 汇总（p50 / p95） ==")
    for k in STAGES + ["首字_总", "整轮_总"]:
        vals = [r[k] for r in rows]
        print(f"{k:<8} p50={fmt(p(vals, 50)):<10} p95={fmt(p(vals, 95))}")

    print("\n== 可直接粘进文档的表格 ==")
    print("| 阶段 | p50 | p95 | 对应事件 |")
    print("|---|---|---|---|")
    event_of = {
        "建连": "start", "规划": "node:planner", "工具": "node:executor_step / tool_step",
        "生成到首字": "首个 content", "流式": "content 首条→末条", "收尾": "done",
    }
    for k in STAGES:
        vals = [r[k] for r in rows]
        print(f"| {k} | {fmt(p(vals, 50))} | {fmt(p(vals, 95))} | {event_of[k]} |")
    vals = [r["首字_总"] for r in rows]
    print(f"| 首字（合计） | {fmt(p(vals, 50))} | {fmt(p(vals, 95))} | 请求→首个 content |")
    vals = [r["整轮_总"] for r in rows]
    print(f"| 整轮（合计） | {fmt(p(vals, 50))} | {fmt(p(vals, 95))} | 请求→done |")
    print(f"\n样本：{len(rows)} 次，同一问题、每轮新 thread_id、单机单进程。"
          f"content 事件中位数 {int(statistics.median([r['content事件'] for r in rows]))} 个/轮。")


if __name__ == "__main__":
    main()

