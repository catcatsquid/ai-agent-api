"""生成 v2 对比报告：题目截图 + 答案"""

import json
from pathlib import Path

OUTPUT_DIR = Path("debug/question_bank_v2")
IMAGE_DIR = OUTPUT_DIR / "images"

with open(OUTPUT_DIR / "questions_2024秋.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>习题库提取 v2 - 题目截图 + 答案</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: #fff; padding: 20px 30px; }
  .header h1 { font-size: 22px; }
  .header .meta { font-size: 13px; color: #aaa; margin-top: 5px; }
  .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
  
  .stats { display: flex; gap: 15px; margin-bottom: 20px; }
  .stat-card { background: #fff; border-radius: 8px; padding: 15px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; text-align: center; }
  .stat-card .num { font-size: 28px; font-weight: bold; color: #3b82f6; }
  .stat-card .label { font-size: 12px; color: #888; margin-top: 3px; }
  
  .q-card { background: #fff; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
  .q-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: #f0f4ff; border-bottom: 2px solid #e0e7ff; }
  .q-header .left { display: flex; align-items: center; gap: 10px; }
  .q-header .q-id { font-weight: bold; font-size: 16px; color: #1e40af; }
  .q-badge { font-size: 11px; padding: 2px 10px; border-radius: 10px; font-weight: bold; }
  .q-badge.选择 { background: #dbeafe; color: #1e40af; }
  .q-badge.综合 { background: #f3e8ff; color: #6b21a8; }
  .q-badge.分析 { background: #fef3c7; color: #92400e; }
  .q-badge.easy { background: #dcfce7; color: #166534; }
  .q-badge.medium { background: #fef3c7; color: #92400e; }
  .q-badge.hard { background: #fee2e2; color: #991b1b; }
  .q-badge.fail { background: #fee2e2; color: #991b1b; }
  
  .q-body { padding: 15px 20px; }
  .q-image { width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 10px; }
  .q-answer { background: #f0fdf4; border-left: 3px solid #22c55e; padding: 10px 15px; margin: 10px 0; font-size: 13px; }
  .q-points { font-size: 12px; color: #6b7280; margin: 8px 0; }
  .q-points li { margin-left: 20px; margin-bottom: 3px; }
  .q-sub { font-size: 12px; color: #4b5563; margin: 8px 0; }
  .q-sub-item { padding: 5px 10px; margin: 3px 0; background: #f9fafb; border-radius: 4px; }
  .q-meta { font-size: 11px; color: #9ca3af; border-top: 1px dashed #e5e7eb; padding-top: 8px; margin-top: 8px; }
  .q-fail { background: #fef2f2; border-left: 3px solid #ef4444; padding: 10px 15px; margin: 10px 0; font-size: 13px; color: #991b1b; }
</style>
</head>
<body>

<div class="header">
  <h1>习题库提取 v2: 题目截图 + 答案分离</h1>
  <div class="meta">2024秋操作系统期末试题 | 21道题 | 题目截图保留原始排版</div>
</div>

<div class="container">
"""

total = len(questions)
success = sum(1 for q in questions if q.get("answer"))
fail = total - success

html += f"""
  <div class="stats">
    <div class="stat-card"><div class="num">{total}</div><div class="label">总题数</div></div>
    <div class="stat-card"><div class="num">{success}</div><div class="label">答案生成成功</div></div>
    <div class="stat-card"><div class="num">{fail}</div><div class="label">答案生成失败</div></div>
  </div>
"""

for i, q in enumerate(questions, 1):
    q_num = q.get("q_num", i)
    qtype = q.get("type", "?")
    difficulty = q.get("difficulty", "?")
    answer = q.get("answer")
    ref_points = q.get("reference_points", [])
    topic = q.get("topic", "?")
    sub_questions = q.get("sub_questions")
    image = q.get("image", "")
    source = q.get("source", "")

    type_class = "选择" if "选择" in qtype else ("综合" if "综合" in qtype else "分析")

    html += f"""
  <div class="q-card">
    <div class="q-header">
      <div class="left">
        <span class="q-id">Q{q_num}</span>
        <span class="q-badge {type_class}">{qtype}</span>
        <span class="q-badge {difficulty}">{difficulty}</span>
      </div>
      <span style="font-size:12px;color:#9ca3af;">{topic}</span>
    </div>
    <div class="q-body">
      <img class="q-image" src="images/{image}" alt="Q{q_num}">
"""

    if answer and answer != "None":
        html += f'<div class="q-answer"><b>答案:</b> {answer}</div>'
    else:
        html += '<div class="q-fail">答案生成失败</div>'

    if ref_points:
        html += '<div class="q-points"><b>参考要点:</b><ul>'
        for p in ref_points:
            html += f'<li>{p}</li>'
        html += '</ul></div>'

    if sub_questions:
        html += '<div class="q-sub"><b>小问:</b>'
        for sq in sub_questions:
            label = sq.get("label", "")
            ans = sq.get("answer", "")
            html += f'<div class="q-sub-item"><b>{label}</b> {ans}</div>'
        html += '</div>'

    html += f'<div class="q-meta">来源: {source} | 页码: {q.get("page","")} | 图片: {image}</div>'
    html += '</div></div>'

html += """
</div>
</body>
</html>
"""

output_file = OUTPUT_DIR / "compare_report_v2.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"v2 对比报告已生成: {output_file}")
