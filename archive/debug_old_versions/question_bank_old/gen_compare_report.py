"""生成试卷原始文本 vs 提取结果对比报告（HTML）"""

import json
from pathlib import Path

OUTPUT_DIR = Path("debug/question_bank_pilot")

# 读取合并文本（原始提取）
merged_text = (OUTPUT_DIR / "merged_text.txt").read_text(encoding="utf-8")

# 读取提取结果
with open(OUTPUT_DIR / "questions_2024秋.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 读取 VLM 描述（从图片文件名推断哪些页做了VLM）
image_dir = OUTPUT_DIR / "images"
vlm_pages = []
if image_dir.exists():
    for f in sorted(image_dir.glob("*.png")):
        # 文件名格式: 2024年秋..._p1.png
        page_num = f.stem.split("_p")[-1]
        vlm_pages.append(int(page_num))

# 生成 HTML
html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>试卷提取对比报告 - 2024秋操作系统</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: #fff; padding: 20px 30px; }
  .header h1 { font-size: 22px; }
  .header .meta { font-size: 13px; color: #aaa; margin-top: 5px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .stats { display: flex; gap: 15px; margin-bottom: 20px; }
  .stat-card { background: #fff; border-radius: 8px; padding: 15px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; }
  .stat-card .num { font-size: 28px; font-weight: bold; }
  .stat-card .label { font-size: 12px; color: #888; margin-top: 3px; }
  .stat-card.high .num { color: #22c55e; }
  .stat-card.medium .num { color: #f59e0b; }
  .stat-card.low .num { color: #ef4444; }
  .stat-card.total .num { color: #3b82f6; }
  
  .section { background: #fff; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
  .section-header { background: #f0f4ff; padding: 12px 20px; border-bottom: 2px solid #e0e7ff; }
  .section-header h2 { font-size: 16px; color: #1e40af; }
  
  .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  .compare-left, .compare-right { padding: 15px 20px; }
  .compare-left { border-right: 1px solid #e5e7eb; background: #fffbeb; }
  .compare-right { background: #f0fdf4; }
  .compare-label { font-size: 12px; font-weight: bold; color: #6b7280; margin-bottom: 8px; text-transform: uppercase; }
  
  .raw-text { font-family: "Courier New", monospace; font-size: 12px; white-space: pre-wrap; word-break: break-all; line-height: 1.6; max-height: 400px; overflow-y: auto; }
  
  .q-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin-bottom: 10px; background: #fff; }
  .q-card .q-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .q-card .q-id { font-weight: bold; color: #1e40af; }
  .q-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: bold; }
  .q-badge.high { background: #dcfce7; color: #166534; }
  .q-badge.medium { background: #fef3c7; color: #92400e; }
  .q-badge.low { background: #fee2e2; color: #991b1b; }
  .q-badge.选择 { background: #dbeafe; color: #1e40af; }
  .q-badge.综合 { background: #f3e8ff; color: #6b21a8; }
  .q-question { font-size: 13px; line-height: 1.6; margin-bottom: 8px; white-space: pre-wrap; }
  .q-options { font-size: 12px; color: #555; margin-left: 15px; margin-bottom: 5px; }
  .q-option { margin-bottom: 2px; }
  .q-answer { font-size: 12px; color: #059669; margin-top: 5px; }
  .q-points { font-size: 11px; color: #6b7280; margin-top: 5px; }
  .q-points li { margin-left: 15px; }
  .q-sub { font-size: 12px; color: #4b5563; margin-left: 15px; margin-top: 5px; }
  .q-note { font-size: 11px; color: #dc2626; margin-top: 5px; }
  .q-meta { font-size: 11px; color: #9ca3af; margin-top: 5px; border-top: 1px dashed #e5e7eb; padding-top: 5px; }
  
  .vlm-info { background: #fff; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .vlm-info h2 { font-size: 16px; color: #1e40af; margin-bottom: 10px; }
  .vlm-list { font-size: 13px; }
  .vlm-item { padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
  .vlm-item img { max-width: 300px; border: 1px solid #e5e7eb; border-radius: 4px; margin: 5px 0; }
  
  .missing { background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 12px; margin-bottom: 10px; }
  .missing h3 { font-size: 14px; color: #dc2626; }
</style>
</head>
<body>

<div class="header">
  <h1>试卷提取对比报告</h1>
  <div class="meta">2024秋操作系统期末考试试题（回忆版） | 10页 | 提取到 15 道题</div>
</div>

<div class="container">
"""

# 统计卡片
high_count = sum(1 for q in questions if q.get("confidence") == "high")
medium_count = sum(1 for q in questions if q.get("confidence") == "medium")
low_count = sum(1 for q in questions if q.get("confidence") == "low")
choice_count = sum(1 for q in questions if q.get("type") == "选择题")
comp_count = sum(1 for q in questions if q.get("type") == "综合题")

html += f"""
  <div class="stats">
    <div class="stat-card total"><div class="num">{len(questions)}</div><div class="label">总题数</div></div>
    <div class="stat-card"><div class="num">{choice_count}</div><div class="label">选择题</div></div>
    <div class="stat-card"><div class="num">{comp_count}</div><div class="label">综合题</div></div>
    <div class="stat-card high"><div class="num">{high_count}</div><div class="label">High 置信度</div></div>
    <div class="stat-card medium"><div class="num">{medium_count}</div><div class="label">Medium 置信度</div></div>
    <div class="stat-card low"><div class="num">{low_count}</div><div class="label">Low 置信度</div></div>
  </div>
"""

# 缺失题目提示
html += """
  <div class="missing">
    <h3>! 缺失: 选择题第10-20题（共11道）因 API 500 错误未提取</h3>
    <p style="font-size:12px;color:#666;margin-top:5px;">原始试卷有20道选择题，当前只提取到前9道。需要重试。</p>
  </div>
"""

# VLM 处理信息
html += """
  <div class="vlm-info">
    <h2>VLM 处理的页面</h2>
    <div class="vlm-list">
"""
if vlm_pages:
    for pg in sorted(vlm_pages):
        img_path = f"images/2024年秋操作系统期末考试试题（回忆版）_p{pg}.png"
        html += f'      <div class="vlm-item">第 {pg} 页 <img src="{img_path}" alt="page {pg}"></div>\n'
html += """    </div>
  </div>
"""

# 对比: 原始文本 vs 提取结果
html += """
  <div class="section">
    <div class="section-header"><h2>对比: 原始提取文本 vs LLM 结构化结果</h2></div>
    <div class="compare">
      <div class="compare-left">
        <div class="compare-label">原始文本 (PyMuPDF 提取 + VLM 描述)</div>
        <div class="raw-text">"""

# 转义HTML
raw_escaped = merged_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
html += raw_escaped

html += """</div>
      </div>
      <div class="compare-right">
        <div class="compare-label">LLM 提取结果 (结构化 JSON)</div>
"""

# 逐题展示
for i, q in enumerate(questions, 1):
    qtype = q.get("type", "?")
    conf = q.get("confidence", "?")
    question = q.get("question", "")
    options = q.get("options")
    answer = q.get("answer")
    ref_points = q.get("reference_points", [])
    topic = q.get("topic", "?")
    difficulty = q.get("difficulty", "?")
    note = q.get("note")
    sub_questions = q.get("sub_questions")
    source = q.get("source", "")

    badge_class = "选择" if qtype == "选择题" else "综合"

    html += f"""
        <div class="q-card">
          <div class="q-header">
            <span class="q-id">Q{i}</span>
            <span class="q-badge {badge_class}">{qtype}</span>
            <span class="q-badge {conf}">{conf}</span>
          </div>
          <div class="q-question">{question}</div>"""

    if options:
        html += '<div class="q-options">'
        for k, v in sorted(options.items()):
            html += f'<div class="q-option"><b>{k}.</b> {v}</div>'
        html += '</div>'

    if answer:
        answer_str = str(answer)
        html += f'<div class="q-answer">答案: {answer_str}</div>'

    if ref_points:
        html += '<div class="q-points">参考要点:<ul>'
        for p in ref_points:
            html += f'<li>{p}</li>'
        html += '</ul></div>'

    if sub_questions:
        html += '<div class="q-sub">小问:'
        for sq in sub_questions:
            html += f'<br>{sq.get("label","")} {sq.get("question","")}'
        html += '</div>'

    if note:
        html += f'<div class="q-note">备注: {note}</div>'

    html += f'<div class="q-meta">topic: {topic} | difficulty: {difficulty} | source: {source}</div>'
    html += '</div>'

html += """
      </div>
    </div>
  </div>
</div>

</body>
</html>
"""

output_file = OUTPUT_DIR / "compare_report.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"对比报告已生成: {output_file}")
print(f"用浏览器打开即可查看")
