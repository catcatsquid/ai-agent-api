"""生成 README 用的架构示意图（Pillow 画）

设计原则：只画主链路，不放实现细节。改布局只改坐标常量。
用法: python debug/make_arch_diagram.py
输出: docs/screenshots/architecture.png
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 540
FONT_DIR = Path("C:/Windows/Fonts")


def _font(size, bold=False):
    for name in (("msyhbd.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")):
        p = FONT_DIR / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size, index=0)
            except OSError:
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_H = _font(19, True)
F_B = _font(16)
F_S = _font(14)

img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)


def box(x, y, w, h, title, lines, fill, edge):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=edge, width=2)
    d.text((x + 14, y + 12), title, font=F_H, fill=edge)
    ty = y + 42
    for ln in lines:
        d.text((x + 14, ty), ln, font=F_B, fill="#3C4043")
        ty += 22


def arrow(x1, y1, x2, y2, color="#5F6368", label="", lx=0, ly=0, dashed=False):
    if dashed:
        n = max(int(math.hypot(x2 - x1, y2 - y1) // 14), 1)
        for i in range(0, n, 2):
            a, b = i / n, min((i + 1) / n, 1.0)
            d.line([x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                    x1 + (x2 - x1) * b, y1 + (y2 - y1) * b], fill=color, width=2)
    else:
        d.line([x1, y1, x2, y2], fill=color, width=2)
    ang = math.atan2(y2 - y1, x2 - x1)
    for s in (1, -1):
        d.line([x2, y2, x2 - 11 * math.cos(ang - s * 0.4), y2 - 11 * math.sin(ang - s * 0.4)],
               fill=color, width=2)
    if label:
        d.text((lx, ly), label, font=F_S, fill=color)


# 第一行：请求链
box(30, 90, 250, 96, "浏览器", ["chat.html 对话", "kb.html 知识库"], "#E8F0FE", "#1A73E8")
box(330, 90, 250, 96, "FastAPI", ["路由 · 中间件", "SSE 五类事件"], "#FFF4E5", "#B06000")
box(630, 90, 280, 96, "LangGraph 编排", ["Plan（默认）/ ReAct", "各一套 checkpointer"], "#E6F4EA", "#137333")
box(960, 90, 250, 96, "工具层", ["17 个 @tool", "Function Calling"], "#F3E8FD", "#7C3AED")

# 第二行：工具背后的两条业务线
box(630, 260, 280, 96, "检索  rag_tool.py", ["混合检索 + rerank", "两级回退"], "#E8F5F9", "#0288D1")
box(960, 260, 250, 96, "题库  qbank.py", ["按课程索引", "分组 · 批改 · 讲题"], "#FCE8E6", "#C5221F")

# 第三行：数据与模型
box(330, 420, 880, 90, "数据与模型",
    ["ChromaDB + BM25  ·  题库 JSON + 题图  ·  智谱 GLM（对话 / 向量 / 图表）  ·  SQLite checkpointer"],
    "#F1F3F4", "#5F6368")

# 第四行：离线
box(30, 420, 250, 90, "离线建库", ["build_knowledge.py", "knowledge_pipeline/"], "#EDE7F6", "#5E35B1")

# 箭头
arrow(280, 138, 330, 138)
arrow(580, 138, 630, 138)
arrow(910, 138, 960, 138)
arrow(1085, 186, 1085, 260)
arrow(770, 186, 770, 260)
arrow(770, 356, 770, 420)
arrow(1085, 356, 1085, 420)
arrow(280, 465, 330, 465, color="#5E35B1", label="入库", lx=286, ly=436)

img.save("docs/screenshots/architecture.png")
print("saved", img.size)