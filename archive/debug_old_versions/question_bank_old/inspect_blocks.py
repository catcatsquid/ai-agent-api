"""检查 PDF 文本块坐标，验证题目截图可行性"""

import pymupdf
from pathlib import Path

doc = pymupdf.open(r"data\操作系统\往年题\2024年秋操作系统期末考试试题（回忆版）.pdf")

# 看前3页的文本块
for pg in range(min(3, doc.page_count)):
    page = doc[pg]
    blocks = page.get_text("dict")["blocks"]
    print(f"\n=== 第{pg+1}页 (共{len(blocks)}个block) ===")
    page_rect = page.rect
    print(f"  页面尺寸: {page_rect.width:.0f} x {page_rect.height:.0f}")

    for i, b in enumerate(blocks):
        if b["type"] == 0:  # text
            bbox = b["bbox"]
            text = ""
            for line in b["lines"]:
                for span in line["spans"]:
                    text += span["text"]
            # 只显示有意义的文本块
            if text.strip():
                y0 = bbox[1]
                y1 = bbox[3]
                print(f"  [{i:2d}] y={y0:.0f}-{y1:.0f} x={bbox[0]:.0f}-{bbox[2]:.0f} | {text[:80]}")
        elif b["type"] == 1:  # image
            bbox = b["bbox"]
            print(f"  [{i:2d}] [IMAGE] y={bbox[1]:.0f}-{bbox[3]:.0f} x={bbox[0]:.0f}-{bbox[2]:.0f}")

doc.close()

# 验证：搜索题号来定位题目边界
print("\n\n=== 题号定位测试 ===")
doc = pymupdf.open(r"data\操作系统\往年题\2024年秋操作系统期末考试试题（回忆版）.pdf")
page = doc[0]

# 搜索题号模式
import re
text_dict = page.get_text("dict")
for b in text_dict["blocks"]:
    if b["type"] == 0:
        for line in b["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text += span["text"]
            # 找选择题题号
            if re.match(r"^\s*(\d+)\.", line_text.strip()):
                bbox = line["bbox"]
                print(f"  题号: {line_text.strip()[:30]} | y={bbox[1]:.0f} x={bbox[0]:.0f}")

doc.close()
