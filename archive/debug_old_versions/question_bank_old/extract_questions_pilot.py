"""习题库提取脚本（试跑版）

用 2024秋操作系统期末试题（回忆版）试跑完整流程:
  1. PyMuPDF 逐页提取文本 + 检测视觉内容（位图 + 矢量绘图）
  2. 有视觉内容的页面 → 整页渲染 → VLM 描述
  3. 合并文本 + VLM描述 → LLM 按题拆分 + 标注完整性 + 生成参考答案
  4. 输出 JSON 到 debug/

用法:
  python debug/extract_questions_pilot.py
"""

import json
import re
import time
import base64
import requests
from pathlib import Path
import pymupdf
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL

# ============================================================
# 配置
# ============================================================

PDF_PATH = r"data\操作系统\往年题\2024年秋操作系统期末考试试题（回忆版）.pdf"
OUTPUT_DIR = Path("debug/question_bank_pilot")
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

VLM_MODEL = "glm-4v-flash"
LLM_MODEL = "glm-4-flash"

# 矢量绘图数量超过此阈值才做整页渲染（省 API 调用）
VECTOR_DRAW_THRESHOLD = 10


# ============================================================
# 1. 文本提取 + 视觉内容检测
# ============================================================

def extract_page_content(doc, page_num: int) -> dict:
    """提取单页内容：文本 + 视觉信息"""
    page = doc[page_num - 1]

    text = page.get_text()
    images = page.get_images(full=True)
    drawings = page.get_drawings()

    has_bitmap = len(images) > 0
    has_vector = len(drawings) >= VECTOR_DRAW_THRESHOLD
    needs_vlm = has_bitmap or has_vector

    return {
        "page_num": page_num,
        "text": text,
        "bitmap_count": len(images),
        "vector_count": len(drawings),
        "has_bitmap": has_bitmap,
        "has_vector": has_vector,
        "needs_vlm": needs_vlm,
    }


def render_page_to_image(doc, page_num: int, zoom: float = 2.0) -> Image.Image:
    """整页渲染为 PIL Image"""
    page = doc[page_num - 1]
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ============================================================
# 2. VLM 描述页面视觉内容
# ============================================================

def describe_page_vlm(img: Image.Image, page_text: str, page_num: int) -> str:
    """用 VLM 描述页面中的图表/表格/示意图"""
    b64 = image_to_base64(img)

    # 截取页面文字作为上下文（帮助 VLM 理解页面主题）
    context = page_text[:800] if page_text else ""

    system_prompt = """你是试题图片理解助手。请用简洁的中文描述这张试卷页面中的视觉内容，重点提取:
1. 如果有表格，请完整描述表格的行列结构（表头、每行数据）
2. 如果有流程图/状态图/示意图，描述各元素及关系
3. 如果有代码，描述代码结构
4. 不要复述纯文字部分已有的内容

直接输出描述，不要多余的开场白。如果页面没有特殊视觉内容，返回空字符串。"""

    user_text = f"这是试卷第{page_num}页。页面提取到的文字如下：\n{context}\n\n请描述页面中的表格、图表等视觉内容。"

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [VLM 错误] 第{page_num}页: {e}")
            return ""
    return ""


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """PIL Image 转 base64"""
    import io
    buf = io.BytesIO()
    # JPEG 压缩大图
    if img.width * img.height > 500000:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        format = "JPEG"
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# 3. 正则清洗
# ============================================================

def clean_exam_text(text: str) -> str:
    """清洗试卷文本：去页眉、页码等"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # 去页眉
    text = re.sub(r"哈尔滨工业大学.*?第\d+\s*页共\d+\s*页", "", text)
    # 去页码（如 "3 / 9"）
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # 去编者信息
    text = re.sub(r"编者：.*\n参与回忆者：.*", "", text)
    text = re.sub(r"Version:.*", "", text)
    text = re.sub(r"免责声明：.*?(?=\n\n|\n（|$)", "", text, flags=re.DOTALL)

    # 压缩空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 4. LLM 按题拆分
# ============================================================

EXTRACT_PROMPT = """你是试题整理专家。下面是一份操作系统期末考试回忆版试题的文本内容（可能包含 VLM 对页面图片的描述）。

请逐题拆分，输出 JSON 数组。每道题包含以下字段:

1. "question": 题目原文（如果是选择题，包含所有选项）
2. "type": 题型，如 "选择题" / "填空题" / "简答题" / "综合题" / "分析题"
3. "options": 选择题选项字典 {"A": "...", "B": "..."}，非选择题为 null
4. "answer": 标准答案（如果有），选择题填选项字母，简答题填答案要点。回忆版可能没有答案，填 null
5. "reference_points": 参考答案要点数组（即使没有标准答案也尽量根据知识点生成），如 ["进程是资源分配的基本单位", "线程是调度的基本单位"]
6. "topic": 知识点分类，如 "操作系统-进程调度" / "操作系统-死锁" / "操作系统-内存管理"
7. "difficulty": "easy" / "medium" / "hard"
8. "confidence": 完整性标注:
   - "high": 题目完整，选项齐全
   - "medium": 基本完整，个别地方可能有笔误
   - "low": 不完整、回忆者说不确定、或者明确说"忘了原题"
9. "note": 备注（如有笔误说明、替代内容说明等）
10. "sub_questions": 如果是大题含多个小问，拆成 [{"label": "(1)", "question": "..."}] 数组；无小问则为 null

特别注意:
- 如果某段文字是回忆者的碎碎念/复习建议/感受（不是题目），直接跳过
- 如果某段文字说"忘了考试原题，此处贴上作业题"，标记 confidence 为 "low"，note 字段说明
- 如果选择题选项有明显笔误（如"C...n = 20"），在 note 中指出并尽量修正
- 表格类题目：如果 VLM 描述了表格结构，把表格数据整合到 question 中
- 输出纯 JSON 数组，不要加 markdown 代码块标记

请输出 JSON:"""


def extract_questions_with_llm(full_text: str) -> list[dict]:
    """用 LLM 从试卷文本中提取题目"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"以下是试卷内容:\n\n{full_text}\n\n请按题拆分并输出 JSON。"},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # 去掉可能的 markdown 代码块标记
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            questions = json.loads(content)
            return questions
        except json.JSONDecodeError as e:
            print(f"  [JSON 解析失败] {e}")
            if attempt < 2:
                print("  重试...")
                time.sleep(2)
                continue
            # 最后一次失败，尝试手动保存原始输出
            (OUTPUT_DIR / "raw_llm_output.txt").write_text(content, encoding="utf-8")
            print(f"  原始输出已保存到 {OUTPUT_DIR / 'raw_llm_output.txt'}")
            return []
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [LLM 错误] {e}")
            return []
    return []


# ============================================================
# 主流程
# ============================================================

def main():
    pdf_name = Path(PDF_PATH).stem
    print(f"试跑: {pdf_name}")
    print("=" * 60)

    doc = pymupdf.open(PDF_PATH)
    total_pages = doc.page_count
    print(f"总页数: {total_pages}")

    # 第1步：逐页提取文本 + 检测视觉内容
    print("\n--- 第1步: 提取文本 + 检测视觉内容 ---")
    pages = []
    for i in range(1, total_pages + 1):
        pc = extract_page_content(doc, i)
        pages.append(pc)
        vlm_flag = "需VLM" if pc["needs_vlm"] else "纯文本"
        print(f"  第{i:2d}页: {pc['text'][:40].strip()[:40]:40s} | "
              f"位图={pc['bitmap_count']} 矢量={pc['vector_count']} | {vlm_flag}")

    # 第2步：有视觉内容的页面 → 整页渲染 + VLM
    print("\n--- 第2步: VLM 描述视觉内容 ---")
    vlm_descriptions = {}
    pages_need_vlm = [p for p in pages if p["needs_vlm"]]
    print(f"需要 VLM 处理的页面: {len(pages_need_vlm)} 页")

    for p in pages_need_vlm:
        pg = p["page_num"]
        print(f"  渲染第{pg}页...", end=" ")
        img = render_page_to_image(doc, pg)

        # 保存图片（方便人工检查）
        img_path = IMAGE_DIR / f"{pdf_name}_p{pg}.png"
        img.save(str(img_path))
        print(f"已保存 {img_path.name}")

        print(f"  VLM 描述第{pg}页...", end=" ")
        desc = describe_page_vlm(img, p["text"], pg)
        vlm_descriptions[pg] = desc
        print(f"({len(desc)} 字)")
        if desc:
            print(f"    预览: {desc[:100]}...")

        time.sleep(2)  # 避免 429

    doc.close()

    # 第3步：合并文本 + VLM描述 → 清洗 → 喂给 LLM
    print("\n--- 第3步: 合并 + 清洗 → LLM 拆题 ---")
    merged_text = ""
    for p in pages:
        pg = p["page_num"]
        merged_text += f"\n\n===== 第{pg}页 =====\n"
        merged_text += p["text"]
        if pg in vlm_descriptions and vlm_descriptions[pg]:
            merged_text += f"\n\n[VLM 页面图片描述]\n{vlm_descriptions[pg]}"

    cleaned_text = clean_exam_text(merged_text)
    print(f"合并后文本: {len(cleaned_text)} 字符")

    # 保存合并文本（方便调试）
    (OUTPUT_DIR / "merged_text.txt").write_text(cleaned_text, encoding="utf-8")
    print(f"合并文本已保存到 {OUTPUT_DIR / 'merged_text.txt'}")

    print("\n调用 LLM 拆题中...", end=" ")
    questions = extract_questions_with_llm(cleaned_text)
    print(f"提取到 {len(questions)} 道题")

    # 给每道题加来源信息
    for q in questions:
        q["source"] = "2024秋操作系统期末试题（回忆版）"
        q["source_file"] = Path(PDF_PATH).name

    # 保存结果
    output_file = OUTPUT_DIR / "questions_2024秋.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"完成！结果已保存到 {output_file}")
    print(f"题目数: {len(questions)}")

    # 打印摘要
    if questions:
        print(f"\n题目摘要:")
        print(f"{'ID':>3} | {'类型':<8} | {'置信度':<8} | {'难度':<6} | 题目预览")
        print("-" * 80)
        for i, q in enumerate(questions, 1):
            preview = q.get("question", "")[:50].replace("\n", " ")
            print(f"{i:3d} | {q.get('type', '?'):<8} | "
                  f"{q.get('confidence', '?'):<8} | "
                  f"{q.get('difficulty', '?'):<6} | "
                  f"{preview}")

        # 统计
        high = sum(1 for q in questions if q.get("confidence") == "high")
        medium = sum(1 for q in questions if q.get("confidence") == "medium")
        low = sum(1 for q in questions if q.get("confidence") == "low")
        print(f"\n置信度分布: high={high}, medium={medium}, low={low}")


if __name__ == "__main__":
    main()
