"""习题库构建模块

从往年试卷 PDF 中提取题目，生成结构化习题库。

流程:
  1. PyMuPDF 定位题号 → 确定每道题的页面坐标范围
  2. 截取题目区域为图片（保留排版、表格、图表）
  3. 有图表/表格的题 → VLM 描述 → LLM 生成答案
     纯文字题 → LLM 直接生成答案
  4. 输出 JSON + 对比报告 HTML

用法:
  from course_agent.question_bank import build_question_bank
  build_question_bank("data/操作系统/往年题/xxx.pdf")

  或命令行:
  python -m course_agent.question_bank data/操作系统/往年题/
"""

import json
import re
import time
import base64
import io
import requests
from pathlib import Path
import pymupdf
from PIL import Image

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL

VLM_MODEL = "glm-4v-flash"
LLM_MODEL = "glm-4-flash"

DATA_DIR = Path("data")
QUESTION_BANK_DIR = DATA_DIR / "question_bank"
IMAGE_DIR = QUESTION_BANK_DIR / "images"

PAGE_HEADER_Y = 60
PAGE_FOOTER_Y = 780
VECTOR_DRAW_THRESHOLD = 3

COURSE_DIRS = {
    "操作系统": DATA_DIR / "操作系统" / "往年题",
    "计算机网络": DATA_DIR / "计算机网络" / "往年题",
}

COURSE_PROMPTS = {
    "操作系统": (
        "你是操作系统课程教师。下面是一道操作系统考试题的文字内容。\n"
        "请生成参考答案并分类。输出 JSON 对象，包含以下字段:\n\n"
        '1. "type": 题型 "选择题" / "简答题" / "综合题" / "分析题"\n'
        '2. "answer": 标准答案。\n'
        "   - 选择题：只填选项字母，如 \"A\"、\"B\"、\"C\"、\"D\"。不要填文字内容\n"
        "   - 综合题/简答题：填 null（答案统一放 sub_questions 里）\n"
        '3. "reference_points": 参考答案要点数组，每条是一句完整的话\n'
        '4. "topic": 知识点分类\n'
        '5. "sub_questions": \n'
        "   - 综合题/大题：必须拆成 [{\"label\": \"(1)\", \"answer\": \"完整答案\"}]，即使只有1个小问也要拆\n"
        "   - 选择题：填 null\n\n"
        "注意:\n"
        "- 只输出 JSON，不要加 markdown 代码块\n"
        "- 如果文字不完整或有乱码，根据已有信息尽力作答\n"
        "- 综合题的 answer 字段必须为 null，答案内容全部放在 sub_questions 里"
    ),
    "计算机网络": (
        "你是计算机网络课程教师。下面是一道计算机网络考试题的文字内容。\n"
        "请生成参考答案并分类。输出 JSON 对象，包含以下字段:\n\n"
        '1. "type": 题型 "选择题" / "简答题" / "综合题" / "分析题"\n'
        '2. "answer": 标准答案。\n'
        "   - 选择题：只填选项字母，如 \"A\"、\"B\"、\"C\"、\"D\"。不要填文字内容\n"
        "   - 综合题/简答题：填 null（答案统一放 sub_questions 里）\n"
        '3. "reference_points": 参考答案要点数组，每条是一句完整的话\n'
        '4. "topic": 知识点分类\n'
        '5. "sub_questions": \n'
        "   - 综合题/大题：必须拆成 [{\"label\": \"(1)\", \"answer\": \"完整答案\"}]，即使只有1个小问也要拆\n"
        "   - 选择题：填 null\n\n"
        "注意:\n"
        "- 只输出 JSON，不要加 markdown 代码块\n"
        "- 如果文字不完整或有乱码，根据已有信息尽力作答\n"
        "- 综合题的 answer 字段必须为 null，答案内容全部放在 sub_questions 里"
    ),
}


# ============================================================
# 1. 定位题目边界
# ============================================================

def locate_questions(doc) -> tuple[list[dict], list[dict]]:
    """扫描所有页，定位每道题的位置

    Returns:
        questions: [{"q_num", "page", "y_start", "y_end", "section", "is_comprehensive", "display_num"}]
        sections: [{"title", "page", "y"}]
    """
    section_pattern = re.compile(r"^[一二三四五六七八九十]+、")
    # 题号：1. 2. 10. 等，排除 "1.0" "0.4" 等小数格式
    q_pattern = re.compile(r"^(\d+)[.．、](?!\d)")

    all_lines = []

    for pg in range(doc.page_count):
        page = doc[pg]
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                line_text = line_text.strip()
                if not line_text:
                    continue

                bbox = line["bbox"]
                if bbox[1] < PAGE_HEADER_Y or bbox[1] > PAGE_FOOTER_Y:
                    continue

                all_lines.append({
                    "text": line_text,
                    "page": pg,
                    "y": bbox[1],
                    "y_end": bbox[3],
                    "x": bbox[0],
                })

    # 识别大题段落
    sections = []
    for ln in all_lines:
        if section_pattern.match(ln["text"]):
            sections.append({
                "title": ln["text"][:60],
                "page": ln["page"],
                "y": ln["y"],
            })

    # 识别小题号
    questions = []
    for ln in all_lines:
        m = q_pattern.match(ln["text"])
        if m:
            q_num = int(m.group(1))
            if q_num == 0:
                continue

            current_section = None
            for s in sections:
                if s["page"] < ln["page"] or (s["page"] == ln["page"] and s["y"] < ln["y"]):
                    current_section = s
                else:
                    break

            questions.append({
                "q_num": q_num,
                "text": ln["text"][:80],
                "page": ln["page"],
                "y_start": ln["y"],
                "section": current_section["title"] if current_section else "未知",
            })

    # 去重
    seen = set()
    deduped = []
    for q in questions:
        key = (q["page"], q["q_num"], q["y_start"] // 10)
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    questions = deduped

    # 计算每题结束位置
    all_items = []
    for s in sections:
        all_items.append({"page": s["page"], "y": s["y"], "type": "section"})
    for q in questions:
        all_items.append({"page": q["page"], "y": q["y_start"], "type": "question", "q": q})
    all_items.sort(key=lambda x: (x["page"], x["y"]))

    for i, item in enumerate(all_items):
        if item["type"] != "question":
            continue
        q = item["q"]
        q["y_end"] = PAGE_FOOTER_Y
        q["end_page"] = q["page"]

        if i + 1 < len(all_items):
            next_item = all_items[i + 1]
            if next_item["page"] == q["page"]:
                q["y_end"] = next_item["y"] - 3
            else:
                q["y_end"] = PAGE_FOOTER_Y
                if next_item["page"] == q["page"] + 1:
                    q["end_page"] = next_item["page"]
                    q["y_end_next_page"] = next_item["y"] - 3
                else:
                    q["end_page"] = q["page"]

    # 区分选择题和综合题
    comp_counter = 0
    for q in questions:
        if "综合" in q["section"]:
            comp_counter += 1
            q["is_comprehensive"] = True
            q["display_num"] = comp_counter
        else:
            q["is_comprehensive"] = False
            q["display_num"] = q["q_num"]

    return questions, sections


# ============================================================
# 2. 截取题目区域
# ============================================================

def trim_whitespace(img: Image.Image, threshold: int = 245) -> Image.Image:
    """裁剪图片四周的白色留白"""
    gray = img.convert("L")
    inverted = gray.point(lambda x: 255 if x < threshold else 0)
    bbox = inverted.getbbox()
    if bbox:
        pad = 5
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(img.width, bbox[2] + pad)
        bottom = min(img.height, bbox[3] + pad)
        img = img.crop((left, top, right, bottom))
    return img


def crop_question(doc, q: dict, pdf_name: str, image_dir: Path) -> dict:
    """截取题目区域，支持跨页拼接"""
    page = doc[q["page"]]
    page_rect = page.rect

    clip = pymupdf.Rect(
        80,
        max(0, q["y_start"] - 3),
        page_rect.width - 60,
        min(page_rect.height, q["y_end"] + 3),
    )

    zoom = 2.5
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip)

    if q.get("end_page", q["page"]) != q["page"]:
        page2 = doc[q["end_page"]]
        clip2 = pymupdf.Rect(
            80,
            PAGE_HEADER_Y,
            page2.rect.width - 60,
            min(page2.rect.height, q.get("y_end_next_page", 400) + 3),
        )
        pix2 = page2.get_pixmap(matrix=mat, clip=clip2)

        img1 = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
        total_height = img1.height + img2.height + 5
        img = Image.new("RGB", (max(img1.width, img2.width), total_height), "white")
        img.paste(img1, (0, 0))
        img.paste(img2, (0, img1.height + 5))
    else:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    img = trim_whitespace(img)

    if q["is_comprehensive"]:
        qid = f"S{q['display_num']}"
    else:
        qid = f"Q{q['display_num']}"

    img_path = image_dir / f"{pdf_name}_{qid}_p{q['page']+1}.png"
    img.save(str(img_path), optimize=True)

    # 提取文字
    text = page.get_text("text", clip=clip)
    if q.get("end_page", q["page"]) != q["page"]:
        page2 = doc[q["end_page"]]
        clip2 = pymupdf.Rect(80, PAGE_HEADER_Y, page2.rect.width - 60, q.get("y_end_next_page", 400) + 3)
        text2 = page2.get_text("text", clip=clip2)
        text = text + "\n" + text2

    return {
        "qid": qid,
        "q_num": q["q_num"],
        "is_comprehensive": q["is_comprehensive"],
        "image_path": str(img_path),
        "page": q["page"] + 1,
        "end_page": q.get("end_page", q["page"]) + 1,
        "cross_page": q.get("end_page", q["page"]) != q["page"],
        "text": text.strip(),
        "section": q["section"],
    }


# ============================================================
# 3. 检测是否需要 VLM
# ============================================================

def has_visual_content(q: dict, doc) -> bool:
    """判断题目区域是否有图表/表格"""
    page = doc[q["page"]]
    clip = pymupdf.Rect(80, max(0, q["y_start"] - 5),
                        page.rect.width - 60, min(page.rect.height, q["y_end"] + 5))

    drawings = page.get_drawings()
    visual_count = 0
    for d in drawings:
        rect = d["rect"]
        if (rect.y0 >= clip.y0 - 5 and rect.y1 <= clip.y1 + 5 and
                rect.x0 >= clip.x0 - 5 and rect.x1 <= clip.x1 + 5):
            visual_count += 1

    return visual_count >= VECTOR_DRAW_THRESHOLD


# ============================================================
# 4. VLM + LLM 生成答案
# ============================================================

def _detect_course(pdf_path: Path) -> str:
    """从文件路径推断课程名称"""
    for course, dir_path in COURSE_DIRS.items():
        if course in str(pdf_path):
            return course
    return "操作系统"


def _image_to_base64(img: Image.Image, max_size: int = 800) -> str:
    ratio = max_size / max(img.width, img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _vlm_describe_question(img: Image.Image, text: str) -> str:
    b64 = _image_to_base64(img)
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "这是试卷中一道有表格或图表的题目截图。"
        "请描述图片中的表格结构（表头、每行数据）和图表内容。"
        "不要复述纯文字部分。直接输出描述。"
    )
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"题目文字:\n{text[:300]}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
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
            return ""
    return ""


def _llm_generate_answer(text: str, vlm_desc: str = "", course: str = "操作系统") -> dict:
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    user_content = f"以下是题目文字:\n{text}"
    if vlm_desc:
        user_content += f"\n\n[VLM 图片描述]\n{vlm_desc}"
    user_content += "\n\n请生成参考答案并分类。"

    prompt = COURSE_PROMPTS.get(course, COURSE_PROMPTS["操作系统"])
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            result = data["choices"][0]["message"]["content"].strip()
            result = re.sub(r"^```(?:json)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(result)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
                continue
            return {}
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if attempt < 2:
                time.sleep(3)
                continue
            return {}
    return {}


def _check_answer_quality(qd: dict, answer_data: dict) -> tuple[bool, str]:
    """检查答案格式是否合规，返回 (needs_review, reason)"""
    if not answer_data:
        return True, "答案生成失败"

    if qd["is_comprehensive"]:
        reason = ""
        if answer_data.get("answer") is not None:
            reason = "综合题 answer 不为 null"
        if not answer_data.get("sub_questions"):
            reason = (reason + "; " if reason else "") + "综合题缺少 sub_questions"
        return (bool(reason), reason) if reason else (False, "")
    else:
        ans_val = str(answer_data.get("answer", "")).strip()
        if not re.match(r"^[A-D]$", ans_val):
            return True, f"选择题答案不是单字母A-D: '{ans_val[:20]}'"
        return (False, "")


# ============================================================
# 5. 生成对比报告
# ============================================================

def _vlm_extract_page(img: Image.Image) -> str:
    """用VLM从纯图片页面提取文字内容"""
    b64 = _image_to_base64(img, max_size=1200)
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "这是考试试卷的一页截图。请提取页面中的所有文字内容，"
        "包括题号、题目、选项、表格内容等。"
        "保持原始格式和编号。如果有表格，转换为文本格式。"
        "直接输出提取的文字内容，不要解释。"
    )
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return ""
    return ""


def _llm_split_questions(page_text: str, course: str) -> list[dict]:
    """用LLM把整页文字拆成独立题目"""
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "你是试卷题目拆分器。下面是从试卷一页提取的文字。"
        "请把文字拆分成独立题目，每道题包含题号和完整内容。"
        '输出 JSON 数组，每个元素: {"q_num": 题号数字, "text": "题目完整文字", "is_comprehensive": true/false}\n'
        "判断标准：选择题/填空题 is_comprehensive=false，大题/综合题/计算题 is_comprehensive=true\n"
        "只输出 JSON 数组，不要解释。"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": page_text[:3000]},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            result = re.sub(r"^```(?:json)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            arr_match = re.search(r"\[.*\]", result, re.DOTALL)
            if arr_match:
                return json.loads(arr_match.group())
            return json.loads(result)
        except (json.JSONDecodeError, Exception):
            if attempt < 2:
                time.sleep(2)
                continue
            return [{"q_num": 1, "text": page_text, "is_comprehensive": True}]
    return [{"q_num": 1, "text": page_text, "is_comprehensive": True}]


def _process_image_only_pdf(doc, pdf_name: str, image_dir: Path, course: str) -> tuple[list[dict], list[dict]]:
    """处理纯图片PDF：逐页VLM提取文字 → LLM拆题 → 截图

    Returns:
        questions: [{"q_num", "page", "is_comprehensive", "section", "display_num"}]
        q_data: [{"qid", "q_num", "image_path", "text", "page", "is_comprehensive", "section"}]
    """
    questions = []
    q_data = []
    q_counter = 0
    comp_counter = 0

    for pg in range(doc.page_count):
        page = doc[pg]
        page_rect = page.rect
        # 整页截图
        zoom = 2.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = trim_whitespace(img)

        print(f"  第{pg+1}页: VLM提取...", end=" ")
        page_text = _vlm_extract_page(img)
        print(f"{len(page_text)}字")

        if not page_text.strip():
            continue

        # LLM拆题
        split_qs = _llm_split_questions(page_text, course)

        for sq in split_qs:
            q_counter += 1
            is_comp = sq.get("is_comprehensive", True)
            if is_comp:
                comp_counter += 1
                display_num = comp_counter
                qid = f"S{display_num}"
            else:
                display_num = q_counter
                qid = f"Q{display_num}"

            # 保存整页截图作为题目图片
            img_path = image_dir / f"{pdf_name}_{qid}_p{pg+1}.png"
            img.save(str(img_path), optimize=True)

            questions.append({
                "q_num": sq.get("q_num", q_counter),
                "page": pg,
                "is_comprehensive": is_comp,
                "section": "综合" if is_comp else "选择",
                "display_num": display_num,
            })
            q_data.append({
                "qid": qid,
                "q_num": sq.get("q_num", q_counter),
                "display_num": display_num,
                "is_comprehensive": is_comp,
                "image_path": str(img_path),
                "text": sq.get("text", page_text),
                "page": pg + 1,
                "section": "综合" if is_comp else "选择",
            })

    return questions, q_data


def generate_report(questions: list[dict], output_dir: Path, image_dir: Path, pdf_name: str):
    """生成 HTML 对比报告"""
    total = len(questions)
    success = sum(1 for q in questions if q.get("answer") or q.get("sub_questions"))
    review = sum(1 for q in questions if q.get("needs_review"))

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>习题库 - {pdf_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }}
  .header {{ background: #1a1a2e; color: #fff; padding: 20px 30px; }}
  .header h1 {{ font-size: 22px; }}
  .header .meta {{ font-size: 13px; color: #aaa; margin-top: 5px; }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
  .stats {{ display: flex; gap: 15px; margin-bottom: 20px; }}
  .stat-card {{ background: #fff; border-radius: 8px; padding: 15px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; text-align: center; }}
  .stat-card .num {{ font-size: 28px; font-weight: bold; color: #3b82f6; }}
  .stat-card .label {{ font-size: 12px; color: #888; margin-top: 3px; }}
  .q-card {{ background: #fff; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
  .q-header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: #f0f4ff; border-bottom: 2px solid #e0e7ff; }}
  .q-header .left {{ display: flex; align-items: center; gap: 10px; }}
  .q-header .q-id {{ font-weight: bold; font-size: 16px; color: #1e40af; }}
  .q-badge {{ font-size: 11px; padding: 2px 10px; border-radius: 10px; font-weight: bold; }}
  .q-badge.选择题 {{ background: #dbeafe; color: #1e40af; }}
  .q-badge.综合题 {{ background: #f3e8ff; color: #6b21a8; }}
  .q-badge.review {{ background: #fee2e2; color: #991b1b; }}
  .q-body {{ padding: 15px 20px; }}
  .q-image {{ width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 10px; }}
  .q-answer {{ background: #f0fdf4; border-left: 3px solid #22c55e; padding: 10px 15px; margin: 10px 0; font-size: 13px; white-space: pre-wrap; }}
  .q-points {{ font-size: 12px; color: #6b7280; margin: 8px 0; }}
  .q-points li {{ margin-left: 20px; margin-bottom: 3px; }}
  .q-sub-item {{ padding: 5px 10px; margin: 3px 0; background: #f9fafb; border-radius: 4px; font-size: 13px; }}
  .q-meta {{ font-size: 11px; color: #9ca3af; border-top: 1px dashed #e5e7eb; padding-top: 8px; margin-top: 8px; }}
  .q-fail {{ background: #fef2f2; border-left: 3px solid #ef4444; padding: 10px 15px; margin: 10px 0; font-size: 13px; color: #991b1b; }}
</style>
</head>
<body>
<div class="header">
  <h1>习题库: {pdf_name}</h1>
  <div class="meta">{total} 道题 | {success} 答案成功 | {review} 需审查</div>
</div>
<div class="container">
  <div class="stats">
    <div class="stat-card"><div class="num">{total}</div><div class="label">总题数</div></div>
    <div class="stat-card"><div class="num">{success}</div><div class="label">答案成功</div></div>
    <div class="stat-card"><div class="num">{review}</div><div class="label">需审查</div></div>
  </div>
""")

    for q in questions:
        qid = q.get("qid", "")
        qtype = q.get("type", "?")
        answer = q.get("answer")
        ref_points = q.get("reference_points", [])
        topic = q.get("topic", "?")
        sub_questions = q.get("sub_questions")
        image = q.get("image", "")
        source = q.get("source", "")
        needs_review = q.get("needs_review", False)
        review_reason = q.get("review_reason", "")

        review_badge = ""
        if needs_review:
            review_badge = f'<span class="q-badge review" title="{review_reason}">需审查</span>'

        html_parts.append(f"""
  <div class="q-card">
    <div class="q-header">
      <div class="left">
        <span class="q-id">{qid}</span>
        <span class="q-badge {qtype}">{qtype}</span>
        {review_badge}
      </div>
      <span style="font-size:12px;color:#9ca3af;">{topic}</span>
    </div>
    <div class="q-body">
      <img class="q-image" src="images/{image}" alt="{qid}">
""")

        if needs_review:
            html_parts.append(f'<div class="q-fail">需人工审查: {review_reason}</div>')

        if answer and answer != "None" and not isinstance(answer, dict):
            html_parts.append(f'<div class="q-answer"><b>答案:</b> {answer}</div>')
        elif isinstance(answer, dict):
            html_parts.append('<div class="q-fail">答案格式异常 (dict): ')
            for k, v in answer.items():
                html_parts.append(f'{k}: {str(v)[:60]} ')
            html_parts.append('</div>')

        if ref_points:
            html_parts.append('<div class="q-points"><b>参考要点:</b><ul>')
            for p in ref_points:
                html_parts.append(f'<li>{p}</li>')
            html_parts.append('</ul></div>')

        if sub_questions:
            html_parts.append('<div class="q-answer"><b>答案:</b></div>')
            for sq in sub_questions:
                label = sq.get("label", "")
                ans = sq.get("answer", "")
                html_parts.append(f'<div class="q-sub-item"><b>{label}</b> {ans}</div>')

        html_parts.append(f'<div class="q-meta">来源: {source} | 图片: {image}</div>')
        html_parts.append('</div></div>')

    html_parts.append("</div></body></html>")

    report_path = output_dir / f"report_{pdf_name}.html"
    report_path.write_text("".join(html_parts), encoding="utf-8")
    return report_path


# ============================================================
# 主流程
# ============================================================

def build_question_bank(pdf_path: str, output_dir: Path = None) -> list[dict]:
    """从单份试卷 PDF 提取题目

    Args:
        pdf_path: 试卷 PDF 路径
        output_dir: 输出目录，默认 data/question_bank/

    Returns:
        题目列表
    """
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    course = _detect_course(pdf_path)

    if output_dir is None:
        output_dir = QUESTION_BANK_DIR
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"提取: {pdf_name} (课程: {course})")
    print("=" * 60)

    doc = pymupdf.open(str(pdf_path))

    # 1. 定位题目
    print("\n[1] 定位题目边界...")
    questions, sections = locate_questions(doc)
    print(f"  大题段落: {len(sections)}")
    for s in sections:
        try:
            print(f"    第{s['page']+1}页: {s['title']}".encode('utf-8', errors='replace').decode('utf-8'))
        except:
            print(f"    第{s['page']+1}页: [编码问题]")
    choice_qs = [q for q in questions if not q["is_comprehensive"]]
    comp_qs = [q for q in questions if q["is_comprehensive"]]
    print(f"  选择题: {len(choice_qs)} 道, 综合题: {len(comp_qs)} 道")

    # 纯图片PDF回退：无文字层时用VLM逐页提取
    if not questions:
        total_text = sum(len(doc[pg].get_text().strip()) for pg in range(doc.page_count))
        if total_text == 0:
            print("  [!] 检测到纯图片PDF，切换到VLM逐页提取模式")
            questions, q_data = _process_image_only_pdf(doc, pdf_name, image_dir, course)
            visual_questions = list(qd["qid"] for qd in q_data)
            print(f"  VLM提取: {len(questions)} 道题")
            # 跳到答案生成步骤
            print("\n[4] 生成参考答案...")
            results = []
            for qd in q_data:
                qid = qd["qid"]
                text = qd["text"]
                print(f"  {qid}: LLM生成...", end=" ")
                answer_data = _llm_generate_answer(text, "", course)
                ans = str(answer_data.get("answer", ""))[:40].replace("\n", " ") if answer_data else "失败"
                print(f"答案={ans}")
                needs_review, review_reason = _check_answer_quality(qd, answer_data)
                if needs_review:
                    print(f"    [!] 需人工审查: {review_reason}")
                results.append({
                    "qid": qid,
                    "q_num": qd["q_num"],
                    "display_num": qd.get("display_num", qd["q_num"]),
                    "type": "综合题" if qd["is_comprehensive"] else "选择题",
                    "course": course,
                    "page": qd["page"],
                    "image": Path(qd["image_path"]).name,
                    "text": qd["text"],
                    "answer": answer_data.get("answer"),
                    "reference_points": answer_data.get("reference_points", []),
                    "topic": answer_data.get("topic", ""),
                    "sub_questions": answer_data.get("sub_questions"),
                    "section": qd.get("section", ""),
                    "source": pdf_name,
                    "source_file": pdf_path.name,
                    "needs_review": needs_review,
                    "review_reason": review_reason,
                })
            # 生成报告并保存
            generate_report(results, output_dir, image_dir, pdf_name)
            json_path = output_dir / f"questions_{pdf_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n完成! 共 {len(results)} 道题")
            print(f"保存到: {json_path}")
            print(f"对比报告: {output_dir / f'report_{pdf_name}.html'}")
            return results

    # 2. 截图
    print("\n[2] 截取题目图片...")
    q_data = []
    for q in questions:
        result = crop_question(doc, q, pdf_name, image_dir)
        result["display_num"] = q.get("display_num", q["q_num"])
        result["is_comprehensive"] = q["is_comprehensive"]
        print(f"  {result['qid']}: {len(result['text'])}字")
        q_data.append(result)

    # 3. VLM 检测
    print("\n[3] 检测图表/表格...")
    visual_questions = []
    for i, q in enumerate(questions):
        if has_visual_content(q, doc):
            visual_questions.append(q_data[i]["qid"])
    print(f"  需要VLM: {visual_questions if visual_questions else '无'}")

    # 4. 生成答案
    print("\n[4] 生成参考答案...")
    results = []
    for qd in q_data:
        qid = qd["qid"]
        text = qd["text"]

        if qid in visual_questions:
            img = Image.open(qd["image_path"])
            print(f"  {qid}: VLM描述...", end=" ")
            vlm_desc = _vlm_describe_question(img, text)
            print(f"({len(vlm_desc)}字) LLM生成...", end=" ")
            answer_data = _llm_generate_answer(text, vlm_desc, course)
        else:
            print(f"  {qid}: 纯文字...", end=" ")
            answer_data = _llm_generate_answer(text, course=course)

        ans = str(answer_data.get("answer", ""))[:40].replace("\n", " ") if answer_data else "失败"
        print(f"答案={ans}")

        needs_review, review_reason = _check_answer_quality(qd, answer_data)
        if needs_review:
            print(f"    [!] 需人工审查: {review_reason}")

        results.append({
            "qid": qid,
            "q_num": qd["q_num"],
            "display_num": qd.get("display_num", qd["q_num"]),
            "type": "综合题" if qd["is_comprehensive"] else "选择题",
            "course": course,
            "page": qd["page"],
            "image": Path(qd["image_path"]).name,
            "text": qd["text"],
            "answer": answer_data.get("answer"),
            "reference_points": answer_data.get("reference_points", []),
            "topic": answer_data.get("topic", ""),
            "sub_questions": answer_data.get("sub_questions"),
            "section": qd["section"],
            "source": pdf_name,
            "source_file": pdf_path.name,
            "needs_review": needs_review,
            "review_reason": review_reason,
        })
        time.sleep(1)

    doc.close()

    # 5. 保存
    output_file = output_dir / f"questions_{pdf_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n保存到: {output_file}")

    # 6. 生成报告
    report_path = generate_report(results, output_dir, image_dir, pdf_name)
    print(f"对比报告: {report_path}")

    print(f"\n完成! 共 {len(results)} 道题")
    return results


def build_all(output_dir: Path = None):
    """批量处理所有课程的往年题"""
    if output_dir is None:
        output_dir = QUESTION_BANK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_questions = []
    for course, exam_dir in COURSE_DIRS.items():
        if not exam_dir.exists():
            print(f"跳过（目录不存在）: {exam_dir}")
            continue

        pdf_files = sorted(exam_dir.glob("*.pdf"))
        print(f"\n[{course}] 找到 {len(pdf_files)} 份试卷")

        for pdf in pdf_files:
            print(f"\n{'='*60}")
            questions = build_question_bank(str(pdf), output_dir)
            all_questions.extend(questions)

    combined_file = output_dir / "question_bank.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n合并习题库: {combined_file} ({len(all_questions)} 道题)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if Path(pdf_path).is_dir():
            build_all(Path(pdf_path))
        else:
            build_question_bank(pdf_path)
    else:
        build_all()
