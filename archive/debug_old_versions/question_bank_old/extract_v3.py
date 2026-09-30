"""习题库提取 v3：修复截图边界 + 跨页 + 综合题

核心修复:
  1. 截图边界：从下一题 y_start 往前留 3px 间距，不再截到下一题
  2. 跨页题：如果一页最后一题后面还有内容且下一页开头是代码/续文，
     合并两页内容为一张图
  3. Q0 误判：题号只接受 >=1 的整数，跳过 0
  4. 综合题：按大题标题（一、二、三...）分段，综合题部分用 "大题号-小题号" 编号
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL

PDF_PATH = r"data\操作系统\往年题\2024年秋操作系统期末考试试题（回忆版）.pdf"
OUTPUT_DIR = Path("debug/question_bank_v3")
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

VLM_MODEL = "glm-4v-flash"
LLM_MODEL = "glm-4-flash"


# ============================================================
# 1. 定位题目边界（支持跨页 + 综合题）
# ============================================================

def locate_questions(doc) -> list[dict]:
    """扫描所有页，定位每道题的位置

    综合题用 "S1" "S2" 编号，选择题用 Q1 Q2...
    跨页题标记 pages=[page1, page2]
    """
    section_pattern = re.compile(r"^[一二三四五六七八九十]+、")
    # 题号：1. 2. 10. 等，不匹配 0 开头
    # 排除 "1.0" "0.4" 等数据（题号后跟数字=小数）
    q_pattern = re.compile(r"^(\d+)[.．、](?!\d)")
    # 小问号：(1) (2)
    sub_pattern = re.compile(r"^[（(]\d+[）)]")

    all_lines = []  # 所有有意义的行

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
                # 跳过页眉页脚
                if bbox[1] < 60 or bbox[1] > 780:
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

    # 识别小题号（跳过 q_num=0）
    questions = []
    for ln in all_lines:
        m = q_pattern.match(ln["text"])
        if m:
            q_num = int(m.group(1))
            if q_num == 0:
                continue  # 跳过 0（表格数据误判）
            # 判断属于哪个大题
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

    # 去重：同一页同一题号只保留第一次
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

    for item in all_items:
        if item["type"] != "question":
            continue
        q = item["q"]
        # 找下一个 item
        q["y_end"] = 780  # 默认到页底
        q["end_page"] = q["page"]  # 默认不跨页

        idx = all_items.index(item)
        if idx + 1 < len(all_items):
            next_item = all_items[idx + 1]
            if next_item["page"] == q["page"]:
                # 同一页，下一题就在下方
                q["y_end"] = next_item["y"] - 3  # 留3px间距
            else:
                # 下一题在下一页，当前题可能跨页
                q["y_end"] = 780
                # 检查下一页开头是不是续文（代码、表格数据等）
                next_page = next_item["page"]
                if next_page == q["page"] + 1:
                    q["end_page"] = next_page
                    q["y_end_next_page"] = next_item["y"] - 3
                else:
                    q["end_page"] = q["page"]

    # 区分选择题和综合题，综合题重新编号
    comp_counter = 0
    for q in questions:
        if "综合" in q["section"]:
            comp_counter += 1
            q["is_comprehensive"] = True
            q["display_num"] = comp_counter  # 综合题从1开始连续编号
        else:
            q["is_comprehensive"] = False
            q["display_num"] = q["q_num"]

    return questions, sections


# ============================================================
# 2. 截取题目区域（支持跨页拼接）
# ============================================================

def crop_question(doc, q: dict, pdf_name: str) -> dict:
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

    # 如果跨页，拼接第二页的部分
    if q.get("end_page", q["page"]) != q["page"]:
        page2 = doc[q["end_page"]]
        clip2 = pymupdf.Rect(
            80,
            60,  # 从页眉下方开始
            page2.rect.width - 60,
            min(page2.rect.height, q.get("y_end_next_page", 400) + 3),
        )
        pix2 = page2.get_pixmap(matrix=mat, clip=clip2)

        # 拼接两张图
        img1 = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
        total_height = img1.height + img2.height + 5
        img = Image.new("RGB", (max(img1.width, img2.width), total_height), "white")
        img.paste(img1, (0, 0))
        img.paste(img2, (0, img1.height + 5))
    else:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 裁剪留白
    img = trim_whitespace(img)

    # 编号
    if q["is_comprehensive"]:
        qid = f"S{q['display_num']}"
    else:
        qid = f"Q{q['display_num']}"

    img_path = IMAGE_DIR / f"{pdf_name}_{qid}_p{q['page']+1}.png"
    img.save(str(img_path), optimize=True)

    # 提取文字
    text = page.get_text("text", clip=clip)
    if q.get("end_page", q["page"]) != q["page"]:
        page2 = doc[q["end_page"]]
        clip2 = pymupdf.Rect(80, 60, page2.rect.width - 60, q.get("y_end_next_page", 400) + 3)
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
    """快速判断题目区域是否有图表/表格"""
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

    return visual_count >= 3


# ============================================================
# 4. VLM + LLM 生成答案
# ============================================================

ANSWER_PROMPT = (
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
)


def trim_whitespace(img: Image.Image, threshold: int = 245) -> Image.Image:
    """裁剪图片四周的白色留白"""
    gray = img.convert("L")
    # 反转：白色变黑(0)，有内容变亮
    inverted = gray.point(lambda x: 255 if x < threshold else 0)
    bbox = inverted.getbbox()
    if bbox:
        # 留 5px 边距
        pad = 5
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(img.width, bbox[2] + pad)
        bottom = min(img.height, bbox[3] + pad)
        img = img.crop((left, top, right, bottom))
    return img


def image_to_base64(img: Image.Image, max_size: int = 800) -> str:
    ratio = max_size / max(img.width, img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_describe_question(img: Image.Image, text: str) -> str:
    b64 = image_to_base64(img)
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


def llm_generate_answer(text: str, vlm_desc: str = "") -> dict:
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    user_content = f"以下是题目文字:\n{text}"
    if vlm_desc:
        user_content += f"\n\n[VLM 图片描述]\n{vlm_desc}"
    user_content += "\n\n请生成参考答案并分类。"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": ANSWER_PROMPT},
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
            # 宽松解析：尝试提取第一个 JSON 对象
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


# ============================================================
# 主流程
# ============================================================

def main():
    pdf_name = Path(PDF_PATH).stem
    print(f"提取: {pdf_name}")
    print("=" * 60)

    doc = pymupdf.open(PDF_PATH)

    # 1. 定位题目
    print("\n[1] 定位题目边界...")
    questions, sections = locate_questions(doc)
    print(f"  大题段落: {len(sections)}")
    for s in sections:
        print(f"    第{s['page']+1}页: {s['title']}")
    print(f"  小题: {len(questions)} 道")
    choice_qs = [q for q in questions if not q["is_comprehensive"]]
    comp_qs = [q for q in questions if q["is_comprehensive"]]
    print(f"    选择题: {len(choice_qs)} 道")
    print(f"    综合题: {len(comp_qs)} 道")
    cross_qs = [q for q in questions if q.get("end_page", q["page"]) != q["page"]]
    print(f"    跨页题: {len(cross_qs)} 道")

    # 2. 截图
    print("\n[2] 截取题目图片...")
    q_data = []
    for q in questions:
        result = crop_question(doc, q, pdf_name)
        result["display_num"] = q.get("display_num", q["q_num"])
        result["is_comprehensive"] = q["is_comprehensive"]
        cross_tag = " [跨页]" if result["cross_page"] else ""
        print(f"  {result['qid']}: 第{result['page']}-{result['end_page']}页{cross_tag}, "
              f"{len(result['text'])}字")
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
    for i, qd in enumerate(q_data):
        qid = qd["qid"]
        text = qd["text"]

        if qid in visual_questions:
            img = Image.open(qd["image_path"])
            print(f"  {qid}: VLM描述...", end=" ")
            vlm_desc = vlm_describe_question(img, text)
            print(f"({len(vlm_desc)}字) LLM生成...", end=" ")
            answer_data = llm_generate_answer(text, vlm_desc)
        else:
            print(f"  {qid}: 纯文字...", end=" ")
            answer_data = llm_generate_answer(text)

        ans = str(answer_data.get("answer", ""))[:40].replace("\n", " ") if answer_data else "失败"
        print(f"答案={ans}")

        # 标注可疑答案
        needs_review = False
        review_reason = ""

        if not answer_data:
            needs_review = True
            review_reason = "答案生成失败"
        elif qd["is_comprehensive"]:
            # 综合题：answer 必须是 null，答案在 sub_questions 里
            if answer_data.get("answer") is not None:
                needs_review = True
                review_reason = "综合题 answer 不为 null"
            if not answer_data.get("sub_questions"):
                needs_review = True
                review_reason = (review_reason + "; " if review_reason else "") + "综合题缺少 sub_questions"
        else:
            # 选择题：answer 必须是单个字母
            ans_val = str(answer_data.get("answer", "")).strip()
            if not re.match(r"^[A-D]$", ans_val):
                needs_review = True
                review_reason = f"选择题答案不是单字母A-D: '{ans_val[:20]}'"

        if needs_review:
            print(f"    [!] 需人工审查: {review_reason}")

        results.append({
            "qid": qid,
            "q_num": qd["q_num"],
            "display_num": qd.get("display_num", qd["q_num"]),
            "type": "综合题" if qd["is_comprehensive"] else "选择题",
            "page": qd["page"],
            "cross_page": qd["cross_page"],
            "image": Path(qd["image_path"]).name,
            "text": qd["text"],
            "answer": answer_data.get("answer"),
            "reference_points": answer_data.get("reference_points", []),
            "topic": answer_data.get("topic", ""),
            "sub_questions": answer_data.get("sub_questions"),
            "section": qd["section"],
            "source": "2024秋操作系统期末试题（回忆版）",
            "source_file": Path(PDF_PATH).name,
            "needs_review": needs_review,
            "review_reason": review_reason,
        })
        time.sleep(1)

    doc.close()

    # 5. 保存
    output_file = OUTPUT_DIR / "questions_2024秋.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n保存到: {output_file}")

    print(f"\n{'='*60}")
    print(f"完成! 共 {len(results)} 道题 (选择题{len(choice_qs)} + 综合题{len(comp_qs)})")

    if results:
        print(f"\n{'ID':>4} | {'类型':<6} | {'跨页':<4} | {'知识点':<20} | 答案预览")
        print("-" * 80)
        for r in results:
            cross = "是" if r.get("cross_page") else ""
            ans = str(r.get("answer", ""))[:30].replace("\n", " ")
            topic = r.get("topic", "?")[:20]
            print(f"{r['qid']:>4} | {r.get('type','?'):<6} | {cross:<4} | {topic:<20} | {ans}")


if __name__ == "__main__":
    main()
