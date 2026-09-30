"""习题库提取 v2：题目截图 + 答案分离

核心改变:
  1. 题目 → 直接截图原始区域（保留排版、表格、图表）
  2. 答案 → 用提取的文字 + 必要时 VLM 处理图表
  3. 不标 confidence，只标 difficulty
  4. 题干和选项不重复

流程:
  1. 搜索每道题的题号位置 → 定位题目 y 范围
  2. 截取该区域为图片 → 保存
  3. 提取该区域文字（用于 LLM 理解题意）
  4. LLM 只负责: 生成参考答案要点 + 标注题型/知识点/难度
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

# ============================================================
# 配置
# ============================================================

PDF_PATH = r"data\操作系统\往年题\2024年秋操作系统期末考试试题（回忆版）.pdf"
OUTPUT_DIR = Path("debug/question_bank_v2")
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

VLM_MODEL = "glm-4v-flash"
LLM_MODEL = "glm-4-flash"


# ============================================================
# 1. 定位题目边界
# ============================================================

def locate_questions(doc) -> list[dict]:
    """扫描所有页，按题号定位每道题的位置

    Returns:
        [{"q_num": 1, "page": 0, "y_start": 233, "y_end": 327, "text": "..."}]
    """
    # 大题标记: 一、二、三、四...
    section_pattern = re.compile(r"^[一二三四五六七八九十]+、")
    # 小题号: 1. 2. ... 20.
    q_pattern = re.compile(r"^(\d+)[.．、]")

    all_items = []  # 所有可定位的行（题号或大题标题）

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

                # 大题标题
                if section_pattern.match(line_text):
                    all_items.append({
                        "type": "section",
                        "text": line_text[:60],
                        "page": pg,
                        "y_start": bbox[1],
                        "y_end": bbox[3],
                    })
                # 小题号
                m = q_pattern.match(line_text)
                if m:
                    q_num = int(m.group(1))
                    all_items.append({
                        "type": "question",
                        "q_num": q_num,
                        "text": line_text[:80],
                        "page": pg,
                        "y_start": bbox[1],
                        "y_end": bbox[3],
                    })

    # 计算每道题的结束位置 = 下一道题/大题的起始位置
    questions = [item for item in all_items if item["type"] == "question"]
    sections = [item for item in all_items if item["type"] == "section"]

    for i, q in enumerate(questions):
        # 找同一页上、y_start 大于当前题的下一个 item
        q["y_end"] = None
        for item in all_items:
            if item["page"] > q["page"]:
                # 下一页了，说明这道题到当前页底
                q["y_end"] = 780  # 页面底部
                break
            if item["page"] == q["page"] and item["y_start"] > q["y_start"] + 5:
                q["y_end"] = item["y_start"]
                break
        if q["y_end"] is None:
            q["y_end"] = 780

    # 合并跨页题（如第3题代码跨了多个block但同一页）
    # 去重：同一题号只保留第一次出现
    seen = {}
    for q in questions:
        if q["q_num"] not in seen:
            seen[q["q_num"]] = q
    questions = sorted(seen.values(), key=lambda x: (x["page"], x["y_start"]))

    return questions, sections


# ============================================================
# 2. 截取题目区域
# ============================================================

def crop_question(doc, q: dict, pdf_name: str) -> dict:
    """截取题目区域为图片，同时提取文字"""
    page = doc[q["page"]]
    page_rect = page.rect

    # 截图区域 (留一点边距)
    clip = pymupdf.Rect(
        80,                          # x0 左边距
        max(0, q["y_start"] - 5),    # y0 题目上方5px
        page_rect.width - 60,        # x1 右边距
        min(page_rect.height, q["y_end"] + 5),  # y1 下一题上方5px
    )

    # 渲染截图
    zoom = 2.5  # 高分辨率
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 保存截图
    img_path = IMAGE_DIR / f"{pdf_name}_q{q['q_num']}_p{q['page']+1}.png"
    img.save(str(img_path), optimize=True)

    # 提取该区域文字
    text = page.get_text("text", clip=clip)

    return {
        "q_num": q["q_num"],
        "image_path": str(img_path),
        "page": q["page"] + 1,
        "text": text.strip(),
    }


# ============================================================
# 3. LLM 生成答案（只处理答案，不拆题）
# ============================================================

ANSWER_PROMPT = (
    "你是操作系统课程教师。下面是一道操作系统考试题的文字内容。\n"
    "请生成参考答案并分类。输出 JSON 对象（不是数组），包含以下字段:\n\n"
    '1. "type": 题型 "选择题" / "简答题" / "综合题" / "分析题"\n'
    '2. "answer": 标准答案。选择题填选项字母（如 "A"），综合题填完整答案\n'
    '3. "reference_points": 参考答案要点数组，每条是一句完整的话，描述一个得分点\n'
    '4. "topic": 知识点分类，如 "操作系统-进程调度"\n'
    '5. "difficulty": "easy" / "medium" / "hard"\n'
    '6. "sub_questions": 如果是大题含多个小问，拆成 [{"label": "(1)", "answer": "..."}]；无小问则为 null\n\n'
    "注意:\n"
    "- 只输出 JSON，不要加 markdown 代码块\n"
    "- 如果文字不完整或有乱码，根据已有信息尽力作答\n"
    "- reference_points 每条要是完整的知识点陈述，不要只写关键词"
)


def image_to_base64(img: Image.Image, max_size: int = 800) -> str:
    """PIL Image 转 base64，压缩到合理大小"""
    # 缩放到最长边 max_size
    ratio = max_size / max(img.width, img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def vlm_describe_question(img: Image.Image, text: str) -> str:
    """第一步：VLM 描述题目图片中的表格/图表"""
    b64 = image_to_base64(img)
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
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
            print(f"    [VLM 错误] {type(e).__name__}: {str(e)[:100]}")
            return ""
    return ""


def llm_generate_answer(text: str, vlm_desc: str = "") -> dict:
    """第二步：纯文字 LLM 生成答案"""
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

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
            return json.loads(result)
        except json.JSONDecodeError as e:
            print(f"    [JSON 解析失败] {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            return {}
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"    [LLM 错误] {type(e).__name__}: {err[:100]}")
            if attempt < 2:
                time.sleep(3)
                continue
            return {}
    return {}


# ============================================================
# 4. 判断题目是否需要 VLM（有图表/表格的题才需要）
# ============================================================

def has_visual_content(q: dict, doc) -> bool:
    """快速判断题目区域是否有图表/表格"""
    page = doc[q["page"]]
    clip = pymupdf.Rect(
        80,
        max(0, q["y_start"] - 5),
        page.rect.width - 60,
        min(page.rect.height, q["y_end"] + 5),
    )

    # 检查矢量绘图
    drawings = page.get_drawings()
    visual_count = 0
    for d in drawings:
        rect = d["rect"]
        if (rect.y0 >= clip.y0 - 5 and rect.y1 <= clip.y1 + 5 and
            rect.x0 >= clip.x0 - 5 and rect.x1 <= clip.x1 + 5):
            visual_count += 1

    return visual_count >= 3  # 3个以上矢量元素才认为有表格/图表


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
    print(f"  大题: {len(sections)} 个")
    for s in sections:
        print(f"    第{s['page']+1}页: {s['text']}")
    print(f"  小题: {len(questions)} 道")

    # 2. 截图 + 提取文字
    print("\n[2] 截取题目图片...")
    q_data = []
    for q in questions:
        result = crop_question(doc, q, pdf_name)
        q_data.append(result)
        print(f"  Q{result['q_num']}: 第{result['page']}页, "
              f"图片={Path(result['image_path']).name}, "
              f"文字={len(result['text'])}字")

    # 3. 判断哪些题需要 VLM
    print("\n[3] 检测哪些题有图表/表格...")
    visual_questions = []
    for i, q in enumerate(questions):
        has_visual = has_visual_content(q, doc)
        if has_visual:
            visual_questions.append(q_data[i]["q_num"])
    print(f"  需要VLM的题: {visual_questions if visual_questions else '无'}")

    # 4. 生成答案
    print("\n[4] 生成参考答案...")
    results = []
    for i, qd in enumerate(q_data):
        q_num = qd["q_num"]
        text = qd["text"]

        if q_num in visual_questions:
            # 两步法：先 VLM 描述图表，再纯文字 LLM 生成答案
            print(f"  Q{q_num}: VLM描述图表...", end=" ")
            img = Image.open(qd["image_path"])
            vlm_desc = vlm_describe_question(img, text)
            print(f"({len(vlm_desc)}字) LLM生成答案...", end=" ")
            answer_data = llm_generate_answer(text, vlm_desc)
        else:
            # 纯文字直接调 LLM
            print(f"  Q{q_num}: 纯文字...", end=" ")
            answer_data = llm_generate_answer(text)

        if answer_data:
            print(f"答案={str(answer_data.get('answer', '?'))[:30]} "
                  f"难度={answer_data.get('difficulty', '?')}")
        else:
            print("失败!")

        results.append({
            "q_num": q_num,
            "page": qd["page"],
            "image": Path(qd["image_path"]).name,
            "text": text,
            "type": answer_data.get("type", "?"),
            "answer": answer_data.get("answer"),
            "reference_points": answer_data.get("reference_points", []),
            "topic": answer_data.get("topic", ""),
            "difficulty": answer_data.get("difficulty", ""),
            "sub_questions": answer_data.get("sub_questions"),
            "source": "2024秋操作系统期末试题（回忆版）",
            "source_file": Path(PDF_PATH).name,
        })

        time.sleep(1)  # 避免 429

    doc.close()

    # 5. 保存
    output_file = OUTPUT_DIR / "questions_2024秋.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n保存到: {output_file}")

    # 摘要
    print(f"\n{'='*60}")
    print(f"完成! 共 {len(results)} 道题")
    if results:
        print(f"\n{'Q':>3} | {'类型':<8} | {'难度':<6} | {'知识点':<20} | 答案预览")
        print("-" * 80)
        for r in results:
            ans = str(r.get("answer", ""))[:30].replace("\n", " ")
            print(f"{r['q_num']:3d} | {r.get('type','?'):<8} | "
                  f"{r.get('difficulty','?'):<6} | "
                  f"{r.get('topic','?'):<20} | {ans}")

        easy = sum(1 for r in results if r.get("difficulty") == "easy")
        medium = sum(1 for r in results if r.get("difficulty") == "medium")
        hard = sum(1 for r in results if r.get("difficulty") == "hard")
        print(f"\n难度: easy={easy}, medium={medium}, hard={hard}")


if __name__ == "__main__":
    main()
