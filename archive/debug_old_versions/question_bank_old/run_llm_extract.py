"""分段调用 LLM 提取题目（修复版）"""

import json
import re
import time
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL

LLM_MODEL = "glm-4-flash"
OUTPUT_DIR = Path("debug/question_bank_pilot")

text = (OUTPUT_DIR / "merged_text.txt").read_text(encoding="utf-8")
print(f"总文本长度: {len(text)} 字符")

# 分成两段: 选择题部分 + 综合题部分
split_marker = "二、综合题"
if split_marker in text:
    idx = text.index(split_marker)
    part1 = text[:idx]
    part2 = text[idx:]
else:
    mid = len(text) // 2
    part1 = text[:mid]
    part2 = text[mid:]

print(f"第一段: {len(part1)} 字符")
print(f"第二段: {len(part2)} 字符")

EXTRACT_PROMPT = (
    "你是试题整理专家。下面是一份操作系统期末考试回忆版试题的文本内容"
    "（可能包含 VLM 对页面图片的描述）。\n\n"
    "请逐题拆分，输出 JSON 数组。每道题包含以下字段:\n\n"
    '1. "question": 题目原文（如果是选择题，包含所有选项）\n'
    '2. "type": 题型，如 "选择题" / "简答题" / "综合题" / "分析题"\n'
    '3. "options": 选择题选项字典 {"A": "...", "B": "..."}，非选择题为 null\n'
    '4. "answer": 标准答案（如果有），选择题填选项字母，简答题填答案要点。回忆版可能没有答案，填 null\n'
    '5. "reference_points": 参考答案要点数组（即使没有标准答案也尽量根据知识点生成）\n'
    '6. "topic": 知识点分类，如 "操作系统-进程调度"\n'
    '7. "difficulty": "easy" / "medium" / "hard"\n'
    '8. "confidence": 完整性标注: "high" / "medium" / "low"\n'
    '9. "note": 备注（笔误说明、替代内容说明等）\n'
    '10. "sub_questions": 大题含多个小问时拆成 [{"label": "(1)", "question": "..."}]；无小问则为 null\n\n'
    "特别注意:\n"
    "- 碎碎念/复习建议/感受 → 直接跳过\n"
    '- "忘了考试原题，此处贴上作业题" → confidence="low"，note 说明\n'
    '- 选项笔误（如"C...n=20"）→ note 指出并修正\n'
    "- 表格数据整合到 question 中\n"
    "- 输出纯 JSON 数组，不要加 markdown 代码块标记"
)

headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json",
}


def call_llm(text_part, label):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"以下是试卷内容:\n\n{text_part}\n\n请按题拆分并输出 JSON。"},
        ],
        "temperature": 0.1,
        "max_tokens": 3000,
    }
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            questions = json.loads(content)
            print(f"  {label}: 提取到 {len(questions)} 道题")
            return questions
        except json.JSONDecodeError as e:
            print(f"  [{label}] JSON 解析失败: {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            (OUTPUT_DIR / f"raw_llm_{label}.txt").write_text(content, encoding="utf-8")
            print(f"  原始输出已保存")
            return []
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [{label}] LLM 错误: {e}")
            return []
    return []


print("\n分段调用 LLM...")
q1 = call_llm(part1, "选择题部分")
time.sleep(2)
q2 = call_llm(part2, "综合题部分")

all_questions = q1 + q2
for q in all_questions:
    q["source"] = "2024秋操作系统期末试题（回忆版）"
    q["source_file"] = "2024年秋操作系统期末考试试题（回忆版）.pdf"

output_file = OUTPUT_DIR / "questions_2024秋.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)

print(f"\n总计: {len(all_questions)} 道题")
print(f"保存到: {output_file}")

if all_questions:
    print(f"\n题目摘要:")
    print(f"{'ID':>3} | {'类型':<8} | {'置信度':<8} | {'难度':<6} | 题目预览")
    print("-" * 90)
    for i, q in enumerate(all_questions, 1):
        preview = q.get("question", "")[:50].replace("\n", " ")
        qtype = q.get("type", "?")
        conf = q.get("confidence", "?")
        diff = q.get("difficulty", "?")
        print(f"{i:3d} | {qtype:<8} | {conf:<8} | {diff:<6} | {preview}")

    high = sum(1 for q in all_questions if q.get("confidence") == "high")
    medium = sum(1 for q in all_questions if q.get("confidence") == "medium")
    low = sum(1 for q in all_questions if q.get("confidence") == "low")
    print(f"\n置信度: high={high}, medium={medium}, low={low}")
