"""课本 PDF 处理器

处理流程:
  课本 PDF → 按章节拆分 → 每章喂给 LLM → 生成结构化知识点笔记
  → 保存为 Markdown → 入库

核心设计思想:
  - 课本内容多而杂，直接全量入库会稀释检索精度
  - 通过 LLM 提炼，将厚书变薄，只保留核心知识点
  - 生成的笔记质量高、结构清晰，检索时相关性更强
  - 笔记可以缓存，重复导入时直接复用
"""

import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pymupdf
import requests

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL


@dataclass
class Chapter:
    """课本章节"""
    index: int
    title: str
    start_page: int
    end_page: int
    text: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


NOTE_SYSTEM_PROMPT = """你是一个课程知识点提炼专家。请根据给定的课本章节内容，提炼出结构化的知识点笔记。

要求:
1. 用 Markdown 格式输出，使用二级标题(##)分知识点模块
2. 每个知识点包含: 定义、核心概念、关键特性/区别、典型应用场景
3. 语言简洁，重点突出，适合复习记忆
4. 保留专业术语的中英文对照（如有）
5. 只提炼核心知识点，不要复述课本中的例子、故事、历史背景等
6. 如果内容是习题、答案、目录、参考文献等非知识点内容，返回空字符串

输出示例:
## 进程与线程
- **进程**: 资源分配的基本单位，有独立地址空间
- **线程**: CPU调度的基本单位，共享进程地址空间
- **区别**: 资源/调度/开销/通信/独立性五个维度

## 进程状态转换
- 三状态模型: 就绪/运行/阻塞
- 转换关系: 就绪→运行(调度)、运行→就绪(时间片)、运行→阻塞(等待IO)、阻塞→就绪(IO完成)
"""


def _llm_generate_note(chapter_text: str, chapter_title: str, max_retries: int = 3) -> str:
    """调用 LLM 生成知识点笔记"""
    # 控制单次输入长度，避免超长
    max_chars = 8000
    if len(chapter_text) > max_chars:
        chapter_text = chapter_text[:max_chars] + "\n...（内容过长，已截断）"

    payload = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "system", "content": NOTE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"章节标题: {chapter_title}\n\n章节内容:\n{chapter_text}\n\n请提炼本章的知识点笔记。",
            },
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [LLM 错误] {e}")
            return ""

    return ""


def _extract_chapters_from_toc(doc: pymupdf.Document) -> list[Chapter]:
    """从 PDF 目录提取章节信息"""
    chapters = []
    toc = doc.get_toc()  # [[level, title, page], ...]

    if not toc:
        return chapters

    # 只取一级章节（level=1）
    # 注意 PyMuPDF 的 TOC 条目是 [level, title, page]：
    # 原来写成 `t[1] >= 1`（拿标题字符串和 1 比较）必然 TypeError，
    # 导致整条课本处理分支 100% 失败。这里比较的是页码 t[2]。
    level1 = [(t[1], t[2]) for t in toc if len(t) >= 3 and t[0] == 1 and t[2] >= 1]

    for i, (title, page) in enumerate(level1):
        end_page = level1[i + 1][1] - 1 if i + 1 < len(level1) else doc.page_count
        # 过滤掉非正文章节（目录、索引、附录等）
        if any(kw in title for kw in ["目录", "索引", "附录", "参考文献", "前言", "致谢"]):
            continue
        chapters.append(Chapter(
            index=i + 1,
            title=title,
            start_page=max(1, page),
            end_page=min(end_page, doc.page_count),
        ))

    return chapters


def _extract_chapters_by_heading(doc: pymupdf.Document) -> list[Chapter]:
    """通过标题样式/格式识别章节（没有目录时的 fallback）"""
    # 简单实现：按固定页数拆分（每 20 页一章）
    # 实际项目中可以用字体大小、样式等更精确地识别
    chapters = []
    pages_per_chapter = 20
    total = doc.page_count
    idx = 0
    for start in range(0, total, pages_per_chapter):
        idx += 1
        end = min(start + pages_per_chapter, total)
        # 尝试从第一页提取标题
        first_page_text = doc[start].get_text()[:200]
        first_line = first_page_text.strip().split("\n")[0].strip() if first_page_text.strip() else f"第{idx}章"
        chapters.append(Chapter(
            index=idx,
            title=first_line[:50],
            start_page=start + 1,
            end_page=end,
        ))
    return chapters


def _clean_textbook_text(text: str) -> str:
    """清洗课本文本，去除页眉页脚页码等噪声"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯数字行（页码）
        if re.match(r"^\s*\d+\s*$", stripped):
            continue
        # 跳过太短的行（可能是页眉页脚）
        if len(stripped) < 3 and not re.match(r"[\u4e00-\u9fa5]", stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_textbook_pdf(
    pdf_path: str,
    output_dir: str = None,
    max_chapters: Optional[int] = None,
    llm_interval: float = 2.0,
) -> list[str]:
    """处理课本 PDF，生成知识点笔记

    Args:
        pdf_path: 课本 PDF 路径
        output_dir: 笔记输出目录（None 则不保存文件）
        max_chapters: 最多处理章节数（用于测试）
        llm_interval: LLM 调用间隔

    Returns:
        list[str]: 每章的知识点笔记 Markdown 文本列表
    """
    with pymupdf.open(pdf_path) as doc:
        book_name = Path(pdf_path).stem

        print(f"  处理课本: {book_name} ({doc.page_count} 页)")

        # 1. 提取章节结构
        chapters = _extract_chapters_from_toc(doc)
        if not chapters:
            print("    未找到目录，按页数粗略拆分")
            chapters = _extract_chapters_by_heading(doc)

        print(f"    识别到 {len(chapters)} 个章节")

        if max_chapters:
            chapters = chapters[:max_chapters]
            print(f"    测试模式：只处理前 {max_chapters} 章")

        notes = []

        for i, ch in enumerate(chapters):
            # 提取章节文本
            ch_text_parts = []
            for p in range(ch.start_page - 1, ch.end_page):
                if p < len(doc):
                    ch_text_parts.append(doc[p].get_text())
            ch.text = _clean_textbook_text("\n".join(ch_text_parts))

            print(f"    [{i+1}/{len(chapters)}] {ch.title} "
                  f"({ch.end_page - ch.start_page + 1}页, {ch.char_count}字)")

            if ch.char_count < 200:
                print(f"      内容太少，跳过")
                continue

            # LLM 生成笔记
            note = _llm_generate_note(ch.text, ch.title)
            if note:
                notes.append(f"# {ch.title}\n\n{note}\n")
                print(f"      笔记: {len(note)} 字")
            else:
                print(f"      生成失败")

            time.sleep(llm_interval)


    # 保存笔记
    if output_dir and notes:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # 合并成一个笔记文件
        full_note = "\n\n---\n\n".join(notes)
        out_file = out_path / f"{book_name}_知识点笔记.md"
        out_file.write_text(full_note, encoding="utf-8")
        print(f"  笔记已保存到: {out_file}")

    print(f"  完成: 生成 {len(notes)} 章笔记，共 {sum(len(n) for n in notes)} 字")
    return notes
