"""PPT PDF 处理器

处理流程:
  PPT PDF → 按页拆分 → 每页提取文字 → 整页渲染成图片 → VLM 理解图文结构
  → 文字 + VLM 描述合并（相辅相成） → 输出按页组织的结构化内容

核心设计思想:
  - 幻灯片是天然的语义单元，同页的文字和图表相辅相成
  - PPT 中的图表多为矢量图形，无法通过 extract_images 拿到
  - 整页渲染 + VLM 理解 = 既能提取文字标签，又能理解结构关系
  - 合并后保留了"图文并茂"的完整语义
  - 按页组织也方便后续分块时保持上下文完整
"""

import re
import io
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pymupdf  # PyMuPDF
from PIL import Image

from knowledge_pipeline.image_understanding import describe_image_vlm, is_worth_processing


@dataclass
class SlideContent:
    """单页 PPT 的内容（文字 + VLM 图文结构描述）"""
    page_num: int
    title: str = ""
    text: str = ""          # 提取的文字
    vlm_description: str = ""  # VLM 对整页图文结构的描述
    has_diagram: bool = False  # 是否包含图表（VLM 判断）

    @property
    def combined(self) -> str:
        """合并文字和 VLM 描述，形成完整的页内容

        文字是骨架（标题、要点、标签），VLM 描述补充结构关系
        （如"箭头从A指向B"、"状态转换关系"等），两者相辅相成。
        """
        parts = []
        if self.title:
            parts.append(f"【标题】{self.title}")
        if self.text.strip():
            parts.append(self.text.strip())
        if self.vlm_description:
            parts.append("【图表与结构】")
            parts.append(self.vlm_description)
        return "\n".join(parts)

    @property
    def char_count(self) -> int:
        return len(self.combined)


def _extract_title(text: str) -> str:
    """从一页文字中尝试提取标题（取第一行有意义的内容）"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    # 标题通常是第一行，且字数不会太多
    first = lines[0]
    if len(first) < 50:
        return first
    return first[:30] + "..."


def _should_use_vlm(text: str, title: str, page_num: int) -> bool:
    """判断一页是否值得调用 VLM

    启发式规则:
      - 跳过: 封面页（第1页且文字很少）
      - 跳过: 目录/大纲页（大量 X.X 格式编号）
      - 处理: 标题含图表关键词（图、流程、状态、架构、模型等）
      - 处理: 文字很少（<150字）且不是封面/目录（可能是图表页）
      - 跳过: 文字多（>300字）且无图表关键词（大概率纯文字页）
      - 跳过: 中等文字量（150~300字）且标题无图表关键词

    最后一档是**保守策略**：这类页面有图有文的概率不高，
    而 VLM 调用要花钱花时间，所以默认跳过。
    （原文档这里写成"处理: 其他情况"，与实现相反 —— 以实现为准。）
    """
    # 封面页
    if page_num == 1 and len(text) < 100:
        return False

    # 目录/大纲页：包含多个 X.X 格式的章节编号
    section_pattern = re.findall(r"\d+\.\d+", text)
    if len(section_pattern) >= 5 and len(text) < 400:
        return False

    # 标题含图表关键词 → 处理
    diagram_keywords = ["图", "流程", "状态", "架构", "模型", "示意", "转换",
                        "结构", "机制", "算法示意", "对比", "实例"]
    if any(kw in title for kw in diagram_keywords):
        return True

    # 文字极少（可能是纯图表页，文字只是标签）→ 处理
    if len(text) < 150:
        return True

    # 其余（含 150~300 字的中等页）都跳过，节省配额
    return False


def _clean_page_text(text: str) -> str:
    """清洗单页文字"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # 去除纯数字行（页码等）
    lines = text.split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*\d+\s*$", ln)]
    # 压缩空行
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()


def process_ppt_pdf(
    pdf_path: str,
    use_vlm: bool = True,
    vlm_interval: float = 3.0,
    max_pages: Optional[int] = None,
    zoom: float = 1.5,
) -> list[SlideContent]:
    """处理 PPT PDF 文件

    Args:
        pdf_path: PDF 文件路径
        use_vlm: 是否用 VLM 理解每页的图文结构
        vlm_interval: VLM 调用间隔秒数（避免限流）
        max_pages: 最多处理页数（用于测试）
        zoom: 渲染缩放比例（1.0 = 72dpi，1.5 = 108dpi，2.0 = 144dpi）

    Returns:
        list[SlideContent]: 每页的结构化内容
    """
    with pymupdf.open(pdf_path) as doc:
        total_pages = min(doc.page_count, max_pages) if max_pages else doc.page_count
        results = []

        print(f"  处理 {Path(pdf_path).name}: 共 {doc.page_count} 页"
              f"{'（测试模式，处理前' + str(total_pages) + '页）' if max_pages else ''}"
              f"{'，VLM 启用' if use_vlm else '，纯文字模式'}")

        for i in range(total_pages):
            page = doc[i]
            page_num = i + 1

            try:
                text = page.get_text()
            except Exception:
                print(f"    [跳过] 第 {page_num} 页文字提取失败")
                results.append(SlideContent(page_num=page_num, text=""))
                continue

            cleaned_text = _clean_page_text(text)
            title = _extract_title(cleaned_text)

            slide = SlideContent(
                page_num=page_num,
                title=title,
                text=cleaned_text,
            )

            # 2. 整页渲染 + VLM 理解图文结构
            if use_vlm and _should_use_vlm(cleaned_text, title, page_num):
                # 渲染整页为图片
                mat = pymupdf.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # 图太小（空白页/装饰）或太大时不值得花一次 VLM 调用
                if is_worth_processing(img):
                    context = (
                        f"这是第 {page_num} 页 PPT 幻灯片，标题可能是：{title}。"
                        f"页面提取到的文字如下：\n{cleaned_text[:500]}\n\n"
                        f"请重点描述页面中的图表、流程图、状态转换图、架构图等视觉内容，"
                        f"以及各元素之间的关系。不需要重复文字已经明确表达的内容。"
                    )
                    desc = describe_image_vlm(img, context=context)
                    if desc:
                        slide.vlm_description = desc
                        slide.has_diagram = True

                    time.sleep(vlm_interval)

            results.append(slide)

            # 进度
            if page_num % 20 == 0 or page_num == total_pages:
                total_chars = sum(s.char_count for s in results)
                vlm_count = sum(1 for s in results if s.vlm_description)
                print(f"    进度: {page_num}/{total_pages} 页, "
                      f"VLM处理: {vlm_count} 页, "
                      f"已生成 {total_chars} 字")


    total_chars = sum(s.char_count for s in results)
    vlm_pages = sum(1 for s in results if s.vlm_description)
    print(f"  完成: {total_pages} 页, VLM 处理 {vlm_pages} 页, 共 {total_chars} 字")

    return results


def slides_to_text(slides: list[SlideContent], merge_pages: int = 1) -> str:
    """将多页 PPT 内容合并成一个长文本，供后续分块使用

    Args:
        slides: 页内容列表
        merge_pages: 几页合并成一个文本块（1 = 每页独立）
    """
    if merge_pages <= 1:
        # 每页之间用分隔符连接，保留页的边界
        parts = []
        for s in slides:
            if s.char_count > 0:
                parts.append(s.combined)
        return "\n\n---\n\n".join(parts)

    # 多页合并
    chunks = []
    for i in range(0, len(slides), merge_pages):
        group = slides[i:i + merge_pages]
        group_text = "\n\n".join(s.combined for s in group if s.char_count > 0)
        if group_text:
            chunks.append(group_text)
    return "\n\n---\n\n".join(chunks)
