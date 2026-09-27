"""文档处理调度器

自动识别文档类型，选择合适的处理器，输出统一格式的文档内容。

支持的文档类型与处理策略:
  - PPT (PDF 放在 ppt 目录下): PPT 处理器（按页+图文合并）
  - 课本 (PDF 放在 课本 目录下): 课本处理器（章节拆分+LLM笔记）
  - 笔记 (MD/DOCX/TXT 或其他路径): 直接加载清洗

用法:
    from knowledge_pipeline.pipeline import process_document

    result = process_document(
        file_path="data/操作系统/ppt/03-OS调度.pdf",
        base_dir="data",
        options={"ppt_use_vlm": True, "textbook_output_dir": "data/notes"},
    )
    # result = {"source": "03-OS调度.pdf", "text": "...", "doc_type": "ppt"}
"""

from pathlib import Path
from typing import Optional

from rag_tool import load_document, clean_text, _classify_doc


def process_document(
    file_path: str,
    base_dir: str,
    ppt_use_vlm: bool = False,
    textbook_output_dir: Optional[str] = None,
    textbook_max_chapters: Optional[int] = None,
) -> dict:
    """处理单个文档文件，返回统一格式的结果

    Args:
        file_path: 文件路径
        base_dir: 基础目录（用于判断文档类型）
        ppt_use_vlm: PPT 是否用 VLM 理解整页图文结构
        textbook_output_dir: 课本笔记输出目录（None 则不保存文件）
        textbook_max_chapters: 课本最多处理章节数

    Returns:
        {"source": 文件名, "text": 处理后的文本, "doc_type": ppt/textbook/notes}
    """
    fp = Path(file_path)
    bp = Path(base_dir)
    doc_type = _classify_doc(fp, bp)

    if doc_type is None:
        return None

    source = fp.name
    suffix = fp.suffix.lower()

    # --- PPT 处理 ---
    if doc_type == "ppt" and suffix == ".pdf":
        from knowledge_pipeline.ppt_processor import process_ppt_pdf, slides_to_text
        slides = process_ppt_pdf(
            str(fp),
            use_vlm=ppt_use_vlm,
        )
        text = slides_to_text(slides, merge_pages=1)
        return {"source": source, "text": text, "doc_type": "ppt"}

    # --- 课本处理 ---
    if doc_type == "textbook" and suffix == ".pdf":
        from knowledge_pipeline.textbook_processor import process_textbook_pdf
        notes = process_textbook_pdf(
            str(fp),
            output_dir=textbook_output_dir,
            max_chapters=textbook_max_chapters,
        )
        text = "\n\n---\n\n".join(notes) if notes else ""
        return {"source": source, "text": text, "doc_type": "textbook"}

    # --- 笔记/文档：直接加载清洗 ---
    text = load_document(str(fp))
    text = clean_text(text)
    return {"source": source, "text": text, "doc_type": doc_type}


def process_directory(
    dir_path: str,
    ppt_use_vlm: bool = False,
    textbook_output_dir: Optional[str] = None,
    skip_textbook: bool = False,
) -> list[dict]:
    """批量处理目录下的所有文档

    Args:
        dir_path: 文档目录
        ppt_use_vlm: PPT 是否启用 VLM 图文结构理解
        textbook_output_dir: 课本笔记输出目录
        skip_textbook: 是否跳过课本

    Returns:
        [{"source": ..., "text": ..., "doc_type": ...}, ...]
    """
    results = []
    p = Path(dir_path)
    if not p.exists():
        print(f"[警告] 目录不存在: {dir_path}")
        return results

    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".pdf", ".docx", ".md", ".txt", ".markdown"):
            continue

        doc_type = _classify_doc(f, p)
        if doc_type is None:
            continue
        if skip_textbook and doc_type == "textbook":
            continue

        print(f"\n处理: {f.name} [{doc_type}]")
        try:
            result = process_document(
                str(f),
                base_dir=str(p),
                ppt_use_vlm=ppt_use_vlm,
                textbook_output_dir=textbook_output_dir,
            )
            if result and result["text"]:
                results.append(result)
                print(f"  完成: {len(result['text'])} 字符")
            else:
                print(f"  跳过: 无内容")
        except Exception as e:
            print(f"  [错误] {e}")

    return results
