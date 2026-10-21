"""文档加载 / 清洗 / 分类（最底层的文件处理，不依赖项目里任何其它模块）

为什么单独抽一个模块：`rag_tool`（检索）和 `knowledge_pipeline.pipeline`（入库）
都要用这几个函数，而它们原来住在 `rag_tool` 里 —— 于是形成

    rag_tool  →  pipeline（函数内延迟导入，用来跑流水线）
    pipeline  →  rag_tool（顶层导入，拿这几个函数）

的**循环依赖**，只是被"函数内延迟导入"掩盖着，导入顺序一变就可能炸。
把共用部分下沉到这里之后，依赖变成单向：

    doc_utils ← rag_tool
    doc_utils ← knowledge_pipeline.pipeline
    rag_tool  → knowledge_pipeline.pipeline（仍然是延迟导入，可选功能）

本模块只依赖标准库 + 按需惰性导入 pypdf / python-docx。
"""

import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------
# PDF 里的"部首字符"要归一成正常汉字（2026-10-18 实测发现）
#
# 现象：新导入的《2-01-原子操作.pdf》《2-02-临界区、互斥与锁.pdf》抽出来是
#     「原⼦操作」「使⽤」
# 其中 U+2F26（KANGXI RADICAL CHILD）之类的**康熙部首/部首补充**字符，跟正常的
# 「子」「用」是**不同码位**。后果有两层：
#   1. 检索命不中 —— 学生/模型说「原子操作」，BM25 与向量都碰不到那份课件
#      （实测：查「原子操作」只回 04-并发与同步 与课本，新课件一条都不出）；
#   2. 模型读到的课件里混着怪字，讲课会照抄。
#
# 归一范围刻意收窄到「部首类 + CJK 兼容表意文字」，**不做整个 NFKC** ——
# 全量 NFKC 会把 ①→1、㎡→m2、全角字母变半角，那些在本项目里是噪声改动。
# ---------------------------------------------------------------


def _build_radical_table() -> dict:
    """部首/兼容表意字 → 正常汉字（只收真的会变的那些）"""
    table = {}
    for cp in (list(range(0x2E80, 0x2EFF + 1))       # CJK Radicals Supplement
               + list(range(0x2F00, 0x2FDF + 1))      # Kangxi Radicals
               + list(range(0xF900, 0xFAFF + 1))):    # CJK Compatibility Ideographs
        fixed = unicodedata.normalize("NFKC", chr(cp))
        if fixed != chr(cp):
            table[cp] = fixed
    return table


_RADICAL_TABLE = _build_radical_table()


def normalize_glyphs(text: str) -> str:
    """把 PDF 抽出来的部首字符归一成正常汉字（幂等，不碰 ①/㎡/全角字母）"""
    if not text:
        return text
    return text.translate(_RADICAL_TABLE)


def load_document(path: str) -> str:
    """加载文档文件，返回纯文本

    支持:
      .pdf   → pypdf 提取
      .docx  → python-docx 提取
      .md/.txt → 直接读取

    Args:
        path: 文件路径
    Returns:
        文件纯文本内容
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix == ".docx":
        return _load_docx(path)
    elif suffix in (".md", ".txt", ".markdown"):
        return _load_text(path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}（仅支持 .pdf/.docx/.md/.txt）")

def _load_pdf(path: str) -> str:
    """用 pypdf 提取 PDF 文本"""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def _load_docx(path: str) -> str:
    """用 python-docx 提取 Word 文档文本"""
    import docx
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def _load_text(path: str) -> str:
    """读取纯文本文件（MD / TXT）"""
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()

def clean_text(text: str) -> str:
    """清洗文档文本

    处理:
      0. **部首/兼容表意字归一**（PDF 里常见的怪字，见上方 `normalize_glyphs`）
      1. 去除多余空行和空格
      2. 合并被换行打断的句子（中文句号/问号/叹号后不应有换行）
      3. 去除页眉页脚模式（如"第X章"/"Page X"等重复行）
      4. 去除纯数字行（页码）
      5. 标准化空白字符
    """
    # 0. 怪字归一：必须在分块/建索引之前做，否则检索永远命不中（见上）
    text = normalize_glyphs(text)

    # 1. 统一换行符，标准化空白
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # 2. 去除纯数字行（页码）
    lines = text.split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*\d+\s*$", ln)]

    # 3. 去除常见页眉页脚模式
    page_patterns = [
        r"^第[一二三四五六七八九十\d]+页$",
        r"^Page\s*\d+$",
        r"^\d+\s*/\s*\d+$",
        r"^目\s*录$",
    ]
    lines = [ln for ln in lines if not any(re.match(p, ln.strip(), re.IGNORECASE) for p in page_patterns)]

    # 4. 合并被换行打断的句子
    text = "\n".join(lines)
    text = re.sub(r"([。！？；])\n([^\n])", r"\1\2", text)
    text = re.sub(r"([.!?])\n([a-z])", r"\1 \2", text)

    # 5. 压缩连续空行（最多保留1个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()#去掉文本最开头和最结尾的空白

def _extract_chapter(file_path: Path, base_dir: Path) -> str:
    """从文件路径提取章节信息

    支持的格式:
      - "02-1-进程与线程之进程" → 进程与线程
      - "Chapter4-Network Layer" → Network Layer
      - "操作系统_补充知识点笔记" → 补充知识点笔记
      - "计算机网络_传输层" → 传输层
      - "《计算机网络》复习整理文档" → 复习整理文档
    """
    name = file_path.stem

    # 匹配 "02-1-进程与线程之进程" 或 "03-OS调度" 格式
    m = re.match(r"^\d+[-_]?\d*[-_](.+)", name)
    if m:
        chapter = m.group(1)
        chapter = re.sub(r"之.+$", "", chapter)
        return chapter

    # 匹配 "Chapter4-Network Layer" 格式
    m = re.match(r"^Chapter\d+[-_](.+)", name, re.IGNORECASE)
    if m:
        return m.group(1)

    # 匹配 "<课程名>_<章节>" 格式（课程清单来自 data/ 目录）
    from course_agent.courses import known_courses

    course_names = known_courses()
    if course_names:
        m = re.match(r"^(?:" + "|".join(map(re.escape, course_names)) + r")[_\-](.+)", name)
        if m:
            return m.group(1)

    # 匹配 "《计算机网络》复习整理文档" 格式
    m = re.match(r"^《.+》(.+)", name)
    if m:
        return m.group(1)

    return ""

def _classify_doc(file_path: Path, base_dir: Path) -> str:
    """根据文件路径判断文档类型

    规则:
      - 路径含 "往年题" → 跳过（返回 None，不入库）
      - 路径含 "课本" → textbook（课本，最低优先级，兜底用）
      - 路径含 "ppt" → ppt（课件，优先级最高）
      - 其他 → notes（笔记等）
    """
    # 两边都要 resolve 后再算相对路径：只判断不 resolve、计算时用原值，
    # 就会出现"判断通过但 relative_to 抛 ValueError"的错配
    try:
        rel_parts = Path(file_path).resolve().relative_to(Path(base_dir).resolve()).parts
    except (ValueError, OSError):
        # 传入绝对路径而 base 是相对路径（文档里就推荐 --add-file "<绝对路径>"），
        # 或 base 不是它的父目录时，退回按全路径判断文档类型
        rel_parts = Path(file_path).parts
    for part in rel_parts:
        if "往年题" in part:
            return None  # 跳过
        if "课本" in part:
            return "textbook"
        if "ppt" in part.lower():
            return "ppt"
    return "notes"
