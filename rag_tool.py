import os
import re
import math
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

import chromadb

from config import ZHIPU_API_KEY, ZHIPU_EMBEDDING_URL, ZHIPU_CHAT_URL
from course_agent.errors import retry_on_rate_limit, EmbeddingError
from course_agent.logger import logger

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 未安装，混合检索降级为纯向量检索")


# ============================================================
# Embedding 函数
# ============================================================

_embedding_headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json",
}


@retry_on_rate_limit(max_retries=3, base_wait=3.0)
def _call_embedding_api(payload: dict, timeout: int = 60) -> dict:
    """调用智谱 embedding-3 API（单次请求），被 retry_on_rate_limit 装饰器保护"""
    resp = requests.post(ZHIPU_EMBEDDING_URL, headers=_embedding_headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_embedding(text: str) -> list[float]:
    """调用智谱 embedding-3 API，返回文本的向量"""
    payload = {"model": "embedding-3", "input": text}
    try:
        data = _call_embedding_api(payload, timeout=60)
        return data["data"][0]["embedding"]
    except Exception as e:
        raise EmbeddingError(f"单条 embedding 失败: {type(e).__name__}: {str(e)[:200]}") from e


def get_embeddings_batch(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """批量调用智谱 embedding-3 API，返回多条文本的向量列表"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": "embedding-3", "input": batch}
        try:
            data = _call_embedding_api(payload, timeout=120)
            batch_embs = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embs)
        except Exception as e:
            raise EmbeddingError(
                f"批量 embedding 失败 (batch {i}-{i+len(batch)}): {type(e).__name__}: {str(e)[:200]}"
            ) from e
        time.sleep(0.5)
    return all_embeddings


# ============================================================
# 文档加载
# ============================================================

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

    # 匹配 "操作系统_补充知识点笔记" 或 "计算机网络_传输层" 格式
    m = re.match(r"^(?:操作系统|计算机网络)[_\-](.+)", name)
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
    rel_parts = file_path.relative_to(base_dir).parts
    for part in rel_parts:
        if "往年题" in part:
            return None  # 跳过
        if "课本" in part:
            return "textbook"
        if "ppt" in part.lower():
            return "ppt"
    return "notes"


def load_documents_from_dir(dir_path: str, skip_textbook: bool = False) -> list[dict]:
    """从目录批量加载文档（递归扫描子目录，跳过往年题）

    Args:
        dir_path: 文档目录路径
        skip_textbook: 是否跳过课本（True 时只加载 ppt 和 notes）
    Returns:
        [{"source": "文件名", "text": "内容", "doc_type": "ppt/textbook/notes"}, ...]
    """
    results = []
    p = Path(dir_path)
    if not p.exists():
        logger.warning("目录不存在: %s", dir_path)
        return results

    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".pdf", ".docx", ".md", ".txt", ".markdown"):
            continue

        doc_type = _classify_doc(f, p)
        if doc_type is None:
            continue  # 跳过往年题
        if skip_textbook and doc_type == "textbook":
            continue  # 跳过课本

        try:
            text = load_document(str(f))
            chapter = _extract_chapter(f, p)
            results.append({
                "source": f.name,
                "text": text,
                "doc_type": doc_type,
                "chapter": chapter,
                "file_path": str(f),
            })
            logger.info("加载 [%s] %s (%d 字符, 章节: %s)",
                        doc_type, f.name, len(text), chapter or "未知")
        except Exception as e:
            logger.warning("跳过文件 %s: %s", f.name, e)
    return results

# ============================================================
#文档清洗
# ============================================================

def clean_text(text: str) -> str:
    """清洗文档文本

    处理:
      1. 去除多余空行和空格
      2. 合并被换行打断的句子（中文句号/问号/叹号后不应有换行）
      3. 去除页眉页脚模式（如"第X章"/"Page X"等重复行）
      4. 去除纯数字行（页码）
      5. 标准化空白字符
    """
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


# ============================================================
# 分块策略
# ============================================================

@dataclass
class ChunkConfig:
    """分块配置"""
    chunk_size: int = 300
    overlap: int = 50
    strategy: str = "fixed"  # "fixed" | "paragraph"


def chunk_text(text: str, config: ChunkConfig = None) -> list[str]:
    """文本分块

    策略:
      fixed    → 固定长度分块（带重叠），适合通用场景
      paragraph → 按段落分块（空行分隔），保持语义完整性
    """
    if config is None:
        config = ChunkConfig()

    if config.strategy == "paragraph":
        return _chunk_by_paragraph(text, config)
    else:
        return _chunk_fixed(text, config)


def _chunk_fixed(text: str, config: ChunkConfig) -> list[str]:
    """固定长度分块（带重叠）"""
    chunks = []
    start = 0
    size = config.chunk_size
    overlap = config.overlap
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _chunk_by_paragraph(text: str, config: ChunkConfig) -> list[str]:
    """按段落分块（空行分隔），超长段落再按固定长度切"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= config.chunk_size:
            chunks.append(para)
        else:
            # 超长段落再按固定长度切
            sub_chunks = _chunk_fixed(para, config)
            chunks.extend(sub_chunks)
    return chunks


# ============================================================
# Chroma 向量数据库
# ============================================================

class KnowledgeBase:
    """知识库管理器：封装 Chroma 向量数据库

    用法:
      # 建库
      kb = KnowledgeBase(persist_dir="./vector_store")
      kb.build_from_dir("./data")

      # 检索
      results = kb.search("什么是TCP三次握手", top_k=3)
    """

    #persist_dir: 库存哪, collection_name: 集合名
    def __init__(self, persist_dir: str = "./vector_store", collection_name: str = "course_kb"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None  # 懒加载
        self._collection = None

    def _get_collection(self):
        """懒加载 Chroma collection（不绑定 embedding function，自己控制 API 调用）"""
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "课程知识库"}
            )
        return self._collection

    # --- BM25 索引（懒加载） ---
    _bm25: Optional[object] = None
    _bm25_corpus: list[str] = []  # 与 BM25 索引对齐的文本列表
    _bm25_ids: list[str] = []     # 对应的 chunk id

    def _get_bm25(self):
        """懒加载 BM25 索引"""
        if not HAS_BM25:
            return None
        if self._bm25 is None:
            collection = self._get_collection()
            if collection.count() == 0:
                return None
            all_data = collection.get()
            corpus = []
            for doc in all_data["documents"]:
                tokens = list(doc)
                corpus.append(tokens)
            self._bm25_corpus = all_data["documents"]
            self._bm25_ids = all_data["ids"]
            self._bm25 = BM25Okapi(corpus)
            logger.info("BM25 索引构建完成: %d 个文档", len(corpus))
        return self._bm25

    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 关键词检索"""
        bm25 = self._get_bm25()
        if bm25 is None:
            return []
        tokens = list(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append({
                "id": self._bm25_ids[idx],
                "text": self._bm25_corpus[idx],
                "bm25_score": float(score),
            })
        return results

    def _hybrid_search(
        self, query: str, top_k: int = 20, doc_types: list[str] = None, chapter: str = None
    ) -> list[dict]:
        """混合检索：向量 + BM25 融合召回"""
        # 向量检索
        vector_results = self.search(query, top_k=top_k, doc_types=doc_types, chapter=chapter)

        # BM25 检索
        bm25_results = self._bm25_search(query, top_k=top_k)

        # 用 id 合并，向量分数归一化
        merged = {}
        max_vec = max((r["score"] for r in vector_results), default=1)
        for r in vector_results:
            merged[r["text"][:100]] = {
                **r,
                "vec_score": r["score"] / max(max_vec, 0.001),
                "bm25_score": 0.0,
            }
        max_bm25 = max((r["bm25_score"] for r in bm25_results), default=1)
        for r in bm25_results:
            collection = self._get_collection()
            chroma_data = collection.get(ids=[r["id"]])
            if not chroma_data["ids"]:
                continue
            meta = chroma_data["metadatas"][0]
            if doc_types and meta.get("doc_type") not in doc_types:
                continue
            if chapter and meta.get("chapter") != chapter:
                continue
            text = chroma_data["documents"][0]
            key = text[:100]
            bm25_norm = r["bm25_score"] / max(max_bm25, 0.001)
            if key in merged:
                merged[key]["bm25_score"] = bm25_norm
            else:
                merged[key] = {
                    "text": text,
                    "source": meta.get("source", ""),
                    "doc_type": meta.get("doc_type", "notes"),
                    "chapter": meta.get("chapter", ""),
                    "score": 0,
                    "vec_score": 0.0,
                    "bm25_score": bm25_norm,
                    "chunk_index": meta.get("chunk_index", 0),
                }
        # 融合分数
        for v in merged.values():
            v["score"] = round(0.6 * v["vec_score"] + 0.4 * v["bm25_score"], 4)
        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    def build_from_dir(
        self,
        docs_dir: str,
        chunk_config: ChunkConfig = None,
        skip_textbook: bool = False,
        use_pipeline: bool = False,
        ppt_use_vlm: bool = False,
        textbook_output_dir: str = None,
    ):
        """从目录加载文档 → 处理 → 分块 → embedding → 入库

        Args:
            docs_dir: 文档目录
            chunk_config: 分块配置
            skip_textbook: 是否跳过课本
            use_pipeline: 是否使用新的知识处理流水线
                False = 旧模式（pypdf 直接提取文字）
                True  = 新模式（PPT 按页+图文合并、课本生成笔记）
            ppt_use_vlm: PPT 是否用 VLM 理解图片内容（较慢，需额外 API 调用）
            textbook_output_dir: 课本笔记输出目录

        关键：自己调 get_embedding() 而不是让 Chroma 内部调，
        这样可以控制 API 调用节奏，避免 429 限流。
        """
        if chunk_config is None:
            chunk_config = ChunkConfig()

        # 选择加载方式
        if use_pipeline:
            from knowledge_pipeline.pipeline import process_directory
            logger.info("使用知识处理流水线")
            docs = process_directory(
                docs_dir,
                ppt_use_vlm=ppt_use_vlm,
                textbook_output_dir=textbook_output_dir,
                skip_textbook=skip_textbook,
            )
        else:
            docs = load_documents_from_dir(docs_dir, skip_textbook=skip_textbook)

        if not docs:
            logger.warning("没有加载到任何文档")
            return 0

        collection = self._get_collection()

        import time
        success_count = 0
        total_chunks = 0

        for doc in docs:
            text = doc["text"] if use_pipeline else clean_text(doc["text"])
            chunks = chunk_text(text, chunk_config)
            source = doc["source"]
            doc_type = doc.get("doc_type", "notes")
            chapter = doc.get("chapter", "")

            try:
                embeddings = get_embeddings_batch(chunks)
            except Exception as e:
                logger.warning("跳过文档 %s: %s", source, e)
                total_chunks += len(chunks)
                continue

            ids = [f"{source}_{i}" for i in range(len(chunks))]
            metas = [
                {
                    "source": source,
                    "doc_type": doc_type,
                    "chapter": chapter,
                    "chunk_index": i,
                    "char_count": len(c),
                }
                for i, c in enumerate(chunks)
            ]
            collection.upsert(
                ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings,
            )
            success_count += len(chunks)
            total_chunks += len(chunks)
            logger.debug("%s: %d 块 (章节: %s)", source, len(chunks), chapter or "未知")

        logger.info("入库完成: %d/%d 个分块，来源: %d 个文件", success_count, total_chunks, len(docs))
        logger.info("存储路径: %s", self.persist_dir)
        return success_count

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_types: list[str] = None,
        chapter: str = None,
    ) -> list[dict]:
        """检索知识库

        自己调 get_embedding() 把 query 向量化，然后用 query_embeddings 查询，
        不让 Chroma 内部调 embedding function。

        Args:
            query: 搜索关键词
            top_k: 返回的文档数
            doc_types: 限定搜索的文档类型列表（如 ["ppt", "notes"]），
                       None 表示不限制
            chapter: 限定章节（如 "进程与线程"），None 表示不限制
        Returns:
            [{"text": "...", "source": "...", "score": 0.85, "chunk_index": 3, "chapter": "..."}, ...]
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_emb = get_embedding(query)

        where_filter = None
        if doc_types and chapter:
            where_filter = {
                "$and": [
                    {"doc_type": {"$in": doc_types}},
                    {"chapter": chapter},
                ]
            }
        elif doc_types:
            where_filter = {"doc_type": {"$in": doc_types}}
        elif chapter:
            where_filter = {"chapter": chapter}

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
        )

        formatted = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = round(1 - distance, 4)
            formatted.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "doc_type": meta.get("doc_type", "notes"),
                "chapter": meta.get("chapter", ""),
                "score": score,
                "chunk_index": meta.get("chunk_index", 0),
            })
        return formatted

    def count(self) -> int:
        """返回知识库中的分块总数"""
        collection = self._get_collection()
        return collection.count()

    def list_documents(self) -> list[dict]:
        """列出知识库中所有文档（按 source 聚合）

        Returns:
            [{"source": "02-1-进程与线程之进程.pdf", "doc_type": "ppt",
              "chapter": "进程与线程", "chunk_count": 15}, ...]
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        all_data = collection.get()
        docs_map = {}
        for i, meta in enumerate(all_data["metadatas"]):
            source = meta.get("source", "unknown")
            if source not in docs_map:
                docs_map[source] = {
                    "source": source,
                    "doc_type": meta.get("doc_type", "notes"),
                    "chapter": meta.get("chapter", ""),
                    "chunk_count": 0,
                }
            docs_map[source]["chunk_count"] += 1

        return sorted(docs_map.values(), key=lambda x: x["source"])

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        docs = self.list_documents()
        total_chunks = sum(d["chunk_count"] for d in docs)

        by_type = {}
        by_chapter = {}
        for d in docs:
            t = d["doc_type"]
            by_type[t] = by_type.get(t, 0) + d["chunk_count"]
            ch = d["chapter"] or "未分类"
            by_chapter[ch] = by_chapter.get(ch, 0) + d["chunk_count"]

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "by_type": by_type,
            "by_chapter": by_chapter,
            "documents": docs,
        }

    def add_document(
        self,
        file_path: str,
        doc_type: str = None,
        chapter: str = None,
        chunk_config: ChunkConfig = None,
    ) -> int:
        """增量导入单个文件，不重建整个知识库

        Args:
            file_path: 文件路径
            doc_type: 文档类型（None 则自动分类）
            chapter: 章节信息（None 则从文件名提取）
            chunk_config: 分块配置
        Returns:
            入库的分块数
        """
        if chunk_config is None:
            chunk_config = ChunkConfig()

        p = Path(file_path)
        if not p.exists():
            logger.warning("文件不存在: %s", file_path)
            return 0

        # 自动分类
        if doc_type is None:
            base_dir = Path("data")
            doc_type = _classify_doc(p, base_dir)
            if doc_type is None:
                logger.info("跳过往年题: %s", p.name)
                return 0

        # 自动提取章节
        if chapter is None:
            chapter = _extract_chapter(p, Path("data"))

        # 计算文件 hash
        file_hash = hashlib.md5(p.read_bytes()).hexdigest()
        processed = self._load_processed()

        # 已处理且 hash 没变 → 跳过
        record = processed.get(str(p))
        if record and record.get("file_hash") == file_hash:
            logger.info("文件未变化，跳过: %s", p.name)
            return 0

        # 先删旧的（如果是更新）
        if record:
            self.remove_document(p.name)

        # 加载 + 清洗 + 分块
        text = clean_text(load_document(str(p)))
        chunks = chunk_text(text, chunk_config)

        if not chunks:
            logger.warning("文件内容为空，跳过: %s", p.name)
            return 0

        # embedding + 入库
        try:
            embeddings = get_embeddings_batch(chunks)
        except Exception as e:
            logger.warning("embedding 失败 %s: %s", p.name, e)
            return 0

        source = p.name
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metas = [
            {
                "source": source,
                "doc_type": doc_type,
                "chapter": chapter,
                "chunk_index": i,
                "char_count": len(c),
            }
            for i, c in enumerate(chunks)
        ]

        collection = self._get_collection()
        collection.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)

        # 更新记录
        processed[str(p)] = {
            "source": source,
            "doc_type": doc_type,
            "chapter": chapter,
            "chunk_count": len(chunks),
            "file_hash": file_hash,
            "added_at": datetime.now().isoformat(),
        }
        self._save_processed(processed)

        logger.info("增量导入完成: %s (%d 块, 章节: %s)", source, len(chunks), chapter or "未知")
        return len(chunks)

    def remove_document(self, source: str) -> bool:
        """从知识库删除某文档的所有分块

        Args:
            source: 文件名（如 "02-1-进程与线程之进程.pdf"）
        Returns:
            是否删除成功
        """
        collection = self._get_collection()

        # 先查是否存在
        existing = collection.get(where={"source": source})
        if not existing["ids"]:
            logger.info("文档不在知识库中: %s", source)
            return False

        collection.delete(where={"source": source})

        # 更新记录
        processed = self._load_processed()
        to_remove = [k for k, v in processed.items() if v.get("source") == source]
        for k in to_remove:
            del processed[k]
        if to_remove:
            self._save_processed(processed)

        logger.info("已删除文档: %s (%d 块)", source, len(existing["ids"]))
        return True

    def add_directory(
        self,
        dir_path: str,
        chunk_config: ChunkConfig = None,
        skip_textbook: bool = False,
    ) -> dict:
        """增量导入目录下的所有新文件

        扫描目录，只处理新增或修改过的文件。

        Returns:
            {"added": 3, "skipped": 5, "failed": 0, "details": [...]}
        """
        if chunk_config is None:
            chunk_config = ChunkConfig()

        p = Path(dir_path)
        if not p.exists():
            logger.warning("目录不存在: %s", dir_path)
            return {"added": 0, "skipped": 0, "failed": 0, "details": []}

        result = {"added": 0, "skipped": 0, "failed": 0, "details": []}

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

            try:
                added = self.add_document(str(f), doc_type=doc_type, chunk_config=chunk_config)
                if added > 0:
                    result["added"] += 1
                    result["details"].append({"file": f.name, "status": "added", "chunks": added})
                else:
                    result["skipped"] += 1
                    result["details"].append({"file": f.name, "status": "skipped"})
            except Exception as e:
                result["failed"] += 1
                result["details"].append({"file": f.name, "status": "failed", "error": str(e)[:100]})
                logger.warning("导入失败 %s: %s", f.name, e)

        logger.info("增量导入目录完成: 新增 %d, 跳过 %d, 失败 %d",
                    result["added"], result["skipped"], result["failed"])
        return result

    def _processed_file_path(self) -> Path:
        """已处理文件清单的路径"""
        return Path(self.persist_dir) / "processed_files.json"

    def _load_processed(self) -> dict:
        """加载已处理文件清单"""
        p = self._processed_file_path()
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_processed(self, data: dict):
        """保存已处理文件清单"""
        p = self._processed_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clear(self):
        """清空知识库"""
        if self._client is None:
            self._get_collection()
        self._client.delete_collection(self.collection_name)
        self._collection = None
        # 清空已处理文件清单
        p = self._processed_file_path()
        if p.exists():
            p.unlink()
        logger.info("已清空 collection: %s", self.collection_name)


# ============================================================
# 对外接口：search_knowledge（供 LangGraph @tool 和 FastAPI 调用）
# ============================================================

# 全局知识库实例（懒加载）
_kb_instance: Optional[KnowledgeBase] = None #变量名:类型注解=初始值

def get_knowledge_base(persist_dir: str = "./vector_store") -> KnowledgeBase:
    """获取全局知识库实例（单例）"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(persist_dir=persist_dir)
    return _kb_instance

def _llm_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """用 LLM 对检索结果重排序

    把候选列表喂给 GLM-4-Flash，让它根据与查询的相关性打分排序
    """
    if len(candidates) <= top_k:
        return candidates[:top_k]

    LLM_MODEL = "glm-4-flash"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    # 构建候选列表文本（编号 + 摘要）
    candidate_texts = []
    for i, c in enumerate(candidates):
        snippet = c["text"][:120].replace("\n", " ")
        candidate_texts.append(f"[{i}] {snippet}")

    prompt = (
        "你是检索重排序器。用户问题和候选文本片段如下，"
        "请按与问题的相关性从高到低输出编号，用逗号分隔，只输出编号。\n\n"
        f"问题: {query}\n\n"
        + "\n".join(candidate_texts)
        + "\n\n输出格式: 0,3,1,2,...（只输出编号，用逗号分隔）"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    try:
        resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        # 解析编号
        indices = []
        for part in re.split(r"[,\s]+", result):
            part = part.strip("[]()")
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates) and idx not in indices:
                    indices.append(idx)
        if not indices:
            return candidates[:top_k]
        reranked = [candidates[i] for i in indices[:top_k]]
        # 补全不足的部分
        if len(reranked) < top_k:
            for c in candidates:
                if c not in reranked:
                    reranked.append(c)
                    if len(reranked) >= top_k:
                        break
        logger.info("LLM rerank: %d → %d", len(candidates), len(reranked))
        return reranked
    except Exception as e:
        logger.warning("LLM rerank 失败，降级为原始排序: %s", str(e)[:100])
        return candidates[:top_k]


def _llm_rewrite_query(query: str) -> list[str]:
    """用 LLM 改写查询，生成更适合检索的关键词查询

    返回原始查询 + 改写后的关键词查询
    """
    LLM_MODEL = "glm-4-flash"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "你是搜索查询改写器。把用户问题改写成 2-3 个适合向量检索和关键词检索的查询变体。\n"
        "要求：保留核心概念，去掉问句词，补全缩写全称，用空格分隔关键词。\n"
        "输出格式：每行一个查询，不要编号，不要解释。\n\n"
        f"用户问题: {query}\n"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }

    try:
        resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        rewrites = [line.strip() for line in result.split("\n") if line.strip()]
        queries = [query] + rewrites[:3]
        # 去重
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        logger.info("LLM 查询改写: %s → %s", query, unique)
        return unique
    except Exception as e:
        logger.warning("LLM 查询改写失败，降级为正则扩展: %s", str(e)[:100])
        return _expand_query(query)


def _is_low_quality_chunk(text: str) -> bool:
    """检测低质量 chunk

    过滤：纯数字/符号、过短、页眉页脚残留
    """
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    # 纯数字和标点
    if re.match(r"^[\d\s\.\-/:，。、]+$", stripped):
        return True
    # 页眉页脚模式
    if re.match(r"^(第[一二三四五六七八九十\d]+页|Page\s*\d+|目录)$", stripped, re.IGNORECASE):
        return True
    # 中文字符占比过低（可能是乱码）
    chinese_count = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff' or c.isalpha())
    if len(stripped) > 0 and chinese_count / len(stripped) < 0.1:
        return True
    return False


def _expand_query(query: str) -> list[str]:
    """查询扩展：从原始查询生成多个搜索变体

    解决 LLM 生成的查询太简短导致检索相似度低的问题。
    例如 "什么是临界区" → ["什么是临界区", "临界区"]
    例如 "临界区 准则" → ["临界区 准则", "临界区"]
    """
    queries = [query]

    # 去掉常见问句词，保留核心概念
    cleaned = re.sub(r"(什么是|是什么|有哪些|怎样|如何|为什么|简述|描述|说明|的|？|\?|请)", " ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned != query:
        queries.append(cleaned)

    # 按空格/逗号拆分成关键词
    keywords = [kw.strip() for kw in re.split(r"[\s,，、]+", cleaned) if len(kw.strip()) > 1]

    # 如果有多个关键词，每个关键词单独搜一次（召回更多相关块）
    if len(keywords) >= 2:
        for kw in keywords:
            if kw not in queries:
                queries.append(kw)

    # 去重
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:4]  # 最多 4 个变体


def search_knowledge(query: str, top_k: int = 5) -> str:
    """检索课程知识库（混合检索 + LLM重排序 + 查询改写）

    流程:
    1. LLM 查询改写: 生成检索变体
    2. 混合检索: 向量 + BM25 多路召回，融合排序
    3. LLM Rerank: 对 top_k*4 候选重排序到 top_k
    4. 两层检索: 先搜 ppt+notes，不足再搜课本
    5. Chunk 质量过滤: 过滤低质量结果
    """
    kb = get_knowledge_base()

    # 1. LLM 查询改写（失败降级为正则扩展）
    queries = _llm_rewrite_query(query)

    # 2. 混合检索（多查询 + 合并去重）
    seen_texts = {}
    recall_k = max(top_k * 4, 20)

    for q in queries:
        results = kb._hybrid_search(q, top_k=recall_k, doc_types=["ppt", "notes"])
        for r in results:
            if _is_low_quality_chunk(r["text"]):
                continue
            text_key = r["text"][:100]
            if text_key not in seen_texts or r["score"] > seen_texts[text_key]["score"]:
                seen_texts[text_key] = r

    all_results = sorted(seen_texts.values(), key=lambda x: x["score"], reverse=True)

    # 3. 第二层：如果第一层结果不足，补充搜课本
    if len(all_results) < 2 or (all_results and all_results[0]["score"] < 0.3):
        for q in queries:
            textbook_results = kb._hybrid_search(q, top_k=5, doc_types=["textbook"])
            for r in textbook_results:
                if _is_low_quality_chunk(r["text"]):
                    continue
                text_key = r["text"][:100]
                if text_key not in seen_texts or r["score"] > seen_texts[text_key]["score"]:
                    seen_texts[text_key] = r
        all_results = sorted(seen_texts.values(), key=lambda x: x["score"], reverse=True)

    if not all_results:
        return "知识库中未找到相关内容。建议使用 web_search 工具联网搜索获取更多信息。"

    # 4. LLM Rerank（候选数 > top_k 时才触发）
    if len(all_results) > top_k:
        all_results = _llm_rerank(query, all_results, top_k=top_k)
    else:
        all_results = all_results[:top_k]

    # 5. 格式化输出
    best_score = all_results[0]["score"] if all_results else 0
    result_count = len(all_results)

    formatted = []
    for i, r in enumerate(all_results[:top_k], 1):
        type_label = {
            "ppt": "课件",
            "textbook": "课本",
            "notes": "笔记",
        }.get(r["doc_type"], r["doc_type"])
        formatted.append(
            f"[{i}] (来源: {r['source']}, 类型: {type_label})\n{r['text']}"
        )

    result_text = "\n\n".join(formatted)

    if best_score < 0.3 or result_count < 2:
        result_text += "\n\n[提示] 知识库中相关内容较少或相关度不高，建议调用 web_search 工具联网搜索补充信息。"

    return result_text



