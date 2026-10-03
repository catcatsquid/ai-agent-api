"""知识库检索与入库 —— 全项目唯一的向量检索入口

## 它负责什么

  · **入库**：把课本 / 课件 / 笔记切成块、算 embedding、写进 ChromaDB；
    另有增量导入（按文件内容 md5 跳过）与删除。
  · **检索**：向量检索 + BM25 关键字检索，融合后排序，可按课程与文档类型过滤。
  · **统计**：给 /kb/stats 提供分块数、类型分布、来源分布。

## 两个容易踩错的地方（都在这里修过）

1. **L2 距离不是相似度**。向量库是 L2 空间，`chromadb` 返回的是**距离**，
   而阈值判断需要**余弦相似度** —— 必须换算（单位向量下 `sim = 1 - d²/2`），
   否则域外查询（d > 1）会算出负分。见 `distance_to_similarity()`。

2. **归一化融合分不能当阈值判据**。混合检索的 `score` 是按最大值归一化的
   （排名第一恒 ≥ 0.6），只能用于**排序**；判断"知识库里到底有没有相关内容"
   必须用**绝对**相似度（`sim` 字段）。见 `search()` 的返回结构与
   `config.RELEVANCE_THRESHOLD`（域内最低 0.665 / 域外最高 0.524）。

## 依赖方向

本模块依赖 `doc_utils`（文档加载/清洗/分类）与 `course_agent.errors`/`logger`。
`knowledge_pipeline.pipeline` 只在**函数内延迟导入**（"用流水线入库"是可选功能）——
这样和 pipeline 之间不构成循环依赖，导入顺序不敏感。

题库相关的读写**不在这里**，走 `course_agent/qbank.py`。
"""

import os
import re
import threading
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

from config import (ZHIPU_API_KEY, ZHIPU_EMBEDDING_URL, ZHIPU_CHAT_URL,
                    RELEVANCE_THRESHOLD)
from course_agent.errors import retry_on_rate_limit, EmbeddingError
from course_agent.logger import logger

# 文档加载/清洗/分类已下沉到顶层 doc_utils（断开与 knowledge_pipeline 的循环依赖）。
# 这里保留同名绑定，历史调用点（含 tests/ 与 debug/ 脚本）不需要改。
from doc_utils import (load_document, clean_text, _classify_doc,  # noqa: F401
                       _extract_chapter)

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

def _collection_space(collection) -> str:
    """读取 collection 实际的向量距离空间（l2 / cosine / ip）"""
    cfg = getattr(collection, "configuration", None) or {}
    hnsw = (cfg.get("hnsw") or {}) if isinstance(cfg, dict) else {}
    return str(hnsw.get("space") or "l2").lower()


def distance_to_similarity(distance: float, space: str = "l2") -> float:
    """把 Chroma 返回的距离换算成余弦相似度

    这是原来最容易被忽略的一处：collection 建的是默认的 L2 空间，
    而代码却按 `score = 1 - distance` 当成余弦相似度用。

    对**已归一化**的向量（本项目 4199 条向量模长实测全为 1.0）有严格关系：
        ||a - b||² = 2 - 2·cos(a,b)   =>   cos = 1 - d²/2

    实测影响（debug/check_threshold.py，域内/域外各 12 条查询）:
        score = 1 - d     : 域内最低 0.181，域外最高 0.024，且域内只有 7/12 过得了 0.3
        cos   = 1 - d²/2  : 域内最低 0.665，域外最高 0.524，24 条样本完全分开
    也就是说旧算法会把一半的**课内**问题误判成"知识库相关度不高"，
    从而每次多做一次课本层检索并提示联网搜索。
    """
    d = float(distance or 0.0)
    if space == "l2":
        sim = 1.0 - (d ** 2) / 2.0
    else:
        # cosine / ip 返回的已经是 1 - cos
        sim = 1.0 - d
    return max(-1.0, min(1.0, sim))


def _is_relative_to(path: Path, base: Path) -> bool:
    """path 是否在 base 之下（不解析符号链接，路径不存在也不报错）

    原来直接 file_path.relative_to(base_dir)：只要传的是绝对路径
    （文档里就推荐 --add-file "<Windows 绝对路径>"）或 base 写成相对路径，
    就会抛 ValueError，整个导入流程崩掉。
    """
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except (ValueError, OSError):
        return False


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
    """固定长度分块（带重叠）

    必须保证游标每轮都前进：原来 `start = end - overlap`，
    当 overlap >= chunk_size 时 start 原地踏步甚至倒退，
    `while start < len(text)` 永不结束且 chunks 无限增长
    （build_knowledge.py --chunk-size 100 --overlap 100 就能触发）。
    """
    chunks = []
    start = 0
    size = max(1, config.chunk_size)
    # overlap 必须小于 size，否则无法推进
    overlap = min(max(0, config.overlap), size - 1)
    if config.overlap != overlap:
        logger.warning("overlap=%s 非法（需 0 <= overlap < chunk_size=%s），已按 %s 处理",
                       config.overlap, size, overlap)
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
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
        # 懒加载都要加锁：FastAPI 的同步端点跑在线程池里，
        # check-then-act 会让并发首次检索重复建 client / 重复全量建 BM25 索引，
        # 更糟的是 _bm25 已就绪而 _bm25_ids 正被替换时会 IndexError。
        # 必须是 RLock（可重入）：_get_bm25 持锁后还会调 _get_collection，
        # 普通 Lock 会在这里自锁死。
        self._init_lock = threading.RLock()

    def _get_collection(self):
        """懒加载 Chroma collection（不绑定 embedding function，自己控制 API 调用）"""
        if self._collection is None:
            with self._init_lock:
                if self._collection is None:
                    self._client = chromadb.PersistentClient(path=self.persist_dir)
                    # 新建 collection 时显式指定 cosine：
                    # 现有库是历史遗留的 l2 空间，Chroma 不会因为这里传了新 metadata
                    # 就改写已存在 collection 的空间（实测验证过，只是静默沿用 l2），
                    # 所以这行对老库无害，对全新环境则能从一开始就用对的距离度量。
                    self._collection = self._client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"description": "课程知识库", "hnsw:space": "cosine"}
                    )
        return self._collection

    # --- BM25 索引（懒加载） ---
    _bm25: Optional[object] = None
    _bm25_corpus: list[str] = []  # 与 BM25 索引对齐的文本列表
    _bm25_ids: list[str] = []     # 对应的 chunk id

    def _invalidate_bm25(self):
        """任何写操作（入库/删除/清空）后都必须调用

        BM25 索引是进程内缓存，原来只在 _bm25 is None 时重建且从不清空，
        于是服务运行期间新导入的文档对关键词检索永远不可见，混合检索名存实亡。
        """
        with self._init_lock:
            self._bm25 = None
            self._bm25_corpus = []
            self._bm25_ids = []
        return None

    def _get_bm25(self):
        """懒加载 BM25 索引"""
        if not HAS_BM25:
            return None
        if self._bm25 is None:
            with self._init_lock:
                if self._bm25 is None:
                    collection = self._get_collection()
                    if collection.count() == 0:
                        return None
                    all_data = collection.get()
                    corpus = [list(doc) for doc in all_data["documents"]]
                    # 先构造局部变量，最后一次性赋值：避免别的线程看到
                    # "索引已就绪但 ids 还没换" 的中间态
                    bm25 = BM25Okapi(corpus)
                    self._bm25_corpus = all_data["documents"]
                    self._bm25_ids = all_data["ids"]
                    self._bm25 = bm25
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
        # 防御：Chroma 这里用的是默认 L2 空间，而 score = 1 - distance 在
        # distance > 1（余弦为负）时会变成负数。一旦最大分 ≤ 0，
        # max(max_vec, 0.001) 会把分母钳到 0.001，向量分被放大到 ±10^4，
        # 0.6/0.4 的权重彻底失效、排序反转。先把整体平移到非负区间。
        merged = {}
        vec_scores = [r["score"] for r in vector_results]
        shift = min(0.0, min(vec_scores)) if vec_scores else 0.0
        max_vec = max((s - shift for s in vec_scores), default=1.0) or 1.0
        for r in vector_results:
            merged[r["text"][:100]] = {
                **r,
                # sim 是绝对余弦相似度（可跨查询比较，用于"相关内容够不够"的判断）；
                # vec_score 是本次查询内的归一化权重（只用于排序融合），两者不能混用。
                "sim": r["score"],
                "vec_score": (r["score"] - shift) / max(max_vec, 1e-6),
                "bm25_score": 0.0,
            }
        max_bm25 = max((r["bm25_score"] for r in bm25_results), default=1.0) or 1.0
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
            bm25_norm = (r["bm25_score"] / max_bm25) if max_bm25 > 0 else 0.0
            if key in merged:
                merged[key]["bm25_score"] = bm25_norm
            else:
                merged[key] = {
                    "text": text,
                    "source": meta.get("source", ""),
                    "doc_type": meta.get("doc_type", "notes"),
                    "chapter": meta.get("chapter", ""),
                    "score": 0,
                    # 纯 BM25 命中没有向量相似度，绝对相关度按 0 计
                    "sim": 0.0,
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
            self._invalidate_bm25()
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
        sources: list[str] = None,
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
            sources: 限定来源文件（文件名列表），None 表示不限制。
                     知识库 metadata 没有课程字段，按课程检索时由调用方
                     用文件名列表来限定，避免两门课互相串味。
        Returns:
            [{"text": "...", "source": "...", "score": 0.85, "chunk_index": 3, "chapter": "..."}, ...]

            score 是**余弦相似度**（越大越相关），已按 collection 的实际距离空间换算 ——
            绝对量纲，可以直接和阈值比较（见 config.RELEVANCE_THRESHOLD）。
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_emb = get_embedding(query)
        space = _collection_space(collection)

        conditions = []
        if doc_types:
            conditions.append({"doc_type": {"$in": list(doc_types)}})
        if chapter:
            conditions.append({"chapter": chapter})
        if sources:
            conditions.append({"source": {"$in": list(sources)}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif conditions:
            where_filter = {"$and": conditions}
        else:
            where_filter = None

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
        )

        formatted = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = round(distance_to_similarity(distance, space), 4)
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
        self._invalidate_bm25()

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
        self._invalidate_bm25()

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
        self._invalidate_bm25()
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
_kb_lock = threading.Lock()


def get_knowledge_base(persist_dir: str = "./vector_store") -> KnowledgeBase:
    """获取全局知识库实例（单例）

    加锁避免并发首次调用各建一个实例（FastAPI 同步端点跑在线程池里）。
    注意：单例一旦建立，后续传入的 persist_dir 会被忽略 —— 这是单例语义的
    必然结果，需要换库请显式构造 KnowledgeBase 或先重置 _kb_instance。
    """
    global _kb_instance
    if _kb_instance is None:
        with _kb_lock:
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

    # 判断"够不够"必须用绝对相似度 sim，不能用归一化的融合分 score：
    # 融合分按本次最大值归一化，排名第一恒 ≥ 0.6，拿它比阈值永远成立，
    # 于是"结果不足就补搜课本"这条分支实际上从来没被走到过。
    def _best_sim(items):
        return max((r.get("sim", 0.0) for r in items), default=0.0)

    # 3. 第二层：如果第一层结果不足，补充搜课本
    if len(all_results) < 2 or _best_sim(all_results) < RELEVANCE_THRESHOLD:
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
    best_sim = _best_sim(all_results)
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

    if best_sim < RELEVANCE_THRESHOLD or result_count < 2:
        result_text += "\n\n[提示] 知识库中相关内容较少或相关度不高，建议调用 web_search 工具联网搜索补充信息。"

    return result_text


