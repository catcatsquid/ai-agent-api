"""RAG 工具函数单元测试（纯函数，不需要 API）"""

import pytest

from rag_tool import (
    ChunkConfig, chunk_text, _chunk_fixed, _chunk_by_paragraph,
    clean_text, _expand_query,
)


class TestChunkConfig:
    """分块配置"""

    def test_defaults(self):
        c = ChunkConfig()
        assert c.chunk_size == 300
        assert c.overlap == 50
        assert c.strategy == "fixed"

    def test_custom(self):
        c = ChunkConfig(chunk_size=500, overlap=100, strategy="paragraph")
        assert c.chunk_size == 500
        assert c.overlap == 100
        assert c.strategy == "paragraph"


class TestChunkFixed:
    """固定长度分块"""

    def test_short_text_single_chunk(self):
        text = "这是一个短文本"
        chunks = _chunk_fixed(text, ChunkConfig(chunk_size=300, overlap=50))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "A" * 700
        chunks = _chunk_fixed(text, ChunkConfig(chunk_size=300, overlap=50))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 300

    def test_overlap_between_chunks(self):
        text = "0123456789" * 100  # 1000 chars
        chunks = _chunk_fixed(text, ChunkConfig(chunk_size=300, overlap=50))
        if len(chunks) >= 2:
            # 第二块的开头应与第一块的结尾有重叠
            assert chunks[1][:50] == chunks[0][-50:]

    def test_empty_text(self):
        chunks = _chunk_fixed("", ChunkConfig())
        assert chunks == []


class TestChunkByParagraph:
    """按段落分块"""

    def test_single_paragraph(self):
        text = "这是第一段。"
        chunks = _chunk_by_paragraph(text, ChunkConfig(chunk_size=300))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_short_paragraphs(self):
        text = "段落一。\n\n段落二。\n\n段落三。"
        chunks = _chunk_by_paragraph(text, ChunkConfig(chunk_size=300))
        assert len(chunks) == 3

    def test_long_paragraph_split(self):
        text = "A" * 500
        chunks = _chunk_by_paragraph(text, ChunkConfig(chunk_size=300, overlap=50))
        assert len(chunks) >= 2

    def test_empty_paragraphs_ignored(self):
        text = "段落一。\n\n\n\n段落二。"
        chunks = _chunk_by_paragraph(text, ChunkConfig(chunk_size=300))
        assert len(chunks) == 2


class TestChunkText:
    """chunk_text 路由函数"""

    def test_default_uses_fixed(self):
        text = "A" * 500
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_paragraph_strategy(self):
        text = "段落一。\n\n段落二。"
        chunks = chunk_text(text, ChunkConfig(strategy="paragraph"))
        assert len(chunks) == 2

    def test_none_config_uses_default(self):
        text = "短文本"
        chunks = chunk_text(text, config=None)
        assert len(chunks) == 1


class TestCleanText:
    """文档清洗"""

    def test_removes_page_numbers(self):
        text = "正文内容\n123\n更多内容"
        result = clean_text(text)
        assert "123" not in result
        assert "正文内容" in result
        assert "更多内容" in result

    def test_removes_page_markers(self):
        text = "第一页\n正文\n第二页"
        result = clean_text(text)
        assert "第一页" not in result
        assert "第二页" not in result

    def test_removes_page_english(self):
        text = "Page 1\ncontent\nPage 2"
        result = clean_text(text)
        assert "Page 1" not in result
        assert "Page 2" not in result

    def test_merges_broken_sentences(self):
        text = "这是一个句子。\n下一句"
        result = clean_text(text)
        assert "句子。下一句" in result

    def test_compresses_blank_lines(self):
        text = "段落一\n\n\n\n\n\n段落二"
        result = clean_text(text)
        # 连续3+空行压缩为最多2个换行
        assert "\n\n\n" not in result

    def test_strips_whitespace(self):
        text = "  内容  \n  更多  "
        result = clean_text(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_removes_toc(self):
        text = "目录\n第一章\n第二章"
        result = clean_text(text)
        assert "目录" not in result


class TestExpandQuery:
    """查询扩展"""

    def test_original_query_always_included(self):
        result = _expand_query("什么是临界区")
        assert "什么是临界区" in result

    def test_cleaned_variant_generated(self):
        result = _expand_query("什么是临界区")
        # 去掉"什么是"后的清理版应该存在
        assert any("临界区" in q for q in result)

    def test_keywords_split(self):
        result = _expand_query("进程调度 算法")
        # 多关键词应拆分
        assert len(result) >= 2

    def test_deduplication(self):
        result = _expand_query("测试")
        # 不应有重复
        assert len(result) == len(set(result))

    def test_max_four_variants(self):
        result = _expand_query("什么是 进程调度 算法 比较")
        assert len(result) <= 4

    def test_single_word(self):
        result = _expand_query("死锁")
        assert "死锁" in result
