"""构建课程知识库

用法:
  # 增量导入（推荐）— 只处理新增或修改的文件
  python build_knowledge.py --incremental --skip-textbook

  # 全量重建（清空后重建）
  python build_knowledge.py --clear --skip-textbook

  # 增量导入单个文件
  python build_knowledge.py --add-file "data/操作系统/ppt/06-内存管理与虚拟内存.pdf"

  # 流水线模式（PPT 按页+图文合并，课本生成笔记）
  python build_knowledge.py --clear --pipeline --skip-textbook

  # 查看统计
  python build_knowledge.py --stats

  # 查看文档列表
  python build_knowledge.py --list

前置条件:
  1. data/ 目录下放好课程文档（PDF / MD / DOCX）
  2. .env 里配置好 ZHIPU_API_KEY
"""

import sys
import argparse
from rag_tool import KnowledgeBase, ChunkConfig, load_documents_from_dir


def main():
    parser = argparse.ArgumentParser(description="构建课程知识库")
    parser.add_argument("--data-dir", default="./data", help="文档目录")
    parser.add_argument("--persist-dir", default="./vector_store", help="向量库存储路径")
    parser.add_argument("--strategy", choices=["fixed", "paragraph"], default="fixed",
                        help="分块策略: fixed(固定长度) / paragraph(按段落)")
    parser.add_argument("--chunk-size", type=int, default=300, help="分块大小")
    parser.add_argument("--overlap", type=int, default=50, help="重叠长度")
    parser.add_argument("--clear", action="store_true", help="清空后重建")
    parser.add_argument("--incremental", action="store_true",
                        help="增量导入：只处理新增或修改的文件")
    parser.add_argument("--add-file", default=None,
                        help="增量导入单个文件")
    parser.add_argument("--skip-textbook", action="store_true", help="跳过课本，只入库 ppt 和 notes")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")
    parser.add_argument("--list", action="store_true", help="列出知识库中所有文档")

    # 流水线模式相关参数
    parser.add_argument("--pipeline", action="store_true",
                        help="使用知识处理流水线（PPT按页处理+课本生成笔记）")
    parser.add_argument("--ppt-vlm", action="store_true",
                        help="PPT 处理时启用 VLM 图片理解（需额外 API 调用，较慢）")
    parser.add_argument("--textbook-notes-dir", default=None,
                        help="课本笔记输出目录（默认同课本所在目录下的 notes/）")

    args = parser.parse_args()

    kb = KnowledgeBase(persist_dir=args.persist_dir)
    config = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )

    # 只看统计
    if args.stats:
        stats = kb.get_stats()
        print(f"知识库统计:")
        print(f"  存储路径: {args.persist_dir}")
        print(f"  文档数: {stats['total_documents']}")
        print(f"  分块数: {stats['total_chunks']}")
        print(f"\n  按类型:")
        for t, c in stats["by_type"].items():
            print(f"    {t}: {c} 块")
        print(f"\n  按章节:")
        for ch, c in stats["by_chapter"].items():
            print(f"    {ch}: {c} 块")
        return

    # 列出文档
    if args.list:
        docs = kb.list_documents()
        if not docs:
            print("知识库为空")
            return
        print(f"知识库文档列表 ({len(docs)} 个文档):")
        print("-" * 80)
        for d in docs:
            print(f"  {d['source']:<40} [{d['doc_type']:<10}] {d['chunk_count']:>3}块  章节: {d['chapter'] or '未分类'}")
        print("-" * 80)
        total = sum(d["chunk_count"] for d in docs)
        print(f"总计: {len(docs)} 个文档, {total} 个分块")
        return

    # 增量导入单个文件
    if args.add_file:
        print(f"增量导入: {args.add_file}")
        added = kb.add_document(args.add_file, chunk_config=config)
        if added > 0:
            print(f"完成! 新增 {added} 个分块")
        else:
            print("文件未变化或跳过")
        return

    # 增量导入目录
    if args.incremental:
        print(f"增量导入目录: {args.data_dir}")
        print(f"分块: {config.strategy}/{config.chunk_size}")
        print("-" * 50)
        result = kb.add_directory(args.data_dir, chunk_config=config, skip_textbook=args.skip_textbook)
        print("-" * 50)
        print(f"新增: {result['added']}, 跳过: {result['skipped']}, 失败: {result['failed']}")
        for d in result["details"]:
            status_icon = {"added": "+", "skipped": "=", "failed": "!"}[d["status"]]
            extra = f" ({d['chunks']}块)" if "chunks" in d else f" ({d.get('error', '')})"
            print(f"  {status_icon} {d['file']:<40} {extra}")
        return

    # 全量重建
    if args.clear:
        print("清空知识库...")
        kb.clear()

    # 检查文档目录
    docs = load_documents_from_dir(args.data_dir, skip_textbook=args.skip_textbook)
    if not docs:
        print(f"\n没有在 {args.data_dir} 找到任何文档!")
        print("请把课程文件（PDF/MD/TXT/DOCX）放到 data/ 目录下，然后重新运行。")
        print("支持多文件、多课程，文件名会作为来源标注。")
        return

    # 建库
    mode = "流水线" if args.pipeline else "经典"
    ppt_mode = "VLM图片理解" if args.ppt_vlm else "纯文字"
    print(f"\n模式: {mode} | PPT: {ppt_mode} | 分块: {config.strategy}/{config.chunk_size}")
    print("-" * 50)

    total = kb.build_from_dir(
        args.data_dir,
        config,
        skip_textbook=args.skip_textbook,
        use_pipeline=args.pipeline,
        ppt_use_vlm=args.ppt_vlm,
        textbook_output_dir=args.textbook_notes_dir,
    )

    print("-" * 50)
    print(f"知识库构建完成! 共 {total} 个文档分块。")
    print(f"存储在: {args.persist_dir}")
    print(f"\n现在可以:")
    print(f"  1. python build_knowledge.py --stats  查看统计")
    print(f"  2. python build_knowledge.py --list  查看文档列表")
    print(f"  3. python build_knowledge.py --incremental  增量导入新文件")
    print(f"  4. 在 course_agent 中调用 search_knowledge() 检索")
    print(f"  5. 启动 API 服务后访问 /kb/stats 查看统计")


if __name__ == "__main__":
    main()
