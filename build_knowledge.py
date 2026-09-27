"""构建课程知识库

用法:
  # 旧模式（直接提取文字，最快）
  python build_knowledge.py --clear --skip-textbook

  # 流水线模式（PPT 按页+图文合并，课本生成笔记）
  python build_knowledge.py --clear --pipeline --skip-textbook

  # 流水线模式 + VLM 图片理解（最全，最慢）
  python build_knowledge.py --clear --pipeline --ppt-vlm --skip-textbook

  # 处理课本（生成知识点笔记后入库）
  python build_knowledge.py --clear --pipeline --textbook-notes-dir ./data/notes

  # 查看统计
  python build_knowledge.py --stats

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
    parser.add_argument("--skip-textbook", action="store_true", help="跳过课本，只入库 ppt 和 notes")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")

    # 流水线模式相关参数
    parser.add_argument("--pipeline", action="store_true",
                        help="使用知识处理流水线（PPT按页处理+课本生成笔记）")
    parser.add_argument("--ppt-vlm", action="store_true",
                        help="PPT 处理时启用 VLM 图片理解（需额外 API 调用，较慢）")
    parser.add_argument("--textbook-notes-dir", default=None,
                        help="课本笔记输出目录（默认同课本所在目录下的 notes/）")

    args = parser.parse_args()

    kb = KnowledgeBase(persist_dir=args.persist_dir)

    # 只看统计
    if args.stats:
        count = kb.count()
        print(f"知识库统计:")
        print(f"  存储路径: {args.persist_dir}")
        print(f"  文档数: {count}")
        if count > 0:
            results = kb.search("测试", top_k=1)
            if results:
                print(f"  示例检索: {results[0]['text'][:80]}...")
        return

    # 清空
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
    config = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )

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
    print(f"  2. 在 course_agent 中调用 search_knowledge() 检索")
    print(f"  3. 启动 agent_api 测试问答")


if __name__ == "__main__":
    main()
