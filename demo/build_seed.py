"""构建「演示数据集」的知识库向量库（随仓库分发，用于零成本复现）

## 它做什么

把 `demo/seed/data/<课程>/ppt/*.md`（**自己写的知识点笔记**，不是课件原件）
切块并向量化，写入 `demo/seed/vector_store/`。这份向量库进 git，
所以别人 clone 后不必自己建库（建库要 20 分钟并消耗 embedding 额度）。

## 什么时候需要重跑

- 改了 `demo/seed/data/` 里的笔记
- **换了 embedding 模型**（向量维度绑定模型：当前是 Embedding-3 / 2048 维，
  换模型后旧向量不可比，必须重建）

## 用法

    python demo/build_seed.py              # 重建
    python demo/build_seed.py --stats      # 只看统计，不重建

需要 `.env` 里有 `ZHIPU_API_KEY`（只用于 embedding，约 60 个分块，成本可忽略）。
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # demo/
PROJECT_ROOT = ROOT.parent                      # 仓库根
sys.path.insert(0, str(PROJECT_ROOT))

SEED_DATA = ROOT / "seed" / "data"
SEED_STORE = ROOT / "seed" / "vector_store"


def main() -> int:
    parser = argparse.ArgumentParser(description="构建演示数据集向量库")
    parser.add_argument("--data-dir", default=str(SEED_DATA), help="笔记目录")
    parser.add_argument("--persist-dir", default=str(SEED_STORE), help="向量库输出目录")
    parser.add_argument("--stats", action="store_true", help="只显示统计")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--strategy", choices=["fixed", "paragraph"], default="paragraph",
                        help="笔记是分节的 markdown，按段落切块检索质量更好（默认）")
    args = parser.parse_args()

    from rag_tool import KnowledgeBase, ChunkConfig   # 延迟导入：先让 --help 可用

    persist = Path(args.persist_dir)

    if args.stats:
        if not persist.exists():
            print(f"[错误] 还没有向量库：{persist}，先跑一次不带 --stats 的", file=sys.stderr)
            return 1
        kb = KnowledgeBase(persist_dir=str(persist))
        stats = kb.get_stats()
        print(f"向量库: {persist}")
        print(f"  文档数: {stats['total_documents']}")
        print(f"  分块数: {stats['total_chunks']}")
        for t, c in (stats.get("by_type") or {}).items():
            print(f"    {t}: {c} 块")
        return 0

    if not Path(args.data_dir).exists():
        print(f"[错误] 找不到 {args.data_dir}", file=sys.stderr)
        return 1

    try:
        from config import ZHIPU_API_KEY  # noqa: F401
    except Exception as e:                  # noqa: BLE001
        print(f"[错误] 读不到配置（.env 里的 ZHIPU_API_KEY）: {e}", file=sys.stderr)
        return 1

    print(f"源目录: {args.data_dir}")
    print(f"输出  : {persist}")

    # 整目录重建。注意**不能**只用 kb.clear()：它删的是 collection，
    # 磁盘上的旧 segment 目录会留下 —— 重建几次就会累积几份几 MB 的垃圾。
    if persist.exists():
        shutil.rmtree(persist)
    kb = KnowledgeBase(persist_dir=str(persist))

    result = kb.add_directory(
        args.data_dir,
        chunk_config=ChunkConfig(chunk_size=args.chunk_size, overlap=args.overlap,
                                 strategy=args.strategy),
        skip_textbook=False,
    )
    print(f"新增: {result.get('added')}, 跳过: {result.get('skipped')}, "
          f"失败: {result.get('failed')}")

    stats = kb.get_stats()
    print(f"完成 —— 文档 {stats['total_documents']} 个，分块 {stats['total_chunks']} 个")
    print("\n文档清单:")
    for d in kb.list_documents():
        print(f"  {d['source']:<28} [{d['doc_type']:<5}] {d['chunk_count']:>3} 块  "
              f"章节: {d['chapter'] or '未分类'}")
    print("\n注意：这份向量库绑定 Embedding-3（2048 维）。换模型必须重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
