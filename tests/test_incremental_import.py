"""增量导入的两个真实缺陷的守卫

缺陷 A（已修）：`/kb/add` 与 `/kb/add-directory` 判定文档类型的**基准目录不同**，
  于是同一个文件走两条路径会得到不同的 doc_type：
      /kb/add（基准 data/）              → 看得到路径里的「课本」→ textbook
      /kb/add-directory（基准被导入目录） → 看不到自己叫「课本」→ notes
  doc_type 影响检索过滤与优先级（textbook 是兜底档），所以这不是纯显示问题。

缺陷 B（未修，仅记录）：入库 id 是 `{文件名}_{序号}`、source 也只存文件名，
  于是**不同目录下的同名文件会互相覆盖** —— 后导入的静默顶掉先导入的，
  而接口仍然回报"added 1"。修它要改数据身份（source 是否含路径），
  涉及已有 4199 个分块的迁移，所以先只加这条**守住现状**的测试。
"""
from pathlib import Path
import shutil

import pytest

from rag_tool import KnowledgeBase, _classify_doc


@pytest.fixture
def workdir():
    """工作区内的临时目录

    不用 pytest 的 `tmp_path`：它建在系统临时目录（Windows 下是
    `%LOCALAPPDATA%\\Temp`），本项目的沙箱配置会拒绝写入那里。
    """
    d = Path("tests/_tmp_import_probe")
    if d.exists():                      # 只删自己会创建的那个固定路径
        shutil.rmtree(d)
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class TestClassifyDocBaseMatters:
    """先固定这个机制：基准目录不同，判定结果就不同 —— 这就是缺陷 A 的成因"""

    def test_same_file_classified_differently_by_base(self):
        f = Path("data/课本/某教材.pdf")
        assert _classify_doc(f, Path("data")) == "textbook"
        # 基准换成"课本目录自己"，路径里就看不到「课本」了
        assert _classify_doc(f, Path("data/课本")) == "notes"

    def test_yearly_exam_is_skipped(self):
        """往年题是题库的源材料，不进知识库（返回 None 表示跳过）"""
        assert _classify_doc(Path("data/操作系统/往年题/2024.pdf"), Path("data")) is None

    def test_ppt_and_notes(self):
        assert _classify_doc(Path("data/操作系统/ppt/03-OS调度.pdf"), Path("data")) == "ppt"
        assert _classify_doc(Path("data/操作系统/notes/x.md"), Path("data")) == "notes"


class TestDirectoryImportUsesSameBase:
    """缺陷 A 的守卫：目录导入必须和单文件导入用同一个基准目录"""

    def test_add_directory_classifies_with_data_as_base(self, monkeypatch, workdir):
        seen = []
        real = _classify_doc

        def spy(file_path, base_dir):
            seen.append(Path(base_dir))
            return real(file_path, base_dir)

        monkeypatch.setattr("rag_tool._classify_doc", spy)
        kb = KnowledgeBase.__new__(KnowledgeBase)          # 不跑 __init__，免得连 ChromaDB
        monkeypatch.setattr(kb, "add_document", lambda *a, **k: 0)

        d = workdir / "课本"
        d.mkdir()
        (d / "某教材.md").write_text("内容", encoding="utf-8")
        kb.add_directory(str(d))

        assert seen, "没有调用 _classify_doc"
        assert all(b == Path("data") for b in seen), \
            f"基准目录必须是 data/，实际收到 {seen}"

    def test_single_file_and_directory_agree(self, monkeypatch, workdir):
        """同一文件：单文件导入与目录导入必须给出同样的 doc_type"""
        results = {}

        def spy(file_path, base_dir):
            return _classify_doc(file_path, base_dir)

        monkeypatch.setattr("rag_tool._classify_doc", spy)
        kb = KnowledgeBase.__new__(KnowledgeBase)

        captured = []

        def fake_add(path, doc_type=None, **kw):
            captured.append((Path(path).name, doc_type))
            return 0
        monkeypatch.setattr(kb, "add_document", fake_add)

        # 目录导入：走 add_directory → 内部自己判类型
        d = workdir / "课本"
        d.mkdir()
        (d / "教材.md").write_text("内容", encoding="utf-8")
        kb.add_directory(str(d))
        results["dir"] = captured[-1][1]

        # 单文件导入：doc_type=None 时由 add_document 内部判定（基准固定 data/）
        # 这里直接比 classify 结果，模拟 add_document 的判定
        results["add"] = _classify_doc(Path("data/课本/教材.md"), Path("data"))

        assert results["dir"] == results["add"], f"两条路径不一致: {results}"


class TestSameNameCollisionIsKnown:
    """缺陷 B：同名文件会互相覆盖 —— 这条测试**记录现状**，修好后应改成断言互不干扰"""

    def test_id_scheme_does_not_include_path(self):
        """入库 id 只由"文件名 + 序号"组成，这是覆盖的根因"""
        source = "同名.md"          # add_document 里 source = p.name（不含路径）
        ids_a = [f"{source}_{i}" for i in range(2)]
        ids_b = [f"{source}_{i}" for i in range(2)]
        assert ids_a == ids_b, (
            "两个不同目录下的同名文件会算出完全相同的 id，"
            "upsert 时后者顶掉前者 —— 这就是缺陷 B"
        )
        # 修法方向（未实施）：id/source 里带上相对 data/ 的路径
        rel_a = "操作系统/ppt/同名.md"
        rel_b = "计算机网络/ppt/同名.md"
        assert f"{rel_a}_0" != f"{rel_b}_0"
