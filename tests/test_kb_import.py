"""知识库导入：**源不限制、目标必须落在 data/ 之下**

背景（用户实测）：从下载目录导入一个文件失败了 —— 因为 `resolve_within_data`
连**读取**路径都要求"必须在 data/ 之下"。可它当初要防的是"越界读机器上任意文件"，
方向错了：真正该守的是**写入位置**。

顺带还有个技术理由必须复制而不是就地读：`_classify_doc` 按"路径相对 data/ 的位置"
判类型（含 ppt→ppt、含 课本→textbook、含 往年题→跳过），外部路径一律落成 notes。

所以这个文件守两件事：
  1. 外面来的文件能被导入，并且**真的被复制进 data/**（落到正确分类的目录）
  2. 写入位置**仍然**被钉在 data/ 之内（路径穿越、越界目标都要被拒）
"""
import shutil
from pathlib import Path
from pathlib import Path as FilePath

import pytest
from fastapi.testclient import TestClient

from config import EXPORT_DIR  # noqa: F401  （仅确保配置项存在）
from course_agent import server
from course_agent.server import (
    _destination_dir,
    _export_root,
    _safe_filename,
    app,
)
from doc_utils import _classify_doc

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "tests" / "_tmp_kb_import"


@pytest.fixture
def env(monkeypatch):
    """把 DATA_ROOT 换到工作区内的临时目录，并替换知识库（不碰真实 Chroma）"""
    if WORK.exists():
        shutil.rmtree(WORK)
    data = WORK / "data"
    (data / "操作系统" / "ppt").mkdir(parents=True)
    (data / "计算机网络" / "课本").mkdir(parents=True)
    (data / "操作系统" / "往年题").mkdir(parents=True)
    outside = WORK / "外面"
    outside.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_ROOT", data.resolve())

    calls = []

    class FakeKB:
        def add_document(self, path, doc_type=None, chapter=None):
            calls.append({"path": FilePath(path), "doc_type": doc_type})
            return 7

    monkeypatch.setattr(server, "get_knowledge_base", lambda: FakeKB())
    try:
        yield {"data": data.resolve(), "outside": outside.resolve(), "calls": calls}
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


# ============================================================
# 文件名与目标目录：两道防线
# ============================================================

class TestFilenameAndTargetSafety:
    @pytest.mark.parametrize("raw, want", [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\win.ini", "win.ini"),
        ("C:/Windows/system32/x.dll", "x.dll"),
        ("普通的名字.md", "普通的名字.md"),
    ])
    def test_directory_parts_are_stripped(self, raw, want):
        """只取文件名部分 —— 路径穿越靠这个挡掉"""
        assert _safe_filename(raw) == want

    @pytest.mark.parametrize("bad", ["", "   ", "..", "."])
    def test_empty_or_dotdot_names_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _safe_filename(bad)

    def test_target_inside_data_is_accepted(self, env):
        dest = _destination_dir("操作系统/ppt")
        assert dest == env["data"] / "操作系统" / "ppt"
        assert dest.is_dir()

    @pytest.mark.parametrize("bad", ["../外面", "..\\外面", "../../tmp", "操作系统/../../外面"])
    def test_target_outside_data_is_rejected(self, env, bad):
        """**这条是核心不变量**：目标目录不许跑到 data/ 外面去"""
        with pytest.raises(ValueError):
            _destination_dir(bad)

    def test_default_target_is_import_dir_and_classifies_as_notes(self, env):
        dest = _destination_dir(None)
        assert dest == env["data"] / server.DEFAULT_IMPORT_DIR
        assert _classify_doc(dest / "x.md", env["data"]) == "notes"


# ============================================================
# 用路径导入：源在 data/ 里就地，源在外面就复制进来
# ============================================================

class TestAddByPath:
    def test_source_inside_data_is_ingested_in_place(self, env):
        src = env["data"] / "操作系统" / "ppt" / "03-调度.md"
        src.write_text("内容", encoding="utf-8")
        resp = TestClient(app).post("/kb/add", params={"file_path": str(src)})
        body = resp.json()
        assert body["code"] == 0, body
        assert body["copied"] is False
        assert env["calls"][-1]["path"] == src.resolve()

    def test_source_outside_data_is_copied_in(self, env):
        """**用户要的就是这条**：从任意位置导入，文件真正落进 data/"""
        src = env["outside"] / "07-文件系统.md"
        src.write_text("# 文件系统\n内容", encoding="utf-8")

        resp = TestClient(app).post(
            "/kb/add", params={"file_path": str(src), "target_dir": "操作系统/ppt"})
        body = resp.json()
        assert body["code"] == 0, body
        assert body["copied"] is True

        landed = Path(body["path"])
        assert landed.parent == env["data"] / "操作系统" / "ppt", "没落到指定目录"
        assert landed.read_text(encoding="utf-8") == "# 文件系统\n内容", "内容不一致"
        assert src.exists(), "源文件不该被搬走（是复制）"
        assert env["calls"][-1]["path"] == landed
        # 落到 ppt 目录 → 分类应该是 ppt（这就是"必须复制"的技术理由）
        assert _classify_doc(landed, env["data"]) == "ppt"

    def test_outside_source_with_traversal_name_lands_inside_data(self, env):
        src = env["outside"] / "正常.md"
        src.write_text("x", encoding="utf-8")
        resp = TestClient(app).post(
            "/kb/add", params={"file_path": str(src), "target_dir": "操作系统/ppt"})
        assert resp.json()["code"] == 0
        landed = Path(resp.json()["path"])
        assert env["data"] in landed.parents, "落点跑到 data/ 外面了"

    def test_missing_source_is_404(self, env):
        resp = TestClient(app).post(
            "/kb/add", params={"file_path": str(env["outside"] / "不存在.md")})
        assert resp.json()["code"] == 404

    def test_bad_target_is_refused(self, env):
        src = env["outside"] / "x.md"
        src.write_text("x", encoding="utf-8")
        resp = TestClient(app).post(
            "/kb/add", params={"file_path": str(src), "target_dir": "../外面"})
        assert resp.json()["code"] == 403
        assert not env["calls"], "被拒的导入不该真的入库"


# ============================================================
# 直接上传：内容经 HTTP 过来，不读机器上的路径
# ============================================================

class TestUpload:
    def test_upload_saves_under_data_and_ingests(self, env):
        client = TestClient(app)
        payload = "# 课本\nTCP".encode("utf-8")
        resp = client.post(
            "/kb/upload",
            params={"target_dir": "计算机网络/课本"},
            files={"file": ("教材.md", payload, "text/markdown")})
        body = resp.json()
        assert body["code"] == 0, body
        landed = Path(body["path"])
        assert landed.parent == env["data"] / "计算机网络" / "课本"
        assert landed.read_bytes() == payload
        assert body["added_chunks"] == 7
        assert _classify_doc(landed, env["data"]) == "textbook"

    def test_upload_to_bad_target_is_refused(self, env):
        client = TestClient(app)
        resp = client.post("/kb/upload", params={"target_dir": "../../外面"},
                           files={"file": ("x.md", b"x", "text/markdown")})
        assert resp.json()["code"] == 403

    def test_upload_with_traversal_filename_stays_inside_data(self, env):
        client = TestClient(app)
        resp = client.post("/kb/upload", params={"target_dir": "操作系统/ppt"},
                           files={"file": ("../../../evil.md", b"x", "text/markdown")})
        body = resp.json()
        assert body["code"] == 0, body
        landed = Path(body["path"])
        assert landed.name == "evil.md", f"文件名没被剥掉目录成分: {landed}"
        assert env["data"] in landed.parents

    def test_upload_does_not_overwrite_same_name(self, env):
        client = TestClient(app)
        first = client.post("/kb/upload", params={"target_dir": "操作系统/ppt"},
                            files={"file": ("同名.md", b"aaa", "text/markdown")}).json()
        second = client.post("/kb/upload", params={"target_dir": "操作系统/ppt"},
                             files={"file": ("同名.md", b"bbb", "text/markdown")}).json()
        assert first["path"] != second["path"], "同名被静默覆盖了"
        assert Path(first["path"]).read_bytes() == b"aaa"
        assert Path(second["path"]).read_bytes() == b"bbb"


class TestSkippedDirsAreExplained:
    """「往年题」不进知识库 —— 必须说清原因，而不是显示"成功 0 个分块"

    这正是用户实测踩到的那个坑：把 os第二次作业.docx 放进
    data/操作系统/往年题/ 再导入，界面显示"导入成功: 0 个分块"，看着像失败。
    """

    def test_add_from_yearly_exam_dir_reports_why(self, env):
        f = env["data"] / "操作系统" / "往年题" / "2024.md"
        f.write_text("x", encoding="utf-8")
        body = TestClient(app).post("/kb/add", params={"file_path": str(f)}).json()
        assert body["code"] == 0, body
        assert body["added_chunks"] == 0
        assert "往年题" in body["skipped_reason"]
        assert not env["calls"], "被跳过的文件不该真的去入库"

    def test_upload_into_yearly_exam_dir_is_refused_and_not_saved(self, env):
        """落到会跳过的目录就先拦住 —— 别存下来却什么也不入库"""
        d = env["data"] / "操作系统" / "往年题"
        before = sorted(p.name for p in d.iterdir())
        resp = TestClient(app).post(
            "/kb/upload", params={"target_dir": "操作系统/往年题"},
            files={"file": ("新题.md", b"x", "text/markdown")})
        body = resp.json()
        assert body["code"] == 400, body
        assert "往年题" in body["msg"]
        assert sorted(p.name for p in d.iterdir()) == before, "文件不该被保存"
        assert not env["calls"]


class TestImportTargetsAreComputedByTheProductionRule:
    def test_lists_dirs_with_doc_type_from_classify_doc(self, env):
        rows = {r["dir"]: r["doc_type"] for r in
                TestClient(app).get("/kb/import-targets").json()["data"]}
        assert rows.get("操作系统/ppt") == "ppt"
        assert rows.get("计算机网络/课本") == "textbook"
        assert rows.get(server.DEFAULT_IMPORT_DIR) == "notes"

    def test_skipped_kinds_are_not_offered(self, env):
        """「往年题」会被 _classify_doc 判成 None（跳过），不该出现在可选项里"""
        dirs = [r["dir"] for r in
                TestClient(app).get("/kb/import-targets").json()["data"]]
        assert not any("往年题" in d for d in dirs), dirs


# ============================================================
# 对话导出目录：可配置
# ============================================================

class TestExportDirIsConfigurable:
    def test_default_is_exports_under_project(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "EXPORT_DIR", "exports")
        assert _export_root() == ROOT / "exports"

    def test_relative_path_resolves_against_project_root(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "EXPORT_DIR", "my_notes/chats")
        assert _export_root() == ROOT / "my_notes" / "chats"

    def test_absolute_path_is_used_as_is(self, monkeypatch):
        import config
        target = WORK / "绝对路径导出"
        monkeypatch.setattr(config, "EXPORT_DIR", str(target))
        assert _export_root() == target.resolve()
