"""课程注册表（`course_agent/courses.py`）的行为

守的是"新增课程 = 建目录"这条设计：目录即真相源，并拒绝非法名与保留名。
"""

import pytest

from course_agent import courses


@pytest.fixture()
def temp_data(tmp_path, monkeypatch):
    """把 DATA_DIR 指到临时目录，避免碰真实 data/"""
    monkeypatch.setattr(courses, "DATA_DIR", tmp_path)
    courses.invalidate_cache()
    yield tmp_path
    courses.invalidate_cache()


class TestKnownCourses:
    def test_reads_course_dirs(self, temp_data):
        (temp_data / "操作系统").mkdir()
        (temp_data / "数据结构").mkdir()
        assert courses.known_courses(refresh=True) == ["操作系统", "数据结构"]

    def test_ignores_reserved_and_hidden_dirs(self, temp_data):
        (temp_data / "操作系统").mkdir()
        (temp_data / "question_bank").mkdir()
        (temp_data / ".cache").mkdir()
        (temp_data / "_tmp").mkdir()
        assert courses.known_courses(refresh=True) == ["操作系统"]

    def test_cache_invalidated_after_create(self, temp_data):
        (temp_data / "操作系统").mkdir()
        assert courses.known_courses(refresh=True) == ["操作系统"]
        courses.create_course("数据结构")
        # 不 refresh 也要能看到新课程（create_course 内部清了缓存）
        assert courses.known_courses() == ["操作系统", "数据结构"]


class TestValidateName:
    @pytest.mark.parametrize("bad", ["", "   ", "a/b", "a\\b", "..", ".git", "question_bank", "x" * 33])
    def test_rejects(self, bad):
        with pytest.raises(courses.CourseNameError):
            courses.validate_name(bad)

    @pytest.mark.parametrize("bad", ["CON", "con", "NUL", "com1", "LPT9", "CON.txt"])
    def test_rejects_windows_device_names(self, bad):
        """实测建出过 data\\CON：目录能建，但读不出、删不掉"""
        with pytest.raises(courses.CourseNameError):
            courses.validate_name(bad)

    @pytest.mark.parametrize("ok", ["数据结构", "编译原理", "AI 导论", "cs-101"])
    def test_accepts(self, ok):
        assert courses.validate_name(ok) == ok

    def test_trims_spaces(self):
        assert courses.validate_name("  数据结构  ") == "数据结构"


class TestCreateCourse:
    def test_creates_three_subdirs(self, temp_data):
        out = courses.create_course("数据结构")
        root = temp_data / "数据结构"
        assert root.is_dir()
        for sub in courses.COURSE_SUBDIRS:
            assert (root / sub).is_dir()
        assert out["created_subdirs"] == list(courses.COURSE_SUBDIRS)

    def test_duplicate_is_rejected(self, temp_data):
        courses.create_course("数据结构")
        with pytest.raises(courses.CourseNameError):
            courses.create_course("数据结构")

    def test_bad_name_creates_nothing(self, temp_data):
        with pytest.raises(courses.CourseNameError):
            courses.create_course("question_bank")
        assert not (temp_data / "question_bank").exists()


class TestCreateCourseHttp:
    """HTTP 层：名字不合法必须返回 400 + 可读原因

    真踩过：`server.py` 的错误分支用了 `JSONResponse` 却没 import，
    于是**所有**校验失败都变成 500「服务内部错误: NameError」——
    前端只能弹这个，用户永远看不到"已经存在 / 是保留名"。
    服务层测试拦不住这类缺陷，所以这几条专打 HTTP 层。
    """

    @pytest.fixture()
    def client(self, temp_data):
        from fastapi.testclient import TestClient

        from course_agent.server import app

        return TestClient(app, raise_server_exceptions=False)

    def test_creates_a_course(self, client, temp_data):
        r = client.post("/kb/courses", json={"name": "数据结构"})
        assert r.status_code == 200
        body = r.json()["data"]
        assert body["course"] == "数据结构"
        assert (temp_data / "数据结构" / "ppt").is_dir()
        # 只回相对位置：绝对路径会把本机盘符/用户名带给前端
        assert body["dir"] == f"{temp_data.name}/数据结构"

    def test_bad_name_returns_400_not_500(self, client):
        r = client.post("/kb/courses", json={"name": "a/b"})
        assert r.status_code == 400
        assert r.json()["msg"]

    def test_duplicate_returns_400(self, client):
        client.post("/kb/courses", json={"name": "数据结构"})
        r = client.post("/kb/courses", json={"name": "数据结构"})
        assert r.status_code == 400
        assert "已经存在" in r.json()["msg"]

    def test_device_name_returns_400_and_writes_nothing(self, client, temp_data):
        r = client.post("/kb/courses", json={"name": "CON"})
        assert r.status_code == 400
        assert not (temp_data / "CON").exists()

    def test_list_reflects_dirs_added_outside_the_api(self, client, temp_data):
        """手工 mkdir / 切分支 / 清理磁盘之后，列表必须立刻反映 —— 缓存不能脏着"""
        assert client.get("/kb/courses").json()["data"] == []
        (temp_data / "数据结构").mkdir()
        names = [c["name"] for c in client.get("/kb/courses").json()["data"]]
        assert names == ["数据结构"]