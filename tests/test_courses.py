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