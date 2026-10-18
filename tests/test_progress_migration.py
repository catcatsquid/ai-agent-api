"""进度账本的 schema 迁移（第 5 项）

场景：将来加字段/改结构，旧 `runtime/progress.json` 会被"按新逻辑静默解释"，
字段丢了也看不出来。迁移函数必须只升不降。
"""
import json

import pytest

from course_agent import progress


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_PATH", tmp_path / "progress.json")
    yield tmp_path / "progress.json"


class TestMigration:
    def test_old_file_without_version_is_upgraded(self, _isolated):
        _isolated.write_text(json.dumps(
            {"courses": {"操作系统": {"sections": {}, "last": ""}}, "papers": {}},
            ensure_ascii=False), encoding="utf-8")
        data = progress.load()
        assert data["version"] == progress._VERSION
        assert data["user"] == "local"
        assert "操作系统" in data["courses"], "迁移不许把已有数据弄丢"

    def test_missing_sections_are_filled(self, _isolated):
        _isolated.write_text('{"version": 0, "courses": {"操作系统": {}}}',
                             encoding="utf-8")
        data = progress.load()
        assert data["courses"]["操作系统"]["sections"] == {}

    def test_newer_version_is_not_downgraded(self, _isolated):
        """文件比程序新 → 只警告，不改写（否则下一次保存会把新字段抹掉）"""
        _isolated.write_text(json.dumps(
            {"version": 99, "future_field": {"x": 1}}, ensure_ascii=False),
            encoding="utf-8")
        data = progress.load()
        assert data["version"] == 99, "把新版本强行降级会丢字段"
        assert data["future_field"] == {"x": 1}

    def test_broken_version_value_does_not_crash(self, _isolated):
        _isolated.write_text('{"version": "not-a-number", "papers": {}}',
                             encoding="utf-8")
        assert progress.load()["version"] == progress._VERSION
