"""对话导出的回归测试

背景（用户实测）：点了"导出当前对话"，页面提示"已导出"，但下载目录里**什么都没有**。

原实现只做浏览器下载：
    a.click();  URL.revokeObjectURL(url);  showToast('对话已导出', 'success')
两个缺陷：
  1. `revokeObjectURL` 紧跟 `click()` —— 下载还没解析这个 blob URL 就把它释放了，
     下载会被静默取消（小文件不一定复现，所以更容易漏）。
  2. 提示**无条件**报成功：下载是异步的，还可能被浏览器/iframe 拦掉，
     前端根本无从得知结果，却告诉用户"已导出"。

修法：以**服务端落盘**为准（这一步的成败前端拿得到），把真实路径显示出来；
浏览器下载降级为顺带触发。所以这里测的是服务端那一半 —— 它是可验证的那一半。
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from course_agent.server import app, EXPORT_ROOT, PROJECT_ROOT, _safe_export_name

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def cleanup_exports():
    """只删本测试新建的文件（不碰用户已导出的内容）"""
    before = set(EXPORT_ROOT.glob("*.md")) if EXPORT_ROOT.exists() else set()
    try:
        yield
    finally:
        if EXPORT_ROOT.exists():
            for f in set(EXPORT_ROOT.glob("*.md")) - before:
                f.unlink(missing_ok=True)


class TestChatExport:
    def test_exports_markdown_to_a_known_path(self, cleanup_exports):
        md = "# 测试对话\n\n## 👤 用户\n\n你好\n"
        r = client.post("/chat/export",
                        json={"title": "测试对话", "markdown": md}).json()
        assert r["code"] == 0, r
        # 关键：返回里必须有**真实路径**，否则用户还是不知道文件在哪
        assert r["path"].startswith("exports/"), r
        assert r["filename"].endswith(".md")
        p = PROJECT_ROOT / r["path"]
        assert p.exists(), f"服务端说导出成功，文件却不在: {p}"
        assert p.read_text(encoding="utf-8") == md
        assert r["bytes"] == len(md.encode("utf-8"))

    def test_renders_from_messages_when_markdown_absent(self, cleanup_exports):
        r = client.post("/chat/export", json={
            "title": "渲染测试",
            "messages": [{"role": "user", "content": "死锁是什么"},
                         {"role": "assistant", "content": "四个必要条件…"}],
        }).json()
        assert r["code"] == 0, r
        text = (PROJECT_ROOT / r["path"]).read_text(encoding="utf-8")
        assert "死锁是什么" in text and "四个必要条件" in text
        assert "👤 用户" in text and "🤖 助手" in text

    def test_empty_is_rejected(self):
        assert client.post("/chat/export", json={"title": "空"}).json()["code"] == 400
        assert client.post("/chat/export",
                           json={"title": "空", "markdown": "   \n "}).json()["code"] == 400

    def test_path_traversal_in_title_is_neutralised(self, cleanup_exports):
        """标题里带 ../ 也不能写到 exports/ 之外"""
        r = client.post("/chat/export",
                        json={"title": "../../../evil", "markdown": "# x"}).json()
        assert r["code"] == 0, r
        p = (PROJECT_ROOT / r["path"]).resolve()
        # 关键：文件必须落在 exports/ 里（分隔符已被替换，所以路径无法逃逸）
        assert p.parent == EXPORT_ROOT, f"跑出 exports/ 了: {p}"
        assert "/" not in r["filename"] and "\\" not in r["filename"]
        assert not (PROJECT_ROOT / "evil.md").exists()

    def test_windows_illegal_chars_are_replaced(self):
        assert _safe_export_name('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
        assert _safe_export_name("") == "对话"
        assert _safe_export_name("   ") == "对话"
        assert _safe_export_name("...") == "对话"
        assert len(_safe_export_name("长" * 200)) <= 60

    def test_two_exports_do_not_overwrite(self, cleanup_exports):
        a = client.post("/chat/export", json={"title": "同名", "markdown": "A"}).json()
        b = client.post("/chat/export", json={"title": "同名", "markdown": "B"}).json()
        assert a["code"] == b["code"] == 0
        assert a["path"] != b["path"], "两次导出不该覆盖同一个文件"
        assert (PROJECT_ROOT / a["path"]).read_text(encoding="utf-8") == "A"
        assert (PROJECT_ROOT / b["path"]).read_text(encoding="utf-8") == "B"


class TestFrontendNoLongerLies:
    """前端那两个缺陷的守卫（读真实页面，不靠人肉检查）"""

    HTML = (PROJECT_ROOT / "course_agent/static/chat.html").read_text(encoding="utf-8")

    def test_does_not_revoke_blob_url_immediately(self):
        """不能紧跟 click() 就 revoke —— 那会静默取消下载"""
        i = self.HTML.index("function exportChat")
        body = self.HTML[i:i + 3000]
        assert "revokeObjectURL" in body, "找不到 revokeObjectURL，测试需要更新"
        # revoke 必须出现在 setTimeout 里（延迟释放）
        idx = body.index("revokeObjectURL")
        window = body[max(0, idx - 200):idx]
        assert "setTimeout" in window, (
            "revokeObjectURL 仍在 click() 之后被立刻调用 —— "
            "这正是'提示成功但没有文件'的元凶")

    def test_calls_the_server_endpoint(self):
        i = self.HTML.index("function exportChat")
        body = self.HTML[i:i + 3000]
        assert "/chat/export" in body, "前端必须调服务端导出（可验证的那一步）"

    def test_reports_failure_instead_of_success(self):
        i = self.HTML.index("function exportChat")
        body = self.HTML[i:i + 3000]
        assert "导出失败" in body, "服务端失败时必须报错，不能报'已导出'"
        # 成功提示里要带路径
        assert "savedPath" in body
