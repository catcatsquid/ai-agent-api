"""前端认证块的回归测试（通过 Node 运行）

为什么放在 pytest 里：第一版的登录框用了 window.prompt()，
在禁用原生弹窗的环境（内嵌浏览器 / 沙箱 iframe）里直接抛
"prompt() is not supported"，页面完全发不出消息。
这条测试直接读取两个页面的**真实内联脚本**，用最小 DOM shim 跑通登录链路，
并硬性禁止 prompt/alert/confirm —— 防止这个 bug 再回来。

没有 Node 时自动跳过（不影响纯 Python 环境跑测试）。
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "js" / "auth_block_test.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="需要 Node.js 才能跑前端脚本测试",
)


def test_frontend_auth_block_behaviour():
    """两个页面的登录链路必须可用，且不依赖任何原生弹窗"""
    assert SCRIPT.exists(), f"测试脚本不存在: {SCRIPT}"
    proc = subprocess.run(
        ["node", str(SCRIPT)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        pytest.fail("前端认证块测试失败：\n" + "\n".join(out.strip().splitlines()[-40:]))
    assert "0 失败" in out, out


def test_frontend_never_uses_native_dialogs():
    """再独立断言一次：页面里不得出现 prompt/alert/confirm 调用

    （与 Node 那条重复是为了在改动页面后能立刻定位到是哪个文件出的问题）
    """
    import re

    for page in ["course_agent/static/chat.html", "course_agent/static/kb.html"]:
        html = (ROOT / page).read_text(encoding="utf-8")
        # 只查调用形态：window.prompt( / alert( / confirm( / prompt(
        for pattern, label in [
            (r"window\.prompt\s*\(", "window.prompt("),
            (r"window\.alert\s*\(", "window.alert("),
            (r"window\.confirm\s*\(", "window.confirm("),
        ]:
            assert not re.search(pattern, html), f"{page} 出现了 {label}"
        # 裸调用：去掉注释后不应再有 "prompt(" / "alert(" 这种调用
        stripped = re.sub(r"//[^\n]*", "", re.sub(r"/\*[\s\S]*?\*/", " ", html))
        for pattern, label in [
            (r"(^|[^.\w])prompt\s*\(", "prompt("),
            (r"(^|[^.\w])alert\s*\(", "alert("),
        ]:
            assert not re.search(pattern, stripped), f"{page} 出现了裸 {label} 调用"


def test_switching_reasoning_mode_resets_thread():
    """切换 Plan/ReAct 必须开新 thread

    两张图各用独立 checkpointer（后端已修），但如果前端切模式还沿用同一个
    thread_id，界面显示的旧历史和后端上下文就对不上 —— 用户会以为"它记得"，
    实际后端是空的。
    """
    import re

    html = (ROOT / "course_agent/static/chat.html").read_text(encoding="utf-8")
    m = re.search(r"function setReasoningMode\([^)]*\)\s*\{(.*?)\n\}", html, re.S)
    assert m, "找不到 setReasoningMode"
    body = m.group(1)
    assert "currentThreadId = null" in body, "切换推理模式没有重置 thread_id"
    assert "currentReasoningMode = mode" in body, "没记录新的模式"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
