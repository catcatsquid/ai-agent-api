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
SCRIPT_CONTEXT = ROOT / "tests" / "js" / "chat_context_test.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="需要 Node.js 才能跑前端脚本测试",
)


def _run_node(script: Path, label: str) -> str:
    assert script.exists(), f"测试脚本不存在: {script}"
    proc = subprocess.run(
        ["node", str(script)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        pytest.fail(f"{label} 失败：\n" + "\n".join(out.strip().splitlines()[-40:]))
    assert "0 失败" in out, out
    return out


def test_frontend_auth_block_behaviour():
    """两个页面的登录链路必须可用，且不依赖任何原生弹窗"""
    _run_node(SCRIPT, "前端认证块测试")


def test_frontend_chat_context_persistence():
    """thread_id **和推理模式**都要持久化到对话上（刷新后能接着聊）

    只存 thread_id 是不够的：模式决定那个 thread 属于哪张图（后端两张图各有
    独立 checkpointer）。模式不恢复的话，刷新后模式回到默认 plan，
    react 会话的 thread_id 在 plan 图里根本不存在 —— 后端静默当成新会话，
    界面显示着历史、模型却一无所知。
    """
    _run_node(SCRIPT_CONTEXT, "前端对话上下文测试")


def test_frontend_imports_server_conversations():
    """服务端保存的会话要被导入本地，且**带全恢复所需的信息**

    导入的每条必须带 threadId 与 reasoningMode：少了任一个，点进去都恢复不了
    （模式错了 thread 就落在另一张图上，后端 404 或静默新开）。
    另外要守住的：本地已有的不重复导、用户删过的不再导回来、拿不到就静默放弃。
    """
    _run_node(ROOT / "tests" / "js" / "server_conversations_test.mjs",
              "前端服务端会话导入测试")


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


def _js_function_body(html: str, signature: str) -> str:
    """按花括号配对切出一个完整函数体

    原来用正则 `\\{(.*?)\\n\\}` —— 它假设"函数体在第一个顶格的 } 结束"，
    而函数里只要有一个顶格的内层 `}`（比如 `if (changed) { ... }`）就会提前截断，
    守卫的检查范围会随代码格式漂移。花括号配对不依赖缩进。
    """
    i = html.index(signature)
    start = html.index("{", i)
    depth = 0
    for j in range(start, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError(f"找不到 {signature} 的结尾花括号")


def test_switching_reasoning_mode_resets_thread():
    """切换 Plan/ReAct 必须开新 thread

    两张图各用独立 checkpointer（后端已修），但如果前端切模式还沿用同一个
    thread_id，界面显示的旧历史和后端上下文就对不上 —— 用户会以为"它记得"，
    实际后端是空的。

    （模式的**持久化**由 tests/js/chat_context_test.mjs 做行为测试；
      这里守的是"切模式要重置 thread"这条只存在于源码结构里的不变量。）
    """
    html = (ROOT / "course_agent/static/chat.html").read_text(encoding="utf-8")
    body = _js_function_body(html, "function setReasoningMode(")
    assert "currentThreadId = null" in body, "切换推理模式没有重置 thread_id"
    assert "applyReasoningMode(" in body, \
        "没有走 applyReasoningMode（状态 / 按钮高亮 / 持久化 都在那里）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
