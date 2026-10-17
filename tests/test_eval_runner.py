"""`eval/runner.py` 自身的两个真实缺陷的回归（都是 2026-10-16 跑基线时踩到的）

跑一条基线评测要几十秒到几分钟、还要真模型额度，所以这里**不跑真评测**：
只把"已经坏过的那两处"钉住 —— 打印路径与编码。两个 bug 的共同点是
**报告明明写成功了，退出码却是 1**（假的失败信号），最坏的情况是一条用例都没跑。
"""
import json
import os
import sys
from pathlib import Path

import pytest

from eval import runner


def _point_runner_at_tmp(monkeypatch, tmp_path, case_id="fake_case"):
    """把 runner 的用例目录/报告目录换成临时目录（绝不碰真的 eval/cases 与仓库）"""
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / f"{case_id}.json").write_text(json.dumps({
        "id": case_id, "why": "测试替身", "mode": "plan",
        "turns": [{"user": "你好", "expect": {}}],
    }, ensure_ascii=False), encoding="utf-8")
    reports = tmp_path / "reports"
    monkeypatch.setattr(runner, "CASES_DIR", cases)
    monkeypatch.setattr(runner, "REPORTS_DIR", reports)

    monkeypatch.setattr(runner, "run_case", lambda case, base, attempts: {
        "id": case["id"], "attempts": [
            {"attempt": 1, "thread_id": "t1",
             "turns": [{"checks": [{"check": "x", "ok": True, "detail": ""}],
                        "ttft_ms": 100, "total_ms": 200, "user": "你好"}]}],
        "checks_total": 1, "checks_passed": 1, "stable_pass": True, "failures": []})
    monkeypatch.setattr(runner, "delete_threads", lambda base, tids, mode="plan": 0)
    return reports


class TestReportPathNeverCrashes:
    """`--out` 的路径不在仓库下（或只是没 resolve）时，曾让整轮评测以退出码 1 结束"""

    def test_path_outside_the_repo_is_printed_as_is(self):
        # 刻意用仓库的**父目录**：pytest 的 tmp_path 在本机落在仓库内的 .tmp 下，
        # 那个"在仓库里"，测不出这条
        outside = runner.ROOT.parent / "outside-report.json"
        assert runner._display_path(outside) == str(outside)

    def test_path_inside_the_repo_is_relative(self):
        inside = runner.ROOT / "eval" / "reports" / "1.7.2.json"
        assert runner._display_path(inside) == str(
            Path("eval") / "reports" / "1.7.2.json")

    def test_a_relative_out_does_not_turn_success_into_failure(
            self, monkeypatch, tmp_path, capsys):
        """端到端：报告写好了、用例全绿 → 必须返回 0（原来这里抛 ValueError）"""
        _point_runner_at_tmp(monkeypatch, tmp_path)
        out = tmp_path / "deep" / "report.json"
        out.parent.mkdir()
        monkeypatch.setattr(sys, "argv", ["eval.runner", "--out", str(out)])
        rc = runner.main()
        assert rc == 0, f"报告写成功了却返回 {rc}（假的失败信号）"
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]["cases"] == 1

    def test_the_default_report_lands_in_the_reports_dir(self, monkeypatch, tmp_path):
        reports = _point_runner_at_tmp(monkeypatch, tmp_path)
        monkeypatch.setattr(sys, "argv", ["eval.runner"])
        from config import APP_VERSION
        assert runner.main() == 0
        assert (reports / f"{APP_VERSION}.json").exists()


class TestTransportRetry:
    """传输层偶发超时不该把整批评测带走（2026-10-16 实测被带走两次）

    这里用**本地**假 HTTP 服务复现"服务端把请求挂住"，而不是把 requests 整层
    mock 掉 —— 判据是"第二/第三次请求真的带着新 thread_id 发出去了"，mock 掉
    requests 就看不见这件事了。
    """

    def _stub_server(self, hang_times: int):
        """返回一个 ThreadingHTTPServer：前 hang_times 次请求直接挂住不响应"""
        import http.server
        import threading

        state = {"seen": [], "n": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                state["seen"].append(body.get("thread_id"))
                state["n"] += 1
                if state["n"] <= hang_times:
                    import time as _t
                    _t.sleep(3)                # 比客户端 timeout 长 → 客户端读超时
                    return
                payload = ("data: " + json.dumps(
                    {"type": "content", "content": "你好"}, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                self.send_response(200)
                # 必须带 charset：requests 对 text/event-stream 没有 charset 时
                # 按 ISO-8859-1 解，中文会变成乱码（生产的 SSE 响应带 charset）
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, state

    def test_timeout_is_retried_with_a_fresh_thread_id(self):
        srv, state = self._stub_server(hang_times=1)
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            turn = runner.stream_turn(base, "你好", "eval-x-0-abc", timeout=1, retries=2)
        finally:
            srv.shutdown()
        assert turn["answer"] == "你好"
        assert len(state["seen"]) == 2, f"没有重试（实际请求 {len(state['seen'])} 次）"
        assert state["seen"][0] == "eval-x-0-abc"
        assert state["seen"][1] != state["seen"][0], (
            "重试换了 thread_id 才干净：上一次可能已经把半截状态写进会话了")
        assert turn["thread_id"] == state["seen"][1], (
            "返回的 thread_id 必须是**实际**用的那个，否则清理会话会漏")

    def test_all_retries_failing_raises_a_clear_error(self):
        srv, state = self._stub_server(hang_times=99)
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            with pytest.raises(RuntimeError, match="传输"):
                runner.stream_turn(base, "你好", "eval-x-0-abc", timeout=1, retries=1)
        finally:
            srv.shutdown()
        assert len(state["seen"]) == 2, "retries=1 应该总共试 2 次"


class TestToolsAnyOfMeansAnyOne:
    """`tools_any_of` 是"任一命中即可" —— 原来被实现成"每个都要命中"

    2026-10-16 在 `react_no_tool_call_leak` 上暴露：用例允许三种取材料方式任选
    （search_knowledge / read_section / list_course_sections），模型调了
    search_knowledge，却被判"没调 read_section / list_course_sections" 而红。
    单元素用例一直没暴露这个 bug。
    """

    def _checks(self, tools, expect):
        turn = {"answer": "答案", "tools": tools, "content_events": 1,
                "answer_source": "content", "ttft_ms": 100, "total_ms": 200}
        return runner.run_checks(expect, turn, {})

    def test_one_hit_is_enough(self):
        checks = self._checks(["search_knowledge"],
                              {"tools_any_of": ["search_knowledge",
                                                "read_section",
                                                "list_course_sections"]})
        assert len(checks) == 1, checks
        assert checks[0]["ok"] is True, checks

    def test_no_hit_fails(self):
        checks = self._checks(["grade_answers"],
                              {"tools_any_of": ["search_knowledge", "read_section"]})
        assert checks[0]["ok"] is False, checks

    def test_single_element_still_works(self):
        assert self._checks([], {"tools_any_of": ["search_knowledge"]})[0]["ok"] is False
        assert self._checks(["search_knowledge"],
                            {"tools_any_of": ["search_knowledge"]})[0]["ok"] is True

    def test_ttft_label_shows_the_real_threshold(self):
        """报告里那条判据必须写出**真实**门槛（原来是写死的 8000，成了假信息）"""
        checks = self._checks([], {"max_ttft_ms": 15000})
        assert "15000" in checks[0]["check"], checks[0]["check"]


class TestOpeningCriteria:
    """开场那轮的三个新判据（W8/W9）：不调工具、必须说到真值、不许反问"""

    def _checks(self, tools, answer, expect):
        turn = {"answer": answer, "tools": tools, "content_events": 1,
                "answer_source": "content", "ttft_ms": 100, "total_ms": 200}
        return runner.run_checks(expect, turn, {})

    def test_tools_exactly_empty(self):
        assert self._checks([], "…", {"tools_exactly_empty": True})[0]["ok"] is True
        assert self._checks(["read_section"], "…",
                            {"tools_exactly_empty": True})[0]["ok"] is False

    def test_must_contain(self):
        assert self._checks([], "你学到死锁那节了",
                            {"must_contain": ["死锁"]})[0]["ok"] is True
        bad = self._checks([], "继续学吧", {"must_contain": ["死锁"]})
        assert bad[0]["ok"] is False and "死锁" in bad[0]["detail"]

    def test_forbid_phrases(self):
        assert self._checks([], "你之前学到哪了？",
                            {"forbid_phrases": ["你之前学到哪"]})[0]["ok"] is False
        assert self._checks([], "你上次在死锁这一节",
                            {"forbid_phrases": ["你之前学到哪"]})[0]["ok"] is True


class TestSeedRequiresAnIsolatedProgressFile:
    """seed 必须 fail-closed：没设 PROGRESS_FILE 就不许跑，否则会写进真实进度"""

    def test_it_refuses_without_progress_file(self, monkeypatch):
        monkeypatch.delenv("PROGRESS_FILE", raising=False)
        with pytest.raises(SystemExit, match="PROGRESS_FILE"):
            runner.apply_seed({"id": "demo", "seed": {"study": [{"course": "操作系统"}]}})

    def test_it_seeds_when_isolated(self, monkeypatch, tmp_path):
        from course_agent import progress

        target = tmp_path / "p.json"
        monkeypatch.setenv("PROGRESS_FILE", str(target))
        monkeypatch.setattr(progress, "PROGRESS_PATH", target)
        summary = runner.apply_seed({"id": "demo", "seed": {
            "study": [{"course": "操作系统", "chapter": "死锁",
                       "section": "死锁四个条件", "source": "03-OS调度.pdf"}],
            "quiz": [{"paper": "某卷", "group_index": 1,
                      "next_group_index": 2, "qids": ["q1"]}]}})
        assert "操作系统" in summary and "某卷" in summary

    def test_no_seed_is_a_no_op(self):
        assert runner.apply_seed({"id": "x"}) == ""


class TestObserveTier:
    """观察档（用户拍板：永不卡发版，但连续 3 版下滑要报警）

    B 类话术行为（岔开提问拉回主线…）没有客观真值，写成硬红只会产生噪声；
    但"不写"就等于没观测。所以：照样算、照样进报告，只是**不计入稳定通过**。
    """

    def _turn(self, tools=(), answer="好的", **over):
        turn = {"answer": answer, "tools": list(tools), "content_events": 1,
                "answer_source": "content", "ttft_ms": 100, "total_ms": 200}
        turn.update(over)
        return turn

    def test_observe_checks_are_marked_and_excluded_from_hard(self):
        checks = runner.run_checks(
            {"answer_min_chars": 1, "observe": {"must_contain": ["死锁"]}},
            self._turn(answer="换个讲法吧"), {})
        hard = [c for c in checks if not c.get("observe")]
        obs = [c for c in checks if c.get("observe")]
        assert len(hard) == 1 and hard[0]["ok"] is True
        assert len(obs) == 1 and obs[0]["ok"] is False, checks

    def test_observe_failure_does_not_fail_the_case(self, monkeypatch):
        """观察档全红也不许把 stable_pass 弄成 False"""
        monkeypatch.setattr(runner, "stream_turn",
                            lambda *a, **k: self._turn(answer="好的"))
        case = {"id": "demo", "mode": "plan", "turns": [
            {"user": "你好",
             "expect": {"answer_min_chars": 1,
                        "observe": {"must_contain": ["死锁"]}}}]}
        result = runner.run_case(case, "http://x", attempts=1)
        assert result["stable_pass"] is True, result
        assert result["observe_hits"] == 0 and result["observe_total"] == 1

    def test_trend_warns_only_on_three_strict_declines(self, monkeypatch, tmp_path):
        def make(stamp, hits):
            return {"started_at": stamp, "cases": [{
                "id": "c", "observations": [
                    {"check": "拉回主线", "ok": i < hits} for i in range(2)]}]}

        monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
        for stamp, hits in (("1", 2), ("2", 1), ("3", 0)):
            (tmp_path / f"{stamp}.json").write_text(
                json.dumps(make(stamp, hits), ensure_ascii=False), encoding="utf-8")
        warnings = runner.observe_trend()
        assert warnings and "拉回主线" in warnings[0], warnings


class TestConsoleEncoding:
    """中文 Windows 的控制台是 cp936，打印 `▶`/`✗` 曾直接 UnicodeEncodeError

    注意这里**不**用 capsys 断言"字符打出来了"：capsys 换掉的是 pytest 自己的
    缓冲流，本来就按 UTF-8 处理，于是"没切编码也照样通过"——那是个安慰剂
    （变异验证时抓到的：把 reconfigure 换成 pass，测试仍然绿）。
    所以直接观察**机制**：`_force_utf8_stdout` 必须用 utf-8 去 reconfigure。
    """

    def test_stdout_is_reconfigured_to_utf8(self, monkeypatch):
        calls = []

        class _RecordingStream:
            def reconfigure(self, **kwargs):
                calls.append(kwargs)

            def write(self, s): pass

            def flush(self): pass

        monkeypatch.setattr(sys, "stdout", _RecordingStream(), raising=False)
        monkeypatch.setattr(sys, "stderr", _RecordingStream(), raising=False)
        runner._force_utf8_stdout()
        assert len(calls) == 2, f"两个流都该被重配，实际 {calls}"
        for kwargs in calls:
            assert kwargs.get("encoding") == "utf-8", kwargs
            assert kwargs.get("errors") == "replace", (
                f"没有 errors=replace：打不出某个字符时整轮评测会白跑，{kwargs}")

    def test_it_survives_streams_without_reconfigure(self, monkeypatch):
        """.reconfigure 不存在的流（被包装/替换过的 stdout）不许让评测崩掉"""
        class _NoReconfigure:
            def write(self, s): pass

            def flush(self): pass

        monkeypatch.setattr(sys, "stdout", _NoReconfigure(), raising=False)
        monkeypatch.setattr(sys, "stderr", _NoReconfigure(), raising=False)
        runner._force_utf8_stdout()      # 不抛异常即通过

    def test_main_calls_it_before_printing_anything(self, monkeypatch, tmp_path):
        """main 必须**先**切编码再打印 —— 否则第一条带装饰符的标题就崩了"""
        _point_runner_at_tmp(monkeypatch, tmp_path)
        order = []
        monkeypatch.setattr(runner, "_force_utf8_stdout", lambda: order.append("reconfigure"))
        real_print = print
        monkeypatch.setattr("builtins.print",
                            lambda *a, **k: (order.append("print"), real_print(*a, **k))[1])
        monkeypatch.setattr(sys, "argv", ["eval.runner"])
        assert runner.main() == 0
        assert order and order[0] == "reconfigure", (
            f"打印发生在切编码之前：{order[:4]}")

    def test_output_survives_a_gbk_console(self, tmp_path):
        """**端到端**：在"控制台是 GBK"的环境里真的跑一遍 main，输出不许崩

        这条是 `▶` 那类装饰符的守卫。为什么必须开子进程：控制台编码是**进程级**的
        （`PYTHONIOENCODING`），在 pytest 进程里没法真实复现 —— 而原来那条用
        capsys 的写法是个安慰剂（capsys 换的是 pytest 自己的 UTF-8 缓冲流，
        把 reconfigure 改成 pass 它照样绿，见下条注释）。

        子进程里把用例目录/报告目录指向临时目录，并换掉 run_case 与 delete_threads：
        **绝不碰真题库、真会话库、真报告**。
        """
        cases = tmp_path / "cases"
        cases.mkdir()
        (cases / "case.json").write_text(json.dumps({
            "id": "case", "why": "编码回归", "mode": "plan",
            "turns": [{"user": "你好", "expect": {}}]}), encoding="utf-8")
        reports = tmp_path / "reports"
        out = tmp_path / "report.json"

        driver = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from eval import runner\n"
            f"runner.CASES_DIR = Path(r'{cases}')\n"
            f"runner.REPORTS_DIR = Path(r'{reports}')\n"
            "def fake_run_case(case, base, attempts):\n"
            "    turn = {'checks': [{'check': 'c', 'ok': False, 'detail': '故意红一条'}],\n"
            "            'ttft_ms': 1, 'total_ms': 2, 'user': '你好'}\n"
            "    return {'id': case['id'], 'attempts': [{'attempt': 1,\n"
            "             'thread_id': 't', 'turns': [turn]}],\n"
            "            'checks_total': 1, 'checks_passed': 0,\n"
            "            'stable_pass': False, 'failures': ['故意红一条']}\n"
            "runner.run_case = fake_run_case\n"
            "runner.delete_threads = lambda base, tids, mode='plan': 0\n"
            "sys.exit(runner.main())\n"
        )
        import subprocess
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0",
                    "PYTHONPATH": str(runner.ROOT),
                    "TMP": str(tmp_path), "TEMP": str(tmp_path)})
        env.pop("ALLOW_TESTS_WITH_SERVER", None)
        p = subprocess.run([sys.executable, "-c", driver, "--out", str(out)],
                           cwd=runner.ROOT, capture_output=True, text=True,
                           env=env, encoding="utf-8", errors="replace")
        assert p.returncode == 1, (          # 用例故意红，所以退出码 1 是"跑完了"
            f"在 GBK 控制台下没跑完：rc={p.returncode}\n"
            f"stdout={p.stdout[-600:]}\nstderr={p.stderr[-600:]}")
        assert out.exists(), f"报告没写出来：{p.stderr[-400:]}"

    def test_source_has_no_symbols_that_gbk_cannot_print(self):
        """另一面：源码里不该再有 GBK 编不出的装饰性符号

        用 AST 取**所有 print 调用里的字符串字面量**（含 f-string 的字面量片段），
        逐个试 `gbk` 编码 —— 这样"以后又加回一个 ▶/✗"会在这里变红。
        注释与 docstring 不参与控制台输出，不检查。
        """
        import ast

        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"):
                continue
            for arg in ast.walk(node):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.append(arg.value)
        assert literals, "一个 print 字面量都没扫到 —— 扫描逻辑坏了"

        bad = []
        for text in literals:
            for ch in text:
                try:
                    ch.encode("gbk")
                except UnicodeEncodeError:
                    bad.append(ch)
        assert not bad, f"print 的字面量里有 GBK 打不出的字符：{sorted(set(bad))}"
