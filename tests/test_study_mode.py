"""学习/复习模式的回归测试

用户的需求（原话要点）：
  · 说"我要复习某门课"或"某章"时，**分节**过一遍，优先参考 PPT；
  · 说"我要学某门课的某章"时，优先参考 PPT 仔细讲那一章；
  · **不要**把"一节包含几个知识点"写死（分节该由模型判断）；
  · 也**不要**做成死板 prompt 或状态机。

所以这套实现刻意是：**素材地图 + 按序读整章 + 一个书签**，
判断权留在模型手里。下面的测试守的就是这个形状。
"""
import json
from pathlib import Path

import pytest

from course_agent import nodes, quiz_service, tools

ROOT = Path(__file__).resolve().parent.parent


class TestSectionMap:
    """分节的"依据"必须来自真实资料，而不是模型的记忆"""

    def test_lists_real_ppts_in_course_order(self):
        out = tools.list_course_sections.invoke({"course": "操作系统"})
        # PPT 文件名就是天然的分节结构
        assert "02-1-进程与线程之进程.pdf" in out
        assert "05-死锁.pdf" in out
        order = [out.index(n) for n in
                 ("02-1-进程与线程之进程.pdf", "03-OS调度.pdf", "05-死锁.pdf")]
        assert order == sorted(order), "课件没有按课程顺序排"

    def test_carries_chapter_and_size(self):
        idx = quiz_service.section_index("操作系统")
        assert idx["ppt"], "拿不到课件清单"
        first = idx["ppt"][0]
        assert first["chapter"], "缺章节名"
        assert first["chunks"] > 0, "缺内容量（模型判断一节的量要用它）"

    def test_does_not_dictate_section_size(self):
        """用户明确要求：不要把"一节几个知识点"写死"""
        out = tools.list_course_sections.invoke({"course": "操作系统"})
        assert "分节由你判断" in out
        # 不能出现这类硬性配额
        for bad in ("每节 3 个", "每节5个", "固定 5 个知识点", "每节必须包含"):
            assert bad not in out

    def test_covers_both_courses(self):
        both = quiz_service.section_index()
        assert "操作系统" in both and "计算机网络" in both
        assert "Chapter4-Network Layer.pdf" in [
            s["source"] for s in both["计算机网络"]["ppt"]]

    def test_notes_and_textbook_are_secondary(self):
        out = tools.list_course_sections.invoke({"course": "操作系统"})
        assert "优先用这些" in out                    # 课件是首选
        assert "只在课件没覆盖时" in out              # 课本/笔记是补充
        # 学生没说从哪开始时要有明确默认：从第 1 份课件开始，别跳到课本去
        assert "从第 1 份课件开始" in out


class TestReadSection:
    """讲一章要按课件**原始顺序**读，不能只取最相似的几句"""

    def test_reads_in_chunk_order(self):
        d = quiz_service.section_chunks("05-死锁.pdf", offset=0, limit=5)
        assert d["returned"] == 5
        idx = [c["index"] for c in d["chunks"]]
        assert idx == sorted(idx), "没有按顺序返回"

    def test_pagination(self):
        d1 = quiz_service.section_chunks("05-死锁.pdf", offset=0, limit=5)
        d2 = quiz_service.section_chunks("05-死锁.pdf", offset=5, limit=5)
        assert d1["has_more"] is True
        assert d1["next_offset"] == 5
        assert d2["offset"] == 5
        assert d1["chunks"][0]["index"] != d2["chunks"][0]["index"]

    def test_last_page_has_no_more(self):
        d = quiz_service.section_chunks("05-死锁.pdf", offset=0, limit=40)
        assert d["has_more"] is False
        assert d["next_offset"] is None

    def test_limit_is_capped(self):
        """上限防止一次把整章灌进上下文"""
        d = quiz_service.section_chunks("06-内存管理与虚拟内存.pdf", limit=9999)
        assert d["limit"] <= 40

    def test_unknown_source_is_handled(self):
        d = quiz_service.section_chunks("不存在的课件.pdf")
        assert d.get("error")

    def test_tool_ships_the_teaching_contract(self):
        out = tools.read_section.invoke({"source": "05-死锁.pdf", "limit": 3})
        assert "以它为准讲解" in out          # 用课件，不用记忆
        assert "本节结束" in out              # 一节讲完要明确收尾
        assert "小测" in out                  # 讲完就测
        assert "全称" in out                  # 新符号要解释
        assert "record_study_position" in out  # 记书签
        assert "节的划分由你" in out            # 粒度不写死（由模型判断）

    def test_short_name_matches(self):
        for given in ("05-死锁", "死锁.pdf", "05-死锁.pdf"):
            out = tools.read_section.invoke({"source": given, "limit": 1})
            assert "05-死锁.pdf" in out, f"「{given}」没对上"


class TestSectionSelectionIsStructural:
    """实测教训：让模型**自己挑并叫出文件名**是不可靠的 ——
    它会 ① 去读补充笔记 ② 编一个"操作系统课件"这种不存在的名字
    ③ 拿不到资料就退回凭记忆讲。

    所以改成：模型只说「哪门课、第几节」，材料由工具认。任何"材料不可用"的情况
    都自动落到本课课件上，绝不让它空手回去自由发挥。
    """

    def test_course_plus_section_needs_no_filename(self):
        out = tools.read_section.invoke({"course": "操作系统", "section": 1, "limit": 2})
        assert "02-1-进程与线程之进程.pdf" in out
        assert "本课课件 1/" in out

    def test_section_index_walks_the_course(self):
        """第 N 节按**课件顺序**走，不是按文件夹里的字符串排序

        ⚠️ 这条原来写死"第 3 份是 03-OS调度"——2026-10-18 新导入《2-01-原子操作》
        与《2-02-临界区、互斥与锁》之后它就红了。那是"测试依赖本地数据"的典型：
        课程加一份课件本来不该让测试失败。改成只守在**顺序关系**上。
        """
        first = tools.read_section.invoke({"course": "操作系统", "section": 1, "limit": 1})
        assert "02-1-" in first and "下一份是第 2 份" in first

        # 把整份索引走一遍，确认 02-1 → 02-2 → 03-OS调度 的**相对顺序**没变
        from course_agent import quiz_service
        order = [p["source"] for p in quiz_service.section_index("操作系统")["ppt"]]
        idx = {name: i for i, name in enumerate(order)}
        assert idx["02-1-进程与线程之进程.pdf"] < idx["02-2-进程与线程之线程.pdf"]
        assert idx["02-2-进程与线程之线程.pdf"] < idx["03-OS调度.pdf"]
        # 第 N 节取到的就是索引里第 N 份
        for n in (1, 3):
            out = tools.read_section.invoke({"course": "操作系统", "section": n, "limit": 1})
            assert f"本课课件 {n}/{len(order)}" in out, out[:80]

    def test_out_of_range_section_falls_back_to_first(self):
        out = tools.read_section.invoke({"course": "操作系统", "section": 99, "limit": 1})
        assert "02-1-" in out

    def test_reading_notes_is_auto_switched_to_ppt(self):
        """光提醒没用（实测模型照样拿笔记讲）→ 直接换材料"""
        out = tools.read_section.invoke({
            "course": "操作系统", "source": "操作系统_补充知识点笔记.md", "limit": 2})
        assert "已自动换材料" in out
        assert "补充笔记" in out
        # 换成了课件：内容应该来自课件而不是笔记
        assert "02-1-进程与线程之进程.pdf" in out

    def test_fabricated_filename_falls_back_to_ppt(self):
        """模型编了个不存在的"操作系统课件" —— 不能让它空手回去凭记忆讲"""
        out = tools.read_section.invoke({
            "course": "操作系统", "source": "操作系统课件", "limit": 2})
        assert "已自动选材料" in out
        assert "02-1-进程与线程之进程.pdf" in out

    def test_no_material_at_all_says_what_to_do(self):
        out = tools.read_section.invoke({"source": "根本没有这个文件"})
        assert "没找到" in out
        assert "read_section(course=" in out, "要给出正确写法"

    def test_args_tolerate_strings(self):
        """模型常把数字传成字符串；实测因此吃过 ValidationError"""
        out = tools.read_section.invoke({"course": "操作系统", "section": "2",
                                         "offset": "0", "limit": "2"})
        assert "02-2-" in out, out[:80]

    def test_args_tolerate_messy_strings(self):
        """实测模型还传过 '2节' 这种 —— schema 层就必须放行，否则拿不到我的容错"""
        out = tools.read_section.invoke({"course": "操作系统", "section": "2节",
                                         "limit": "1块"})
        assert "02-2-" in out, out[:80]

    def test_negative_offset_is_clamped(self):
        out = tools.read_section.invoke({"course": "操作系统", "offset": -5, "limit": 2})
        assert "本次第 1~" in out


class TestChapterLookup:
    """学生按章提问是最自然的说法（"我要学死锁那一章"）——
    实测这条路一开始没做，模型就去读了第 1 份课件、还谎称内容是死锁那份的。"""

    @pytest.mark.parametrize("course,chapter,expect", [
        ("操作系统", "死锁", "05-死锁.pdf"),
        ("操作系统", "OS调度", "03-OS调度.pdf"),
        ("操作系统", "进程与线程", "02-1-进程与线程之进程.pdf"),
        ("操作系统", "内存管理与虚拟内存", "06-内存管理与虚拟内存.pdf"),
        ("计算机网络", "Transport Layer", "Chapter5-Transport Layer.pdf"),
        ("计算机网络", "Network Layer", "Chapter4-Network Layer.pdf"),
    ])
    def test_chapter_resolves_to_the_right_ppt(self, course, chapter, expect):
        out = tools.read_section.invoke({"course": course, "chapter": chapter,
                                         "limit": 1})
        assert "按章定位" in out
        assert expect in out, f"「{chapter}」没有定位到 {expect}"

    def test_unknown_chapter_lists_available_ones(self):
        out = tools.read_section.invoke({"course": "操作系统", "chapter": "量子计算"})
        assert "没有叫" in out
        assert "死锁" in out, "要列出可用章名，便于自我纠正"

    def test_chapter_takes_priority_over_section(self):
        out = tools.read_section.invoke({"course": "操作系统", "chapter": "死锁",
                                         "section": 1, "limit": 1})
        assert "05-死锁.pdf" in out, "给了章名就该按章走，不能被 section 抢走"


class TestScopedRetrieval:
    """「优先参考 PPT」「不串课」都靠检索过滤，而不是靠 prompt 请求"""

    def test_scoped_to_one_file(self):
        out = tools.search_knowledge.invoke({"query": "互斥使用",
                                             "source": "05-死锁.pdf"})
        assert "只在《05-死锁.pdf》里检索" in out

    def test_scoped_to_course(self):
        out = tools.search_knowledge.invoke({"query": "进程调度", "course": "操作系统"})
        assert "已在「操作系统」的资料内检索" in out

    def test_relevance_is_displayed_from_the_right_field(self):
        """kb.search() 的 score 是绝对余弦（没有 sim 字段）——
        曾经按 sim 取，显示出来全是 0.000。"""
        hits = __import__("rag_tool").get_knowledge_base().search(
            "死锁", top_k=2, sources=["05-死锁.pdf"])
        assert hits and hits[0].get("score", 0) > 0.3
        rendered = tools._format_hits(hits)
        assert "相关度=0.000" not in rendered
        assert "相关度=" in rendered

    def test_format_hits_handles_hybrid_shape_too(self):
        """_hybrid_search 返回的是 sim + 归一化 score，也得能正确显示"""
        hybrid_like = [{"source": "x.pdf", "text": "t", "sim": 0.71, "score": 0.99}]
        assert "相关度=0.710" in tools._format_hits(hybrid_like)

    def test_search_knowledge_keeps_the_grounding_contract(self):
        out = tools.search_knowledge.invoke({"query": "死锁", "course": "操作系统"})
        assert "以这些片段为准" in out
        assert "全称" in out, "契约里要要求解释缩写/符号"


class TestStudyBookmark:
    """书签只是记忆 —— 讲什么、怎么分节、何时进下一节都由模型判断"""

    def test_records_position(self):
        r = json.loads(tools.record_study_position.invoke({
            "course": "操作系统", "chapter": "死锁",
            "section": "死锁的四个必要条件", "source": "05-死锁.pdf",
            "mode": "学习", "status": "讲解中"}))
        assert r["recorded"] == {
            "mode": "学习", "course": "操作系统", "chapter": "死锁",
            "section": "死锁的四个必要条件", "source": "05-死锁.pdf",
            "status": "讲解中"}

    def test_guide_covers_the_pedagogy(self):
        r = json.loads(tools.record_study_position.invoke(
            {"course": "操作系统", "section": "x"}))
        g = r["guide"]
        for kw in ("只讲这一节", "本节结束", "小测", "错在哪", "重讲", "不要硬推"):
            assert kw in g, f"契约缺「{kw}」"

    def test_executor_extracts_it_into_session(self):
        r = tools.record_study_position.invoke({"course": "操作系统",
                                                "section": "死锁概述"})
        plan = [{"tool": "record_study_position", "args": {}}]
        results = [nodes.ToolMessage(content=r, tool_call_id="plan_0",
                                     name="record_study_position")]
        sess = nodes._extract_study_session({}, plan, results)
        assert sess["study"]["course"] == "操作系统"
        assert sess["study"]["section"] == "死锁概述"

    def test_garbage_does_not_break(self):
        plan = [{"tool": "record_study_position", "args": {}}]
        for bad in ["不是 json", "{}", '{"recorded": {}}']:
            results = [nodes.ToolMessage(content=bad, tool_call_id="plan_0",
                                         name="record_study_position")]
            assert nodes._extract_study_session({}, plan, results) is None

    def test_bookmark_survives_int_section(self):
        """模型把 section 传成数字不该报 ValidationError（实测报过）"""
        r = tools.record_study_position.invoke({"course": "操作系统", "section": 3})
        assert json.loads(r)["recorded"]["section"] == "3"

    def test_chapter_in_course_field_is_corrected(self):
        """实测模型会写 course="死锁"（把章名当课程）—— 要自我纠正，
        否则书签里的课程字段被污染，下一轮的"当前学习位置"就跟着错。"""
        r = json.loads(tools.record_study_position.invoke(
            {"course": "死锁", "section": "死锁的必要条件"}))["recorded"]
        assert r["course"] == "操作系统", r
        assert r["chapter"] == "死锁", r

    def test_read_position_is_remembered_automatically(self):
        """从 read_section 的返回里自动记住"读到哪了"（含课程）

        实测教训：第一版按自然语言措辞解析，结果 study 里**没有 course** ——
        而 planner 是 `if study.get("course")` 才注入"当前学习位置"，
        于是整块提示被跳过，模型照样不知道读到哪了、又跳去下一份课件。
        """
        out = tools.read_section.invoke({"course": "操作系统", "chapter": "死锁",
                                         "limit": 15})
        assert "[study-state]" in out, "工具必须给出机器可读的一行"
        plan = [{"tool": "read_section", "args": {}}]
        results = [nodes.ToolMessage(content=out, tool_call_id="plan_0",
                                     name="read_section")]
        sess = nodes._extract_study_session({}, plan, results)
        study = sess["study"]
        assert study["course"] == "操作系统", "没有 course，planner 那块提示就白写了"
        assert study["chapter"] == "死锁"
        assert study["last_source"] == "05-死锁.pdf"
        assert study["next_offset"] == 15
        assert "第 1-15 块" in study["last_read"]

    def test_read_position_alone_reaches_the_planner(self, monkeypatch):
        """端到端：只经过 read_section（没有 record_study_position），
        planner 也必须看到"刚读到哪、下一步用哪个 offset"

        这条就是上面那个 bug 的回归测试 —— 之前 study 缺 course，整块被跳过。
        """
        out = tools.read_section.invoke({"course": "操作系统", "chapter": "死锁",
                                         "limit": 15})
        plan = [{"tool": "read_section", "args": {}}]
        results = [nodes.ToolMessage(content=out, tool_call_id="plan_0",
                                     name="read_section")]
        session = nodes._extract_study_session({}, plan, results)

        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({"messages": [nodes.HumanMessage(content="继续讲")],
                            "plan": [], "tool_rounds": 0, "session": session})
        p = captured["prompt"]
        assert "当前学习位置" in p, "planner 没看到学习位置（course 丢了）"
        assert "offset=15" in p
        assert "05-死锁.pdf" in p

    def test_last_page_records_no_next_offset(self):
        out = tools.read_section.invoke({"course": "操作系统", "chapter": "死锁",
                                         "limit": 40})
        plan = [{"tool": "read_section", "args": {}}]
        results = [nodes.ToolMessage(content=out, tool_call_id="plan_0",
                                     name="read_section")]
        sess = nodes._extract_study_session({}, plan, results)
        assert sess["study"]["next_offset"] is None

    def test_planner_tells_it_to_finish_the_current_ppt(self, monkeypatch):
        """"继续讲"要接着读同一份课件，不能跳下一份"""
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({
            "messages": [nodes.HumanMessage(content="继续讲")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"mode": "学习", "course": "操作系统",
                                  "chapter": "死锁", "section": "死锁的概念",
                                  "source": "05-死锁.pdf",
                                  "last_source": "05-死锁.pdf",
                                  "last_read": "第 1~15 块",
                                  "next_offset": 15, "status": "讲解中"}}})
        p = captured["prompt"]
        assert "offset=15" in p
        # 要给**可直接照抄的一句调用** —— 描述式提示实测不跟
        assert 'read_section(course="操作系统", source="05-死锁.pdf", offset=15' in p
        assert "回到 section=1" in p

    def test_planner_shows_the_bookmark_next_turn(self, monkeypatch):
        """下一轮 planner 必须看得到"学到哪了"，否则"下一节"会重新开始"""
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({
            "messages": [nodes.HumanMessage(content="下一节")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"mode": "学习", "course": "操作系统",
                                  "chapter": "死锁", "section": "死锁的四个必要条件",
                                  "source": "05-死锁.pdf", "status": "已通过"}}})
        p = captured["prompt"]
        assert "当前学习位置" in p
        assert "操作系统" in p and "死锁" in p
        assert "下一节" in p, "要告诉它学生说下一节时该接着讲"

    def test_bookmark_and_drill_cursor_coexist(self):
        """书签与刷题游标互不干扰（同一份 session 里两套状态）"""
        session = {"paper": "某卷", "next_group_index": 2}
        r = tools.record_study_position.invoke({"course": "计算机网络",
                                                "section": "传输层概述"})
        plan = [{"tool": "record_study_position", "args": {}}]
        results = [nodes.ToolMessage(content=r, tool_call_id="plan_0",
                                     name="record_study_position")]
        merged = nodes._extract_study_session(session, plan, results)
        assert merged["paper"] == "某卷"                  # 刷题游标还在
        assert merged["study"]["course"] == "计算机网络"   # 书签也记上了


class TestPromptContracts:
    """prompt 是原则，不是清单；且工具数必须和注册数一致"""

    def test_planner_routes_study_requests_to_the_right_tools(self):
        p = nodes.PLANNER_PROMPT
        assert "list_course_sections" in p
        assert "read_section" in p
        assert "record_study_position" in p
        # 复习/学习不能走刷题工具
        assert "不要" in p and "list_papers" in p

    def test_planner_tool_count_matches_registry(self):
        n = len(tools.tools_by_name)
        assert f"你有 {n} 个工具可用" in nodes.PLANNER_PROMPT

    def test_synthesizer_has_study_principles(self):
        p = nodes.SYNTHESIZER_PROMPT
        assert "学习/复习模式" in p
        for kw in ("一次只讲一节", "本节结束", "小测", "错在哪", "重讲",
                   "新符号", "作业辅导", "不串课"):
            assert kw in p, f"教学原则缺「{kw}」"

    def test_principles_are_not_hardcoded_counts(self):
        """用户明确说：分节粒度交给模型，不要写死"""
        p = nodes.SYNTHESIZER_PROMPT
        assert "由你判断" in p
        for bad in ("每节固定", "每节 3 个", "每节必须讲 5"):
            assert bad not in p

    def test_no_state_machine_in_the_study_flow(self):
        """用户明确说不要状态机 —— 书签是记忆，流程仍由模型决定

        这条原来截取 `_extract_study_session` 的**前 1200 字符**再做字符串匹配。
        实测：该函数现在有 3400+ 字符，在**距函数头 3000 字符处**塞进
        `stage` / `step_index`，全量 594 条测试依然全绿 —— 它有 65% 的内容
        从来没被这条守卫看过。（同一个目录里另一条测试的注释就警告过
        "别用字符数窗口当函数边界，那就是一种数据快照"，这条当时漏改了。）

        现在改成按 **AST** 取该函数的标识符与字符串常量：
        注释不参与 AST，所以既骗不过，也不受函数长度影响。
        """
        import ast

        tree = ast.parse((ROOT / "course_agent/nodes.py").read_text(encoding="utf-8"))
        func = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef)
                     and n.name == "_extract_study_session"), None)
        assert func is not None, "找不到 _extract_study_session"

        idents, strings = set(), set()
        for n in ast.walk(func):
            if isinstance(n, ast.Name):
                idents.add(n.id)
            elif isinstance(n, ast.Attribute):
                idents.add(n.attr)
            elif isinstance(n, ast.arg):
                idents.add(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                strings.add(n.value)

        assert "recorded" in strings, "函数没在读工具返回的 recorded 字段"
        forbidden = {"stage", "step_index", "must_be", "state_machine"}
        hit = forbidden & (idents | strings)
        assert not hit, f"学习流程里出现了状态机味道的东西: {sorted(hit)}"
