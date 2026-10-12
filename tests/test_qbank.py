"""题库访问层 + 批改解析的回归测试

这些用例对应修复前的真实 bug：
1. _qnum 从 qid 里取到"年份"，导致按题号批改永远匹配不上
2. 判断题答案完全无法解析
3. 纯字母去重把 "ABCDA" 变成 "ABCD"
4. 一句带英文解释的话被拆成多个答案套到前几题
5. question_bank.json 漏了两份试卷，题库只加载到 125 题
6. 出题接口把 text/answer 一起返回给模型（泄题）
"""

import json

import pytest

from course_agent import qbank
from conftest import requires_images


@pytest.fixture(scope="module")
def papers():
    return {p["source"]: p for p in qbank.list_papers()}


def _group(source, qtype, idx=1):
    g = qbank.get_group(source, qtype, idx)
    qs = [q for q in qbank.paper_questions(source) if q["qid"] in g["qids"]]
    order = {qid: i for i, qid in enumerate(g["qids"])}
    qs.sort(key=lambda q: order[q["qid"]])
    return g, qs


class TestLoading:
    def test_loads_every_per_paper_file(self):
        """题库必须合并**全部** questions_*.json，不能只读某一个文件

        原先的 bug 是只读 question_bank.json —— 那是个过期快照，漏了整整两份卷子，
        题库只加载到 125/191。这里断言的是**语义不变量**：加载结果必须覆盖所有
        单卷文件里的每一道题（按 source+qid 比对），不写死题数，
        以后增删卷子都不用改这条测试。
        """
        expected = set()
        for f in qbank.QUESTION_BANK_DIR.glob("questions_*.json"):
            for q in json.loads(f.read_text(encoding="utf-8-sig")):
                expected.add((q.get("source"), q.get("qid")))
        got = {(q.get("source"), q.get("qid")) for q in qbank.load_questions()}
        assert got == expected, \
            f"漏了 {len(expected - got)} 题，多了 {len(got - expected)} 题"

    def test_no_duplicate_source_qid(self):
        qs = qbank.load_questions()
        keys = [(q.get("source"), q["qid"]) for q in qs]
        assert len(keys) == len(set(keys))

    def test_both_courses_present(self, papers):
        courses = {p["course"] for p in papers.values()}
        assert "操作系统" in courses
        assert "计算机网络" in courses

    def test_every_per_paper_file_is_listed(self, papers):
        """每份单卷文件都必须在 list_papers 里出现

        原实现只看得见合并文件里的卷子（50 题那份计网真题就是这么丢的）。
        """
        for f in qbank.QUESTION_BANK_DIR.glob("questions_*.json"):
            stem = f.stem.replace("questions_", "")
            assert stem in papers, f"{stem} 没出现在试卷列表里"


class TestNumbering:
    def test_q_num_not_year(self):
        """qid 形如 '2026OS_Q1'，题号必须是 1 而不是 2026"""
        qs = qbank.paper_questions("2026春操作系统期末试题（回忆版）")
        choices = [q for q in qs if q["type"] == "选择题"]
        assert qbank._qno(choices[0]) == 1
        assert max(qbank._qno(q) for q in choices) == 20

    def test_section_numbers_restart_per_type(self):
        """题号按题型分段从 1 开始：选择题 1..N、综合题 1..M"""
        qs = qbank.paper_questions("2019秋本部计算机网络模拟题")
        nums = qbank.section_numbers(qs)
        choices = [nums[q["qid"]] for q in qs if q["type"] == "选择题"]
        comp = [nums[q["qid"]] for q in qs if q["type"] == "综合题"]
        assert choices == list(range(1, 26))
        assert comp == [1, 2, 3, 4]

    def test_section_numbers_are_unique_per_paper(self):
        """整卷范围内 (题型, 题号) 唯一 —— 这正是不能用 display_num 的原因"""
        for src in ["2019秋本部计算机网络模拟题", "2024年秋操作系统期末考试试题（回忆版）"]:
            qs = qbank.paper_questions(src)
            nums = qbank.section_numbers(qs)
            keys = [(q["type"], nums[q["qid"]]) for q in qs]
            assert len(keys) == len(set(keys)), f"{src} 题号有冲突"
            assert len(nums) == len(qs)

    def test_handles_dirty_float_q_num(self):
        """源数据里有 q_num=4.1 这种脏数据，不能让它排到队尾或抛异常"""
        assert qbank._qno({"q_num": 4.1}) == 4
        assert qbank._qno({"q_num": True, "display_num": 7}) == 7
        assert qbank._qno({}) == 999

    def test_paper_questions_sorted_by_type_then_number(self):
        qs = qbank.paper_questions("2019秋本部计算机网络模拟题")
        assert [q["type"] for q in qs[:5]] == ["选择题"] * 5
        assert qs[0]["q_num"] == 1


class TestGrouping:
    @pytest.mark.parametrize("qtype,size", [
        ("选择题", 5), ("填空题", 5), ("判断题", 5), ("综合题", 1),
    ])
    def test_group_size_matches_requirement(self, qtype, size):
        """选择/填空/判断 5 题一组，综合题 1 题一组"""
        assert qbank.group_size(qtype) == size
        g = qbank.get_group("2019秋本部计算机网络模拟题", qtype, 1)
        assert len(g["questions"]) == size

    def test_group_index_walk_covers_everything(self):
        source = "2019秋本部计算机网络模拟题"
        seen = []
        idx = 1
        while True:
            g = qbank.get_group(source, group_index=idx)
            if g.get("finished"):
                break
            seen.extend(g["qids"])
            assert g["has_more"] == (idx < g["group_count"])
            idx += 1
        assert len(seen) == 40
        assert len(set(seen)) == 40

    def test_out_of_range_group_is_explicit(self):
        g = qbank.get_group("2019秋本部计算机网络模拟题", group_index=999)
        assert g["finished"] is True


class TestNoAnswerLeak:
    def test_group_payload_has_no_text_or_answer(self):
        """出题只给题号+题型+知识点+图片，绝不带题干和答案"""
        g = qbank.get_group("2019秋本部计算机网络模拟题", "选择题", 1)
        for item in g["questions"]:
            assert set(item) == {"no", "type", "topic", "image_url"}
            assert "text" not in item and "answer" not in item

    def test_image_url_is_encoded(self):
        """图片 URL 必须百分号编码（题库里有中文文件名）"""
        g = qbank.get_group("2026春操作系统期末试题（回忆版）", "综合题", 1)
        url = g["questions"][0]["image_url"]
        assert url.startswith("/qb_images/")
        assert "%" in url or url.isascii()

    @requires_images
    def test_image_file_actually_exists(self):
        """数据完整性检查：URL 指向的图片文件必须真的在

        这条需要本地图片数据（被 .gitignore 排除），缺失时自动跳过 ——
        否则新克隆/CI 上会看着像代码坏了。
        """
        from urllib.parse import unquote
        from pathlib import Path

        g = qbank.get_group("2026春操作系统期末试题（回忆版）", "综合题", 1)
        url = g["questions"][0]["image_url"]
        path = qbank.IMAGES_DIR / unquote(url.split("/qb_images/")[1])
        assert Path(path).exists(), f"图片不存在: {path}"


class TestParsing:
    def test_numbered_answers_hit_the_right_questions(self):
        """修复前：'3.C 4.D 5.A' 会被当成前 3 题的答案"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("3.C 4.D 5.A", members)
        by_no = {qbank._display_no(q): q["qid"] for q in members}
        assert set(amap) == {by_no[3], by_no[4], by_no[5]}

    def test_judge_answers_are_parsed(self):
        """修复前：判断题答案一个都匹配不上"""
        _, members = _group("2019秋本部计算机网络模拟题", "判断题", 1)
        amap, notes = qbank.parse_submission("1.对 2.错 3.对 4.错 5.对", members)
        assert len(amap) == 5
        assert list(amap.values()) == ["对", "错", "对", "错", "对"]
        assert notes == []

    def test_judge_compact_form(self):
        _, members = _group("2019秋本部计算机网络模拟题", "判断题", 1)
        amap, _ = qbank.parse_submission("对错对错对", members)
        assert list(amap.values()) == ["对", "错", "对", "错", "对"]

    def test_letters_are_not_deduplicated(self):
        """修复前 normalize_choice 去重，'ABCDA' 只剩 4 个答案"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("ABCDA", members)
        assert len(amap) == 5
        assert list(amap.values()) == ["A", "B", "C", "D", "A"]

    def test_prose_with_extra_letters_is_not_shredded(self):
        """修复前：'第1题我选C，因为A和B都不对' 被拆成 C/A/B 三题"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("第1题我选C，因为A和B都不对", members)
        assert len(amap) == 1
        assert list(amap.values()) == ["C"]

    def test_multi_select_answer_kept(self):
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("1.AC 2.B", members)
        assert list(amap.values()) == ["AC", "B"]

    def test_group_relative_numbering(self):
        """第 2 组里学生说 '1.A 2.B'，应落到组内第 1、2 题"""
        g, members = _group("2019秋本部计算机网络模拟题", "选择题", 2)
        amap, _ = qbank.parse_submission("1.A 2.B", members, {q["qid"]: i + 1 for i, q in enumerate(members)})
        assert set(amap) == {members[0]["qid"], members[1]["qid"]}

    def test_group_relative_numbering_with_section_numbers(self):
        """真实场景：第 2 组显示的是 6~10 号题，学生却常写「1.A 2.B」

        传入的题号表是按题型编号（6~10），所以要能在查不到时退回"组内第几题"。
        少了这个兜底，第 2 组往后的填空/判断题答案会整组解析失败。
        """
        paper = "2019秋本部计算机网络模拟题"
        all_qs = qbank.paper_questions(paper)
        numbers = qbank.section_numbers(all_qs)
        group2 = [q for q in all_qs if q["type"] == "选择题"][5:10]
        assert [numbers[q["qid"]] for q in group2] == [6, 7, 8, 9, 10]

        amap, _ = qbank.parse_submission("1.A 2.B", group2, numbers)
        assert set(amap) == {group2[0]["qid"], group2[1]["qid"]}
        # 按真实题号写也要认
        amap2, _ = qbank.parse_submission("6.A 7.B", group2, numbers)
        assert set(amap2) == {group2[0]["qid"], group2[1]["qid"]}

    def test_group_relative_numbering_for_judge_group(self):
        """判断题（非选择题路径）也要能用「组内第几题」作答；这里回归的是非选择题路径"""
        paper = "2019秋本部计算机网络模拟题"
        all_qs = qbank.paper_questions(paper)
        numbers = qbank.section_numbers(all_qs)
        judge = [q for q in all_qs if q["type"] == "判断题"]
        if len(judge) < 2:
            pytest.skip("该卷判断题不足两题")
        amap, _ = qbank.parse_submission("1.对 2.错", judge[:5], numbers)
        assert len(amap) == 2, "非选择题组也要能用组内序号作答"

    # ---- 用户实测反馈的格式（"写得太死了"）----

    @pytest.mark.parametrize("text,expected", [
        ("AAAAA", ["A"] * 5),
        ("aaaaa", ["A"] * 5),
        ("A A A A A", ["A"] * 5),
        ("A,A,A,A,A", ["A"] * 5),
        ("AAAAAAAAAAAA", ["A"] * 5),        # 多打了几个 A
        ("全都选A", ["A"] * 5),              # 自然说法
        ("全选B", ["B"] * 5),
        ("我选AAAAA", ["A"] * 5),            # 带中文
        ("1A2A3A4A5A", ["A"] * 5),
        ("1.A 2.A 3.A 4.A 5.A", ["A"] * 5),
        ("1.D2.B3.D4.A5.C", ["D", "B", "D", "A", "C"]),   # 连写没空格
        ("D B D A C", ["D", "B", "D", "A", "C"]),
        ("ABCDA", ["A", "B", "C", "D", "A"]),             # 每个字母不同，按位解析
    ])
    def test_natural_answer_formats_are_recognised(self, text, expected):
        """这些都是学生会真的打出来的写法，必须都能识别"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, notes = qbank.parse_submission(text, members)
        assert [amap.get(q["qid"]) for q in members] == expected, f"{text!r} 解析结果不对: {notes}"

    def test_chinese_question_numbers(self):
        """「第一题A第二题B」这种写法也要认"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("第一题D第二题B第三题D", members)
        assert [amap.get(q["qid"]) for q in members[:3]] == ["D", "B", "D"]
        assert len(amap) == 3, "只答了三题就只解析三题"

    def test_partial_same_letter_is_not_expanded_to_all(self):
        """只答了 3 个 A 时不能当成"整组都选 A"

        长度门槛就是为这个设的：给够题目数量的重复字母才采信。
        """
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("AAA", members)
        assert len(amap) == 3

    def test_prose_with_letters_still_safe(self):
        """放宽格式后，"因为A和B都不对"仍不能被拆成多个答案"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("第1题我选C，因为A和B都不对", members)
        assert list(amap.values()) == ["C"]

    # ---- E 选项（用户实测："一题答案给了 E，它没识别出来"）----

    @pytest.mark.parametrize("text,expected", [
        ("1.A 2.B 3.C 4.D 5.E", ["A", "B", "C", "D", "E"]),
        ("ABCDE", ["A", "B", "C", "D", "E"]),
        ("A B C D E", ["A", "B", "C", "D", "E"]),
        ("全都选E", ["E"] * 5),
        ("1.E", ["E", None, None, None, None]),
        ("1.A 2.E 3.C", ["A", "E", "C", None, None]),
    ])
    def test_fifth_option_E_is_recognised(self, text, expected):
        """选项字母原来只认 A-D，E 会被直接丢掉

        丢掉比"认不出"更危险：按位解析时 E 一丢，后面所有答案都会错位。
        """
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission(text, members)
        got = [amap.get(q["qid"]) for q in members]
        assert got == expected, f"{text!r} 解析结果不对: {got}"

    def test_english_prose_is_not_treated_as_option_string(self):
        """放宽到 A-H 后必须挡住英文句子

        "the answer is B" 里的 a/n/s/w/e/r 都会被当成选项字母，
        若不加形状校验就会整串错位。
        """
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        for text in ["the answer is B", "I think it is A", "maybe C or D"]:
            amap, _ = qbank.parse_submission(text, members)
            # 只能落到兜底的"整段给第 1 题"，绝不能按位拆成多个答案
            assert len(amap) == 1, f"{text!r} 被误拆成 {amap}"

    def test_too_many_letters_gives_up_instead_of_misaligning(self):
        """「ABCDEFGHIJ」（10 个字母 / 5 道题）不能退化成"整段给第 1 题"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, notes = qbank.parse_submission("ABCDEFGHIJ", members)
        assert amap == {}, "解析不出来就如实返回空，不能凭空当成某一道题的答案"
        assert notes, "要给出说明"

    def test_unparseable_answer_string_does_not_become_question_one(self):
        """只含字母数字的短标记解析不了时，也不能塞给第 1 题"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        for text in ["ZZZZZ", "ABCDEFGHIJ", "1234567890"]:
            amap, _ = qbank.parse_submission(text, members)
            assert amap == {}, f"{text!r} 被硬塞成了答案: {amap}"

    def test_question_whose_answer_is_E_can_be_graded(self):
        """答案字段是 E 的题以前判不了（"选项答案缺失或不是字母"）"""
        q = {"qid": "X", "type": "选择题", "answer": "E", "topic": "t", "text": "..."}
        right = qbank.grade_answer(q, "E", no=1)
        wrong = qbank.grade_answer(q, "A", no=1)
        assert right["correct"] is True
        assert wrong["correct"] is False
        assert wrong["correct_answer"] == "E"

    def test_out_of_range_option_is_judged_wrong_not_unjudgeable(self):
        """四选项题答 E 要**正常判错**，不能变成"判不了/需人工核对"

        用户实测：那道题只有四个选项，E 是乱答的，期望就是判错。
        关键区别在 correct 的取值：
          False → 判错，会告诉学生正确答案（期望）
          None  → 判不了，学生会看到"需要人工核对"（不期望）
        以前范围写死 A-D 时，E 根本提取不出来，结果连"判错"都做不到。
        """
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        for q in members:
            r = qbank.grade_answer(q, "E", no=1)
            assert r["correct"] is False, f"{q['qid']} 答 E 应判错，实际 {r['correct']}"
            assert r["correct_answer"] == q["answer"], "要给出正确答案供讲解"
            assert "正确答案是" in r["explanation"]

    @pytest.mark.parametrize("text", ["1.E 2.E 3.E 4.E 5.E", "EEEEE", "全都选E"])
    def test_out_of_range_option_in_bulk_is_graded_not_skipped(self, text):
        """整组乱答 E：5 题都要被批改，一题都不能"没认出"而跳过"""
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission(text, members)
        assert len(amap) == 5, f"{text!r} 只解析出 {len(amap)} 个答案"
        for q in members:
            r = qbank.grade_answer(q, amap[q["qid"]], no=1)
            assert r["correct"] is False

    def test_multi_select_still_works_with_wider_range(self):
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        amap, _ = qbank.parse_submission("1.BE 2.A", members)
        assert list(amap.values()) == ["BE", "A"]

    def test_whole_paper_answers_land_on_the_right_type(self):
        """整卷批改时，一份选择题答案必须落到选择题上，不能落到综合题上

        真实踩过：2019 那份计网模拟题里 display_num=1 同时属于
        选择题/填空题/判断题/综合题四种题型，按 display_num 查表会把「1.D」
        判到综合题，并把综合题的参考答案整段泄给学生。
        """
        src = "2019秋本部计算机网络模拟题"
        qs = qbank.paper_questions(src)
        amap, _ = qbank.parse_against_paper("1.D 2.B 3.D 4.A 5.C", qs)
        types = {q["qid"]: q["type"] for q in qs}
        assert len(amap) == 5
        assert all(types[qid] == "选择题" for qid in amap), \
            f"落到了非选择题上: {[(types[q], a) for q, a in amap.items()]}"

    def test_single_letter_rejected_for_non_choice(self):
        """非选择题不接受单个选项字母（那是定位错了的信号）"""
        _, members = _group("2019秋本部计算机网络模拟题", "填空题", 1)
        amap, notes = qbank.parse_submission("1.A 2.B", members)
        assert amap == {}
        assert any("不是选择题" in (n.get("warning") or "") for n in notes)


class TestGrading:
    def test_choice_grading(self):
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        q = members[0]
        right = qbank.grade_answer(q, q["answer"])
        wrong = qbank.grade_answer(q, "Z" if q["answer"] != "Z" else "A")
        assert right["correct"] is True
        assert wrong["correct"] is False

    def test_wrong_answer_carries_material_for_explanation(self):
        """答错时必须带上参考要点

        只回一句"正确答案是 B"学生学不到东西；而模型手里没有依据就没法讲，
        只能空泛地鼓励他再试试 —— 这正是用户反馈的问题。
        """
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        q = members[0]
        wrong_letter = next(c for c in "ABCD" if c != (q["answer"] or "").upper())
        r = qbank.grade_answer(q, wrong_letter)
        assert r["correct"] is False
        assert r.get("reference_points"), "错题应带参考要点，供模型讲解"
        assert all(len(p) <= 240 for p in r["reference_points"])

    def test_correct_answer_does_not_carry_clutter(self):
        _, members = _group("2019秋本部计算机网络模拟题", "选择题", 1)
        q = members[0]
        r = qbank.grade_answer(q, q["answer"])
        assert r["correct"] is True
        assert "reference_points" not in r

    def test_judge_grading_normalizes_variants(self):
        _, members = _group("2019秋本部计算机网络模拟题", "判断题", 1)
        q = members[0]
        variants = {"正确": "对", "√": "对", "T": "对", "错": "错", "×": "错", "F": "错"}
        expected = qbank.normalize_judge(q["answer"])
        for text, _ in variants.items():
            r = qbank.grade_answer(q, text)
            if qbank.normalize_judge(text) is not None:
                assert r["correct"] is (qbank.normalize_judge(text) == expected)

    def test_fill_in_blank_misassigned_answer_is_flagged(self):
        """真题里填空题 answer 被抽成了选项字母 C，不能拿去和学生答案硬比"""
        _, members = _group("2019秋本部计算机网络模拟题", "填空题", 1)
        q = members[0]
        basis = qbank.grading_basis(q)
        assert basis["misassigned"] is True
        r = qbank.grade_answer(q, "（1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1）")
        assert r.get("needs_llm") is True
        assert r["correct_answer"] != "C"
        assert "错位" in r.get("basis_warning", "")
        # 喂给阅卷模型的依据里不能出现那个错位的字母
        assert qbank.build_llm_basis(q)["answer"] == ""
        assert qbank.build_llm_basis(q)["answer_unreliable"] is True

    def test_comprehensive_question_uses_sub_answers_and_reference_points(self):
        """综合题 answer 为 null，必须回退到 sub_questions / reference_points"""
        qs = qbank.paper_questions("2026春操作系统期末试题（回忆版）")
        q = next(x for x in qs if x["type"] == "综合题")
        assert not q.get("answer")
        basis = qbank.grading_basis(q)
        assert basis["usable"] is True
        assert basis["sub_answers"] or basis["reference_points"]
        r = qbank.grade_answer(q, "我的作答内容")
        assert r.get("needs_llm") is True

    def test_every_question_in_bank_is_gradable(self):
        """每道题都至少要有一种批改依据"""
        bad = [q["qid"] for q in qbank.load_questions() if not qbank.grading_basis(q)["usable"]]
        assert bad == [], f"这些题缺批改依据: {bad[:10]}"


class TestPaperResolution:
    def test_fuzzy_match_missing_suffix(self):
        p, _ = qbank.resolve_paper("2026春操作系统期末试题")
        assert p and p["source"] == "2026春操作系统期末试题（回忆版）"

    def test_fuzzy_match_short_alias(self):
        p, _ = qbank.resolve_paper("2019网模拟")
        assert p and "2019秋本部计算机网络模拟题" == p["source"]

    def test_fabricated_paper_name_is_rejected(self):
        """模型爱编卷名。编出来的年份对不上任何卷子时必须拒绝，
        而不是模糊命中最像的那份，把学生带到他没选的卷子上。"""
        p, cands = qbank.resolve_paper("2026春计算机网络模拟题")
        assert p is None
        assert cands, "拒绝时要给出候选试卷名让学生确认"

    def test_unknown_name_returns_candidates(self):
        p, cands = qbank.resolve_paper("完全不存在的卷子")
        assert p is None
        assert cands == [] or all(isinstance(c, str) for c in cands)

    def test_outline_reports_group_counts(self):
        out = qbank.paper_outline("2019秋本部计算机网络模拟题")
        by = {s["type"]: s for s in out["sections"]}
        assert by["选择题"]["group_count"] == 5
        assert by["综合题"]["group_size"] == 1
        assert by["综合题"]["group_count"] == 4
