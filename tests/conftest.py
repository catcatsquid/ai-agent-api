"""pytest 共享配置

除了把项目根目录加进 sys.path，这里还负责一件**容易被忽略的事**：
告诉测试哪些"数据集"是可选的。

背景：这个仓库的 `data/` 里只有题库 JSON 是文本、可跟踪；
课程课件/课本/图片/向量库加起来 300 MB，是刻意 gitignore 的本地数据。
于是"别人 clone 下来跑测试"和"CI"都会缺这些数据 ——
以前的做法是直接报红，看起来像代码坏了，其实是数据没带。

所以这里统一检测并**跳过**依赖缺失数据的用例（skip，不是 xfail），
并把原因写清楚。这样：
  · 本地（数据齐全）跑满全部用例，一项不跳；
  · CI / 新克隆（只有 JSON）跳过依赖本地数据的用例，其余照跑。
（刻意不写测试条数：写死的数字必然过期，要看数量跑一次 pytest 就有。）
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# 确保项目根目录在 path 上
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QB_DIR = ROOT / "data" / "question_bank"
IMAGES_DIR = QB_DIR / "images"
VECTOR_DIR = ROOT / "vector_store"

# 学习/复习模式的"分节结构"只来自这两门课，与 quiz_service.section_index 的默认课程一致。
COURSES = ("操作系统", "计算机网络")
# 会被 quiz_service.course_sources() 认作课程资料的后缀（保持一致，别各写一套）。
SOURCE_SUFFIXES = (".pdf", ".docx", ".md", ".txt", ".pptx")


def _has_question_bank() -> bool:
    return bool(list(QB_DIR.glob("questions_*.json")))


def _has_images() -> bool:
    return IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.png"))


def _has_vector_store() -> bool:
    """向量库里是否**真的有已入库的文档**

    只看 chroma.sqlite3 在不在是不够的：`get_knowledge_base()` 一旦被调用
    就会自动建出一个空库（约 190 KB），于是"有没有向量库"永远为真 ——
    实测 .ci-sim 里就跑出过这种空库，会让本该跳过的用例转成"运行然后失败"。
    所以直接数 embeddings 表的行数（真实库 4199 行，空库 0 行）；
    万一换 Chroma 版本后读不出这张表，退回按体积判断（真实库几十 MB）。
    """
    db = VECTOR_DIR / "chroma.sqlite3"
    if not db.exists():
        return False
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] > 0
        finally:
            con.close()
    except Exception:
        return any(p.is_file() and p.stat().st_size > 1_000_000
                   for p in VECTOR_DIR.rglob("*"))


def _course_source_files(course: str) -> list:
    """data/<课程>/ 下会被 course_sources() 收进来的文件"""
    root = ROOT / "data" / course
    if not root.exists():
        return []
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES]


def _has_course_files() -> bool:
    """有没有任何课程资料文件（课本/课件/笔记，任一即可）"""
    return any(_course_source_files(c) for c in COURSES)


def _has_course_ppt() -> bool:
    """课件是否"可用"：磁盘上有课件**且**它已入库

    section_index() 的 ppt 列表是「磁盘文件名 × 向量库文档」的**交集**：
      · 文件名来自 course_sources()（扫 data/<课程>/）；
      · 是否算课件看 doc_type，而 doc_type 由路径里有没有 "ppt" 决定。
    两者缺一 ppts 都是空，read_section 就回「没有可用课件」，
    依赖真实课件名的用例必然失败 —— 这正是 CI 变红的原因。

    这里刻意复用 doc_utils._classify_doc，而不是照抄一遍规则：
    入库和"该不该跳过"必须同一套判据，否则就会重演
    "两条导入路径对同一个文件给出不同 doc_type"那个 bug（见 76acf8f）。
    """
    from doc_utils import _classify_doc

    for course in COURSES:
        for p in _course_source_files(course):
            if _classify_doc(p, ROOT / "data") == "ppt":
                return True
    return False


AVAILABLE = {
    "question_bank": _has_question_bank(),
    "images": _has_images(),
    "vector_store": _has_vector_store(),
    "course_files": _has_course_files(),
    # 课件文件与向量库缺一不可，所以这里把 vector_store 一起并进来
    "course_ppt": _has_course_ppt() and _has_vector_store(),
}

_REASONS = {
    "question_bank": "缺少题库数据（data/question_bank/questions_*.json）—— 它应该被提交进仓库",
    "images": "缺少题图（data/question_bank/images/）—— 它被 .gitignore 排除，属于本地数据",
    "vector_store": "缺少向量库（vector_store/）—— 属于本地数据，需先跑 build_knowledge.py",
    "course_files": "缺少课程资料（data/<课程>/ 下的课本/课件/笔记）—— 属于本地数据，未提交进仓库",
    "course_ppt": "缺少课件或向量库 —— 分节结构依赖「磁盘课件 × 已入库文档」的交集，需本地跑 build_knowledge.py",
}

# 哪些测试依赖哪份数据。key 支持三种粒度：
#     "test_x.py"                    → 整个文件
#     "test_x.py::TestClass"         → 整个类
#     "test_x.py::TestClass::test_y" → 单个用例
# 匹配是**逐层取并集**：文件级说"要题库"，某个类再说"还要课件"，
# 两者都会生效。这样就不必为了少数几条用例把整个文件跳过。
DATA_NEEDS = {
    # ---- 题库 JSON（已提交，任何克隆都有）----
    "test_qbank.py": ("question_bank",),
    "test_qa_mode_tools.py": ("question_bank",),
    "test_qb_single_source.py": ("question_bank",),
    "test_auth.py": ("question_bank",),
    "test_server_security.py": ("question_bank",),
    "test_tutor_agent.py": ("question_bank",),
    "test_quiz_service.py": ("question_bank",),
    "test_drill_cursor.py": ("question_bank",),

    # ---- 只需要"课程资料文件"存在（这些用例把 KB 换成了假对象）----
    # course_sources() 扫不到文件时，search_knowledge 就不带 sources 过滤，
    # 于是"按课程检索不能串味"那条断言必然失败。
    "test_tutor_agent.py::TestMissedQuestionsRecovery::test_knowledge_search_filters_by_course":
        ("course_files",),
    "test_tutor_agent.py::TestMissedQuestionsRecovery::test_knowledge_search_falls_back_when_topic_is_not_in_that_course":
        ("course_files",),
    "test_tutor_agent.py::TestCustomQuizIntent::test_topic_inside_current_course_keeps_the_label":
        ("course_files",),

    # ---- 需要「真课件 + 向量库」（section_index 的分节结构）----
    # 这些用例的断言里带着真实课件名（05-死锁.pdf、Chapter5-Transport Layer.pdf …），
    # 没有课件数据就无法成立。不能改成假数据绕过去 ——
    # 它们本来就是照着"用户实测踩到的那份课件"写的回归测试，
    # 换成编出来的文件名，测的就不是当初那个 bug 了。
    "test_study_mode.py::TestChapterLookup": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_last_page_has_no_more": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_limit_is_capped": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_pagination": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_reads_in_chunk_order": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_short_name_matches": ("course_ppt",),
    "test_study_mode.py::TestReadSection::test_tool_ships_the_teaching_contract": ("course_ppt",),
    "test_study_mode.py::TestSectionMap::test_carries_chapter_and_size": ("course_ppt",),
    "test_study_mode.py::TestSectionMap::test_covers_both_courses": ("course_ppt",),
    "test_study_mode.py::TestSectionMap::test_lists_real_ppts_in_course_order": ("course_ppt",),
    "test_study_mode.py::TestSectionMap::test_notes_and_textbook_are_secondary": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_args_tolerate_messy_strings": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_args_tolerate_strings": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_course_plus_section_needs_no_filename": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_fabricated_filename_falls_back_to_ppt": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_negative_offset_is_clamped": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_out_of_range_section_falls_back_to_first": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_reading_notes_is_auto_switched_to_ppt": ("course_ppt",),
    "test_study_mode.py::TestSectionSelectionIsStructural::test_section_index_walks_the_course": ("course_ppt",),
    "test_study_mode.py::TestScopedRetrieval::test_relevance_is_displayed_from_the_right_field": ("course_ppt",),
    "test_study_mode.py::TestScopedRetrieval::test_scoped_to_course": ("course_ppt",),
    "test_study_mode.py::TestScopedRetrieval::test_scoped_to_one_file": ("course_ppt",),
    "test_study_mode.py::TestStudyBookmark::test_chapter_in_course_field_is_corrected": ("course_ppt",),
    "test_study_mode.py::TestStudyBookmark::test_last_page_records_no_next_offset": ("course_ppt",),
    "test_study_mode.py::TestStudyBookmark::test_read_position_alone_reaches_the_planner": ("course_ppt",),
    "test_study_mode.py::TestStudyBookmark::test_read_position_is_remembered_automatically": ("course_ppt",),
    "test_surface_parity.py::TestSharedImplementation::test_tutor_and_bot_agree_on_section_index": ("course_ppt",),
    # 下面三条随测试一起搬到了 test_tutor_reference.py（拆分：让文件名与内容对上）
    "test_tutor_reference.py::TestTutorStudyFlow::test_tutor_corrects_chapter_in_course_field": ("course_ppt",),
    "test_tutor_reference.py::TestTutorStudyFlow::test_tutor_read_section_records_position_automatically": ("course_ppt",),
    "test_tutor_reference.py::TestTutorStudyFlow::test_tutor_read_section_returns_same_content_shape": ("course_ppt",),
    # 下面四条留在这个文件，但类名从 TestStudentSessionFindings 换成了 TestSharedCoreBehaviour
    "test_surface_parity.py::TestSharedCoreBehaviour::test_chapter_lookup_beats_section": ("course_ppt",),
    "test_surface_parity.py::TestSharedCoreBehaviour::test_chinese_chapter_maps_to_english_ppt": ("course_ppt",),
    "test_surface_parity.py::TestSharedCoreBehaviour::test_section_means_chapter_number_for_chapter_named_courses": ("course_ppt",),
    "test_surface_parity.py::TestSharedCoreBehaviour::test_section_still_means_position_when_no_chapter_numbering": ("course_ppt",),
}

# 供个别用例显式使用（例如"每道题的图片必须存在"这种数据完整性断言）
requires_images = pytest.mark.skipif(
    not AVAILABLE["images"], reason=_REASONS["images"])
requires_question_bank = pytest.mark.skipif(
    not AVAILABLE["question_bank"], reason=_REASONS["question_bank"])


def _needs_for(nodeid: str) -> list:
    """把 nodeid 归一化成 "文件名::类::用例"，再逐层取并集

    归一化而不是直接用 nodeid：nodeid 的前缀随调用方式变
    （`pytest tests/` 是 tests/xxx.py，`cd tests && pytest` 是 xxx.py），
    只保留文件名 + :: 之后的部分就与调用位置无关了。
    """
    head, *rest = nodeid.split("::")
    rel = "::".join([Path(head).name] + rest)
    needs: list = []
    for key, datasets in DATA_NEEDS.items():
        if rel == key or rel.startswith(key + "::"):
            needs.extend(d for d in datasets if d not in needs)
    return needs


def pytest_collection_modifyitems(config, items):
    """按 AVAILABLE 跳过依赖缺失数据的用例，而不是让它们报红"""
    for item in items:
        missing = [n for n in _needs_for(item.nodeid) if not AVAILABLE[n]]
        if missing:
            item.add_marker(pytest.mark.skip(
                reason="；".join(_REASONS[n] for n in missing)))


def pytest_report_header(config):
    """在测试开头声明哪些数据可用 —— 免得有人看到 skip 就以为测试坏了"""
    parts = [f"{k}={'有' if v else '无'}" for k, v in AVAILABLE.items()]
    return "数据集: " + ", ".join(parts)


# ============================================================
# 守卫：整轮测试不许往**真实**的会话库里写
# ============================================================

_REAL_CKPT_FILES = ("checkpoints_plan.sqlite3", "checkpoints_react.sqlite3")


def _real_thread_ids() -> dict:
    """真实 runtime/checkpoints_*.sqlite3 里现存的 thread 集合（只读）"""
    import sqlite3

    out = {}
    for name in _REAL_CKPT_FILES:
        path = ROOT / "runtime" / name
        ids: set = set()
        if path.exists():
            try:
                # 只读模式：守卫本身绝不能改动被观察的东西
                con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
                try:
                    ids = {r[0] for r in con.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints")}
                except sqlite3.OperationalError:
                    ids = set()          # 表还没建（= 从没写过）
                finally:
                    con.close()
            except sqlite3.Error:
                ids = set()
        out[name] = ids
    return out


@pytest.fixture(scope="session", autouse=True)
def _real_conversation_store_stays_clean():
    """跑完测试后，真实的会话库里不能多出任何 thread

    为什么需要：1.2.0 把 checkpointer 换成了落盘 SQLite。在那之前，测试里对
    模块级 graph / react_graph 的 `update_state` 只写进程内存、重启就没了；
    现在它们会**持久化**成真实会话 —— 用户会在会话列表里看到
    `isolation-test-plan-write` 这种垃圾（实测发生过）。

    这条守卫让"下次再有人这么写"立刻暴露，而不是等用户发现侧栏里多了几条怪会话。
    写会话的测试应当注入临时库 saver（见 test_graph_isolation.py 的 tmp_graphs）。
    """
    before = _real_thread_ids()
    yield
    after = _real_thread_ids()
    added = {name: sorted(after[name] - before[name])
             for name in before if after[name] - before[name]}
    assert not added, (
        "测试往**真实的** runtime/checkpoints_*.sqlite3 里写了会话，"
        f"用户会在会话列表里看到这些垃圾：{added}。"
        "请给写入型测试注入临时库 saver（别用模块级的 graph / react_graph 写状态）。")


def _real_data_files() -> set:
    """真实 data/ 下现存的文件集合（只读；跳过测试自己用的临时目录）"""
    data = ROOT / "data"
    if not data.exists():
        return set()
    return {p.relative_to(data).as_posix()
            for p in data.rglob("*")
            if p.is_file() and "_tmp_" not in p.as_posix()}


@pytest.fixture(scope="session", autouse=True)
def _real_data_dir_stays_clean():
    """跑完测试后，真实的 data/ 下不能多出（或少掉）任何文件

    为什么需要：`data/` 是**用户的课程资料与题库**，不是测试的草稿纸。
    而 1.7.0 起 `/kb/add` 会把 data/ 之外的文件**复制进** data/ —— 于是
    "拿 README.md 这类真实文件去测越界导入"会真的往用户数据目录里塞文件并入库。

    这条守卫把"测试碰了真实数据"变成一次失败，而不是等用户发现资料里多了个怪文件。
    写文件的测试请把 `server.DATA_ROOT` 指到工作区内的临时目录。
    """
    before = _real_data_files()
    yield
    after = _real_data_files()
    added = sorted(after - before)
    removed = sorted(before - after)
    assert not added and not removed, (
        f"测试改动了**真实的** data/ 目录：新增 {added}，减少 {removed}。"
        "请把 server.DATA_ROOT 指到临时目录，别往用户的课程资料目录里写。")


def _local_server_is_running() -> bool:
    """本机是不是我们那个服务在跑（用 /health 的特征字段认，避免误判别的程序）"""
    import json
    import urllib.request

    for url in ("http://127.0.0.1:8003/health", "http://127.0.0.1:8000/health"):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("status") == "healthy" and "knowledge_chunks" in body:
                return True
        except Exception:
            continue
    return False


@pytest.fixture(scope="session", autouse=True)
def _no_server_running_during_tests():
    """跑测试时**不能**有服务在跑 —— Chroma 是单写者，并发写会把知识库写坏

    代价是实测出来的：2026-10-16 我在服务运行着的时候跑了一遍全量测试，
    两个进程同时写 `vector_store/`，把元数据段写坏 —— 之后**任何**读操作都抛
    `Failed to apply logs to the metadata segment`，最后只能整库重建
    （4199 块重新 embedding）。更阴的是：清掉写前日志后"读"恢复了，
    而"写"变成**静默丢弃**（upsert 不报错但什么都不落盘）。

    想同时跑就先把服务停掉；确实需要并存时设 `ALLOW_TESTS_WITH_SERVER=1`。
    """
    if os.environ.get("ALLOW_TESTS_WITH_SERVER") == "1":
        return
    if _local_server_is_running():
        pytest.exit(
            "检测到本地服务正在运行（8003/8000 的 /health 有响应）。\n"
            "Chroma 是单写者：服务与测试同时写 vector_store/ 会把知识库写坏"
            "（实测踩过，代价是整库重建）。\n"
            "请先停掉服务再跑测试；确需并存请设 ALLOW_TESTS_WITH_SERVER=1。",
            returncode=1)
