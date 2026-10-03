"""pytest 共享配置

除了把项目根目录加进 sys.path，这里还负责一件**容易被忽略的事**：
告诉测试哪些"数据集"是可选的。

背景：这个仓库的 `data/` 里只有题库 JSON 是文本、可跟踪；
课本/PPT/图片/向量库加起来 300 MB，是刻意 gitignore 的本地数据。
于是"别人 clone 下来跑测试"和"CI"都会缺这些数据 ——
以前的做法是直接报红，看起来像代码坏了，其实是数据没带。

所以这里统一检测并**跳过**依赖缺失数据的用例（skip，不是 xfail），
并把原因写清楚。这样：
  · 本地（数据齐全）跑满 404 项；
  · CI / 新克隆（只有 JSON）跳过图片相关的少数几项，其余照跑。
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


def _has_question_bank() -> bool:
    return bool(list(QB_DIR.glob("questions_*.json")))


def _has_images() -> bool:
    return IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.png"))


def _has_vector_store() -> bool:
    return VECTOR_DIR.exists() and any(
        p.name == "chroma.sqlite3" for p in VECTOR_DIR.rglob("*"))


AVAILABLE = {
    "question_bank": _has_question_bank(),
    "images": _has_images(),
    "vector_store": _has_vector_store(),
}

_REASONS = {
    "question_bank": "缺少题库数据（data/question_bank/questions_*.json）—— 它应该被提交进仓库",
    "images": "缺少题图（data/question_bank/images/）—— 它被 .gitignore 排除，属于本地数据",
    "vector_store": "缺少向量库（vector_store/）—— 属于本地数据，需先跑 build_knowledge.py",
}

# 哪些测试文件依赖哪份数据。集中写在这里，避免每个文件各写一遍 skipif。
DATA_NEEDS = {
    "test_qbank.py": ("question_bank",),
    "test_qa_mode_tools.py": ("question_bank",),
    "test_qb_single_source.py": ("question_bank",),
    "test_auth.py": ("question_bank",),
    "test_server_security.py": ("question_bank",),
    "test_tutor_agent.py": ("question_bank",),
    "test_quiz_service.py": ("question_bank",),
    "test_drill_cursor.py": ("question_bank",),
}

# 供个别用例显式使用（例如"每道题的图片必须存在"这种数据完整性断言）
requires_images = pytest.mark.skipif(
    not AVAILABLE["images"], reason=_REASONS["images"])
requires_question_bank = pytest.mark.skipif(
    not AVAILABLE["question_bank"], reason=_REASONS["question_bank"])


def pytest_collection_modifyitems(config, items):
    """按 AVAILABLE 跳过依赖缺失数据的用例，而不是让它们报红"""
    for item in items:
        fname = Path(str(item.fspath)).name
        for need in DATA_NEEDS.get(fname, ()):
            if not AVAILABLE[need]:
                item.add_marker(pytest.mark.skip(reason=_REASONS[need]))


def pytest_report_header(config):
    """在测试开头声明哪些数据可用 —— 免得有人看到 skip 就以为测试坏了"""
    parts = [f"{k}={'有' if v else '无'}" for k, v in AVAILABLE.items()]
    return "数据集: " + ", ".join(parts)
