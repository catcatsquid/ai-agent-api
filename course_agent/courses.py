"""课程注册表 —— 课程清单的唯一真相源

## 为什么真相源是磁盘

课程的学习资料天然是 `data/<课程>/` 下的目录（`往年题` / `课本` / `ppt`）。
所以"新增一门课"最省力的定义就是 `mkdir`：一旦目录存在，检索（`course_sources()`
按目录反查文件名）、题库（按 `course` 字段索引）、进度账本、Agent 的课程识别
全部自动跟上，不需要改配置、不需要重启。

反过来，如果把课程清单放在 `config.COURSES` 里，就会出现"目录建了、配置忘了改"
的不一致 —— 而那种不一致不会报错，只会让新课程的资料入不了库。

## 保留目录

`data/` 下不全是课程：题库 JSON 在 `data/question_bank/`，演示数据集在
`demo/seed/data/`。这些名字列在 `RESERVED` 里，新增课程时会拒绝。

## 用户可见的规则

Web 上**不预先**罗列允许/禁止什么（那是内部约定，不是给用户的说明书）。
用户万一踩线，抛 `CourseNameError`，接口层转 4xx，前端弹窗显示原因。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#: 新增课程时自动建好的三个子目录（与现有两门课保持一致）
COURSE_SUBDIRS = ("往年题", "课本", "ppt")

#: `data/` 下不是课程的目录
RESERVED = {"question_bank", "images", "runtime", "static", "demo", "seed", "logs"}

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_LEN = 32

#: Windows 的设备名。同名目录**能建出来，却读不出、删不掉**（实测 `mkdir data\CON`
#: 成功，之后连 ls 都列不出来，只能 `cmd /c rmdir` 收尾），所以在这里直接拒绝。
#: 带扩展名的写法同样被系统占用，所以只比对第一个点之前的部分。
_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL",
                 *(f"COM{i}" for i in range(1, 10)),
                 *(f"LPT{i}" for i in range(1, 10))}

_cached: list[str] | None = None


class CourseNameError(ValueError):
    """课程名不合法（给用户看的那种错误）"""


def _is_course_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.startswith((".", "_")) and p.name not in RESERVED


def known_courses(refresh: bool = False) -> list[str]:
    """当前所有课程名（按目录名排序）

    缓存到进程内；`refresh=True` 或调用 `invalidate_cache()` 时重扫。
    新增课程后必须让缓存失效，否则 Agent 这一轮还看不到它。
    """
    global _cached
    if _cached is not None and not refresh:
        return list(_cached)
    if not DATA_DIR.is_dir():
        _cached = []
        return []
    _cached = sorted(p.name for p in DATA_DIR.iterdir() if _is_course_dir(p))
    return list(_cached)


def invalidate_cache() -> None:
    global _cached
    _cached = None


def is_known(name: str) -> bool:
    return (name or "").strip() in known_courses()


def course_dir(name: str) -> Path:
    return DATA_DIR / name


def validate_name(name: str) -> str:
    """校验课程名并返回规整后的名字；不合法抛 CourseNameError"""
    raw = (name or "").strip()
    if not raw:
        raise CourseNameError("课程名不能为空")
    if len(raw) > _MAX_LEN:
        raise CourseNameError(f"课程名最多 {_MAX_LEN} 个字")
    if _ILLEGAL.search(raw):
        raise CourseNameError('课程名不能包含 \\ / : * ? " < > | 这些字符')
    if raw in (".", "..") or raw.startswith("."):
        raise CourseNameError("课程名不能以点开头")
    if raw in RESERVED or raw.lower() in {r.lower() for r in RESERVED}:
        raise CourseNameError(f"「{raw}」是系统保留名，换一个")
    if raw.split(".")[0].upper() in _DEVICE_NAMES:
        raise CourseNameError(f"「{raw}」是 Windows 保留设备名，换一个")
    return raw


def create_course(name: str) -> dict:
    """建一门课：`data/<课程>/{往年题,课本,ppt}`

    Returns:
        {"course": 名称, "dir": 目录, "created_subdirs": [...]}

    Raises:
        CourseNameError: 名字不合法 / 与保留名或已有课程冲突
    """
    course = validate_name(name)
    root = course_dir(course)
    if root.exists():
        raise CourseNameError(f"「{course}」已经存在")

    made: list[str] = []
    try:
        for sub in COURSE_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
            made.append(sub)
    except OSError as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise CourseNameError(f"建目录失败：{exc}") from exc

    invalidate_cache()
    # 只回相对位置：绝对路径会把本机的用户名/盘符带给前端，没有任何用处
    return {"course": course, "dir": f"{DATA_DIR.name}/{course}",
            "created_subdirs": made}