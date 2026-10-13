"""锁文件与声明文件必须一致，且当前环境必须**就是**锁文件描述的那套

背景：`requirements.txt` 是人改的**直接依赖**声明；真正决定装什么的是
`requirements.lock`（`uv pip compile --universal` 生成的完整锁定：含所有传递依赖、
版本精确、并带环境标记）。两份文件一旦不同步，"可复现"就是假的 ——
本地与 CI 会各装一套，而 CI 会在你没改代码的情况下变红。

三条断言各守一件事：
  ① 声明里写死版本的直接依赖，锁文件里必须是同一个版本（忘了重新生成 → 红）
  ② 当前环境装着的版本，必须与锁文件里**对本平台生效**的条目一致
     （CI 若退回从 requirements.txt 装、或有人手工升了包 → 红）
  ③ 锁文件必须带环境标记 —— 否则它只是"某一台机器"的快照（用 pip freeze 生成 → 红）
"""
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parent.parent
DECLARED = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock"

REGENERATE = ("uv pip compile --universal --python-version 3.12 "
              "requirements.txt -o requirements.lock")

# 形如 name==1.2.3 或 name==1.2.3 ; sys_platform == "win32"（行尾注释已去掉）
_ENTRY_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(?P<ver>[^\s;]+)"
    r"\s*(?:;\s*(?P<marker>.+?))?\s*$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def parse_requirements(path: Path) -> dict:
    """解析 requirements 格式 → {规范化包名: (版本, 标记或 None)}"""
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _ENTRY_RE.match(line)
        if m:
            out[_normalize(m.group("name"))] = (m.group("ver"), m.group("marker"))
    return out


def lock_entries() -> dict:
    assert LOCK.exists(), (
        f"requirements.lock 不存在 —— 请生成：\n  {REGENERATE}")
    return parse_requirements(LOCK)


def applies_here(marker) -> bool:
    """这个条目在当前平台上是否生效（求值环境标记）"""
    if not marker:
        return True
    try:
        return bool(Marker(marker).evaluate())
    except Exception:
        return True     # 标记看不懂就当它生效：宁可多检查一个，也不漏掉


class TestLockMatchesDeclarations:
    def test_every_pinned_direct_dependency_is_in_the_lock(self):
        declared, lock = parse_requirements(DECLARED), lock_entries()
        missing, mismatched = [], []
        for name, (ver, _) in declared.items():
            if name not in lock:
                missing.append(f"{name}=={ver}")
            elif lock[name][0] != ver:
                mismatched.append(f"{name}: 声明 {ver} / 锁文件 {lock[name][0]}")
        assert not missing, (
            f"requirements.txt 里声明了、锁文件里却没有：{missing}\n"
            f"  —— 改了声明就要重新生成锁文件：{REGENERATE}")
        assert not mismatched, (
            f"声明与锁文件的版本不一致：{mismatched}\n"
            f"  —— 重新生成：{REGENERATE}")


class TestInstalledEnvironmentMatchesTheLock:
    def test_installed_versions_match_the_lock(self):
        lock = lock_entries()
        bad = []
        for name, (ver, marker) in sorted(lock.items()):
            if not applies_here(marker):
                continue
            try:
                here = installed_version(name)
            except PackageNotFoundError:
                bad.append(f"{name}: 锁文件要求 {ver}，实际**没装**")
                continue
            if here != ver:
                bad.append(f"{name}: 锁文件 {ver} / 实际 {here}")
        assert not bad, (
            "当前环境与锁文件不一致 —— 应当用 `pip install -r requirements.lock` 安装：\n  "
            + "\n  ".join(bad[:20])
            + (f"\n  …另有 {len(bad) - 20} 处" if len(bad) > 20 else ""))

    def test_the_lock_carries_platform_markers(self):
        """锁文件必须**通用**，不能是"我这台机器"的快照

        用 pip freeze 生成的文件里没有环境标记，于是 uvloop（Linux 专有）会被带到
        Windows、pywin32（Windows 专有）会被带到 Linux —— 两边各炸一次。
        断言写在**解析出来的标记**上，而不是搜文件文本，注释骗不过它。
        """
        markers = [m for _, m in lock_entries().values() if m]
        assert markers, (
            f"锁文件里没有任何环境标记 —— 它大概是平台专属的快照（别用 pip freeze 生成）。\n"
            f"  正确生成方式：{REGENERATE}")
        assert any("sys_platform" in m for m in markers), (
            "锁文件里的标记没有涉及 sys_platform —— 跨平台差异（uvloop / pywin32）"
            f"不会被正确处理。正确生成方式：{REGENERATE}")
