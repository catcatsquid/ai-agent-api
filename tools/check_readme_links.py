"""README 链接自检：锚点是否都有对应标题、相对路径是否真的存在。

用法：python tools/check_readme_links.py README.md
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md")
root = path.resolve().parent
s = path.read_text(encoding="utf-8")

heads = [re.sub(r"^#+\s*", "", l).strip()
         for l in s.split("\n") if re.match(r"^#{2,3}\s", l)]


def slug(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[`*_~\[\]()（）【】。，、：:；;！!？?\"'—\-\./\\|]+", "", t)
    return re.sub(r"\s+", "-", t)


anchors = {slug(h) for h in heads}
problems = []

for a in re.findall(r"\]\(#([^)]+)\)", s):
    if a not in anchors:
        problems.append(f"失效锚点 #{a}")

for target in set(re.findall(r"\]\((?!\#|https?:)([^)]+)\)", s)):
    if not (root / target).exists():
        problems.append(f"文件不存在 {target}")

total = len(re.findall(r"\]\(", s))
print(f"标题 {len(heads)} 个 · 锚点 {len(anchors)} 个 · 链接 {total} 处")
if problems:
    print("问题：")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("链接全部有效")
