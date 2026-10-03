"""依赖方向与导入顺序的守卫

循环依赖的真正症状不是"报错"，而是**导入顺序敏感** ——
单跑测试总是绿，换个入口（比如先 import pipeline）就可能炸。
所以这里既做静态断言，也真的用一个干净子进程按"反向顺序"导入一次。
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _top_level_imports(path: str) -> list:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return [ast.unparse(n) for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))]


def _nested_imports(path: str) -> list:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    top = {id(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    return [ast.unparse(n) for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in top]


class TestNoCircularDependency:
    """rag_tool ↔ knowledge_pipeline.pipeline 的环必须保持断开

    原来的环：
        rag_tool  →  pipeline（函数内延迟导入）
        pipeline  →  rag_tool（顶层导入 load_document/clean_text/_classify_doc/_extract_chapter）
    """

    def test_pipeline_does_not_import_rag_tool(self):
        imports = _top_level_imports("knowledge_pipeline/pipeline.py")
        offenders = [s for s in imports if "rag_tool" in s]
        assert not offenders, f"pipeline 又顶层导入了 rag_tool: {offenders}"

    def test_pipeline_uses_doc_utils(self):
        imports = _top_level_imports("knowledge_pipeline/pipeline.py")
        assert any("doc_utils" in s for s in imports), \
            "pipeline 应该从 doc_utils 拿文档处理函数"

    def test_rag_tool_does_not_import_pipeline_at_top_level(self):
        imports = _top_level_imports("rag_tool.py")
        offenders = [s for s in imports if "knowledge_pipeline" in s]
        assert not offenders, f"rag_tool 顶层又导入了 pipeline: {offenders}"

    def test_rag_tool_still_lazily_uses_pipeline(self):
        """流水线是 rag_tool 的可选功能，延迟导入要保留（否则又是环）"""
        nested = _nested_imports("rag_tool.py")
        assert any("knowledge_pipeline" in s for s in nested), \
            "rag_tool 应当仍在函数内延迟导入 pipeline"

    def test_doc_utils_has_no_project_dependencies(self):
        """最底层模块只能依赖标准库 —— 否则环会以别的形式回来"""
        imports = _top_level_imports("doc_utils.py")
        for s in imports:
            for bad in ("course_agent", "rag_tool", "knowledge_pipeline", "config"):
                assert bad not in s, f"doc_utils 不该依赖 {bad}: {s}"

    def test_rag_tool_reexports_the_moved_functions(self):
        """历史调用点（tests/、debug/ 脚本）用的是 rag_tool.xxx，不能断"""
        import rag_tool
        for name in ("load_document", "clean_text", "_classify_doc", "_extract_chapter"):
            assert hasattr(rag_tool, name), f"rag_tool 少了 {name}（历史调用点会挂）"

    def test_importing_pipeline_first_works(self):
        """按"反向顺序"导入一次：这正是循环依赖会炸的场景"""
        code = ("import knowledge_pipeline.pipeline as p;"
                "import rag_tool;"
                "print('ok')")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        assert r.returncode == 0, f"先导入 pipeline 失败了:\n{r.stderr[-800:]}"
        assert "ok" in (r.stdout or "")

    def test_importing_rag_tool_first_works(self):
        code = ("import rag_tool;"
                "import knowledge_pipeline.pipeline as p;"
                "print('ok')")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        assert r.returncode == 0, f"先导入 rag_tool 失败了:\n{r.stderr[-800:]}"
        assert "ok" in (r.stdout or "")
