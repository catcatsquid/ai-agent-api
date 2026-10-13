"""两个"用户可见但一直没修"的问题的守卫

① `/qb/search` 与 `/qb/topics` 的数据源曾经不一致
   —— 知识点下拉来自完整题库，结果来自一个只装了 125 题的过期快照文件
     （当时完整题库是 191 题，两者差 66 题），
     于是有一批知识点点进去**永远 0 条结果**。
     （该快照文件已删除；题数后来也做过一次人工筛选，现在两者恒等。）
② `/health` 不在限流豁免表里 —— 健康检查会被 429 判死，
   而我原来的测试恰好断言的是没意义的 "/"，所以测试是绿的。
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from course_agent import middleware, qbank
from course_agent.server import app, ALL_PAPERS_SENTINEL, _load_questions_for_request

ROOT = Path(__file__).resolve().parent.parent

client = TestClient(app, raise_server_exceptions=False)


class TestQuestionBankSingleSource:
    def test_default_search_uses_the_merged_layer(self):
        """默认（"全部题库"）必须走 qbank，而不是读那个只有 125 题的合并文件"""
        questions, err = _load_questions_for_request(ALL_PAPERS_SENTINEL)
        assert err is None
        assert len(questions) == len(qbank.load_questions())

    def test_search_total_matches_the_merged_layer(self):
        """搜索接口的总数必须等于题库真实总数

        修复前这个接口读的是那个过期快照文件（只有 125 题），
        而知识点来自 191 题的完整题库，于是有一批知识点点进去恒为 0 条结果。
        这里断言的是**两个入口的一致性**，不写死题数 —— 题库增删卷子都不用改。
        """
        search = client.post("/qb/search",
                             json={"filename": ALL_PAPERS_SENTINEL}).json()
        assert search["count"] == len(qbank.load_questions())

    def test_search_sources_cover_every_paper(self):
        """搜索结果必须覆盖全部卷子（修复前只覆盖过期快照里那几份）"""
        search = client.post("/qb/search",
                             json={"filename": ALL_PAPERS_SENTINEL}).json()
        found = {q.get("source") for q in search["data"]}
        all_sources = {p["source"] for p in qbank.list_papers()}
        assert found == all_sources, \
            f"搜索结果少了这些卷子: {all_sources - found}"

    def test_every_topic_is_searchable(self):
        """知识点下拉里的每一项都必须能搜出结果（修复前有一批是 0 条）"""
        data = client.get("/qb/topics").json()["data"]
        topics = data.get("topics") or []
        assert topics, "拿不到知识点列表"
        empty = []
        for name in topics[:40]:
            r = client.post("/qb/search", json={"topic": name,
                                                "filename": ALL_PAPERS_SENTINEL}).json()
            if r["count"] == 0:
                empty.append(name)
        assert not empty, f"这些知识点搜不出任何题: {empty[:5]}"

    def test_named_file_still_works(self):
        """显式指定单卷文件时仍然读那个文件（/qb/list 的下拉要用）"""
        names = [f["filename"] for f in client.get("/qb/list").json()["data"]]
        per_paper = [n for n in names if n.startswith("questions_")]
        assert per_paper, "没有单卷文件"
        r = client.post("/qb/search", json={"filename": per_paper[0]}).json()
        assert r["count"] > 0

    def test_missing_file_reports_404(self):
        r = client.post("/qb/search", json={"filename": "不存在.json"}).json()
        assert r["code"] == 404

    def test_list_reports_real_total_for_merged_entry(self):
        """下拉里"全部题库（合并）"的条数必须报真实总数（191），不能报文件里的 125"""
        data = client.get("/qb/list").json()["data"]
        merged = [d for d in data if d["filename"] == ALL_PAPERS_SENTINEL]
        assert merged, "找不到合并项"
        assert merged[0]["count"] == len(qbank.load_questions())


class TestHealthIsRateLimitExempt:
    def test_health_is_in_the_quiet_paths(self):
        assert "/health" in middleware.QUIET_PATHS

    def test_health_survives_a_tight_rate_limit(self, monkeypatch):
        """把上限压到 1，健康检查仍然必须每次都通"""
        monkeypatch.setattr(middleware, "RATE_LIMIT_PER_MINUTE", 1)
        codes = [client.get("/health").status_code for _ in range(4)]
        assert 429 not in codes, f"/health 被限流了: {codes}"

    def test_business_endpoint_still_gets_limited(self, monkeypatch):
        """反过来：普通接口该限流还是要限流（别把限流整个关掉）"""
        monkeypatch.setattr(middleware, "RATE_LIMIT_PER_MINUTE", 1)
        codes = [client.get("/qb/topics").status_code for _ in range(4)]
        assert 429 in codes, f"业务接口没有被限流: {codes}"

    def test_only_one_exempt_list_exists(self):
        """限流豁免表只允许有一份定义，且两个消费方都必须复用它

        原来请求日志和限流**各硬编码了一份**，而且两份都漏了 /health
        （放的是没意义的 "/"）—— 于是探活会被 429 判死，而我的测试恰好
        断言的是 "/"，测试还是绿的。

        这条守卫原来写的是 `src.count("QUIET_PATHS") >= 3`（整文件字符串计数）。
        实测**能被骗过**：让 RateLimitMiddleware 改回内联豁免表（漏掉 /health，
        把上面那个 bug 原样复现），只要补两句提到 QUIET_PATHS 的注释，
        计数照样 ≥ 3、字符串断言也照样过 —— 单跑这条测试是绿的。
        （整份文件一起跑时会被下面那条行为测试抓住，所以严重性不高；
          但一条"自己不会失败"的守卫不该留着。）

        现在改成按 **AST** 断言：
          · QUIET_PATHS 顶层赋值**有且只有一处**；
          · 两个消费方的类体里都**引用了**它。
        注释不参与 AST，两类绕过都不成立。
        """
        import ast

        src = (ROOT / "course_agent/middleware.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        defs = [n for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "QUIET_PATHS"
                        for t in n.targets)]
        assert len(defs) == 1, f"QUIET_PATHS 有 {len(defs)} 份定义，应只有一份"

        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        for cls_name in ("RequestLogMiddleware", "RateLimitMiddleware"):
            cls = classes.get(cls_name)
            assert cls is not None, f"找不到 {cls_name}"
            used = {n.id for n in ast.walk(cls) if isinstance(n, ast.Name)}
            assert "QUIET_PATHS" in used, \
                f"{cls_name} 没有复用 QUIET_PATHS（自己又写了一份豁免表？）"

        limiter = classes["RateLimitMiddleware"]
        literals = {n.value for n in ast.walk(limiter)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for bad in ("/openapi.json", "/favicon.ico"):
            assert bad not in literals, f"限流里还留着硬编码的豁免路径: {bad}"
