"""Locust 压测脚本 - Week8

使用方式:
  1. 先启动 API 服务: python -m uvicorn course_agent.server:app --port 8003
  2. 启动 Locust Web 界面:
     python -m locust -f tests/locustfile.py --host http://localhost:8003
  3. 打开浏览器: http://localhost:8089
  4. 设置用户数和孵化率，开始压测

  无界面模式（直接跑完出报告）:
     python -m locust -f tests/locustfile.py --host http://localhost:8003 \
       --users 50 --spawn-rate 10 --run-time 30s --headless --html docs/benchmark_report.html

注意:
  - 问答接口会真实调用 LLM，压测会烧钱，QAUser 默认 weight=0
  - 默认只压测轻量级 GET 接口
  - 要压测问答接口请把 QAUser 的 weight 改成 1
"""

import time
import random
from locust import HttpUser, task, between, tag


class QuickUser(HttpUser):
    """轻量级压测用户 - 只打 GET 接口

    测 API 框架本身的性能基线，不涉及 LLM 调用。
    weight=10 意味着大部分用户是这种。
    """

    weight = 10
    wait_time = between(0.5, 1.5)

    @tag("health")
    @task(10)
    def health_check(self):
        self.client.get("/", name="健康检查")

    @tag("health")
    @task(3)
    def docs_page(self):
        self.client.get("/docs", name="API文档页")

    @tag("kb")
    @task(5)
    def kb_stats(self):
        with self.client.get("/kb/stats", name="知识库统计", catch_response=True) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("kb")
    @task(3)
    def kb_documents(self):
        with self.client.get("/kb/documents", name="文档列表", catch_response=True) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("qb")
    @task(3)
    def qb_list(self):
        with self.client.get("/qb/list", name="题库列表", catch_response=True) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("qb")
    @task(2)
    def qb_topics(self):
        with self.client.get("/qb/topics", name="题库知识点", catch_response=True) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("page")
    @task(2)
    def chat_page(self):
        with self.client.get("/chat", name="聊天页面", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("page")
    @task(1)
    def kb_page(self):
        with self.client.get("/kb", name="知识库页面", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"status={r.status_code}")


class SearchUser(HttpUser):
    """检索接口压测用户 - 打 KB/QB 搜索接口

    不涉及 LLM 调用，但会触向量检索，中等负载。
    weight=3 意味着约 23% 的用户是这种。
    """

    weight = 3
    wait_time = between(1, 3)

    KB_QUERIES = [
        "什么是进程调度",
        "页面置换算法",
        "死锁的四个必要条件",
        "TCP三次握手",
        "虚拟内存",
        "磁盘调度算法",
        "信号量与PV操作",
        "分页与分段",
        "CSMA/CD协议",
        "银行家算法",
    ]

    QB_QUERIES = ["死锁", "TCP", "进程", "调度", "内存", "路由", "分页", "文件系统"]

    @tag("kb_search")
    @task(3)
    def kb_search(self):
        q = random.choice(self.KB_QUERIES)
        with self.client.post(
            "/kb/search",
            json={"query": q, "top_k": 5},
            name="知识库检索",
            catch_response=True,
        ) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @tag("qb_search")
    @task(2)
    def qb_search(self):
        q = random.choice(self.QB_QUERIES)
        with self.client.post(
            "/qb/search",
            json={"keyword": q, "filename": "question_bank.json"},
            name="题库搜索",
            catch_response=True,
        ) as r:
            if r.status_code == 200 and r.json().get("code") == 0:
                r.success()
            else:
                r.failure(f"status={r.status_code}")


class QAUser(HttpUser):
    """问答接口压测用户 - 会真实调用 LLM！

    慎用！每次调用都会消耗 Token 额度。
    weight=0 表示默认关闭，要测问答接口请改成 1。
    """

    weight = 0
    wait_time = between(5, 10)

    QUESTIONS = [
        "什么是进程调度？",
        "解释一下什么是死锁",
        "页面置换算法有哪些？",
        "TCP三次握手过程是什么？",
        "什么是虚拟内存？",
    ]

    @tag("qa")
    @task(1)
    def ask_question(self):
        q = random.choice(self.QUESTIONS)
        with self.client.post(
            "/chat/stream",
            json={
                "query": q,
                "thread_id": f"loadtest-{random.randint(1, 100)}",
                "mode": "plan",
            },
            name="问答接口(SSE)",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()  # 被限流也是正常情况
            else:
                response.failure(f"status={response.status_code}")
