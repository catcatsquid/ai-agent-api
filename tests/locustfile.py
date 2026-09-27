"""Locust 压测脚本 - Week9-3

使用方式:
  1. 先启动 API 服务: python -m uvicorn course_agent.server:app --port 8003
  2. 启动 Locust Web 界面:
     python -m locust -f tests/locustfile.py --host http://localhost:8003
  3. 打开浏览器: http://localhost:8089
  4. 设置用户数和孵化率，开始压测

  无界面模式（直接跑完出报告）:
     python -m locust -f tests/locustfile.py --host http://localhost:8003 \
       --users 50 --spawn-rate 10 --run-time 30s --headless --html report.html

注意:
  - 问答接口会真实调用 LLM，压测会烧钱，谨慎使用！
  - 默认只压测健康检查和登录接口（轻量级）
  - 要压测问答接口请把 QAUser 的 weight 调高
"""

import time
from locust import HttpUser, task, between, tag
from locust.clients import HttpSession


class QuickUser(HttpUser):
    """轻量级压测用户 - 只打健康检查和登录

    用来测 API 框架本身的性能基线，不涉及 LLM 调用。
    weight=5 意味着 83% 的用户是这种轻量用户。
    """

    weight = 5
    wait_time = between(0.5, 1.5)  # 每个请求之间等 0.5~1.5 秒

    @tag("health")
    @task(10)  # 数字越大，执行频率越高
    def health_check(self):
        """健康检查 - 最轻量的接口"""
        self.client.get("/", name="健康检查")

    @tag("health")
    @task(3)
    def docs_page(self):
        """Swagger 文档页"""
        self.client.get("/docs", name="API文档页")

    @tag("auth")
    @task(2)
    def login(self):
        """登录获取 JWT"""
        with self.client.post(
            "/auth/login",
            json={"username": f"loadtest-{int(time.time())}"},
            name="登录接口",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0:
                    response.success()
                    # 保存 token，后面可以用
                    self.token = data.get("token")
                else:
                    response.failure(f"登录失败: {data.get('msg')}")
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("auth")
    @task(1)
    def invalid_login(self):
        """空用户名登录 - 测试错误路径"""
        with self.client.post(
            "/auth/login",
            json={"username": ""},
            name="登录-空用户名",
            catch_response=True,
        ) as response:
            # 预期返回 code=400，这是正常的
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @task(1)
    def not_found(self):
        """404 接口 - 测试异常处理性能"""
        with self.client.get(
            "/api/nonexistent/path",
            name="404接口",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()  # 404 是预期的，算成功
            else:
                response.failure(f"预期404，实际{response.status_code}")


class QAUser(HttpUser):
    """问答接口压测用户 - 会真实调用 LLM！

    ⚠️  慎用！每次调用都会消耗 Token 额度。
    weight=1 意味着只有约 17% 的用户是这种。
    想完全禁用就把 weight 改成 0。
    """

    weight = 0  # 默认关闭，要测问答接口请改成 1 或更大
    wait_time = between(3, 8)  # 问答比较慢，间隔长一点

    def on_start(self):
        """每个用户启动时先登录拿 token"""
        import random
        username = f"qa-user-{random.randint(1, 10000)}"
        resp = self.client.post(
            "/auth/login",
            json={"username": username},
        )
        if resp.status_code == 200:
            self.token = resp.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @tag("qa")
    @task(1)
    def ask_question(self):
        """问答接口 - 会调用 LLM，慎用"""
        if not self.token:
            return  # 登录失败就不调了

        questions = [
            "什么是机器学习？",
            "Python 和 Java 有什么区别？",
            "解释一下什么是神经网络",
            "数据库索引有什么用？",
            "什么是 RESTful API？",
        ]
        import random
        q = random.choice(questions)

        with self.client.post(
            "/qa/ask",
            json={
                "question": q,
                "thread_id": f"loadtest-thread",
                "use_plan": False,
            },
            headers=self.headers,
            name="问答接口",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and data.get("answer"):
                    response.success()
                else:
                    response.failure(f"返回异常: {data.get('msg', '未知错误')}")
            elif response.status_code == 429:
                response.success()  # 被限流也是正常情况
            else:
                response.failure(f"状态码: {response.status_code}")


# ============================================================
# 压测场景配置
# ============================================================
# 可以通过 --tags 参数只跑某类接口:
#   python -m locust -f tests/locustfile.py --host http://localhost:8003 --tags health
#   python -m locust -f tests/locustfile.py --host http://localhost:8003 --tags auth
#   python -m locust -f tests/locustfile.py --host http://localhost:8003 --tags qa
# ============================================================
