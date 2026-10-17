# 知识寻回BOT —— 课程答疑与刷题 Agent

> **把课件、课本、历年真题 PDF 变成会出题、会批改、会讲题、讲不清就联网查的对话式助教。**
> 覆盖操作系统 / 计算机网络两门课：4199 块知识分块 + 125 道结构化真题（题图 + 批改依据），
> 学生一句自然语言即可「刷题 → 交答案 → 被批改 → 听不懂就追问 → 换个话题也行」。

`Python 3.12` · `LangGraph` · `FastAPI` · `ChromaDB` · `rank_bm25` · `GLM-4-Flash` · `Docker`

| 844 项测试 | 29 条行为评测用例 | 17 个工具 | 26 个 API | CI 全绿 |
|:--:|:--:|:--:|:--:|:--:|
| 全离线可跑 · 守卫值守 | 判据化 · 分版本存档 | Function Calling | SSE 流式 | Docker 一键 |

---

## Demo

<!-- TODO(投递前必做)：放 8–12 秒无声循环 GIF，内容＝下面黄金路径的第 2→4 步。
     文件 demo/demo.gif，控制在 5 MB 以内。项目页没有动图，招聘方停留通常 < 10 秒。
     生成 GIF 后，把下面这行取消注释（现在先注释掉，避免仓库首页出现坏图）。 -->

<!-- ![演示](demo/demo.gif) -->

> **演示动图待补**（`demo/demo.gif`）｜录屏与素材清单见 [`demo/README.md`](demo/README.md)

真实会话形态（`exports/` 里导出的真机对话，不是设计稿）：

```text
你：我想刷操作系统的题
它：2024年秋操作系统期末试题（回忆版）(26题：选择20 综合6) / 2025年秋… / 2026春…
    └ list_papers —— 题量与题型构成来自题库，不是模型编的

你：2024 年那份
它：第 1/10 组（选择题 5 题）1. 进程调度 [题图] 2. 线程状态转换 [题图] … 共 5 张真题原图
    └ get_paper_questions —— 只返回题号/题型/知识点/题图

你：ACBDA
它：1.A 正确 / 2.C 正确 / 3.B 正确 / 4.D 正确 / 5.A 错，正确答案 D  ↓ 只讲错的那道
    └ grade_answers

你：第 5 题为什么选 D
它：……此刻才去取参考答案
    └ explain_question

你：死锁的四个必要条件是什么
它：……带来源引用作答，命中课本/课件原文
    └ search_knowledge（混合检索 + Rerank）

你：2025 年图灵奖给了谁
它：库里没有 → 自动联网查，并说明"这不是课程资料里的内容"
    └ web_search
```

本地启动后访问 `http://127.0.0.1:8003/chat` 直接体验。完整演示脚本见 [`demo/README.md`](demo/README.md)。

---

## 核心能力

| 能力 | 实现 |
|------|------|
| RAG 混合检索 | 向量 + BM25 融合（0.6/0.4）→ LLM 查询改写（保留原查询）→ LLM Rerank；课件/笔记优先，不足回退课本 |
| Agent 双推理模式 | Plan-and-Execute 与 ReAct 在同一张 LangGraph 图上，`mode` 参数切换 |
| 工具调用 | 17 个工具（检索 / 刷题 / 学习复习 / 统计 / 进度），Function Calling 由模型自主规划 |
| 刷题与批改 | 分组出题（选择 5 题/组）、多种答案格式解析（`1.D`、`DBDAC`、`全都选A`）、自动批改、按点讲解 |
| SSE 流式输出 | `start` / `node` / `content` / `done` / `error` 五类事件：节点进度 + 正文逐字 + done 校准 |
| 多轮对话 | `thread_id` + checkpoint 持久化；滑动窗口 + LLM 摘要压缩治理长上下文 |
| Web 管理后台 | 知识库概览 / 检索测试 / 增量导入 / 题库检索（原生 JS，零框架） |

---

## 最值得看的三个设计

只看三分钟的话，看这三段。

### 1. 评测是地基，不是附属品 —— 它抓出过一个「上线即残废」的 bug

`pytest` 回答"代码对不对"，回答不了"学生是不是 35 秒才看到第一个字"。所以另建一层 [`eval/`](eval/README.md)：
脚本化用例 + **全部机器可判定**的判据 + 按版本归档、可 `git diff` 的报告。

- 29 条用例覆盖出题 / 批改 / 讲题 / 学习 / 复习 / 意图矩阵，每条必须写 `why`（防的是哪个真实缺陷）
- 判据示例：`no_stem_leak`（题面不许进正文，用题库原文切片比对）、`images_all_from_bank`（不许编图片链接）、
  `no_internal_artifacts`（写给模型看的 JSON 不许播给学生）、`require_streaming`、`max_ttft_ms`、`tools_exactly_empty`
- 三条防自欺纪律：**必须有"最低门槛"判据**（否则空回答会让所有"不许出现 X"的判据真空通过）、
  **评测集要被变异验证**（改坏代码必须变红）、**不许为了让评测变绿而改判据**

它抓到的那个 bug：

> v1.7.1 的报告里，12 轮对话的 `content` 事件数**全是 0**。根因：自写的 `SimpleLLM` 不是 LangChain ChatModel
> （只实现 `invoke`/`bind_tools`），而 `stream_mode="messages"` 只认实现了 `_stream` 的 Runnable ——
> **逐字流式在生产里从未生效**，学生等约 33 秒后整段答案弹出。
> 640+ 项测试没抓到，因为流式用例用的是自带 `_stream` 的假模型，**从没跑过生产那个 LLM 对象**；
> 前端又有 `done.answer` 兜底，于是"能用，只是慢"。修完后加纪律：`tests/test_llm_streaming.py`
> **只对生产 llm 对象**下判据，并做变异验证。

### 2. 把「呈现约束」放进**工具返回值**，而不是 system prompt

刷题 Agent 最容易出三类事故：**泄题**（还没答就写出答案）、**缺图**（真题没图没法做）、
**编链接**（曾编出 `example.com/net1.jpg`）。常规做法是在 prompt 里写"不许泄题"，本项目换了条路：

- `get_paper_questions` 的返回值只有题号 / 题型 / 知识点 / 题图 —— 题干和参考答案**根本不进上下文**，模型想泄也没得泄
- 每个工具结果的 `_guide` 字段说明"这一步该怎么呈现"（报出组内几题、把图贴出来、不要念答案）
- 于是 system prompt 只描述人设与能力，交互保持自然，而硬约束跟着数据走、不会被上下文冲掉

这条设计有 8 条评测用例专门守着。

### 3. 检索阈值：归一化融合分**不能**当判据

融合分按最大值归一化，排名第一恒 ≥ 0.6，**只能排序**；判断"库里到底有没有相关内容"必须用绝对相似度。
而 ChromaDB 返回的是 L2 **距离**不是相似度，单位向量下要换算 `sim = 1 - d²/2`，否则域外查询（`d>1`）会算出负分。

实测边界：**域内最低 0.665 / 域外最高 0.524** → 阈值取 0.55，不足则两级回退（先回退课本 → 仍不足则联网）。

---

## 架构

```text
                  ┌───────────── FastAPI (course_agent/server.py) ─────────────┐
 chat.html        │ /qa/stream SSE   /kb/* 知识库   /qb/* 题库   /auth/login    │
 kb.html          │ 限流 · CORS · 请求日志 · trace_id · JWT/API Key（默认关闭） │
                  └───────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
  plan（默认）      │  planner ──▶ executor ──▶ [route] ──▶ synthesizer ──▶ END│
                    │      │            │                                   │
  react             │   llm ──▶ should_continue ──▶ tools / force_search     │
                    │                 └──────循环，轮数硬上限──────┘         │
                    │        17 个工具：检索 / 刷题 / 学习复习 / 统计 / 进度   │
                    └───────┬─────────────────────────────┬─────────────────┘
                            │                             │
        ┌───────────────────┴──────────┐   ┌──────────────┴──────────────┐
        │ rag_tool.py                  │   │ qbank.py                    │
        │ 混合检索 · Rerank · 查询改写   │   │ 125 题 · 题图 · 批改依据      │
        │ ChromaDB 4199 chunks (2048d) │   │ 6 份真题 · 89 个知识点        │
        └──────────────────────────────┘   └─────────────────────────────┘
```

检索流水线：

```
问题 → LLM 查询改写（保留原查询）→ 每个变体混合检索 → 合并去重 + 低质块过滤
     → 结果不足回退课本 → LLM Rerank → 格式化输出（含补搜提示）
```

---

## 可靠性设计（为什么它不是 demo 级玩具）

- **防 Agent 失控**：工具调用轮数硬上限（默认值会放大到数千次调用，实测踩过）；guard node 强制课程问题先检索，防止模型凭记忆作答
- **防泄题纵深防御**：出题接口只返回题号/知识点/题图，题面与答案不进上下文；仅在批改或学生说「不会」时按需取回
- **上下文治理**：读时裁剪（state 永保完整历史）、摘要缓存按消息 id 摊还、tool 消息与 `tool_calls` 严格配对
- **SSE 四层过滤**：来源白名单 + 内部调用标签 + 类型白名单 + 非空过滤，工具返回原文不会泄漏进正文频道
- **路径安全**：所有文件操作经 `resolve()` + `relative_to(data/)` 校验，防线设在 API 边界而非库层
- **可观测性**：`contextvars` 存 `trace_id`，贯穿中间件 / 日志 / 错误处理；三层错误模型 + 指数退避重试 + 降级

---

## 量化效果

| 指标 | 结果 | 口径 |
|------|------|------|
| 首字响应 TTFT | 33s → **6.5s**（p50） | v1.7.1 → v1.7.2，eval 实测，同上同判据 |
| `content` 逐字事件 | 0 个/轮 → 517–613 个/轮 | 同上 |
| 行为评测判据 | 202 / 204 通过 | v1.7.3 存档（6 用例 × 3 attempts） |
| 补搜提示准确率 | 域内/域外 12 条样本 **12/12** | 修复前 7/12 |
| 测试 | **844** 项 pytest + 52 项前端断言 + 服务探活 | CI 每次 push 自动跑 |

> 评测走真实 HTTP + SSE + 真模型；判据全部机器可判（工具轨迹 / 泄漏原文比对 / 流式门槛），报告按版本存档可 diff。
> v1.7.3 之后用例已扩到 29 条，报告待重跑归档 —— 见 Roadmap。

---

## 快速开始

```bash
pip install -r requirements.lock   # 锁文件保证与 CI 一致（uv --universal 生成，跨平台）
cp .env.example .env               # 填智谱 GLM API Key
python build_knowledge.py --clear --skip-textbook
python -m uvicorn course_agent.server:app --port 8003
```

| 入口 | URL |
|------|-----|
| 聊天页面 | http://127.0.0.1:8003/chat |
| 知识库管理 | http://127.0.0.1:8003/kb |
| API 文档 | http://127.0.0.1:8003/docs |

Docker：`docker compose up -d --build`（向量库与数据目录经 volume 持久化）。

```bash
# 一行验证流式是否真在流（看 content 是否逐字出现、TTFT 多少）
curl -N -X POST http://127.0.0.1:8003/qa/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"进程和线程的区别","mode":"plan"}'
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/qa/stream` | SSE 流式问答（唯一流式端点） |
| POST | `/qa/ask` | 非流式问答 |
| POST | `/auth/login` | 登录获取 JWT（默认关闭，本地免登录） |
| GET/POST/DELETE | `/kb/*` | 知识库管理（统计/文档/检索/浏览/增量导入/上传/删除） |
| GET/POST | `/qb/*` | 题库（列表/搜索/知识点与来源） |
| GET/DELETE | `/chat/conversations*` | 会话列表/详情/删除 |

---

## 工程化

- **测试四层**：单元测试（外部依赖全 mock）/ 接口契约测试 / 历史 bug 回归 / 架构守卫（依赖方向、图隔离、锁文件同步）
- **三条环境守卫**（session 级）：真实会话库不被测试写入、真实 `data/` 不被测试改动、服务运行中拒绝开跑（防 Chroma 并发写损坏）
- **数据可用性**：300MB 课件/向量库不入库，依赖缺失的用例自动 skip 而非报红，新克隆即可跑通
- **CI**：GitHub Actions 每次 push 跑 pytest + 前端测试 + 服务冒烟；`txt / lock / 环境`三方一致性有测试守着
- **压测**：Locust 三级虚拟用户（只读 / 检索 / 问答），问答级默认关闭防烧额度

---

## 已知限制与设计取舍

- 课件/课本 PDF 因版权不入库，知识库需本地构建；题库 JSON 入库可直接体验刷题
- 另保留一份手写 tool-calling 循环（`tutor_agent.py`）作为冻结的对照实现：与 LangGraph 路径共用 `quiz_service.py`，无 HTTP 入口
- 首响仍有约 6.5 秒 —— planner + 检索是静默的，模型"检索完才开始写"。方向：压检索耗时 / 把进度也流给学生
- ReAct 偶发"嘴上说要调工具、实际没调"：协议帧泄漏已兜住，补一轮真调用未做
- 当前为单用户本地部署设计；多用户需在会话与配额层扩展
- 模型层依赖智谱 GLM API，未做模型侧微调（应用层定位）

---

## Roadmap

- [ ] **重跑 29 条评测用例、归档 v1.11 报告**（当前最该补的一件事）
- [ ] 检索层 Recall@K / MRR：把"混合检索比纯向量好多少"变成数字
- [ ] ReAct vs Plan-and-Execute 同题对比：token 消耗 / 轮次 / 正确率
- [ ] 首响压到 3 秒内
- [ ] 全局学习进度、刷题位置记忆与错题本

---

## 项目结构

```
course_agent/          # Agent 核心：graph / nodes / tools / server / memory / qbank / quiz_service
knowledge_pipeline/    # 知识处理流水线（PPT/课本 → 分块入库，含 VLM 图片理解）
rag_tool.py            # 混合检索 + Rerank + 查询改写
build_knowledge.py     # 知识库构建 CLI（薄壳，逻辑在 KnowledgeBase）
tests/                 # 844 项测试 + conftest 守卫 + Locust 压测
eval/                  # 行为评测（runner + 29 条用例 + 按版本存档的报告）
demo/                  # 演示脚本与素材
data/                  # 课程数据（gitignore，仅题库 JSON 入库）
docs/                  # 文档
archive/               # 历史版本归档
```

旧版说明（学习路线视角）保留在 [README_old.md](README_old.md)；评测细节见 [eval/README.md](eval/README.md)；
踩坑与取舍的完整复盘见 [docs/design-decisions.md](docs/design-decisions.md)。
