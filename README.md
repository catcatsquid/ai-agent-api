# 知识寻回BOT —— 课程资料变成会讲课、会出题、会批改的助教

> 把课件、课本、历年真题 PDF 变成一个真能用的课程助教：按章节讲、分组出题、批改、讲题、记进度，讲不清就联网查。
>
> 仓库自带 985 项离线测试（另有 107 项前端断言）和 34 条行为评测用例（走真实 HTTP + SSE）。
> 测试与首响口径的唯一定义在 [eval/reports/INDEX.md](eval/reports/INDEX.md)，
> 路由与工具数在 [docs/architecture.md](docs/architecture.md)。

`Python 3.12` · `LangGraph` · `FastAPI` · `ChromaDB` · `rank_bm25` · `GLM-4-Flash` · `Docker`

**目录**

- [Demo](#demo)

- [核心能力](#核心能力)

- [架构](#架构)

- [快速开始](#快速开始从零复现)

- [当前状态与已知限制](#当前状态与已知限制)

- [测试与工程化](#测试与工程化)

- [Roadmap](#roadmap)

- [设计取舍](#设计取舍)

***

## Demo

**视频（52 秒）：[demo/demo-full.mp4](demo/demo-full.mp4)**

| RAG 混合检索                                    | 问答：先检索再作答（1 个工具调用）                       |
| ------------------------------------------- | ---------------------------------------- |
| ![检索测试页](docs/screenshots/03-retrieval.png) | ![问答：先检索再作答](docs/screenshots/04-qa.png) |

| 出题：题号与知识点取自题库                        | 学习进度：学到第几块、刷到第几组                           |
| ------------------------------------ | ------------------------------------------ |
| ![出题页](docs/screenshots/01-quiz.png) | ![学习进度页](docs/screenshots/02-progress.png) |

本地启动后访问 `http://127.0.0.1:8003/chat` 直接体验。不想装环境可走
[Docker 演示镜像](#docker)，一条命令就能看到检索与刷题在工作。

***

## 核心能力

| 场景   | 能力                                    |
| ---- | ------------------------------------- |
| 学习   | 按章节讲、书签续读、章末小测、随时不懂就问                 |
| 解题   | 分组出题、多种答案格式批改、按点讲解、按知识点抽题、错题重做、自拟练习   |
| 复习   | 知识点速查、针对薄弱环节出题练习                      |
| 知识问答 | 先检索再作答；库里没有就两级回退，仍不足则联网               |
| 进度   | 跨会话、跨窗口学习进度保存；换场景不断线                  |
| 意图路由 | 一句话切换场景（学习/解题/复习），不需要手动切模式开关          |
| 知识库  | 上传、路径导入、目录浏览、检索测试、删除                  |
| 数据入库 | 知识库：清洗→切块→向量化入 Chroma（增量导入）；题库：试卷按题裁图→VLM 读图表+LLM 出答案→结构化 JSON |

### 技术栈

| 层  | 选型                                                                  |
| -- | ------------------------------------------------------------------- |
| 接入 | FastAPI · SSE 五类事件（`start` / `node` / `content` / `done` / `error`） |
| 编排 | LangGraph：Plan-and-Execute 与 ReAct 两张图，各自独立 checkpointer            |
| 检索 | ChromaDB + BM25 混合融合 → LLM rerank → 两级回退                            |
| 模型 | `glm-4-flash`（对话）· `embedding-3`（向量）· `glm-4v-flash`（图表理解）          |
| 状态 | SQLite checkpointer 持久化 + JSON 进度账本                                 |
| 前端 | 原生 JS，零框架（`chat.html` / `kb.html`）                                  |

***

## 架构

![架构总览](docs/screenshots/architecture.png)

端到端流程：`POST /qa/stream` 进来 → planner 出计划（课程问题强制先检索）→ executor 按计划调用 17 个 `@tool`
中的一个或多个（检索、取题、批改……）→ synthesizer 把工具结果写成答案，逐字从 `content` 事件推给前端 → `done` 收尾。
两张图（Plan-and-Execute / ReAct）只在编排层有差别，业务逻辑共用 `course_agent/quiz_service.py`。

模块分工、路由清单、层间依赖、两条数据线的构建链路：见 [docs/architecture.md](docs/architecture.md)。

**工程防线**：

| 防线    | 做法                                                              |
| ----- | --------------------------------------------------------------- |
| 防失控   | 工具调用轮数硬上限；guard node 强制课程问题先检索，不许凭模型记忆作答                        |
| 上下文治理 | 读时裁剪、摘要缓存按消息 id 摊还、tool 消息与 `tool_calls` 严格配对                   |
| 不泄漏   | SSE 正文四层判据（来源白名单 + 内部调用标签 + 类型白名单 + 非空），工具返回原文不进正文频道            |
| 路径安全  | 所有文件操作 `resolve()` + `relative_to(data/)`，防线设在 API 边界而非库层       |
| 可观测   | `contextvars` 存 `trace_id` 贯穿中间件 / 日志 / 错误处理；三层错误模型 + 指数退避 + 降级 |

***

## 快速开始（从零复现）

### 0. 前置

| 需要         | 要求         | 为什么                                                     |
| ---------- | ---------- | ------------------------------------------------------- |
| Python     | **3.12.x** | 锁文件按 3.12 生成；3.13 有包缺 wheel，3.11 会静默降级依赖                |
| Git        | 任意版本       | 也可以直接下 ZIP（解压后**多一层目录**，别踩）                             |
| 磁盘         | ≥ 300 MB   | 课件 + 向量库；课件不入库，需自备                                      |
| 智谱 API Key | —          | 免费申请约 2 分钟：[docs.bigmodel.cn](https://docs.bigmodel.cn) |

### 1. 拿到代码并建虚拟环境

```bash
git clone <仓库地址> ai-agent-learning && cd ai-agent-learning

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.lock
```

> **别跳过 venv。** 用系统 Python 装，很容易和别的项目的 `chromadb` / `pydantic` 版本打架，
> 而报错都是"import 失败"，极难定位。锁文件保证你装出来的和 CI、镜像完全一致。

### 2. 配置 Key

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

打开 `.env`，**只需填一行**：

```
ZHIPU_API_KEY=你的key
```

其余全部可选：`API_KEY` / `JWT_SECRET`（鉴权，本地不用）、`RATE_LIMIT_PER_MINUTE`（限流）、
`PROGRESS_FILE`（学习进度文件，演示时用它隔离真实进度）。

### 3. 准备课程数据

> **只想先看看效果？** 用 [Docker 演示镜像](#docker) 可以跳过第 3、4 步。

课件 PDF、试卷 PDF、题图、题库 JSON、向量库**都不入库**（版权 + 体积，合计约 300 MB），需要自己准备。目录结构固定：

```
data/操作系统/ppt/*.pdf          # 课件
data/计算机网络/**/*.pdf
data/<课程>/往年题/*.pdf          # 试卷，用来生成题库
```

- **想刷题** → 先有题库：`python -m course_agent.question_bank data/操作系统/往年题/`

- **想问答** → 放好课件，跑第 4 步建库

- 题图缺失时，出题那一轮会没图（题库 JSON 里有引用，但文件不在）—— 这是预期，不是 bug

### 4. 建库（一次性）

```bash
python build_knowledge.py --clear --skip-textbook
python build_knowledge.py --stats        # 验证：分块数必须 > 0
```

- 会按批调用 embedding，**消耗额度**，全量约 20 分钟

- `--skip-textbook` 跳过三本课本（最慢的部分）；要全量就去掉它

- 想先快跑通：`--data-dir ./data/操作系统` 只入一门课

- 各步细节（分类 / 分块 / VLM / md5 增量）见 [docs/architecture.md](docs/architecture.md)

### 5. 启动

```bash
python -m uvicorn course_agent.server:app --port 8003
```

| 入口     | URL                          |
| ------ | ---------------------------- |
| 聊天页面   | <http://127.0.0.1:8003/chat> |
| 知识库管理  | <http://127.0.0.1:8003/kb>   |
| API 文档 | <http://127.0.0.1:8003/docs> |

### 6. 怎么确认"真的跑对了"

```bash
python -m pytest tests/ -q          # 全离线；缺数据的用例 skip 而不是报红
curl http://127.0.0.1:8003/health   # knowledge_chunks > 0 = 库建好了

curl -N -X POST http://127.0.0.1:8003/qa/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"进程和线程的区别","mode":"plan"}'
  # 看 content 是否逐字出现；整段一次性出现 = 流式没生效
```

### 7. 起不来？先查这五条

| 现象                                | 大概率原因                    | 解法                               |
| --------------------------------- | ------------------------ | -------------------------------- |
| `import` 失败 / pydantic 版本冲突       | 没在 venv 里装               | 删 `.venv` 重来，确认命令行有 `(.venv)` 前缀 |
| `/health` 里 `knowledge_chunks: 0` | 没建库 / `--data-dir` 指错    | 重跑第 4 步                          |
| 问答报 401 或额度错误                     | `.env` 里 Key 没填、填错行、带了引号 | 只留 `ZHIPU_API_KEY=xxx` 一行        |
| 端口 8003 被占                        | 上一次的服务没关                 | 换 `--port 8004`，或关掉旧进程           |
| 出题没有图                             | 题图未入库（`.gitignore`）      | 属预期，见第 3 步                       |

### Docker

仓库带两份构建文件，用途不同：

| <br />     | 演示镜像                                            | 完整部署                             |
| ---------- | ----------------------------------------------- | -------------------------------- |
| 构建文件       | `Dockerfile.demo` + `docker-compose.demo.yml`   | `Dockerfile`                     |
| 数据         | 挂 `demo/seed/`（随仓库分发：5 份自写笔记 + 11 道原创题 + 预建向量库） | 挂用户自己的 `data/` 与 `vector_store/` |
| 需要 API Key | 不需要，向量库已预建                                      | 需要，建库要调 embedding                |
| 用途         | 别人 clone 下来一条命令就能看到检索与刷题在工作                     | 装自己的课件，当日常工具用                    |

演示镜像：

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

起来就能看 `/chat`。`demo/seed/data` 是**只读**挂载，所以 `/kb/add`、`/kb/upload` 会失败 —— 故意的，
防止演示时改坏数据集；`demo/seed/vector_store` 可写（Chroma 打开 sqlite 要写 `-wal`/`-shm`），
`runtime` 用命名卷保存会话与进度，删掉卷就清空演示状态。

自己的完整部署：`Dockerfile` 装了 gcc/g++ 并 `COPY data/`，它对应的编排不随仓库分发（各人挂载路径不同），
用 `docker run` 或自己的 compose 挂 `./data` 与 `./vector_store` 即可。**前提是宿主机先建过库**，
否则起来检索全空（`/health` 会显示 `knowledge_chunks: 0`）。

***

## 当前状态与已知限制

| 指标        | 当前                                      | 口径                                                        |
| --------- | --------------------------------------- | --------------------------------------------------------- |
| 版本        | **v1.11.2**                             | `config.py` 的 `APP_VERSION`，与 git tag、GitHub Release 三方对齐 |
| 首字响应 TTFT | **7.5s**（p50）/ 16.6s（p95）               | 1.11.1 归档，真实 HTTP + SSE + 真模型                             |
| 单轮总时长     | 24.2s（p50）                              | 1.11.1 归档                                                 |
| 行为评测      | 判据 **900 / 903** 通过；29 条用例中 **26 条稳定**  | 每用例 3 次尝试，判据全部机器可判                                        |
| 补搜提示准确率   | 域内/域外 12 条样本 **12/12**                  | 1.7.x 时期修的单项，之后未重测                                        |
| 测试        | **985** 项 pytest + **107** 项前端断言 + 服务探活 | CI 每次 push 自动跑                                            |

> 未稳定的三条：`intent_followup_reference`（该检索却没调工具）、`knowledge_answer_streams`（调了章节读取而非检索）、
> `quiz_b11_wrong_redo`（该练错题却去取整卷）—— 都是"工具没调对"，不是回归，也没有内容质量判据在红。
>
> 历史对照只留一个：v1.7.1 → v1.7.2 首响 33s → 6.5s（`content` 事件 0 → 517–613 个/轮）。
> 1.7.1 当时还没有逐字流式，首字 ≈ 整轮，所以基线取的是轮时长 p50（32.9s）而不是 TTFT，证据在两份归档报告里。
> 更早的 1.7.3 存档是 6 条用例 / 202–204 条判据，分母不同，别和上表混着比。

**已知限制**

1. 课件、课本、题库、题图因版权与体积不入库（约 300 MB），都要本地准备；题库由试卷 PDF 生成，知识库需本地建库
2. 文档解析只有文本层：没有 OCR，也没有表格结构识别，扫描件和复杂表格会退化。题库那条线用"每题裁一张图"绕开了它，
   知识库那条线没有绕开
3. 首响 7.5 秒，是"先检索再作答"的代价：planner 与检索必须在写答案之前跑完，这段是静默的 —— 暂时不是能调掉的延迟
4. ReAct 偶发"嘴上说要调工具、实际没调"：已补一轮强制调用，残留是流式打字时可能闪过一下工具调用原文。
   根治要给 SSE 加回退缓冲，那会让流式变粗，所以保留现状
5. 评测要真模型、真额度，因而不进 CI，报告靠手动跑完归档 —— 这也是它做不到每次提交自动更新的原因；
   34 条用例里 29 条已归档，重跑归档在 Roadmap 里
6. 讲得好不好没有度量：34 条判据只管"该调的工具调了没、该出现的出现没"，不管"讲得清不清楚"
7. 单用户本地部署设计，多用户需要在会话与配额层扩展

***

## 测试与工程化

- **测试四层**：单元测试（外部依赖全 mock）/ 接口契约测试 / 历史 bug 回归 / 架构守卫（依赖方向、图隔离、
  锁文件同步、26 条路由的端点清单）

- **三条环境守卫**（session 级）：真实会话库不被测试写入、真实 `data/` 不被测试改动、服务运行中拒绝开跑
  （防 Chroma 并发写损坏）

- **数据可用性**：课件、试卷、题库、向量库都不入库，依赖缺失的用例自动 skip 而非报红，新克隆即可跑通
  （刷题相关的用例除外）

- **CI**：GitHub Actions 每次 push 跑 pytest + 前端断言（`tests/js/*.mjs`）+ 服务冒烟；
  `txt / lock / 环境`三方一致性有测试守着

- **可复现交付**：锁文件用 `uv pip compile --universal`（带 `sys_platform` 标记，Windows/Linux 同一份）→
  CI 与镜像都从它装；镜像分层先依赖后源码（改代码不触发重装）；`vector_store/` 与 `data/` 走 volume
  （重建要耗 embedding 额度）；healthcheck 用 `python -c requests` 而非 `curl`（slim 镜像里没有 curl）

- **压测**：Locust 三级虚拟用户（只读 / 检索 / 问答），问答级默认关闭防烧额度

***

## Roadmap

- [ ] **重跑 34 条用例、归档当前版本报告** —— 最新归档停在 1.11.1 的 29 条

- [ ] 检索层 Recall\@K / MRR：把"混合检索比纯向量好多少"变成数字

- [ ] ReAct 与 Plan-and-Execute 同题对比：token / 轮次 / 准确率 / 失败模式分类

- [ ] 给知识库分块加 `course` 元数据，去掉"按文件名反查课程"这层耦合（只改 metadata，不用重新 embedding）

- [ ] 新增课程后能在 Web 上传试卷 PDF 并生成题库

- [ ] 讲得好不好：LLM 裁判 + 人工标注校准，先拿 10–20 条对齐再当参考值

- [x] 全局学习进度、刷题位置记忆与错题本 —— 1.8.0 / 1.9.0 已交付

***

## 设计取舍

- **题库用历年真题，不用模型自拟题** —— 复习阶段刷真题对考试最有效；模型自拟的题没有权威答案，
  批改就变成自己给自己打分

- **状态放在模型之外** —— 出题结果、题图、学习进度都存在显式状态和服务端账本里，不指望模型从对话里记住；
  换个窗口打开，能接着上次的位置

- **课程问题强制先检索，一切以课件为准** —— 答案以本课程课件为来源，而不是模型的通用记忆；
  课件不够退到课本，仍不足才联网

为什么同时保留 Plan-and-Execute 和 ReAct 两条图、为什么冻结 `frozen/tutor_agent.py` 这个对照实现，
以及踩过的坑（逐字流式在生产里从未生效、题库加载少了一半、Plan 图编造题目），见
[docs/design-decisions.md](docs/design-decisions.md)；评测细节见 [eval/README.md](eval/README.md)。 
