# 架构与代码结构

> 这是 [README](../README.md) 的展开版：README 只留一张图和"一次请求怎么走"，模块级细节在这里。
> 路由清单与 `tests/test_route_surface.py` 的 `EXPECTED_ROUTES` 必须一致（26 条）——改路由要同时改测试与本页。

![架构总览](screenshots/architecture.png)

## 一次提问怎么走

1. `POST /qa/stream` 收 `query` + `mode`（`plan` / `react`），选图。
2. **planner**：先出执行计划。课程问题过 guard，强制把检索工具排进计划 —— 不许凭模型记忆作答。
3. **executor**：按计划调 `course_agent/tools.py` 里的 17 个 `@tool`。业务逻辑不下沉在这里：
   刷题在 `quiz_service.py`，检索在 `rag_tool.py`，题库读取在 `qbank.py`。
4. **synthesizer**：把工具结果写成答案。SSE 侧按 `start` → `node`（节点进度）→ `content`（逐字正文）→ `done`，
   异常走 `error`。
5. **状态**：两张图各一套 SQLite checkpointer；学习进度另记 JSON 账本，跨会话、跨窗口沿用。

## 两条数据流水线

构建**离线**跑（CLI），服务只读产物。两条线互不相干：知识库给"讲、查"用，题库给"出题、批改"用。

### 知识库（向量）

1. **分类**：`doc_utils._classify_doc` 按文件所在路径判 `ppt` / `textbook` / `notes`；`--skip-textbook` 直接跳过课本。
   章节由 `_extract_chapter` 从文件名推断，入库时作为 `chapter` 元数据（`pipeline` 分支一度漏传这个字段，
   导致章节过滤与章节列表全空，已修）。
2. **处理**（只有 `--pipeline` 走这一步，否则就是 `load_document` + `clean_text`）：
   - PPT：`ppt_processor` 按页抽文本 → 每页渲染成 PNG → `--ppt-vlm` 时交给 VLM 读图文结构（箭头、流程图、
     状态转换这类"文字层拿不到"的信息）→ 文本与描述按页合并
   - 课本：`textbook_processor` 按章节拆 → 每章送 LLM 生成结构化 Markdown 笔记（可落盘缓存，重跑直接复用）
3. **分块**：`ChunkConfig(chunk_size=300, overlap=50, strategy=fixed|paragraph)`。
   `overlap` 必须 `< chunk_size`，否则游标不前进（`--chunk-size 100 --overlap 100` 可复现，现已钳制并告警）。
4. **入库**：`get_embeddings_batch`（`config.EMBEDDING_BATCH_SIZE=20`）→ Chroma 持久化集合，`hnsw:space=cosine`。
   metadata 只有 `source` / `doc_type` / `chapter` —— **课程没有进 metadata**，靠文件名反查（Roadmap 里要补）。
5. **增量**：`vector_store/processed_files.json` 记每个文件的 md5，未变就跳过；`--add-file` 只处理一个文件。
6. **读**：`/kb/stats` 出文档数、分块数、类型与章节分布。检索的两个坑（L2 距离要换算成余弦相似度、
   归一化后的融合分不能当"有没有相关内容"的阈值）写在 `rag_tool.py` 顶部注释里。

### 题库（结构化）

1. **定位**：`locate_questions(doc)` 用 PyMuPDF 按题号正则找每题的边界，`is_comprehensive` 区分选择/综合题。
2. **裁图**：每题渲染一张 PNG（保留排版、图表、公式）→ `data/question_bank/images/`。
3. **理解与作答**：按题型分派 —— 有图表或版式复杂时先 VLM 描述、再让 LLM 写答案；纯文字题 LLM 直接写；
   拿不准的题打 `needs_review` + `review_reason`。整卷是扫描图（无文本层）时走 `_process_image_only_pdf`
   的整页 VLM 兜底。
4. **落盘**：`questions_<卷名>.json`，字段 `qid` / `q_num` / `display_num` / `type` / `course` / `page` / `image` /
   `text` / `answer` / `reference_points` / `topic` / `sub_questions` / `section` / `source` / `needs_review` / `images`。
5. **读**：`qbank.py` 只扫 `data/question_bank/questions_*.json`，按 `(source, qid)` 去重合并；题号用结构化的
   `q_num`，不再从 `qid` 里正则抠数字（`2026OS_Q1` 会被抠成 2026）。`data/question_bank/` 是保留目录，
   `courses.py` 不会把它当成一门课。
6. **出题要剥掉题干与答案**：出题轮的返回值只含题号 / 题型 / 知识点 / 图片，`text` 与 `answer` 不下发，
   `quiz_no_leak` 用例守着这条。分组固定为单选/多选/判断 5 题一组、综合题 1 题一组，不让模型去数。

---

## 代码结构

| 层 | 位置 | 负责什么 |
|---|---|---|
| 入口 | `course_agent/server.py` | HTTP 入口：26 个路由、SSE 推流、中间件挂载、静态页 |
| 编排 | `course_agent/graph.py` | 两张图的组装与 checkpointer（Plan / ReAct 各一套库文件） |
| | `course_agent/state.py` | `AgentState` 定义 |
| | `course_agent/nodes.py` | 节点实现：planner / executor / synthesizer / llm / tool，以及各类行为守卫 |
| 业务 | `course_agent/quiz_service.py` | 刷题业务的唯一实现：取题、分组、批改、讲题、按知识点抽题 |
| | `course_agent/tools.py` | 17 个 `@tool` 的薄适配；业务逻辑一律下沉到 `quiz_service` |
| | `course_agent/qbank.py` | 题库读取层：扫描 `questions_*.json`、按课程索引、解析答案与分组 |
| | `course_agent/question_bank.py` | 离线题库构建：PDF → 题号定位 → 裁图 → VLM/LLM → 结构化 JSON |
| 状态 | `course_agent/progress.py` | 学习进度账本：书签、刷题位置、错题，跨会话与跨窗口 |
| | `course_agent/memory.py` | 上下文治理：滑动窗口、摘要压缩、摘要缓存按消息 id 摊还 |
| | `course_agent/courses.py` | 课程注册表：课程名来自 `data/*/` 目录，加课不改代码 |
| 检索 | `rag_tool.py` | 全项目唯一检索入口：混合检索、rerank、查询改写、两级回退 |
| | `doc_utils.py` | 文档加载 / 清洗 / 分类（ppt · textbook · notes）、章节推断 |
| | `build_knowledge.py` | 建库 CLI：增量（md5 跳过未变）/ 全量 / 单文件 |
| | `knowledge_pipeline/` | 离线流水线：PPT 按页 + 课本分块 + VLM 图表理解 |
| 横切 | `course_agent/llm.py` | 智谱 HTTP 直调 + `SimpleLLM`（`BaseChatModel` 子类）+ 流式 |
| | `course_agent/auth.py` · `middleware.py` · `logger.py` · `errors.py` | 鉴权 / 限流与 trace_id / 结构化日志 / 三层错误模型与退避 |
| 前端 | `course_agent/static/` | `chat.html` / `kb.html`（原生 JS，零框架） |
| 对照 | `frozen/tutor_agent.py` | 冻结的手写 tool-calling 循环（对照实现，不参与运行） |
| 测试 | `tests/` · `eval/` · `demo/` | 单元/契约/守卫（全离线）· 行为评测用例与归档报告 · 演示脚本与数据集 |

### 层间的一个反向依赖（如实记录）

根目录的 `config.py` / `doc_utils.py` / `rag_tool.py` / `build_knowledge.py` 是服务启动前就存在的**库层与 CLI 入口**，
`course_agent/` 是**服务层**。方向基本单向（服务 → 库），**唯一例外**是 `rag_tool.py` 反过来用
`course_agent/errors.py` 的 `retry_on_rate_limit` 和 `course_agent/logger.py`。

已知的环 `rag_tool ↔ knowledge_pipeline` 已经断开，由 `tests/test_dependency_direction.py` 守着：
`doc_utils.py` 只许依赖标准库，先 import 哪一侧都不许炸。

## 对外接口

对表面很窄 —— 一共 **6 组**端点、26 个路由（含 `/`、`/chat`、`/kb` 三个页面），
与 `tests/test_route_surface.py::EXPECTED_ROUTES` 一一对应：

| 组 | 端点 | 说明 |
|---|---|---|
| 问答 | `POST /qa/stream` · `POST /qa/ask` | 流式 / 非流式 |
| 知识库 | `GET·POST·DELETE /kb/*` | 统计 / 文档 / 检索测试 / 增量导入 / 上传 / 浏览 / 删除 / 课程注册表 |
| 题库 | `GET·POST /qb/*` | 列表 / 搜索 / 知识点与来源 |
| 会话 | `GET·DELETE /chat/conversations*` · `POST /chat/export` | 列表 / 详情 / 删除 / 导出 |
| 认证 | `POST /auth/login` | 换 JWT（默认关闭，本地免登录） |
| 运维 | `GET /health` · `GET /progress` | 探活 / 只读进度快照 |

> 这份清单由测试守着：删路由、加路由都必须显式改 `EXPECTED_ROUTES`，否则 CI 红。
> 之所以加这道守卫：1.1.0 合并流式端点时整段替换误删了 `POST /tutor/stream`，此后四个版本没人发现 ——
> 界面上没有入口、测试也不走 HTTP，于是文档和代码一直一起保持着那个错误说法。
