# 变更日志

本文件记录每个**已发布版本**的面向使用者变更。

- 格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)
- 版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)：`1.1.0` 加功能，`1.0.1` 只修 bug，架构级改动才动 `2.0.0`
- 版本号的**唯一真源**是 `config.py` 的 `APP_VERSION`，与 git tag、GitHub Release 三者对齐
- 本文件**只写"变了什么"**。为什么这么改、当时怎么判断的、踩了什么坑、怎么验证的，
  属于复盘，另有一批文档在 `docs/` 里 —— 但它们目前**不随仓库分发**
  （原因见 [docs/README.md](docs/README.md)）。
  两者刻意不重复 —— 同一个事实写两遍，最后必然有一处过期。

## [未发布]

（下一次改动先写在这里，发版时移到下面并补上版本号与日期）

## [1.4.1] - 2026-10-13

一轮"自我清洁"：清掉 1.2.0 把 checkpointer 换成落盘 SQLite 时带出来的两处遗留。

### 修复

- **测试会往真实会话库里写垃圾**（1.2.0 引入的）。`test_graph_isolation.py` 那两条
  "写入再读回"的测试用的是**模块级** `graph` / `react_graph` —— 也就是真实的库文件。
  于是每跑一次测试，用户的会话列表里就多两条 `isolation-test-plan-write` 这种会话。
  还是 `MemorySaver` 的年代，这些写入只落在进程内存里、重启就没了；**落盘之后就持久化了**，
  而这一点当时没被意识到。
  → 改成注入临时库 saver（`tmp_graphs` 夹具），覆盖不变。

- **加了一条会话级守卫**（`tests/conftest.py`）：整轮测试跑完时，真实的
  `runtime/checkpoints_*.sqlite3` 里**不能多出任何 thread**，否则报错并列出泄漏的
  thread_id。变异验证：把那两条测试改回用模块级图 → 守卫立刻报出泄漏的两条。
  守卫**只读**打开被观察的库 —— 它自己不改动它观察的东西。

### 变更

- **黑名单自我清洁**：删会话时先记 `dismissedThreads`（万一服务端删失败，这是保险），
  等服务端**确认**删掉后把这条收回去 —— 那个 thread 已经不存在，不可能再被导回来。
  之前它只增不减。反例也守住：服务端没删成功时，这条记录**必须留着**。

### 清理

- 删掉 `runtime/plan_cursors.json`（1.0.x 的游标落盘副本；1.2.0 之后已无任何代码读写它）。
- 用 `DELETE /chat/conversations/...` 清掉会话库里已有的两条测试垃圾。

### 验证

- Python **631 passed**（含新守卫）；前端 auth_block 52/0、chat_context 14/0、
  server_conversations **35/0**（原 31，新增自我清洁 4 条）。
- 两处变异验证：测试污染守卫、自我清洁回收，都正确变红。

## [1.4.0] - 2026-10-13

**会话删除做成端到端的**：前端删掉一条会话，服务端那份也真的清掉 ——
这同时补上了 checkpoint 的清理策略（它是 append-only 的）。

### 变更

- 新增 **`DELETE /chat/conversations/{thread_id}?mode=plan|react`**：按 thread
  整条清掉 `checkpoints` 与 `writes` 两张表（用 `SqliteSaver.delete_thread`，
  实测 6 行 → 3 行）。已不存在时返回 404。

  **为什么这就算清理策略**：checkpoint 每轮对话都会多一行，长期用下去库会一直涨。
  `delete_thread` 让"用户删掉不要的会话"本身就完成回收 —— 这个个人项目
  不必再加定时清理任务。

- 前端 `deleteChat` 会**同时**删服务端，但**不等它返回**（删会话不该被网络卡住）：
  - 先把 thread_id 记进 `dismissedThreads` —— 万一服务端删失败，下次加载也不会导回来；
  - 网络失败静默降级：本地照删；
  - 404 视为"服务端已经干净了"（本来就没有）；
  - 没有 thread 的本地对话（还没聊过）不发请求。

### 测试

- `test_conversations.py` 增加删除端点 5 条：删干净（checkpoint 归零 + 列表消失 +
  详情 404）、**不误删别的会话**、未知 thread 是 404、**模式不对必须 404 且不能真删**、
  重复删除是 404。
- `tests/js/server_conversations_test.mjs` 扩到 31 条，新增删除路径：本地删 +
  发出 DELETE + 带对 mode、网络失败仍本地删掉且记进 dismissed、没有 thread 不发请求、
  404 视为已干净。
  **变异验证**：去掉 `deleteChat` 里的服务端删除调用 → 2 条断言变红；
  把导入的 `reasoningMode` 写死成 plan → 1 条变红。
- `test_auth.py` 的"必须鉴权"清单加上 DELETE。
- 全量 **631 passed**（原 625）；JS：auth_block 52/0、chat_context 14/0、
  server_conversations 31/0。

## [1.3.0] - 2026-10-13

**把服务端会话真正用起来**：前端现在能取回"服务端存着、本地没有"的会话 ——
换设备、清缓存之后不再等于会话全丢。

### 变更

- 新增 **`GET /chat/conversations/{thread_id}?mode=plan|react`**：取回一个会话里
  **适合展示**的消息，供前端恢复历史。只返回"学生问的"和"模型答的"，
  三类一律过滤：
  - `SystemMessage`（含压缩摘要 `SUMMARY_MARK`）—— 内部产物，不是谁的原话；
  - `ToolMessage` —— 检索原文 / 参考答案 / 批改 JSON，**正是要藏起来的东西**；
  - 带 `tool_calls` 的 `AIMessage` —— 内部协议声明（ReAct 的中间轮也是）。

  `mode` 必须传对：读错图返回 **404**，而不是静默给一份空历史
  （1.0.2 修的就是这类"静默失忆"）。

- 前端启动时导入服务端会话（`importServerConversations`）：把"服务端有、本地没有"的
  补进 `chats`，之后就走原来的 `loadChat` 路径（threadId 与 reasoningMode 都在，
  能直接接着聊）。刻意做得很保守：
  - 拿不到就静默放弃 —— 离线时本地功能必须照旧可用；
  - 单条详情失败不影响其它会话；
  - 本地已有同 thread 的跳过（连详情都不请求）；
  - 没有可展示消息的不导入（避免导入出一条空对话）；
  - 正在流式回答时不发任何请求。

- 侧栏用 `☁` 标出从服务端导入的会话。
- **删掉的会话不再被导回来**：`dismissed_threads` 记一笔 thread_id ——
  否则用户删了它、下次加载又出现，看起来像"删不掉"。

### 测试

- `tests/test_conversations.py` 增加详情端点的 7 条，其中最关键的一条是
  **工具结果绝不通过这个接口回传**（检索原文 / 参考答案 / 批改 JSON）。
- 新增 `tests/js/server_conversations_test.mjs`（18 条）：读**真实页面**抽函数进 vm，
  桩 `fetch`/`localStorage`，覆盖正常导入、不重复导、删过的不再导、网络失败静默放弃、
  单条失败不影响其它、空消息跳过、流式中不打扰。
  **变异验证**：把导入时的 `threadId` 去掉 → 2 条断言变红。
- `test_frontend_auth.py` 接入这个 node 测试，CI 会自动跑到。
- 全量 **625 passed**（原 617）；JS：auth_block 52/0、chat_context 14/0、新增 18/0。

### 顺手修掉测试工具里的一个坑

`tests/js/*.mjs` 里"按花括号抽函数"的辅助函数**会把 `async` 丢掉** ——
抽出来的函数体里有 `await`，于是直接
`SyntaxError: await is only valid in async functions`。
改成保留 `async` 前缀。（另外两个文件抽的都是同步函数，所以之前一直没暴露。）

## [1.2.0] - 2026-10-13

把 checkpointer 换成落盘的 SQLite —— **会话状态第一次活过进程重启**，
并在此基础上加了服务端会话列表。顺带删掉 1.0.x 那套"给游标单独落盘"的补丁。

### 变更

- **`MemorySaver` → `SqliteSaver`**（`langgraph-checkpoint-sqlite==3.0.3`）。
  state 现在写在 `runtime/checkpoints_plan.sqlite3` / `checkpoints_react.sqlite3`
  （两张图**各一个库文件**，沿用"防止跨模式串味"的隔离），`runtime/` 已被 gitignore。
  两个实现细节：`check_same_thread=False`（LangGraph 在别的线程跑节点、FastAPI 也会并发）、
  模块级长连接（不是每次都开）。

  ⚠️ 版本必须配得对：**3.1.x 要求 `langgraph-checkpoint>=4.3.0`，而 langgraph 1.2.11
  带的是 4.2.0**，装上会在 **import 时**就炸
  （`ImportError: cannot import name 'writes_sort_key'`）。3.0.3 的约束是 `>=3,<5.0.0`，
  与 4.2.0 相容。这条写进 requirements.txt 的注释了。

- **删掉 `runtime/plan_cursors.json` 那一整套**（`CURSOR_FILE` / `_save_cursor` /
  `_load_cursor` / `_thread_id`）。游标只存在 `state["session"]`，由 checkpointer
  落盘保管 —— **真源从两个变回一个**，1.0.1 修过的那类"读写不对称"从根上消失了。
  `state.py` 的注释也从"这是补偿"改写成"这是设计"。

- 新增 **`GET /chat/conversations`**：列出服务端保存的会话（thread_id / mode /
  标题 / checkpoint 数 / 更新时间），最近的在前。鉴权走
  `AuthGuardMiddleware` 的默认拒绝，并已加进 `test_auth.py` 的"必须鉴权"清单
  （它会吐出每个会话的第一句提问）。前端目前仍把会话列表存在 localStorage，
  改用这个接口是下一步。

- `checkpoint_id → 时间` 的换算：它是 **UUIDv6**，前 48 位是 60 位时间戳（1582 起、
  100 纳秒）的高位。实测三个相隔 2.5 秒写入的 checkpoint，解出来与实际写入时间
  **差 0.0 秒**；解析失败返回 `None`，排序仍由 checkpoint_id 自身的单调性保证。

### 测试

- 新增 `tests/test_conversations.py`（12 条）：用临时库替换模块级 checkpointer
  （**不碰真实的 runtime/ 数据**），覆盖标题/分区/排序/limit/端点形状与参数校验；
  并用**真实 checkpointer 写出来的 id** 验证时间解码（不手造样本）。
- `test_reported_issues.py` 的游标那组重写：核心换成
  **模拟重启**（同一个库文件、新 saver、新图 → state 必须还在）——
  这正是 MemorySaver 时代会拿到空 state 的场景。
  三个只对磁盘副本有意义的用例（round-trip / 未知 thread / 坏 JSON）删掉。
- `test_graph_isolation.py`：除了"两个 checkpointer 不是同一个对象"，
  再加一条**必须落在不同库文件**上 —— 否则绕一圈还是会撞。
- 全量 **617 passed**（原 604）；前端 auth_block 52/0、chat_context 14/0。

### 已知取舍

- checkpoint 是 **append-only**：每轮对话都会多一行，长期跑需要清理策略
  （`SqliteSaver` 自带 `delete_thread`，暂未接）。
- 会话标题取"最新 state 里的第一条用户提问"，被摘要压缩过之后会往后挪
  （要拿最初那一问得逐条扫历史，代价不值得）。

## [1.1.0] - 2026-10-13

合并两个重复的流式端点，并把正文事件的判据从"类型黑名单"改成"来源白名单"。
**顺带堵住两个实测确认的泄漏** —— 它们本来就要在合并之后才变成线上行为。

### 变更

- **合并两个流式端点**：`/chat/stream` 是 `/qa/stream` 的逐行复制品（约 100 行），
  已删除。live 前端本来调的就是 `/qa/stream`（`chat.html`），所以**前端一行没改**；
  而 `/qa/stream` 换成 `updates + messages` 之后，live 路径**白得了逐字流式** ——
  原来它用默认的 `values` 模式，答案只在 `done` 里一次性给出。
  进度事件抽出成 `_node_progress_events` 一份实现（原来两个端点各写一份）。
- `/qa/stream` 的 `content`（逐字）与 `done.answer`（权威值）分工写进 docstring：
  前者是打字机效果，后者保证"模型不流式时答案也一定能到"。

### 修复（两个实测确认的泄漏）

正文事件的判据原来是**类型黑名单**（ToolMessage / 带 tool_calls / 空 content 之外
一律放行）。实测发现它挡不住两件事：

- **节点内部的 LLM 调用也会被推出去。** messages 模式会把图内**每一次** LLM 调用
  都推给前端，包括 `compress_history` 在 planner/synthesizer **内部**跑的对话摘要。
  它带着**同一个** `langgraph_node`，所以按节点做白名单也挡不住；给那次调用加
  `stream=False` 也没用（只是从 token 流变成一条整消息，消息照样在流里）。
  → 现在给那次摘要调用打 `INTERNAL_LLM_TAG`，流层据此丢弃。
- **节点返回的完整 AIMessage 会被再推一遍。** `synthesizer_node` 返回的正是一条
  完整 AIMessage，放行就会让**答案在气泡里出现两遍**（token 一遍 + 整条一遍）。
  → 现在只放行 `AIMessageChunk`。

⚠️ 这两个漏洞在旧的 `/chat/stream` 上**一直存在**，只是因为 live 前端调的是
`values` 模式的 `/qa/stream` 而没暴露。合并端点会让它变成线上行为，所以必须先修。

新的四层判据（顺序即优先级）：
① 来源白名单 `metadata["langgraph_node"]`（未知来源**不放行**，fail closed）；
② 内部 tag；③ 只放行 `AIMessageChunk`；④ content 非空。

### 测试

- 新增 `tests/test_stream_contract.py`：用假模型搭出「planner + synthesizer
  （内部含一次打了 tag 的摘要调用）」的**真实 LangGraph 图**，把每个 chunk 喂给
  过滤函数，端到端断言用户最终看到的正文**只有答案** —— 没有摘要、没有计划 JSON、
  且答案只出现一次。含一条**反向验证**（关掉来源白名单则计划必然泄漏），
  否则前面几条可能只是碰巧通过。
- 新增：断言摘要调用**确实**打了 tag（用 `_FakeLLM.with_config` 做行为断言）。
- `TestSseContentEvent` 按新契约重写：覆盖四层判据与"按图区分答案节点"。
- 修正我自己 1.0.1 写的那条接线守卫（1.0.2 审计的遗留项）：改用 **AST** 只数真实
  调用与切片 —— 原写法枚举 `content[:120]` / `content[:100]` 两个字面量，
  换成 `content[:150]` 就漏。**改完当场翻车一次**：第一版用 `src.count(...)`，
  被我自己在注释里提了一句就撑到 3，正是刚审完的那个毛病。
- `tests/locustfile.py` 的压测目标从 `/chat/stream` 改为 `/qa/stream`。

### 验证

Python **605 passed**（原 595 + 新契约测试 3 条 + 契约/守卫相关新增）；
前端 auth_block 52/0、chat_context 14/0。

## [1.0.2] - 2026-10-13

补齐 1.0.1 那个只修了一半的缺陷；另做了一次"守卫本身失效"的审计并重写三条。

### 修复

- **刷新后 react 会话仍然失忆**（1.0.1 只修了一半）：`thread_id` 持久化了，但
  **推理模式**没有。模式决定那个 thread 属于**哪张图**（plan / react 后端各有独立
  checkpointer），刷新后模式回到默认 `plan`，react 会话的 `thread_id` 在 plan 图里
  根本不存在 —— 后端**静默**当成新会话（不报错）：界面显示着历史，模型却一无所知。
  现在模式随 `localStorage` 持久化，并记进每条对话（`chat.reasoningMode`）；
  `loadChat` 时先恢复模式、再恢复 thread。

### 测试 —— 一次"守卫自己失效"的审计

起因：1.0.1 里发现 `assert "def find_chapter" not in src` 挡不住 `def _find_chapter`
（多一个下划线，字符串就不匹配）。于是把仓库里所有"读源码做字符串断言"的守卫过了一遍，
并用**变异测试**（故意植入它该拦住的违规，看它还绿不绿）逐条验证。

三条**能被绕过**的，已重写：

| 守卫 | 原来的写法 | 实测（全量测试） | 现在 |
|---|---|---|---|
| `test_memory.py` 历史口径一致 | 函数源码里找 `not isinstance(m, ToolMessage)` | 删掉真实过滤、只在注释里留那句话 → **594 条仍全绿** | 行为断言：看 `compress_history` 实际收到哪些消息 |
| `test_study_mode.py` 无状态机 | 取函数**前 1200 字符**（函数实际 3415 字符） | 在距函数头 3000 字符处塞 `stage`/`step_index` → **仍全绿** | AST：取该函数的标识符与字符串常量（注释不参与 AST） |
| `test_qb_single_source.py` 豁免表唯一 | 整文件 `count("QUIET_PATHS") >= 3` | 改回内联表（漏 `/health`，复现原始 bug）＋两句注释 → 单跑仍绿（整文件跑时被行为测试兜住） | AST：顶层赋值恰好一处 ＋ 两个消费方都引用它 |

两条**方向相反**的脆弱点（只会假红、不会放过违规）评估后**保留不动**：
`test_frontend_auth.py` 里那组禁用原生弹窗的断言查的是带注释的原文、
`test_chat_export.py` 用了 3000 字符窗口 —— 两者的失败信息都会自己说明"测试需要更新"，
改动它们的收益小于扰动风险。另确认 `test_dependency_direction.py`（用 `ast`）与
`tests/js/auth_block_test.mjs`（真的执行抽出来的代码 + DOM 桩）本来就没问题。

### 新增

- `tests/js/chat_context_test.mjs`：前端对话上下文持久化的**行为测试**
  （vm + 桩 `localStorage`/DOM，跑真实页面的 `rememberThreadId` / `loadChat` /
  `createNewChat`），14 项断言。同样用变异验证过：把持久化那行删掉会红两条。
- `test_frontend_auth.py` 里那条"切模式要重置 thread"的守卫改用**花括号配对**取函数体
  （原来用正则 `\{(.*?)\n\}`，函数里出现顶格的内层 `}` 就会提前截断）。

## [1.0.1] - 2026-10-12

一轮代码审查后的修复：4 个真实缺陷 + 一处重复实现清理。都不改变对外接口。

### 修复

- **进度事件不再回显工具返回内容**（共 5 处：`/qa/ask`、`/qa/stream` 的 executor 与
  tools 两个分支、`/chat/stream` 的同两处）。原来都是 `content[:120]` 原样截断，
  会把三类不该出现在「思维过程」面板里的东西播出去：课件原文（PDF 抽出来的乱文字）、
  出题阶段的题面、批改结果里的参考答案。现在统一走 `quiz_service.tool_summary`，
  只给语义标签 —— 手写助教侧本来就是这么做的，这次是把它下沉成共享实现。
  顺带修掉 `/qa/ask` 里连工具名都丢了（写死 `"tool": "executed"`）的问题。
- **BOT 侧章节纠偏失效**：`tools.py` 里 `find_chapter` 有一份旧副本，
  不带中英对照表（`传输层 → transportlayer`）。模型把章名填进 `course` 时
  （实测会发生），BOT 侧纠不回「计算机网络」而助教侧能纠 ——
  书签的课程字段留错，两头的学习记录对不上。改为调用 `quiz_service.find_chapter`。
- **重启后只更新学习书签会抹掉磁盘上的刷题游标**：`executor_node` 组装新 session 时
  合并基线只取 `state["session"]`（重启后为空），而 `_save_cursor` 是整条替换 ——
  于是磁盘上的卷名/组号被清掉，「继续下一组」随即失效（正是这套落盘机制本来要修的症状）。
  改为与读取口径一致：state 优先、磁盘兜底。
- **前端不持久化 `thread_id`**：刷新页面或切换对话再切回来，前端历史还在，
  但请求不带 `thread_id`，后端开新会话 —— 界面上有上下文、模型却一无所知。
  现在 `thread_id` 存在对话对象里并随 `localStorage` 持久化，`loadChat` 时恢复。

### 变更

- **文档移出仓库**：`docs/` 原有的文档与报告（项目总览、架构审查与代码阅读地图、
  缺陷与修复报告、方法论、案例复盘、题库工程、面试自测清单、部署指南等 **18 个文件**）
  已移到仓库外暂存，不再随仓库分发。原因：这批文档多为 **AI 辅助整理成稿**，
  需要逐篇核实结论与措辞后再决定哪些放回。已核实的内容会逐步迁回；
  期间**以代码与测试为准**（见 [docs/README.md](docs/README.md)）。
- **清掉 `tools.py` 里的重复实现与死代码**：`_as_int`（零引用）、
  `_course_of`（零引用）、`_course_ppt_list`（只被那份旧副本用）、`_find_chapter`。
  该文件 619 → 490 行，回到它自己 docstring 里写的"只做薄适配"。
- **删掉前端 `case 'token'` 死分支**：全仓库没有任何地方发出 `type: "token"`，
  而且它和 `case 'content'` 的代码逐行相同。

### 测试

- 新增 11 条守卫：两头纠偏一致性（参数化 3 例）、"tools.py 不许自己实现章节查找"、
  重启后游标不被抹掉、进度摘要不回显内容（参数化 3 例 + 三种降级路径）、
  以及两个端点的接线检查。
- 加固一条**本来失效的守卫**：`assert "def find_chapter" not in src` 挡不住
  `def _find_chapter`（多一个下划线字符串就不匹配），正是它让上面那条分叉长期没被发现。

## [1.0.0] - 2026-10-12

首个基线版本。从一个 SSE 聊天 Demo，长成带**学习/复习模式、题库、可观测性与 CI** 的完整应用。

### 新增

**学习 / 复习模式**

- 按课件分节讲 → 讲完小测 → 逐题批改 → 跨轮记进度
- 学习与复习在讲解粒度、检索范围上有明确差异
- 分节结构取自课件文件名自带的信息（`02-1-进程与线程之进程.pdf`、`Chapter5-Transport Layer.pdf`），
  **一节讲多少由模型按内容判断**，不写死粒度
- 「继续讲」接着读同一份课件，不跳到下一份

**手写助教**（`course_agent/tutor_agent.py`）

- 一条不依赖框架 Agent 的独立教学链路
- 与 BOT 共用同一套分节与定位实现（`list_course_sections` / `read_section` / `record_study_position`），
  两个运行面能力对齐

**题库**

- 6 卷 125 题的构建、清洗与结构化（`data/question_bank/`），含题目图片
- 按卷出题、逐题批改、练习计数

**检索**

- 混合检索：向量 + BM25 融合
- 两级检索：先搜课件/笔记，不足再补课本
- 按课程过滤 + **绝对相似度**判定，避免两门课串味

**服务与界面**

- RAG 问答 API、多工具 Agent API（FastAPI + SSE 流式）
- 知识库管理 Web 界面（`/kb`）：上传、增量导入、检索查看
- 学习 / 复习界面（`/chat`）
- 对话导出：服务端落盘并返回真实路径

**可观测与部署**

- trace_id 全链路追踪
- 对话记忆：滑动窗口裁剪 + 摘要压缩
- Docker / docker-compose、部署文档、压测报告

### 修复

**安全**

- `/auth/login` 原先**从不读取 password** —— 任意用户名 + 任意密码都能换到 token，
  于是「需要鉴权」的写接口（`/kb/add` 等）对匿名者等于不设防。现在必须匹配 `ADMIN_PASSWORD`
- 未配置 `ADMIN_PASSWORD` / `JWT_SECRET` 时登录 **fail closed 返回 503**，不再无条件签发
- `/kb/add` 的路径校验（`resolve_within_data()`）与鉴权**解耦**：关掉鉴权后，`data/` 之外的文件仍被 403 拒绝
- `/health` 纳入限流与请求日志的豁免表。原先两处**各硬编码一份**且都漏了它，健康检查会被 429 判死

**学习流程**

- 讲完不收尾 / 编造图片链接 / 凭记忆讲课（不读课件）—— 三个结构性问题
- 批改失败、假工具名泄漏、幽灵步骤「工具1」
- 复习模式不检索、卷名解析失败、重启后刷题游标丢失
- 状态注入位置不一致，导致 BOT 与手写助教行为不一致（补三条守卫 + 两条路由规则）
- 找不到课件/资料时给出**可自我纠正**的提示，而不是空手回去自由发挥

**工程**

- 「导出对话」提示成功但文件从未生成
- 增量导入的两条路径对同一个文件给出不同 `doc_type`（判定基准统一为 `data/`）
- 跨轮会话缺陷、工具层重复实现

### 变更

- 界面撤掉「LLM助教」模式切换（保留手写助教代码与接口）
- 题库由 191 题精简到 125 题（删掉两份试卷）
- 清理工作区遗留数据与临时文件；`.gitignore` 改为白名单（题库 JSON 入库，图片不入库）

### 工程

- 测试扩展到 28 个文件：学习/复习模式、题库、检索、鉴权、安全、依赖方向、两个运行面的能力对齐等
- 新增 CI（GitHub Actions）：pytest + 前端断言 + 服务探活
- 缺数据的用例统一改为**跳过**（`tests/conftest.py`），不再让「新克隆跑测试」看起来像代码坏了

### 文档

- 建立文档索引（docs/README.md）
- 项目总览、架构审查与代码阅读地图、缺陷与修复报告
- 方法论：怎么优化一个 Agent；Agent 灵活性成因、检查表与瘦身
- 案例复盘：手写助教的完整生命周期、框架版 Agent 悄悄变坏、Self-Refine 尝试与移除
- 题库工程、部署指南、面试自测清单

### 已知限制

- **课件 / 课本 / 向量库（约 300 MB）不入库**：新克隆（含 CI）跑测试会跳过依赖本地数据的用例。
  要看完整结果需本地准备数据并先跑 `build_knowledge.py`
- 题目图片（约 130 MB）同样不入库
- `docs/` 里面试三篇（总览 / 自测清单 / 架构审查）内容有重叠，尚未整理

---

[未发布]: https://github.com/catcatsquid/ai-agent-api/compare/v1.4.1...HEAD
[1.4.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.4.1
[1.4.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.4.0
[1.3.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.3.0
[1.2.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.2.0
[1.1.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.1.0
[1.0.2]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.2
[1.0.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.1
[1.0.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.0
