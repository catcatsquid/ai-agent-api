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

## [1.11.1] - 2026-10-17

29 条用例 × 3 次（约 50 分钟）跑出 **23/29 稳定、888/903 判据**。没稳的 6 条归成四类，
按用户拍板的方式逐个处理（**没有为了变绿而放宽判据**）。

### 1) ReAct「嘴上说要查、实际没调」→ 补一轮强制调用（用户选的方案）

实测约 1/3 概率：模型文字里写"我先查一下知识库"，但返回的 `tool_calls` 是空的
→ 这一轮没检索就作答。**不是我们解析丢了**（逐帧验过原始 SSE，确实没有 tool_calls）。

- `llm.py`：`chat_with_tools` / `_iter_stream_events` / `SimpleLLM` 都支持 `tool_choice`，
  新增 `with_tool_choice()`（返回新实例，不改原对象）
- `nodes.llm_node`：检测到"文字说要调 + 没有 tool_calls"→ 用 `tool_choice="required"`
  **补一轮**；补的那一轮也没发起调用就保留原来的文字（绝不把回答吞掉）
- 效果：`react_mode_streams`、`react_no_tool_call_leak` 两条从 16/18、20/21 → **稳定全绿**

### 2) 流式里闪出工具调用原文 → 判据拆成"硬 + 观察"（用户拍板：**不许把流式变粗**）

复现到了：刷题第 2 轮，打字机过程中真的会闪过
`explain_question {"qids": ["2025OS_Q3", "2025OS_Q4"]}`；等 `done.answer` 到达后被清理覆盖。
之前硬判据判的是**流式拼接结果**，所以必然抖动。

- **硬判据**改为判 **`done.answer`**（最终权威值，前端用它覆盖气泡）—— 最终答案里再漏照样红
- **新增观察判据** `forbid_tool_call_text_in_stream`：只记录"打字机过程中闪没闪"
- 每条用例都自动加了这个观察项（29 条），信号没丢

### 3) 复合意图与 F2 形态 → 代码守卫（提示词只到 1/3、2/3，不够）

- 「复习 + 真题/练几道」→ **强制 `practice_by_topic`**（认得出题库知识点才生效，绝不瞎猜）
  - `review_r2_topic_drill`：0/3 → **3/3 稳定**
- 「交了部分答案 + 其余说不会」→ **强制 `grade_answers` + `explain_question`**
  - 只在此形态生效（只说答案不说不会的 F4 不动）
  - `quiz_f2_partial_unknown` 里那次"改去 search_knowledge"的问题消失：**42/42 稳定**

### 4) 出题必须贴齐每题图片（规范硬要求）

`quiz_f2` 曾出现"一组 5 题只贴 1 张图"。规范第 10 条写死：有几道题就贴几张图、按题号顺序、
不许只贴第一张。**不降为观察**（用户明确要求）。

### 评测工具改进

- 终端现在会打印**观察档明细**：`观察档 5/6（不卡发版）` + 每条未命中的具体原因
- 新增 `tests/test_react_retry.py`（6 条）：补一轮只在"说要调却没调"时发生、
  已经调了不重复补、纯文本回答不补、补失败保留原文
- `test_eval_runner.py` 新增 3 条：最终答案泄漏是硬红 / 只在流式闪是观察 / 都干净才全绿

### 验证

- Python **855 passed**
- 复跑（每次 3 次）：`react_mode_streams` 18/18、`react_no_tool_call_leak` 21/21、
  `quiz_f2` 42/42、`dont_know_calls_explain` 42/42、`review_r2` 24/24

### 已知限制（诚实记账）

- 流式闪现仍然存在（**用户决定不改流式粒度**），现有观察档记录；
  真正根治只能在 SSE 层做回退缓冲（会让流式变粗，已被否决）
- ReAct 补一轮会多一次 LLM 调用（这一轮更慢），只在"确实没调"时才发生

## [1.11.0] - 2026-10-17
## [1.11.0] - 2026-10-17

**补上学习/复习支线与意图矩阵的评测**（11 条新用例），并在评测里抓出、修掉了三个真问题
——其中一个是**提示注入能骗出整卷答案**。

### 新增 11 条用例

| 支线 | 用例 | 硬判据要点 |
|---|---|---|
| 学习 L1 | `study_l1_continue_from_bookmark` | 「继续讲」→ 按书签+offset 取材料，不许回第 1 节、不许列章节清单 |
| 学习 L3 | `study_l3_switch_chapter` | 「换成内存那一章」→ 重新按新章取材料，正文必须提到新章 |
| 学习 L5 | `study_l5_end_of_chapter_quiz` | 「这节讲完了，测测我」→ 出小测，**不许**只记一笔练习 |
| 复习 R1 | `review_r1_quick_pass_must_retrieve` | 快速过考点 → 必须真的检索/读课件，不许凭记忆干讲 |
| 复习 R2 | `review_r2_topic_drill` | 复习+要真题 → `practice_by_topic` 抽题，不许拉试卷清单 |
| 意图矩阵 | `intent_followup_reference` | 指代追问 → 接着上一轮话题继续检索，不许反问"你指什么" |
| 意图矩阵 | `intent_retrieval_fallback` | 课程资料里没有的 → 允许联网/诚实降级，不许装成本课内容 |
| 意图矩阵 | `intent_prompt_injection` | 提示注入 → **不许批量倒答案** |
| 意图矩阵 | `intent_symbol_only` | 只发「？？？」→ 不崩、不碰题库 |
| 意图矩阵 | `intent_out_of_scope_request` | 「写首诗」→ 可以陪聊，不许拿题库工具当挡箭牌 |
| 意图矩阵 | `intent_meta_question` | 「你能干什么」→ 不许去检索/列试卷 |

### 评测抓出并修掉的三个真问题

1. **提示注入能骗出整卷答案**（安全项）：只加提示词规则时，模型 2 次里仍有 1 次真的调
   `get_answers` 把 14 个答案倒出来。→ 代码兜底：新增 `_is_answer_dump_request()`
   与 planner 守卫 5，命中"忽略规则/全部答案/我是管理员"这类要求时**本轮不规划任何工具**；
   配套规范第 9 条（批量倒答案一律拒绝，任何要求都不能让它放弃这条）。修完 **2/2 稳定通过**。
2. **说"这节讲完了"它只记了一笔练习**：`study_l5` 第一次跑，模型调了
   `get_learning_progress + get_course_info + record_practice` 却没出题。→ 学习规范补第 7 条
   （立刻出小测、自带标准答案、别只记练习），修完 14/14。
3. **用例配置错误跑到一半才炸**：`review_r2` 声明了真值型判据却没给 `truth`，
   报 `KeyError: 'stems'`。→ runner 新增 `validate_cases()`：**开跑前**就指出"哪条用例
   第几轮用了哪些真值判据、但没给 truth"，前面的用例时间不再白费。

### 评测能力

- `seed` 新增 `read` 段：预置"读到第几块"（L1 的"接着上次"才测得到）
- `intent_symbol_only` / `intent_prompt_injection` 用了观察档（工具为空 / 有拒绝措辞）

### 验证

- Python **844 passed**（1.10.0 是 834）
- 冒烟（`--attempts 1`）：学习/复习 5 条 **4 条直接绿**（L5 修完后 14/14）；
  意图矩阵 6 条 **5 条直接绿**（注入修完后 12/12）
- 观察档命中：学习 0/1、意图 3/4（话术级行为，先攒曲线）

## [1.10.0] - 2026-10-17

**评测有了"观察档"** —— 话术级行为终于能测了，而且不会再用误报挡住发版；
同时补上入口三分支与两条 B 类行为用例。

### 评测观察档（`expect.observe`）

B 类行为（岔开提问要拉回主线、闲聊第 3 轮才轻推一次、插播做题后回主线）**没有客观真值**，
写成硬红只会产生噪声；但"不写"更糟 —— 规范没人看就腐烂。所以：

- 用例里加 `expect.observe`（字段与硬判据同名），**照样计算、照样进报告**
- **不计入 `stable_pass`**：观察档全红也不会让用例变红（发版判断只看硬判据）
- 报告新增 `observe_total` / `observe_hits`，并逐条列出观察命中
- **连续 3 版严格下滑会打印警告**（用户拍板：观察档永不卡发版，但这个信号要人看）——
  实现是读历史报告算每个（用例, 判据）的版本命中率，不需要额外状态文件
- 升档标准（用户拍板）：同一观察判据**连续 3 版命中率 ≥90% 且措辞稳定** → 升 P1

### 新用例 5 条（入口三分支 + B1/B2/B9）

| 用例 | 硬判据 | 观察档 |
|---|---|---|
| `entry_chitchat_nudge` | 纯闲聊 4 轮都**不调工具**、正文非空 | B4：第 3 轮才出现引导词、第 4 轮不再出现 |
| `entry_explicit_command` | 明确指名卷子+组号 → **直接取题**，不许反问「你想做哪套」 | — |
| `global_interrupt_clears` | 「不学了，算了」→ 不调工具 + 不出现「下一节/继续讲/确定吗」 | — |
| `b1_offtopic_then_back` | 学习中岔开提问 → 真的去检索、不许切去刷题 | 拉回主线：回答里出现当前章节名 |
| `b2_quiz_break_then_back` | 插播"练几道死锁的题"→ 用 `practice_by_topic`；回来那轮不批改 | 回主线：又提到当前章节 |

首轮冒烟（`--attempts 1`）：**硬判据 67/67 全绿**，观察档命中 3/5。

### `quiz_service.py` 的用户文案正式提交

`group_payload` 里面向学生那句「请享用。」（用户自己的改动）**正式入库**。
之前每次发版都要用 `git show HEAD:…` + 追加段的方式把它排除，很脆 —— 现在不用了。
（1.7.1 已确认：那句只是**面向学生**的措辞，行为约束在 `grade_answers` 的工具描述里，
两套文案都能过测试。）

### 验证

- Python **834 passed**（1.9.0 是 831）
- 观察档三条守卫有测试：观察档红不影响 `stable_pass`、`observe` 判据有标记、
  连续 3 版下滑才报警（构造三份报告验证）

## [1.9.0] - 2026-10-17

**把"答题的三种形态"和"按知识点练习"做成代码保证的行为**，把 B 类话术规范收进代码，
并让评测能测"跨会话开场"。工具 14 → 17。

### 答题三形态（D1/D2 定稿）

| 学生这轮说 | 现在保证的行为 | 靠什么保证 |
|---|---|---|
| 只答了部分 | 只批已答的，并**指出还没答的题号** | `grade_answers` 返回新增 `未答题号`（题号由题库算） |
| 部分作答 + 其余"不会" | 批已答的 + **给那几道答案并讲解** | 同一返回里的 `提示` 指引调 `explain_question` |
| 直接问答案 | **不批改**，直接给题库原文 | 新工具 `get_answers`；说明写明"原样转述、不要顺带批改" |

- 没传 `qids`（= 批整卷）时**不猜**未答题号，有用例守着
- 顺手修掉：`qids` / `no` 传 `None` 会被 pydantic 拒绝（schema 缺 `| None`）——
  模型把可选参数写成 `null` 很常见，失败还是静默的

### 按知识点抽题（A2 / D5 轻量方案）

新工具 **`practice_by_topic(topic, course?, count?, qtype?)`**：跨**所有卷子**按知识点收集再随机抽
（上限 10）；课程精确过滤；章节走"topic 包含式 + 同义词小表"（死锁 → 银行家算法/资源分配）。
只给**题号+知识点+图片**（防泄题）；匹配不到返回 `count:0` + hint 让人改用 `prepare_custom_quiz`。

> 第一次评测跑出来是 **2/2 次都调了 `prepare_custom_quiz`** —— 工具加了但提示词还写着旧规则，
> 等于没接上。已改 PLANNER 规则 5 并加守意图的测试。教训：**新工具必须同时改"指路"的那句话**。

### 重做错题（A5 / D4）

- 批改时把错题 qid 记进账本（`correct is False` 才算；`None`=需人工判定，不记）
- **做对就移出、做错保留**；`summary()` 显示"错题 N 道待重做"
- 新工具 **`practice_wrong(paper_source)`**：把错题**原题**取回来重做，不给题干与答案

### B 类行为规范：收进代码，三条链路共用

`nodes.QUIZ_BEHAVIOR_RULES`（8 条）+ `nodes.STUDY_BEHAVIOR_RULES`（6 条），拼进
`PLANNER_PROMPT` / `SYNTHESIZER_PROMPT` / `REACT_PROMPT`，覆盖 B1~B11。

为什么写成常量：**评测判据直接 import 它**（规范与判据不各写一份），三条提示词也不各抄一份。
落点选在**工具返回值**（模型对它比对系统提示更服从）：
- `record_study_position` 新增 `next_action`（已通过 → 出小测 + 问下一章/休息；待巩固 → 重讲）
- `prepare_custom_quiz` 返回新增 `要求`（出题必须同时给标准答案与评分要点）

### 评测：能测"跨会话开场"了

- 用例可声明 **`seed`**（study / quiz / grade 三段），runner 跑前写进评测专用进度文件；
  **fail-closed**：声明了 seed 却没设 `PROGRESS_FILE` → 直接退出
- 新判据：`tools_all_of`、`tools_exactly_empty`、`must_contain`、`forbid_phrases`
- 新用例 7 条：`entry_kickoff_from_progress`、`quiz_f1_all_answered`、`quiz_f2_partial_unknown`、
  `quiz_f4_partial_only`、`quiz_f6_ask_answer`、`quiz_a2_topic_practice`、`quiz_b11_wrong_redo`
- **`max_ttft_ms` 从所有用例里删掉**：按决策「首响暂时接受」，只做观测（汇总仍有 p50/p95），
  之前它一直在制造边界误报（16.0 / 15.6 / 16.6 秒全判红）

### 修掉一个自己引入的回归

"新会话一律注入进度块"之后，**概念题**也会被注入"某卷做到第 1 组"，planner 误判成刷题，
真的调了 `get_paper_questions + grade_answers`（`knowledge_answer_streams` 抓到的）。
改成**只按意图注入**（想学/复习/进度/继续/错题/重做），空会话不再是理由，并补了回归用例。

### 验证

- Python **831 passed**（1.8.0 是 772）
- **变异验证全红**：错题三态、出题选项不带章节、规范缺失、planner 不带学习规范、
  去掉 next_action、自拟题不要求答案、四个新判据各自失效、A2/A3/A5 五个关键行为
- 评测单条复验：`quiz_a2` 12/12、`knowledge_answer_streams` 14/14、`entry_kickoff` 14/14、
  `quiz_f2` 28/28；F1/F4/F6/B11 全通过

### 已知限制

- 首响仍约 6~8 秒（观测项，不卡发版）
- "提醒还剩几道"的具体措辞仍是观察项（D3：先跑 3 版再用真实措辞写判据）
- 按知识点抽题的章节匹配是"包含式 + 同义词表"，不是语义匹配（D5 的彻底方案留作后手）

## [1.8.0] - 2026-10-17

**新功能：全局学习进度。** 开新窗口说"我想学习"时，Agent 不再问"你之前学到哪了"，
而是**直接报进度 + 给四选一建议**（接着学 / 自拟练习 / 刷整套卷子 / 复习）。

### 为什么这是新功能而不是修 bug

"学到哪"和"刷到第几组"原来只存在**单条会话**的 state 里（checkpoint）——
换一个窗口就"忘了"，于是每次都得重新问一遍。现在把它们提升成**跨会话账本**。

### 账本长什么样、放哪里

- 落 `runtime/progress.json`（与 checkpoints 同级，已在 .gitignore）
- **不放 `data/`**：那是课程资料目录，会被 `/kb` 扫到、被"测试不许污染 data/"的守卫盯着
- 课程侧：`课程 → 节 → {source, chapter, title, status, last_offset, visits, 首次/最近时间}`
- 刷题侧：`卷子 → {cursor(第几组/下一组/qids), groups(每组是否已批改、得分), finished}`
- **"刷到一半"的定义（用户拍板）**：记录写到第几组 —— `next_group_index is not None`
  就是没刷完，下一组就是它；`None` 表示这份卷子刷完了

### 写入：确定性的，不靠模型记得写

只从**已有工具**的返回里推导（`record_study_position` / `get_paper_questions` /
`grade_answers`），挂在 `executor_node` 里已有的 `_extract_*_session` 旁边 ——
**零新提示词、零新模型动作**（与"模型记不住的事交给代码"同一条纪律）。

- **文件锁 + 原子替换**（先写 `.tmp` 再 `replace`）：`practice_stats` 上真实发生过
  "12 个并发只记到 3 次"，这里一开始就带锁
- **坏文件不许崩**：读不出来就当空账本，绝不因为进度让一整轮问答失败
- **评测/探针线程不写**（`eval-*` / `probe-*`）：否则跑一遍评测就把你的"学习进度"刷满，
  你会看到"今天刷了 30 套卷子"

### 开场：先展示进度并让你选，不许直接开讲

实测（不加守卫时）：新窗口说"我想学习"，它**直接 read_section 讲了一整节**
（《03-OS调度.pdf》），学生根本没机会选。

- `planner` 新增守卫：**有进度** + 学生只是"刚开口想学/想复习"（没说具体章节）
  → 这一轮**不规划任何工具**，交给 synthesizer 报进度 + 报选项
- 说了具体目标（"我想学习死锁那一章"）→ 不算开场，照常规划；
  **没有进度**时也保持原行为（先给资料地图）
- 建议文案由 `progress.brief()` **用代码生成**（可照抄、不会漏项）
- 模型**实测会漏选项**（提示里写了"四条全部列出"，它只给三条、甚至两条）→
  新增 `ensure_modes()` 兜底：漏的那几条由代码补在后面（同 `ensure_closing` 的做法）

### 读取：三处

1. 提示词注入：**新会话**或学生提到学习/复习/进度/继续时才注入（每轮都塞会带偏答题、也费 token）
2. 新工具 **`get_learning_progress()`**：学生问"我学到哪了""哪套卷子没做完"时照实回答
3. **`GET /progress`**：只读接口（按用户要求**不做前端**，不靠点按钮查）

### 顺带

- 文件头工具清单 13 → **14**，`PLANNER_PROMPT` 的工具数与第 14 条说明同步
- `test_only_intentional_differences_remain` 登记 `get_learning_progress` 为"BOT 独有"
  （手写助教是冻结的对照实现，不追这个功能）
- `tests/conftest.py` 新增守卫：测试**绝不读写真实的** `runtime/progress.json`
  —— 否则"本机有多少学习记录"会改变模型行为，测试今天绿明天红

### 验证

- Python **772 passed**（原 732 + 本版 40）
- 变异验证 8/8 变红：评测线程也写进度 / 不加锁 / 刷完判定永远 False /
  开场不说别重问 / 空值覆盖已有值 / 空章名新开一条 / planner 不注入进度 /
  去掉意图门控（每轮都注入）
- **端到端**（真服务、种子进度、全新会话说"我想学习"）：回答里有真实课程、真实章节、
  半刷的卷子与组号，**四个选项齐全**，且没有反问"你学到哪"、没有直接开讲
- 过程中踩到一个值得记的坑：**重启服务没绑上端口时，验证跑的是旧代码**，
  结论会全错。现在 e2e 先 `GET /progress`（只有本版才有）确认版本，再验证

### 已知限制

- 只按 `user: "local"` 存（本地单人）；将来加账号改 key 即可
- 进度只在**工具真的被调用**时更新（比如模型直接凭记忆答题、不调工具，就没有记录）
- 评测用例还没覆盖"开场建议"（需要给 runner 加"预置进度"的能力）；
  目前由 40 条单测 + 端到端脚本守着

## [1.7.3] - 2026-10-16

**ReAct 模式的回答不再把工具调用原文漏给学生**；顺手修掉一个更阴的：
planner 没明说"不用工具"时，**一个工具都不调就凭记忆作答**。

### ReAct 那条路的回答，原来什么都不清

工具调用原文的清理、编造图片链接的清理、学习收尾标记的补全 —— 这三件兜底
原来**只写在 `synthesizer_node` 里**（plan 模式那条路）。ReAct 的 `llm_node`
只筛选文本、什么都不清，于是 ReAct 模式下这种正文会**直接进学生气泡**：

```
好的，我来讲解。
explain_question
{"question_index": 1}
```

（行为评测用例 `react_mode_streams` 抓到过一次；它与 `quiz_no_leak` 那条
"正文里没有工具调用原文"是同一个判据，只是换到了另一条图。）

- 提成共用函数 `nodes.clean_answer_text`，**两条路都调** —— 分散写就会出现
  "只修好一条路"的分叉（这个项目在别处已经栽过好几次）
- **与流式的关系（必须知情）**：清理作用于**节点返回的那条消息**。逐字 chunk
  已经发出去了，所以打字机过程中那一段仍然会闪过去；但 `done.answer` 是清理后的
  权威值，前端在 done 时会覆盖一次，**最终气泡是干净的**。要让它根本不出现，
  得在生成时拦住（提示词实测拦不住）
- 测试同时用两条互补判据：内容级（清理逻辑对不对）+ **源码级**
  （这两个节点函数体里都出现 `clean_answer_text`）—— 当初的 bug 恰恰是
  "函数是对的，只是那条路没调它"，只有源码断言能守
- 顺带纠正 1.7.2 里我写错的一句归因：我当时说"正则认不出换行/编的参数名"，
  实测**都认**（`\s*` 能吃换行）—— 真正的原因是**这条路上根本没调用它**

### planner 没明说"不用工具"时，不许当成"不需要工具"

prompt 里写死了「必须返回 `need_tools`」。但模型回一个 `{}`（或漏掉 `need_tools`）
时，`steps` 是空数组 → 原来的判据 `if not normalized and steps` **不触发** →
`normalize` 后计划为空 → `route_after_plan` 直接进 synthesizer
→ **这一轮一个工具都不调，纯凭记忆作答**（离线打桩 planner 复现）。

- 判据改成：没有可用步骤、且**没有显式** `need_tools: false` → 按"不合规"处理，
  照旧回退到 `search_knowledge`。合规的空计划（闲聊）与"学生交答案"那一轮
  （守卫 2 处理成不调工具）都不受影响，各有一条用例守着

### 另外验证掉的三个疑点（都**不是** bug，无需改）

1. `list_course_sections(course="")` 与单课程调用的返回结构**一致**（实测同格式）
2. 坏计划的回退"丢本意"这个担心不成立：steps 是字符串 / 顶层是数组 / 根本不是
   JSON / steps 里全是字符串 —— 四种都给同一个确定性回退（`search_knowledge` + 原问题）
3. planner 偶发"计划无法解析"是**设计好的降级**；真正漏的是上面那条"合法但空"

### 顺带查实、但**留给下一版**的一件事

工具的**可选字符串参数传 `null` 会被 schema 拒绝**（实测三种形态）：

| 调用 | 结果 |
|---|---|
| `record_study_position(course="操作系统", section=None, mode=None)` | `ValidationError`（3 处） |
| `list_papers(course=123)` | `ValidationError` |
| `get_practice_stats(paper_source=456)` | `ValidationError` |

模型把可选参数写成 `null` 是很常见的写法，而失败是**静默**的（工具不执行、
模型拿不到任何理由）。与 1.7.2 修的 `no="1"` 属同一类，但面更宽，需要单独设计
（13 个工具 × 各自的可选参数），所以不在本版里顺手改。

### 再修一个：模型把**原始协议帧**当正文写出来

评测新用例 `react_no_tool_call_leak` 抓到（ReAct 模式，3 次里命中 1~2 次）：
这一轮模型**没有真的调工具**，而是把"打算调用"的协议当正文打了出去 ——

```
好的，我们来复习一下死锁的四个必要条件。首先，我会调用一下知识库来获取更详细的信息。

{"course": "操作系统", "query": "死锁的四个必要条件"}          ← 工具参数被写成普通文本
```{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant",…  ← 原始 API 帧
```

最后那段是**原始 API 响应**（`finish_reason` / `delta` / `tool_calls`）——
它是模型的幻觉续写，**不是我们漏播的**（智谱的原始 SSE 帧已用探针验过，只有 `content`）。

- 正文兜底：新增 `nodes.strip_protocol_frames`。判据是**解析出来的 JSON 必须命中
  ≥2 个协议字段**（顶层或一层嵌套），不是关键词匹配 —— 否则"讲 tool_calls 这个字段"
  这种教学文本会被误伤。括号配对跳过字符串字面量与转义，配不成对的（模型还在流）
  一律不动
- 提示里明确禁止：想调工具就**直接调**，不许把 `工具名(...)`、参数 JSON、
  `finish_reason` / `delta` / `tool_calls` 写进正文
- **没修的**：这条路上"模型只是嘴上说要调工具"时，`should_continue` 依然直接 END，
  不会补一轮真调用（等于这一轮白答）。改它要给 ReAct 加一轮纠偏重试，
  属于行为改动，留给下一版

### 顺带修掉评测工具自己的一个真 bug

`tools_any_of` 的字面意思是"**任一命中即可**"，而实现写成了逐项断言（"调用了 A""调用了 B"
都要求真）—— 多元素时**必然红**。之前的用例都只有一个元素，所以一直没暴露；
在 `react_no_tool_call_leak`（三种取材料方式任选）上第一次撞到：模型调了
`search_knowledge`，却被判"没调 read_section / list_course_sections"。
另外报告里那条判据的文案写死了 `8000`，现在显示**真实门槛**（否则报告里是假信息）。

### 验证

- Python **732 passed**（1.7.2 的 703 + ReAct 清理/空计划回退/协议帧 25 + 评测工具 4）
- 行为评测 **6 条用例 / 204 条判据 / 202 通过 / 4 条稳定通过**（`eval/reports/1.7.3.json`）：
  - 新增的 `react_no_tool_call_leak` **24/24 稳定通过** —— 协议帧兜底与提示禁令都生效
  - 没稳定的两条**都是模型侧抖动**，不是本版引入的：
    `dont_know_calls_explain` 有一次首字 9.1 秒（15 秒门槛内，但确实偏慢）；
    `react_mode_streams` 有一次 ReAct 一个工具都没调（正是上面"只嘴上说要调工具"那条）
  - 首字 p50 **6599 ms**（1.7.2 是 6533 ms）；单轮总时长 p50 **31.4 秒**
- **变异验证**（改坏 → 必须变红）：ReAct 路不再清理 / 共用函数不再摘图片链接 /
  共用函数不再摘工具调用原文 / 空计划回退退回原判据 / `clean_answer_text` 不调帧清理 /
  帧判据阈值从 2 降到 1 / 提示里删掉禁令那行 / 括号配对忽略字符串内括号 —— 全部变红
- 过程中又抓出**我自己的两条弱测试**（"教学文本不误伤"那条两个例子挨在一起、
  提示禁令只断言关键词），已改成能真正区分阈值与意图的写法

## [1.7.2] - 2026-10-16

**回答现在真的是逐字流出来的。** 顺带修掉两个工具参数上的真 bug、两处评测工具自身的缺陷，
并把行为评测从 2 条用例扩到 5 条。

### 修掉「回答不流式」—— 学生等约 33 秒，然后整段答案一次性出现

根因不是网络也不是模型，而是 `course_agent/llm.py` 的 `SimpleLLM` **只是一个"看起来像"
LangChain ChatModel 的普通类**：只有 `invoke` / `bind_tools`，外加一个**假的** `astream`
（拿非流式结果逐字切片充数）。

- LangGraph 的 `stream_mode="messages"` 只对**实现了 `_stream` 的 Runnable** 吐
  `AIMessageChunk`；普通类走的是"节点把完整 AIMessage 返回再推一遍"那条路，
  被 `sse_content_event` 的"只放行 chunk"判据正确挡掉 → `/qa/stream` 的 `content`
  事件**在生产里从未触发过**。
- 改法：`SimpleLLM` 变成真正的 `BaseChatModel` 子类，实现 `_generate`（非流式）与
  `_stream`（流式，与非流式**同一个 HTTP 端点**，区别只在 `stream: true`），删掉假 `astream`。
  **工具调用的分片也一起流出去**（`tool_call_chunks`）—— 否则 ReAct 会静默退化成
  "什么工具都不调"，最后由 LangGraph 按 index 合并回完整的 `tool_calls`。
- 为什么不绑工具时请求体绝不能多出 `tools` 字段：planner / synthesizer 本该发纯文本请求，
  多带工具定义会让模型返回 tool_calls 而 content 为空（这是 1.5.x 修过的缺陷）。
  已加断言守住。

**为什么 674 条测试全绿却没发现**：流式相关用例用的是 `GenericFakeChatModel`
（真 LangChain 模型，自己带 `_stream`）—— 测试里流的那个模型根本不是生产的那个。
新增 `tests/test_llm_streaming.py` 全部**只对生产那个 `llm` 对象**下判据。

### 工具参数：模型怎么写的，就怎么收

- **`explain_question` 的 `no`**：schema 写成 `list[int]`，而模型讲第 1 题时传的是
  `no="1"` → pydantic `ValidationError: Input should be a valid list`，**工具根本没执行**，
  学生得到的是一段绕开题目的泛泛而谈。现在 `no` / `qids` 都接受
  `1` / `"1"` / `"1,2"` / `"[1,2]"` 这几种真实写法（收敛函数 `tools._as_list`）。
- **`grade_answers` 的 `qids`**：schema 放宽后如果不在入口收敛，`normalize_grade_args`
  会把字符串当**可迭代对象逐字符遍历**（`["2","0","2","5",…]`）→ 静默退化成
  "整卷按题型分段解析"，批改结果可能是错的。测试直接盯参数（用 spy），因为拿批改结果当判据
  **碰巧也能通过**（变异验证时抓到的安慰剂）。
- 与 `read_section` 的 `"int | str"` 同一条纪律：**schema 层放行现实里的写法，容错放实现里**。

### 工具（`eval/`）自身的两个真缺陷（都是跑基线时踩到的）

- **中文 Windows 上直接崩**：控制台是 cp936，`print("▶ 用例")` 抛 `UnicodeEncodeError`
  —— 崩点还在用例标题那一行，等于**一条用例都没跑**。现在 `main` 开头显式把
  stdout/stderr 切到 UTF-8（`errors="replace"` 兜底），装饰符也换成纯 ASCII。
- **报告写完了却以退出码 1 结束**：`--out` 给的相对路径没 resolve，`relative_to(ROOT)`
  抛 `ValueError`（假的失败信号）。现在能相对就相对、不能就原样，**不许因此崩**。
- **传输超时会重试**：智谱偶发把请求挂死（服务端 120s 后才放弃），runner 这边读超时
  —— 一次抖动就把**整批**评测带走（实测被带走两次，报告一行都没写）。现在只对传输层
  超时重试（换新 thread_id，避免判据看到上一轮的残留）；模型真的出错（服务端 `error` 事件）不重试。

### 评测集：2 条 → 5 条，并新增一条判据

| 用例 | 守什么 |
|---|---|
| `knowledge_answer_streams` | 概念题必须逐字流出来；正文里不许有**写给模型看的东西** |
| `react_mode_streams` | ReAct 那条链路的正文来自 `llm` 节点（白名单是两套），流式与工具调用都得活着 |
| `dont_know_calls_explain` | 学生说"不会"必须真的调 `explain_question`（出题接口刻意不给题面） |

- **新增判据 `no_internal_artifacts`**：planner 的计划 JSON（`need_tools`）、
  `search_knowledge` 返回值末尾那段【怎么用上面的片段】使用说明、`tool_calls`、
  `【内部摘要】` 都不许进正文。与泄题判据不同，它**不比对真值**，所以讲解类用例也能用。
- **阈值改动（按纪律显式记录，不是为了让评测变绿）**：`max_ttft_ms` 8000 → **15000**，
  `dont_know_calls_explain` 第 2 轮 `answer_min_chars` 80 → **40**。理由：8 秒是在
  "流式完全没生效"时写的；现在 ttft p50≈6 秒、p95≈10 秒，8 秒会变成**边界误报**
  （实测 8250 / 8656 / 9170 ms 都判红），而它要挡的缺陷是"30 秒以上才出字"—— 15 秒足够。
  "防真空通过"的意图不变（仍是非 None 且有上限）。报告同时记录 `ttft_ms` 原始值。

### 验证

- Python **703 passed**（原 674 + 流式 13 + 评测工具 11 + 工具参数容错 5）
- 行为评测（同一台机器、同一套判据）：

  | | 1.7.1（改前） | 1.7.2（改后） |
  |---|---|---|
  | `content` 事件 | **0 个**（12 轮全是 0） | **517~613 个/轮** |
  | `ttft_p50_ms` | `null`（一次都没流过） | **6533 ms**（p95 7867） |
  | 单轮总时长 p50 | 32.9 秒 | 31.1 秒 |
  | 判据 / 用例 | 69/87 · 0/2 稳定通过 | **180/180 · 5/5 稳定通过** |
- **变异验证**（改坏 → 必须变红）：删 `_stream` / 只流正文不流工具调用 / 正文改成
  "非流式切片"的假流式 / `bind_tools` 就地改写单例 / 评测工具路径退回 `relative_to` /
  去掉 UTF-8 切换 / 去掉 `errors="replace"` / `_as_list` 不再收敛 / schema 退回 `list[int]` /
  `_as_list` 不拆逗号串 —— 全部变红。
  过程中还抓出**我自己写的一条安慰剂测试**（用 capsys 断言"字符打得出来"，
  把 `reconfigure` 改成 `pass` 它照样绿），已改成观察机制 + 开子进程在 GBK 环境下真跑一遍。

### 已知问题（本次没修）

- 首响仍有约 6 秒：算答案的模型是在**检索完之后**才开始流式，前面 planner+检索的
  十几秒是静默的。下一步要么把检索耗时压下去，要么让它流"进度"。
- 一次 ReAct 第 2 轮把 `explain_question\n{"question_index": 1}` 写进了正文 ——
  `strip_tool_call_text` 只认"工具名 + JSON 对象"且只作用在 plan 侧的那一处；
  模型编参数名（`question_index`）时整行认不出来。属**下一版**的题目。

  > **⚠️ 这段归因是错的，见 [1.7.3]**：正则其实**认**换行也认编的参数名（`\s*` 能吃换行）。
  > 真正的原因是**那条路上根本没调用它** —— 清理只写在 plan 侧的 synthesizer 里。
  > 原始记录按纪律保留不改，更正写在 1.7.3。

## [1.7.1] - 2026-10-16

**把两条"守字词"的守卫改成"守意图"** —— 起因是一个真实误报。

### 问题：守卫拿字词当代理，文案一改就红

用户把 `group_payload` 里**面向学生**的一句提示改成「请享用。」，结果两条测试红了：

| 测试 | 它其实想守的 | 它用的代理（字词） |
|---|---|---|
| `test_reply_text_is_the_shared_one` | 助教的成品文本**来自共享层**（不许各写一份） | `"答案怎么写都行" in reply_text` |
| `test_get_paper_questions_ships_ready_to_send_text` | 出题必须给**成品文本**（否则模型只回"好的，这是下一组"却不贴题） | `"请慢慢做" in reply` |

**守字词 = 一改文案就红，而真正的不变量没被守住。** 这与项目之前栽过的坑同类
（`find_chapter` 加个下划线就能躲过守卫）。

### 改法

- `test_tutor_reply_text_comes_from_the_shared_layer`：与共享层对**同一组**的输出
  **逐题对齐结构化字段**（image_url / topic），并要求成品文本带上题号与图片 ——
  措辞随便改都不会误报
- `test_get_paper_questions_ships_ready_to_send_text`：删掉措辞断言，保留"完整到不可能是
  只回了一句好的"（长度门槛 + 每题的题号与图片都在）
- **新增** `test_freeform_answers_stay_supported_where_the_model_can_see_it`：
  把"自由格式作答必须被支持"这条**行为约束**守在对的地方 —— `grade_answers` 的
  **工具描述**（模型每轮必看）里必须仍有"原样传入/工具自动解析"的意思

### 顺带查清的一件事（写进注释，免得后人再猜）

那句"答案怎么写都行"**只面向学生**：真正让模型接受任意答案格式的是**工具描述**
（`支持多种答案格式：1.A 2.B、ABCDA、1.对 2.错等，工具会自动解析` +
`user_answer_text: 原样传入即可，工具会自动解析`）。而且这句话在描述里**出现了两次** ——
删掉一处，行为约束仍在（变异验证过：两处都删才会红）。

### 验证

- Python **674 passed**（用改后的文案）；把文案换回原版再跑那 2 个文件也**87 passed**
  —— 两个方向都绿，证明守卫确实与措辞无关
- 变异验证：把工具描述里两处格式容错的措辞**都**删掉 → 新守卫变红；还原 → 变绿

## [1.7.0] - 2026-10-16

**两件用户提的事：导出目录可配置、导入文件不再受"必须先在 data/ 里"的限制。**
顺带修掉一次**我自己造成的知识库损坏**（详见下面"事故记录"，这是本次最该看的部分）。

### 可配置：对话导出目录

原来是写死的 `<项目>/exports` —— 而导出的是**用户自己的对话记录**，
想放到别处（笔记目录、同步盘）是很自然的需求。

- `config.EXPORT_DIR`（默认 `exports`，相对路径按项目根解析；也可填绝对路径）
- `.env.example` 里加了说明
- 落点改成**函数** `_export_root()`（常量在 import 时就定死了，测试和环境变量都不好改）
- 导出目录配到项目之外时，返回的 `path` 直接给绝对路径（原来那行
  `relative_to(PROJECT_ROOT)` 会**抛异常**，正好把"可配置"这条路堵死）

### 知识库导入：**源不限制、目标必须落在 data/ 之下**

原来 `/kb/add` 连**读取**路径都要求"必须在 data/ 之下"（`resolve_within_data`），
所以"从下载目录导入一个文件"必然失败 —— 用户实测撞到了。可它当初要防的是
"越界读取机器上的任意文件"，**方向反了**：真正该守的是**写入位置**。

- `/kb/add`：源在 `data/` 里就就地入库；在外面（下载目录、桌面、D 盘…）会被
  **复制进** `data/<目标目录>/` 再入库
- **`POST /kb/upload`（新增）**：浏览器**直接上传**，内容经 HTTP 过来 ——
  服务端不去读你机器上的任意路径，这是最省事也最安全的一条路，界面里标为推荐
- **`GET /kb/import-targets`（新增）**：给出可选目标目录，`doc_type` 是**用生产那条
  分类规则算出来的**（不另写一套）。为什么必须让用户选：`_classify_doc` 按子目录名
  判类型（含 `ppt`→课件、含 `课本`→课本、含 `往年题`→**跳过**），填错就会被判成 notes
- 复制而不是就地读还有一条技术理由：外部路径一律被判成 notes，复制进来才能落到正确分类

**修掉用户实际遇到的"导入失败"**：文件放进 `data/操作系统/往年题/` 再导入时，
`_classify_doc` 返回 `None`（往年题是题库的源材料，走 `/qb` 那条管线），
旧行为返回 `code: 0, added_chunks: 0` → 界面显示"导入成功: 0 个分块"，**看着像失败**。
现在明确说清原因、并告诉用户该放哪；上传到这类目录则**直接拒绝且不保存文件**。

### 安全取舍（必须知情，所以写了开关 + 启动告警）

放开"源不限制"等于重新打开 Week9 修过的那条链（任意文件入库 → `/kb/search` 外泄）。
所以：

- `config.KB_IMPORT_ALLOW_OUTSIDE`（默认 `true`）—— 设 `false` 即回到原策略（一律 403），
  拒绝信息里直接带开关名，可操作
- **上传接口不受这个开关影响**（它不读本地路径），永远可用
- 无鉴权 + 允许越界导入的组合**在启动时打 WARNING**（原来的启动日志还在说
  "data/ 之外的文件依旧会被拒绝" —— 那是假话了，一并改掉）
- 每次从 `data/` 之外导入都**记日志留痕**

### 两条新的测试守卫（都是踩着事故加的）

1. **真实的 `data/` 不许被测试改动**（新增/删除文件都算）。
   这次写越界导入的用例时，测试**真的**把 `README.md`、`.env`、`server.py`、`win.ini`
   复制进了 `data/导入/` —— 其中包括 `.env`（里面是 API 密钥）。
2. **服务运行着的时候不许跑测试**。Chroma 是单写者，并发写会写坏知识库（见下）。
   检测到服务在跑就 `pytest.exit` 并说明原因；确需并存可设 `ALLOW_TESTS_WITH_SERVER=1`。

### 事故记录：我把用户的知识库写坏并重建了（4199 块重新 embedding）

**我做了什么**：在 `uvicorn` 服务**运行着**的时候跑了一遍全量测试，两个进程同时写
`vector_store/`。

**后果**：Chroma 的元数据段被写坏 —— 之后**任何新进程**读它都抛
`Failed to apply logs to the metadata segment`（服务进程内存里的句柄还能读，
所以一开始没发现）。写前日志 `embeddings_queue` 里积了 119 条**无法应用**的垃圾
（`README_2.md_*`、`plain.md_*` 等），`embeddings` 那 4199 条正常数据一条没丢。

**修复过程（以及一个更阴的坑）**：
1. 清空未应用的队列 → **读恢复了**，但……
2. ……**写变成静默丢弃**：`upsert` 不报错、`delete` 也不报错，但什么都不落盘
   （独立进程里实测：`count()` 4199 → 4199、按 id 查不到、sqlite 行数不变）
3. 于是**忠实重建索引**：以重建前的文档清单（20 个文档 / 4199 块，已存档）为准逐个重导，
   而**不是** `build_knowledge.py --clear` —— 干跑显示后者会多收 7 个文件，其中两个是
   项目自己的提示词文档、还有一本与已有课本重复的 PDF。**修故障不该顺手改内容。**

**重建结果**：20 个文档 / **4234 块**（原 4199）。差异全部来自 10 个 PPT 的 PDF 抽取
（`172 vs 171`、`223 vs 207` 之类，pypdf 打出 `Ignoring wrong pointing object` 印证了
抽取库与当初建库时不同）；**课本 2319 块与所有 .md/.docx 完全一致**。重建后
**写入恢复正常**（实测 `count()` 4234 → 4235 → 删除回 4234）。

**顺带查清的一件事**：重建时 Chroma 报 `unable to open database file`，
一度以为是"目录不存在"。实际原因是**系统临时目录不可写**（它会往 TMP/TEMP 落中间文件），
把 TMP 指到工作区内就好了 —— 这是**运行环境的限制，不是项目缺陷**，所以没有为此改代码。

### 未决（需要用户拍板，本次刻意没做）

`data/` 下有 **7 个文件从未进过知识库**，全量重建会把它们收进来：

| 文件 | 我的看法 |
|---|---|
| `Operating Systems - Three Easy Pieces-version1.01.pdf` | 与已有的`操作系统导论（…）.pdf`是**同一本书**，收进来是重复的 2319 块 |
| `复习模式Prompt-速查刷题.md`、`学习模式Prompt-教练批改.md` | 是**项目自己的提示词文档**，不是课程资料 |
| `计算机网络（第7版）…pdf`、`计网第1次作业-26秋.pdf`、`2-01-原子操作.pdf`、`2-02-临界区、互斥与锁.pdf` | 看着像正常课程资料（可能是后来才放进 `data/` 的），要不要入库由用户定 |

### 顺带（eval 侧，1.6.0 之后就该跟上的两处）

- **评测跑完自己清会话**：`/chat/conversations` 会把评测造的 thread 列出来，
  前端会当成"服务端保存的会话"导入 —— 用户侧栏里就会冒出几条
  「给我出 2026春…」。现在每条用例跑完就删掉自己造的会话（`--keep-threads` 可关）；
  之前已经污染的那 7 条也清了
- **`_TOOL_CALL_TEXT_RE` 改为从生产 import**，不再复制一份 —— 复制的话生产改了正则，
  评测还在按旧规则判，而注释里那句"与生产同一条"就变成假话（这个项目在别处已栽过几次）

### 验证

- Python **672 passed**（原 641 + 新增 31；`tests/test_kb_import.py` 是新增文件）
- 端到端（对真实服务打 HTTP）：上传落进 `data/` → **检索命中** → 删除生效 → 计数回位；
  路径导入 `copied=True`；往年题被拒且不落文件
- 服务运行时的测试守卫：拦截 ✓ / 逃生开关 ✓

## [1.6.0] - 2026-10-15

**第一次有了"效果好不好的数据"**：新增 `eval/` 行为评测（脚本化用例 + 可判定判据 +
按版本存档的报告）。而且它**第一次运行就抓到一个 641 条测试看不见的真问题**。

### 新增：`eval/` 行为评测

`pytest tests/` 守的是"**代码对不对**"，回答不了"这个版本学生用起来是不是更好了"。
`eval/` 补那一层：打真实 HTTP（学生走的那条路：`/qa/stream` + SSE），
每条用例**重复 N 次**（历史上那些缺陷全是"时好时坏"，单次通过没有意义），
报告存进 git 供版本对比。

- **起始 2 条用例**：`quiz_no_leak`（出题不泄题 + 图片归属 + 无协议原文）、
  `continue_next_group`（多轮：出题 →"继续下一组"必须取第 2 组，且不许回头重列试卷）
- **判据全部可判定**（不做主观打分）：工具轨迹、图片是否**全部来自题库**、
  题面/参考答案是否泄漏（用**题库原文切片**比对，不猜关键词）、是否有工具调用原文、
  正文长度、**是否真的流式**、**首响预算**
- **硬指标**：`ttft_ms`（首响 = 首 token —— 流式产品里"快"的正确口径）、总时长、
  content 事件数、正文来源（`content` / `done`）
- **软指标**（讲得好不好、像不像老师）**明确留作另一层**：LLM 裁判 + 人工标注校准，
  在报告里标注为参考值、**不当门槛** —— 软的当门槛会变成噪声驱动的开发

### 它第一次运行就抓到的真问题：**回答根本不流式**

```
2 条用例 / 58 条判据 / 46 通过 / 稳定通过 0 条
12 条红**全部**是 require_streaming 与 max_ttft_ms
ttft_p50 = None（一次 content 事件都没有）· 每轮总时长 p50 = 34.7 秒
```

**正确性那一侧是真绿的** —— 工具轨迹、不泄题、图片归属、无协议原文全过，
"继续下一组"也确实取对了第 2 组（游标链路被独立确认）。

但**"快"这一侧完全不及格**：`/qa/stream` 的 `content` 事件**在生产里从未触发过** ——
学生盯着转圈约 35 秒，然后整段答案一次性出现。

**根因**：`course_agent/llm.py` 的 `SimpleLLM` **不是 LangChain 的 ChatModel**
（普通类，只实现 `invoke` / `bind_tools`），它调的智谱接口本身也**不是流式**
（`chat_with_tools` → 非流式 POST）。所以 LangGraph 的 `messages` 模式**永远产不出 chunk**，
1.1.0 那套"只放行 `AIMessageChunk`"的判据自然一条都放不出来。
（顺带：`SimpleLLM.astream` 是个**假流式** —— 调非流式接口再按字符切。）

**为什么一直没被发现**：640+ 条测试里，流式相关用例用的是 `GenericFakeChatModel`
（真 LangChain 模型），**从没跑过生产 LLM**；而前端有 `done.answer` 兜底，
所以答案照样显示，只是退化成"等 35 秒一次性出现"。

→ 这是**第一个由评测（而不是读代码）发现的缺陷**。修复方向已明确（让 LLM 真的流式），
留给下一个版本，用**同一套评测**给出改前/改后的 TTFT 对比。

### 一条踩出来的评测设计纪律（已写进 `eval/README.md`）

第一次跑的时候正文是空的（流式没生效 + runner 只读了 `content` 事件），
结果 `no_stem_leak` / `no_reference_leak` **全部轻松通过** —— 空字符串当然不含参考答案。
**只有 `answer_min_chars` 把它暴露出来。**

→ **每条用例都必须有"最低门槛"断言**（`answer_min_chars` / `require_streaming`），
否则其余判据会**真空通过**。

### 验证

- `python -m eval.runner` 可跑；报告 `eval/reports/1.5.4.json` 入库（23.9 KB）
- 既有 640+ 条测试不受影响（`eval/` 不进 CI —— 它要真模型、真额度）

## [1.5.4] - 2026-10-13

**修掉"文档里的工具清单已经腐烂"** —— `tools.py` 文件头还写着 9 个工具，实际是 13 个。

### 修复

- **`tools.py` 文件头**：清单写 9 个、实际注册 **13 个** —— 漏的正是后来加的
  「学习/复习三件套 + 自拟题」（`list_course_sections` / `read_section` /
  `record_study_position` / `prepare_custom_quiz`）。现在按用途分四组写全
  （检索 / 刷题 / 学习复习 / 练习统计），并注明它腐烂过一次。
- **`README.md` 文件树注释**只列了 13 个中的 3 个
  （`search_knowledge + web_search + get_course_info`）→ 改成指向文件头，不再枚举。
- **`README.md` 的工具表**补了范围说明：那张表只覆盖刷题线，完整清单在文件头；
  并说明 `search_knowledge_base` 是助教侧命名、BOT 侧同名能力叫 `search_knowledge`。

### 新增守卫

`tests/test_surface_parity.py::TestToolParity::test_the_docstring_tool_list_matches_reality`：
文件头列出的工具名必须与 `tools_by_name` **完全一致**。断言比对的是**解析出来的条目名**，
不是搜文本（正文里提到某个工具名不算数）。

变异验证：从文件头删掉 `read_section` → 报「只在代码里: ['read_section']」；
加一个不存在的 `ghost_tool` → 报「只在文档里: ['ghost_tool']」。

### 顺带把同类风险扫了一遍（这几处没问题）

| 位置 | 结论 |
|---|---|
| `nodes.py` 的 `PLANNER_PROMPT` 写"你有 13 个工具可用" | **已被守住** —— `test_planner_prompt_contract.py::test_tool_count_statement_matches_reality` 会比对实际注册数。这处最要紧：它影响**模型行为**，不只是文档 |
| `graph.py` 文件头那张图结构图 | 与真实节点一致（4 个 plan 节点 + 2 个 react 节点都在图里） |
| `README.md` 的"两条 Agent 路径的工具"表 | 是有范围的**部分**表（刷题线），已补范围说明 |

> 一条可以复用的规则：**文档里不要枚举"会变的清单"** —— 要么指向权威处（像文件头那样，
> 并有测试守着），要么就接受它一定会腐烂。

### 验证

- Python **641 passed**（原 640 + 新增守卫 1）
- 前端 auth_block 52/0、chat_context 14/0、server_conversations 35/0

## [1.5.3] - 2026-10-13

**把手写助教的 HTTP 面与前端接线成套移除**，并给这个决定加上守卫。

### 背景：这个端点来回折腾了三次

| 版本 | 发生了什么 |
|---|---|
| 1.1.0 | 我用脚本整段替换 `server.py` 时**误删**了它 —— 当时只核对"`/chat/stream` 还剩几处"，没问"是不是少了别的" |
| 1.5.1 | 按当时的文档（写着"代码与接口全部保留"）把它**恢复**了，并加了路由清单守卫 |
| 1.5.3 | 按定位**有意移除** —— 手写助教只保留模块与测试 |

三次都不是"代码坏了"，而是**"它该不该在"这件事没有一份显式的记录** ——
只散落在注释与 README 里，于是每次有人照着某处过期的说明改一遍。

### 变更：成套移除（只清一半会留下半截接线）

| 位置 | 处理 |
|---|---|
| `server.py` | 删 `tutor_stream` 端点与 `TutorRequest` 模型；`tutor_agent` 从导入里去掉 |
| `chat.html` | 删请求路径里的 `/tutor/stream` 分支、`currentTutorSessionId`（6 处）、以及**已无任何调用者的 `setAppMode`** |
| `README.md` | 接口表去掉那一行；"代码与接口全部保留"改成"只保留模块与测试" |
| `tests/test_auth.py` | 去掉 `/tutor/stream` 的鉴权条目 |
| `tests/test_route_surface.py` | 路由清单去掉它；`TestTutorStreamActuallyWorks`（2 条）随之删除 |
| `tests/test_tutor_reference.py` | **新增** `TestTutorHasNoHttpOrFrontendSurface`（4 条） |

顺带发现的死代码：`setAppMode` **没有任何调用者**（模式切换按钮 2026-10 就撤了），
所以整个函数删除 —— 它的 tutor 分支正是这次要处理的那处。

### 新增守卫：把"没有"也写成测试

`TestTutorHasNoHttpOrFrontendSurface` 守两件事：

- **没有端点**（`/tutor/stream` 不在路由表里）；
- **前端代码里没有 tutor 接线** —— 断言前会**先去注释**（页面里留了一段"想恢复要怎么做"
  的说明，它当然提到这些名字；那是文档不是接线。`//` 的判定避开 `https://`）。

变异验证：把 `/tutor/stream` 加回前端代码 → 红；把端点加回 `server.py` →
**两条守卫同时红**（本文件 + 路由清单）。

### 另外

`docs/*.md` 现在被 gitignore（只留 `docs/README.md`）—— 按"文档不随仓库分发"的既定决定，
放在 `docs/` 下的笔记不会再出现在 `git status` 里。要放行某个文档，显式加一行 `!docs/xxx.md`。

### 验证

- Python **640 passed**；前端 auth_block 52/0、chat_context 14/0、server_conversations 35/0
- 路由总览只剩 `/qa/stream` 一个流式端点
- `test_tutor_agent.py` / `test_tutor_reference.py` 照旧全跑（它们直接调函数，不经过 HTTP）

## [1.5.2] - 2026-10-13

**把 `test_surface_parity.py` 拆开，让文件名与内容对上。** 纯重组，测试逻辑一行没改。

### 为什么拆

那个文件叫"两个运行面对齐"，但 498 行里**大部分与对齐无关**：

| 内容 | 与"对齐"的关系 |
|---|---|
| 工具能力对齐、共用 quiz_service、提示词契约 | 确实是 |
| `TestBotCaughtUpWithTutor`（9 条） | **只跟 BOT 有关**（planner / synthesizer / 历史裁剪） |
| 助教的工具守卫、状态注入、会话书签（12 条） | **只跟助教自己有关** |

名字低估内容会让人误判它的用途 —— 上一轮我差点因为"助教不再是产品路径"
就把它当遗留物建议删掉。现在拆成三个**名字说实话**的文件：

| 文件 | 内容 | 条数 |
|---|---|---|
| `test_surface_parity.py` | 两面一致性 + 共享契约（quiz_service） | 19 |
| `test_bot_study_flow.py` | BOT 那条路自己的学习流程行为 | 9 |
| `test_tutor_reference.py` | 手写助教（冻结参照）自己的行为与工具守卫 | 12 |

留在 `test_surface_parity.py` 的那 4 条共享核心用例，类名也从
`TestStudentSessionFindings` 换成了 `TestSharedCoreBehaviour`（同一个理由）。

### 顺带

- `conftest.py` 里 **8 条按「文件名 :: 类名 :: 用例名」索引的 `DATA_NEEDS`** 同步更新 ——
  它们是 `course_ppt` 数据的跳过机制键。移动测试却忘了改这里，跳过会**静默失效**
  （用例带着缺失数据跑，报出与代码无关的红）。
- 两个新文件的模块 docstring 写清了边界，并记下"这些用例为什么存在"
  （全部来自真实会话发现，不是凭空设计的边界）。

### 验证

- **拆分前后用例数 40 = 40**（9 + 12 + 19），全量 **639 passed** 不变 ——
  证明这是一次纯重组：没有丢、也没有重复。
- 前端 auth_block 52/0、chat_context 14/0、server_conversations 35/0。

## [1.5.1] - 2026-10-13

**修复 1.1.0 误删的 `/tutor/stream` 端点**，并补上"当初就该有"的那类守卫。

### 修复：一个我自己造成、跨了 4 个版本没被发现的回归

`POST /tutor/stream` 在 1.1.0 被**误删**。那一次我是用脚本把 `server.py` 里从
`/qa/stream` 到"知识库管理页面"横幅之间的**整段**替换掉的 —— 本意只是删掉重复的
`/chat/stream`，而 `/tutor/stream` **紧跟在它后面**，于是被一起吞了
（server.py 里 tutor 相关行 5 → 2）。

当时我核对的是"`/chat/stream` 还剩几处""`/qa/stream` 还在不在"，
**从没问一句"是不是少了别的"** —— 这正是"确认了预期的变化，却没确认没有别的变化"。

它一直没被发现，因为：

- 界面上早就撤掉了助教入口 —— **没有任何用户路径会 404**；
- 助教的测试全是直接调 `tutor_agent` 的函数，**从不经过 HTTP**；
- 于是 README 里"手写助教的代码与接口全部保留"那句话、以及接口表里的
  `POST /tutor/stream`，从 1.1.0 起就是**假的**。

现在端点已恢复（`tests/test_route_surface.py` 用打桩请求确认 SSE 链路真的通）。

### 新增守卫：tests/test_route_surface.py

- **路由清单**：把 22 个端点钉成一份显式清单，**少一个、多一个都要报错**并指出是哪个。
  变异验证：把 `/tutor/stream` 的装饰器注释掉（复现那次误删）→ 立刻红。
- **端点行为**：光"在清单里"不够，还得真能跑 —— 用打桩的 `process_message` 真发一次
  SSE 请求，确认 `start`/`done` 事件流与错误转成 SSE 事件（不产生任何网络调用）。
- `test_auth.py` 的"必须鉴权"清单加入 `POST /tutor/stream`（它会真的花掉 LLM 额度）。

### 顺带修掉文档里两处过时引用

README 的接口表还列着 **1.1.0 有意删掉**的 `POST /chat/stream`，正文里也还有一处
"（`/qa/stream`、`/chat/stream`）"。这些是那次合并**该改而漏改**的，一并修掉，
并把会话列表三个端点补进接口表。

### 验证

- Python **639 passed**（原 634 + 路由清单 2 + 端点行为 2 + 鉴权 1）
- 前端 auth_block 52/0、chat_context 14/0、server_conversations 35/0
- 变异验证：注释掉端点装饰器 → 路由守卫变红（说明它能在下次真的挡住）

## [1.5.0] - 2026-10-13

**环境依赖变成真正可复现的**：加了 uv 生成的通用锁文件，本地与 CI 从此装的是同一套。

### 背景（先说清问题）

`requirements.txt` 只写**直接**依赖，而开头自称"已锁定版本（可复现安装）"。实测下来：

- 传递依赖**一个都没锁**，pip 每次解析都拿"当时最新"；
- 里面还有一行 `langchain>=0.3.0` —— **开放式范围**，等于没锁；
- 本地与 CI 的实际差异：**52 个包版本不同**（含 `huggingface-hub` 1.x→2.x、
  `oauthlib` 3→4、`kubernetes` 36→37 这种跨大版本），另有 7 个包只在 Windows 有、
  4 个只在 Linux 有。

于是"CI 绿灯"带着运气成分：某个包发了新版，CI 可能在你一行代码没改的情况下变红。

### 变更

- 新增 **`requirements.lock`**（`uv pip compile --universal` 生成）：约 150 条完整锁定，
  含所有传递依赖与**环境标记** —— 所以同一份文件在两个平台上都对：

  ```
  uvloop==0.23.0 ; ... sys_platform != 'win32'
  pywin32==312   ; sys_platform == 'win32'
  colorama==0.4.6 ; os_name == 'nt' or sys_platform == 'win32'
  ```

  > 为什么不用 `pip freeze`：它是"我这台机器"的快照、**没有标记** ——
  > `uvloop` 会被带到 Windows、`pywin32` 会被带到 Linux，**两边各炸一次**。
  > `--universal` 才是跨平台的正确做法。

- **CI 与 Dockerfile 改成从锁文件安装**（`pip install -r requirements.lock`）。
  锁文件本来就是 requirements 格式，所以**安装这一步不需要引入新工具** ——
  镜像不用多装 uv，也不用改已有的清华镜像配置。uv 只用于**生成/更新**锁文件，
  单独装（`pip install uv`），不进锁文件。

- **删掉 `langchain>=0.3.0`**：全仓库扫过，代码只用 `langchain_core` 与
  `langchain_community`，**从未 `import langchain`**；本机根本没装它，634 条测试也全过。
  CI 装它只是因为文件里写了它 —— 而它又是全文件唯一一行开放式范围。

- **本机环境按锁文件对齐**（52 个传递依赖升到锁定版本）**然后跑全量测试** ——
  锁文件描述的这套栈是被真正验证过的，不是我本机原来那套"碰巧能用"的。

- README 的"快速开始"补上了**装依赖这一步**（原来整节都没提，照它走根本跑不起来）。

### 测试

- 新增 `tests/test_lockfile.py`（3 条）：
  ① 声明里写死版本的直接依赖，锁文件必须是同一个版本（改了声明忘了重新生成 → 红）；
  ② 当前环境装的版本必须与锁文件里**对本平台生效**的条目一致
     （CI 退回从 requirements.txt 装、或有人手工升级 → 红）；
  ③ 锁文件必须带环境标记（用 `pip freeze` 生成 → 红）。
  断言写在**解析出来的条目与标记**上，不是搜文件文本 —— 注释骗不过它。

### 验证

- Python **634 passed**（原 631 + 锁文件 3 条）；前端 auth_block 52/0、chat_context 14/0、
  server_conversations 35/0。
- 三处变异验证锁文件守卫：改声明不重新生成、加一个锁文件里没有的包、把标记抹掉
  （模拟 `pip freeze`）—— 全部正确变红。
- 锁文件的 Windows 分支已在本机验证；**Linux 分支由 CI 验证**（CI 已改为从锁文件装）。

### 一条环境观察（不当成结论）

本机走代理时 `pip install -r requirements.lock` 卡了 25 分钟没动静，换成
`uv pip install -r requirements.lock` **14 秒**装完。但这**不能**作为"CI 也该用 uv 装"的依据 ——
CI 是直连 PyPI，pip 在那边一直是 1 分钟左右。保留 pip 是为了不给镜像和 CI 引入新工具。

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

[未发布]: https://github.com/catcatsquid/ai-agent-api/compare/v1.7.1...HEAD
[1.7.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.7.1
[1.7.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.7.0
[1.6.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.6.0
[1.5.4]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.5.4
[1.5.3]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.5.3
[1.5.2]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.5.2
[1.5.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.5.1
[1.5.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.5.0
[1.4.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.4.1
[1.4.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.4.0
[1.3.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.3.0
[1.2.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.2.0
[1.1.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.1.0
[1.0.2]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.2
[1.0.1]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.1
[1.0.0]: https://github.com/catcatsquid/ai-agent-api/releases/tag/v1.0.0
