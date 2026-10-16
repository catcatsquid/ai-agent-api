# 行为评测（`eval/`）

一套**脚本化用例 + 可判定判据 + 按版本存档的报告**，用来回答一个问题：

> 这个版本，学生用起来是不是更好了？

`pytest tests/`（640+ 条）回答的是"**代码对不对**"—— 单元、契约、结构不变量。
它看不见"回答是不是 35 秒才出现""有没有开始流式"这类**产品事实**。
这个目录补的就是那一层。

---

## 一、怎么跑

```bash
# 1. 起服务（评测走学生真正走的那条路：HTTP + SSE）
python -m uvicorn course_agent.server:app --port 8003

# 2. 跑评测
python -m eval.runner                      # 全部用例 × 3 次
python -m eval.runner --case quiz_no_leak --attempts 5
python -m eval.runner --base http://127.0.0.1:8003 --out eval/reports/try.json
```

- **不进 CI**：它要真模型、真额度（和 `pytest tests/` 分开，那套必须离线可跑）
- **每条用例重复 N 次**：历史上那些缺陷全是"时好时坏"，单次通过没有意义
- 退出码：全部用例"稳定通过"才 0，否则 1 —— 可以直接用来卡发版
- **跑评测与 `pytest` 别同时来**：评测要往真实会话库写 thread（跑完自动清），
  而 `pytest` 有一条"真实会话库不许变脏"的守卫 —— 同时跑会让那条守卫变红
  （判据本身是对的，但它会把评测造的 thread 当成测试漏出来的垃圾）。
  服务在跑时也能跑评测（评测只打 HTTP）；跑 `pytest` 时服务要么停掉，要么设
  `ALLOW_TESTS_WITH_SERVER=1`（见 `tests/conftest.py`）。

## 二、用例格式

`eval/cases/*.json`（JSON 而不是 YAML：不引入新依赖；"为什么存在"用 `why` 字段写）：

```json
{
  "id": "quiz_no_leak",
  "why": "出题时题面与参考答案绝不能进正文 …… 覆盖两个真实缺陷：(1) 工具结果被当正文播出去；(2) 模型自己编图片链接",
  "mode": "plan",
  "truth": {"paper_source": "2026春操作系统期末试题（回忆版）", "group_index": 1},
  "turns": [
    {
      "user": "给我出 2026春操作系统期末试题（回忆版）的第 1 组题",
      "expect": {
        "tools_any_of": ["get_paper_questions"],
        "tools_none_of": ["grade_answers", "explain_question"],
        "forbid_tool_call_text": true,
        "min_images": 3,
        "images_all_from_bank": true,
        "no_stem_leak": true,
        "no_reference_leak": true,
        "no_answer_letter_leak": true,
        "answer_min_chars": 10,
        "require_streaming": true,
        "max_ttft_ms": 8000
      }
    }
  ]
}
```

- `why` **必写**：用例必须来自真实缺陷或真实需求，不是"容易测的东西"（否则会挑软柿子）
- 多轮用例：`turns` 里依次追加，共用同一个 `thread_id`
- `truth`：判据需要**真值**时声明来源（这里=拿题库原文比对"有没有泄题"，不猜关键词）
- `group_offset`（某一轮上）：这一轮指的是第几组，供真值比对用

## 三、判据清单（全部**可判定**，不做主观打分）

| 判据 | 防的是什么缺陷 |
|---|---|
| `tools_any_of` / `tools_none_of` | 该调的没调（"我不会"不调 explain_question）；不该调的调了（讲课时去调 grade_answers） |
| `forbid_tool_call_text` | 把内部协议当正文写出来：`get_paper_questions {"paper_source": …}` |
| `min_images` | 出题不给图 / 给少了 |
| `images_all_from_bank` | **模型自己编图片链接**（曾编出 `example.com/net1.jpg`） |
| `no_stem_leak` | 题面进正文（学生还没答就看到题）—— 用题库原文切片比对 |
| `no_reference_leak` | 参考答案进正文 —— 同上 |
| `no_answer_letter_leak` | 直接说"答案：B"（只在"明确给答案"的语境里判，避免误伤"B 选项说…"） |
| `answer_min_chars` | 空回答 / 兜底套话 |
| **`no_internal_artifacts`** | **写给模型看的东西被当正文播出去** —— planner 的计划 JSON（`need_tools`）、`search_knowledge` 返回值末尾那段【怎么用上面的片段】使用说明、`tool_calls`、`【内部摘要】`。与泄题判据不同，它**不比对真值**，所以讲解类用例（没有卷子原文可比）也能用 |
| **`require_streaming`** | **流式失效** —— 正文来自 `done` 而不是逐字 `content` |
| **`max_ttft_ms`** | **首响太慢**（学生盯着转圈） |
| `args`（部分实现） | 工具参数是否正确（当前 `/qa/stream` 不回报参数，退化为"调了这个工具"） |

## 四、指标（每次运行都记录）

**硬指标**（自动、零成本）：

- `ttft_ms`：**首响时间**（流式产品里"快"= 首 token，不是总时长）
- `total_ms`：整轮时长
- `content_events`：逐字片段数（0 = 没流式）
- `answer_source`：正文来自 `content` 还是 `done`
- `tools`：工具轨迹（顺序）
- 判据通过数 / 用例稳定通过数

**软指标（另一层，还没做）**：讲得切不切题、像不像老师、有没有共情。
做法：LLM 裁判 + **人工标注校准**（先用 10 条人工打分的对齐裁判），
并且在报告里**明确标注为参考值、不当门槛** —— 软指标不稳，当门槛会变成噪声驱动的开发。

## 五、报告与版本对比

`eval/reports/<APP_VERSION>.json`，**进 git** —— 这是"版本有没有变好"的唯一依据。

```json
{
  "app_version": "1.5.4",
  "attempts": 3,
  "summary": {"cases": 2, "cases_stable_pass": 0, "checks_total": 58,
              "checks_passed": 46, "ttft_p50_ms": null, "turn_total_p50_ms": 34736},
  "cases": [{"id": "...", "attempts": [{"turns": [{"checks": [...], "ttft_ms": ...}]}]}]
}
```

对比方式：`git diff v1.5.4 v1.5.5 -- eval/reports/`，或写个小脚本比 `summary`。

## 六、三条防自欺规则（第一条是踩出来的）

1. **必须有不变量门槛，否则其余判据会"真空通过"。**
   第一次跑的时候正文是空的（流式没生效 + 我只读了 `content`），结果
   `no_stem_leak` / `no_reference_leak` **全部轻松通过** —— 空字符串当然不含参考答案。
   只有 `answer_min_chars` 把这件事暴露出来。
   → 所以**每条用例都要有 `answer_min_chars`、`require_streaming` 这类"最低门槛"断言**。
2. **评测集自己要被变异验证**：故意改坏代码 → 对应用例必须变红。
   没有这一步，评测集就是安慰剂（与 `tests/` 里那些守卫同一条纪律）。
3. **不许为了让评测变绿而改判据**：改判据要在报告/CHANGELOG 里显式记录。

**已记录的判据改动**：

| 版本 | 改动 | 理由 |
|---|---|---|
| 1.7.2 | `max_ttft_ms` 8000 → **15000** | 8 秒是"流式完全没生效"时写的。现在 p50≈6.5 秒，8000 变成**边界误报**（实测 8250 / 8656 / 9170 ms 都判红）；它要挡的缺陷是"30 秒以上才出字"。门槛仍在（非 None + 有上限），"防真空通过"的意图不变 |
| 1.7.2 | `dont_know_calls_explain` 第 2 轮 `answer_min_chars` 80 → **40** | 讲一道题的思路本来就可能是 57 字；80 字是在拿字数当质量，不是防"空回答" |
| 1.7.2 | 新增 `no_internal_artifacts` | planner 的计划 JSON、`search_knowledge` 的使用说明都不许进正文（讲解类用例没有真值可比，所以它不比对真值） |

## 七、当前基线（v1.7.2，2026-10-16）

```
5 条用例 / 180 条判据 / **180 通过** / 稳定通过 5 条
ttft_p50 = 6533 ms · p95 = 7867 ms · 每轮 content 事件 517~613 个
每轮总时长 p50 = 31.1 秒（与 1.7.1 的 32.9 秒基本持平 —— 流式改的是"什么时候开始出字"）
```

对比 1.7.1（同一台机器、同一套判据）：

| | 1.7.1 | 1.7.2 |
|---|---|---|
| `content` 事件 | **0 个**（12 轮全是 0） | 517~613 个/轮 |
| `ttft_p50_ms` | `null` | **6533 ms** |
| 判据 / 用例 | 69/87 · 0/2 稳定通过 | 180/180 · 5/5 稳定通过 |

**1.7.1 的根因**（已在 1.7.2 修掉）：`course_agent/llm.py` 的 `SimpleLLM`
不是 LangChain 的 ChatModel（普通类，只实现 `invoke`/`bind_tools`，外加一个假的
`astream`），而 `stream_mode="messages"` 只认实现了 `_stream` 的 Runnable ——
所以 `content` 事件**在生产里从未触发过**，学生等约 33 秒后整段答案一次性出现。

> 这也解释了为什么它一直没被发现：640+ 条测试里，流式相关的用例用的是
> `GenericFakeChatModel`（真 LangChain 模型，自己带 `_stream`），**从没跑过生产 LLM**；
> 而前端有 `done.answer` 兜底，所以答案照样显示 —— 只是退化成"等 35 秒一次性出现"。
> 现在 `tests/test_llm_streaming.py` **只对生产那个 `llm` 对象**下判据（用假 HTTP 层），
> 并由变异验证保证它真的会红。

## 八、怎么加用例

来源优先级（越高越该先写）：

1. **CHANGELOG 里记录过的真实缺陷** —— 每条都能变成一个用例
2. **学生/用户真实抱怨**（"我没看到题目""不是这个意思""你在说什么"）
3. **产品的硬约束**（不泄题、给图片、进度正确）

加的时候问三句：这条防的是什么？谁能判它？**怎么证明它能判**（变异一下会不会红）？
