"""对话记忆管理

两种策略:
  1. 滑动窗口裁剪: 消息太多时直接丢掉最老的（保留系统提示 + 最近 N 条）
  2. LLM 摘要压缩: 用 LLM 把老对话总结成一段摘要，保留摘要 + 最近几轮

使用方式（推荐用 compress_history，它是唯一实现）:
  from course_agent.memory import compress_history

  # 在节点里调用 LLM 之前：老对话压成摘要文本，返回要保留的最近消息
  summary, recent = compress_history(state["messages"], llm)

summarize_old_messages 是保留的兼容写法，内部同样走 compress_history。

为什么不直接在 state 上改?
  LangGraph 的 messages 用 add_messages reducer，追加容易修改难。
  我们在"读出来用"的时候做压缩，state 里还是完整历史，
  这样既不会爆 token，又不丢完整历史（需要时还能查）。
"""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from config import MAX_MESSAGES, ENABLE_SUMMARY, SUMMARY_THRESHOLD, SUMMARY_KEEP_LAST_N
from course_agent.logger import logger


# 摘要文本的标记。放在消息里/拼进 system prompt 时靠它识别，
# 这样别的地方（如 synthesizer）能判断"这段是历史摘要"而不是用户或助教说的话。
SUMMARY_MARK = "【对话历史摘要】"


# ============================================================
# 工具函数：估算 token 数
# ============================================================

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数量

    中文模型没有 tiktoken 那么精确，用经验公式:
      - 中文: 1 token ≈ 1.5 个字
      - 英文: 1 token ≈ 4 个字母
    所以取个折中: 字符数 / 2.5 ≈ token 数

    这个估算是给裁剪用的，不需要特别精确。
    """
    if not text:
        return 0
    return max(1, len(text) // 2)


def count_messages_tokens(messages: list[BaseMessage]) -> int:
    """统计一组消息的总 token 数（估算）"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        # 工具调用的参数也算 token
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(str(tc.get("args", "")))
    return total


# ============================================================
# 策略一：滑动窗口裁剪
# ============================================================

def trim_messages(
    messages: list[BaseMessage],
    max_messages: int = None,
    keep_system: bool = True,
) -> list[BaseMessage]:
    """滑动窗口裁剪消息

    规则:
      1. 系统消息（SystemMessage）永远保留
      2. 保留最近 max_messages 条非系统消息
      3. 保持消息顺序不变

    Args:
        messages: 完整的消息列表
        max_messages: 最多保留多少条非系统消息（默认用配置）
        keep_system: 是否保留系统消息

    Returns:
        裁剪后的消息列表
    """
    if max_messages is None:
        max_messages = MAX_MESSAGES

    # 分离系统消息和普通消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # 裁剪普通消息（超过 max_messages 才裁）
    if len(other_msgs) > max_messages:
        trimmed_count = len(other_msgs) - max_messages
        other_msgs = other_msgs[-max_messages:]
        logger.info(
            "消息裁剪: %d条 → %d条（裁掉最老的%d条）",
            len(messages), max_messages + len(system_msgs) if keep_system else max_messages,
            trimmed_count,
        )

    if keep_system:
        return system_msgs + other_msgs
    else:
        return other_msgs


# ============================================================
# 策略二：LLM 摘要压缩
# ============================================================

SUMMARY_PROMPT = """你是一个对话摘要助手。请将上面的对话历史总结成一段简洁的摘要。

要求:
1. 保留关键信息：用户问了什么、AI 回答了什么、调用了哪些工具
2. 按时间顺序描述，用第三人称
3. 不要遗漏重要的知识点和结论
4. 摘要控制在 300 字以内
5. 如果有多个话题，用"1.xxx 2.xxx"的格式列出

直接输出摘要内容，不要加多余的话。"""


# 摘要结果缓存。
# 我们是"读时压缩"（state 里保留完整历史），所以一旦超过阈值，
# 之后每一轮都会重新走一次摘要 —— 而摘要请求要把几十条老消息全发一遍，
# 比正常请求还贵。这里按"这段历史是否已经摘要过"做摊还：
# 同一段对话每新增 _RESUMMARY_STEP 条老消息才真正重算一次。
_SUMMARY_CACHE: dict = {}
_RESUMMARY_STEP = 10
_MAX_SUMMARY_CACHE = 128


def _history_key(old_msgs: list[BaseMessage]) -> str | None:
    """给一段老历史生成稳定的缓存键；拿不到可靠标识时返回 None（不缓存）

    用消息 id：LangGraph 的 add_messages 会给每条消息分配 id，
    且在后续轮次从 checkpoint 反序列化时保持不变，所以它既能跨轮命中，
    又能天然区分不同会话。

    为什么不退化成"首条消息内容的哈希"：两段不同的对话如果开头一模一样，
    就会共用同一份摘要 —— 测试里立刻暴露了这个问题。
    宁可不缓存，也不冒串用摘要的风险。
    """
    return str(getattr(old_msgs[0], "id", None) or "") or None


def _cache_summary(key: str, covered: int, text: str):
    _SUMMARY_CACHE[key] = (covered, text)
    while len(_SUMMARY_CACHE) > _MAX_SUMMARY_CACHE:
        _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))


# 内部工具型 LLM 调用打的 tag —— 流层据此把它们丢掉（见 server.sse_content_event）。
#
# 为什么必须有：LangGraph 的 messages 模式会把图内**每一次** LLM 调用都推出去，
# 包括本模块在 planner / synthesizer **内部**跑的这次摘要调用。实测确认：
#   · 它带着**同一个** langgraph_node 流出去 → 光按节点做白名单挡不住；
#   · 给这次调用加 stream=False 也没用 —— 只是从 token 流变成一条整消息，消息照样在流里。
# 摘要属于"内部工具调用"，从来不该出现在用户的对话流里，所以给它打个 tag。
INTERNAL_LLM_TAG = "internal_llm"


def compress_history(
    messages: list[BaseMessage],
    llm=None,
    threshold: int = None,
    keep_last_n: int = None,
) -> tuple[str, list[BaseMessage]]:
    """把历史压缩成 (摘要文本, 要保留的最近消息)

    这是摘要压缩的**唯一实现**，synthesizer 节点直接用它。

    规则:
      1. 未开启摘要 / 没传 llm / 消息数没超阈值 → 退化为滑动窗口，摘要为空字符串
      2. 超过阈值 → 老消息交给 LLM 总结成一段文本，保留最近 keep_last_n 条
      3. LLM 失败 → 降级为滑动窗口，绝不影响正常对话

    Returns:
        (summary_text, recent_messages)。summary_text 为空表示没有摘要，
        调用方按"无摘要"处理即可。
    """
    if not messages:
        return "", []

    if threshold is None:
        threshold = SUMMARY_THRESHOLD
    if keep_last_n is None:
        keep_last_n = SUMMARY_KEEP_LAST_N

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # 未触发摘要时也要走滑动窗口。
    # 注意不能"没超阈值就原样返回" —— 那等于在阈值以下完全没有窗口保护，
    # 25 条消息会全量塞给模型（阈值 30 时就是这种情况）。
    if not ENABLE_SUMMARY or llm is None or len(other_msgs) <= threshold:
        return "", trim_messages(messages, max_messages=MAX_MESSAGES, keep_system=False)

    old_msgs = other_msgs[:-keep_last_n]
    recent_msgs = other_msgs[-keep_last_n:]
    if not old_msgs:
        return "", recent_msgs

    # 已经摘要过且新增不多，直接复用，避免每轮都重发几十条老消息给模型
    cache_key = _history_key(old_msgs)
    if cache_key:
        cached = _SUMMARY_CACHE.get(cache_key)
        if cached and len(old_msgs) - cached[0] < _RESUMMARY_STEP:
            logger.debug("复用已有对话摘要（已覆盖 %d 条，当前 %d 条）", cached[0], len(old_msgs))
            return cached[1], recent_msgs

    conversation_text = _format_conversation(old_msgs)
    summary_prompt = f"以下是一段对话历史：\n\n{conversation_text}\n\n{SUMMARY_PROMPT}"

    try:
        from course_agent.errors import safe_llm_invoke

        response = safe_llm_invoke(
            # 打 tag：这次摘要调用是内部工具调用，不该流给用户
            llm.with_config(tags=[INTERNAL_LLM_TAG]),
            [SystemMessage(content="你是一个简洁的摘要助手。"),
             HumanMessage(content=summary_prompt)],
            context="conversation_summary",
        )
        summary_text = (response.content or "").strip()
        if summary_text:
            logger.info(
                "对话摘要: %d条老消息 → 摘要约%d字，保留最近%d条",
                len(old_msgs), len(summary_text), keep_last_n,
            )
            if cache_key:
                _cache_summary(cache_key, len(old_msgs), summary_text)
            return summary_text, recent_msgs
        logger.warning("对话摘要返回空内容，降级为滑动窗口")
    except Exception as e:
        # 摘要失败就降级成简单裁剪，不能影响正常对话
        logger.warning("对话摘要失败，降级为滑动窗口裁剪: %s", e)

    return "", trim_messages(messages, max_messages=MAX_MESSAGES, keep_system=False)


def summarize_old_messages(
    messages: list[BaseMessage],
    llm=None,
    threshold: int = None,
    keep_last_n: int = None,
) -> list[BaseMessage]:
    """LLM 摘要压缩老消息（compress_history 的列表版包装）

    规则:
      1. 系统消息永远保留
      2. 如果消息数没超过 threshold，退化为滑动窗口裁剪
      3. 超过后：老消息 → LLM 总结成一段摘要 → 摘要消息 + 最近 keep_last_n 条

    Returns:
        [系统消息] + [摘要消息（如果触发了摘要）] + [最近 N 条消息]
    """
    summary_text, recent = compress_history(
        messages, llm, threshold=threshold, keep_last_n=keep_last_n
    )
    result = [m for m in messages if isinstance(m, SystemMessage)]
    if summary_text:
        result.append(SystemMessage(
            content=f"{SUMMARY_MARK}\n{summary_text}\n\n（以上是之前对话的总结，下面是最近的对话内容）"
        ))
    result.extend(recent)
    return result


def _format_conversation(messages: list[BaseMessage]) -> str:
    """把消息列表格式化成可读的对话文本，给 LLM 摘要用"""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc.get("name", "") for tc in msg.tool_calls]
                lines.append(f"AI: {content} [调用工具: {', '.join(tool_names)}]")
            else:
                lines.append(f"AI: {content}")
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool")
            content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"[工具 {name} 返回]: {content_preview}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"[系统]: {msg.content[:100]}...")
        else:
            lines.append(f"[{type(msg).__name__}]: {str(msg.content)[:100]}")
    return "\n".join(lines)


# ============================================================
# 便捷函数：获取对话轮数
# ============================================================

def count_turns(messages: list[BaseMessage]) -> int:
    """统计对话轮数（HumanMessage 的数量 = 用户问了多少次）"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def get_last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None
