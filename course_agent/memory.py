"""对话记忆管理

两种策略:
  1. 滑动窗口裁剪: 消息太多时直接丢掉最老的（保留系统提示 + 最近 N 条）
  2. LLM 摘要压缩: 用 LLM 把老对话总结成一段摘要，保留摘要 + 最近几轮

使用方式:
  from course_agent.memory import trim_messages, summarize_old_messages

  # 在节点里调用 LLM 之前，先裁剪/摘要
  messages = trim_messages(state["messages"], max_messages=20)
  # 或者更高级的摘要压缩
  messages = summarize_old_messages(state["messages"], llm)

为什么不直接在 state 上改?
  LangGraph 的 messages 用 add_messages reducer，追加容易修改难。
  我们在"读出来用"的时候做裁剪，state 里还是完整历史，
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


def summarize_old_messages(
    messages: list[BaseMessage],
    llm=None,
    threshold: int = None,
    keep_last_n: int = None,
) -> list[BaseMessage]:
    """LLM 摘要压缩老消息

    规则:
      1. 系统消息永远保留
      2. 如果消息数没超过 threshold，直接返回（不摘要）
      3. 超过后：老消息 → LLM 总结成一段摘要 → 摘要 + 最近 keep_last_n 条

    Args:
        messages: 完整的消息列表
        llm: LLM 实例（必须传，用来生成摘要）
        threshold: 超过多少条消息触发摘要
        keep_last_n: 保留最近多少条（不被摘要）

    Returns:
        [系统消息] + [摘要消息（如果触发了摘要）] + [最近 N 条消息]
    """
    if not ENABLE_SUMMARY or llm is None:
        return trim_messages(messages, max_messages=MAX_MESSAGES)

    if threshold is None:
        threshold = SUMMARY_THRESHOLD
    if keep_last_n is None:
        keep_last_n = SUMMARY_KEEP_LAST_N

    # 分离系统消息和普通消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(other_msgs) <= threshold:
        return messages  # 没超阈值，不用摘要

    # 老消息（要被摘要的部分）和新消息（保留的部分）
    old_msgs = other_msgs[:-keep_last_n]
    recent_msgs = other_msgs[-keep_last_n:]

    # 构造摘要请求
    conversation_text = _format_conversation(old_msgs)
    summary_prompt = f"以下是一段对话历史：\n\n{conversation_text}\n\n{SUMMARY_PROMPT}"

    try:
        from course_agent.errors import safe_llm_invoke

        response = safe_llm_invoke(
            llm,
            [SystemMessage(content="你是一个简洁的摘要助手。"),
             HumanMessage(content=summary_prompt)],
            context="conversation_summary"
        )
        summary_text = response.content

        logger.info(
            "对话摘要: %d条老消息 → 摘要约%d字，保留最近%d条",
            len(old_msgs), len(summary_text), keep_last_n,
        )

        # 把摘要包装成一条系统消息
        summary_msg = SystemMessage(
            content=f"【对话历史摘要】\n{summary_text}\n\n（以上是之前对话的总结，下面是最近的对话内容）"
        )

        result = []
        if system_msgs:
            result.extend(system_msgs)
        result.append(summary_msg)
        result.extend(recent_msgs)
        return result

    except Exception as e:
        # 摘要失败就降级成简单裁剪，不能影响正常对话
        logger.warning("对话摘要失败，降级为滑动窗口裁剪: %s", e)
        return trim_messages(messages, max_messages=keep_last_n)


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
