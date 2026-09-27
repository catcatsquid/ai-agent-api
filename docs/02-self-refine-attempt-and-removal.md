# Self-Refine 自检功能：尝试 → 踩坑 → 最终移除

## 背景

Day 3 深入阶段，想在 ReAct 循环基础上加一个 Self-Refine（自检优化）节点，让 Agent 回答问题后先自检一次质量，再返回给用户。

这是 LangGraph "加节点=加功能" 的典型演示。

---

## 尝试的过程

### 第一版：无脑追加

**思路**：加一个 `self_refine` 节点，LLM 回答后走自检，自检结果追加到 messages。

```
START → llm → route_after_llm → tools → llm → ...
                          ↘ self_refine → END
```

**代码**：
```python
def self_refine_node(state):
    messages = state["messages"] + [HumanMessage(content=REFINE_PROMPT)]
    response = llm.invoke(messages)
    return {"messages": [response], "refine_count": refine_count}
```

**问题**：LLM 自检后回复"通过"两个字，这个"通过"被追加到 messages，用户看到的答案是"通过"而不是真正的回答。

### 第二版：判断"通过"关键词

**思路**：检查 LLM 自检回复，如果是"通过"就不追加消息，否则追加改进版答案。

**代码**：
```python
if response.content.strip() == "通过":
    return {"refine_count": refine_count}  # 不加消息
else:
    return {"messages": [response], "refine_count": refine_count}  # 追加改进版
```

**问题**：
- 依赖"通过"这个关键词，LLM 不一定每次都说"通过"
- 如果 LLM 说"质量很好"或"没问题"，判断就失败了
- 过度耦合 LLM 的输出措辞

### 第三版：改 Prompt，统一追加

**思路**：改 REFINE_PROMPT，让 LLM 质量好时原样输出原答案，质量不好时输出改进版。不管哪种都追加到 messages，最后一条永远是最终答案。

**代码**：
```python
REFINE_PROMPT = """...
如果回答质量没问题，直接把原答案原样输出一遍。
如果需要改进，直接输出改进后的完整回答。"""

def self_refine_node(state):
    response = llm.invoke(messages)
    return {"messages": [response], "refine_count": refine_count}
```

**问题**：
- 每次都追加一条消息，messages 里有两条几乎一样的答案（原答案 + 自检输出）
- 浪费一次 LLM 调用（质量好的情况下等于白调一次）
- route_after_llm 的逻辑变复杂：要判断"调工具 / 自检 / 结束"三选一

---

## 遇到的核心矛盾

1. **LangGraph 的 add_messages reducer 只能追加，不能替换**
   - 想用自检结果替换原答案？做不到，只能追加
   - 追加后 messages 里有两条答案，取最后一条

2. **自检节点的路由逻辑变复杂**
   - 原来只有两个分支：调工具 / 结束
   - 加自检后变成三个：调工具 / 自检 / 结束
   - 还要加 `refine_count` 状态字段防止无限自检

3. **步数上限和自检的冲突**
   - 步数到上限时直接 END？那回答没自检
   - 步数到上限时走自检？那还在调工具时也会走自检（没有最终答案可自检）
   - 要区分"还在调工具"和"已经给出回答"两种情况

4. **图结构的设计困惑**
   - `self_refine → END`：自检完直接结束，不会回到 LLM → `route_after_llm` 里的"自检过了 → END"是死代码
   - `self_refine → llm`：自检完回到 LLM，可以多次自检，但容易无限循环
   - 两种方案都不干净

---

## 最终决定：移除自检功能

### 原因

1. **核心矛盾没解决**：add_messages 只能追加不能替换，导致自检结果和原答案共存
2. **收益不明显**：对于课程知识库问答，LLM 一次性回答质量已经够好，自检多调一次 API 反而增加延迟
3. **理解成本高**：自检节点的路由逻辑比 ReAct 循环本身还复杂，得不偿失
4. **时机不对**：当前阶段应该先把 ReAct 循环跑通跑稳，而不是加高级功能

### 移除的改动

| 文件 | 删了什么 |
|---|---|
| `state.py` | `refine_count` 字段 |
| `nodes.py` | `REFINE_PROMPT`、`self_refine_node`、`MAX_REFINE`，`route_after_llm` 回到简单二选一 |
| `graph.py` | `self_refine` 节点和对应的边 |

### 移除后的图结构

```
START → llm → route_after_llm → tools → llm → ...
                          ↘ END
```

回到最干净的 ReAct：调工具或结束，两条路。

---

## 经验总结

1. **框架特性限制设计**：LangGraph 的 `add_messages` reducer 是追加式的，想做"替换"语义很别扭。如果要做自检，需要换一种 State 设计（比如单独存 `answer` 字段而不是只靠 messages）。

2. **先跑通再加功能**：Self-Refine 是"锦上添花"的功能，ReAct 循环才是"雪中送炭"的核心。先把核心跑稳，高级功能等需要时再加。

3. **死代码要识别**：`self_refine → END` 的图结构导致 `route_after_llm` 里"自检过了 → END"永远不会执行。这种死代码在图结构里不容易发现，要画图理清执行路径。

4. **节点数和复杂度的关系**：每加一个节点，路由逻辑复杂度翻倍（2个节点→2条路，3个节点→可能6条路）。加节点前要想清楚路由逻辑能不能 hold 住。

---

## 相关文件

- `paper_agent/state.py`
- `paper_agent/nodes.py`
- `paper_agent/graph.py`

## 时间线

- 尝试时间：2026-09-10
- 移除时间：2026-09-10
