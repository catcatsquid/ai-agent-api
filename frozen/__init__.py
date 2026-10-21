"""冻结的对照实现：不参与运行，只作为等价性验证的参照物与测试对象。

`tutor_agent.py` 是手写的 tool-calling 循环。它的 HTTP 入口已删除，
`tests/test_tutor_reference.py` 守着"它不许出现在路由表与前端接线里"。
保留原因见 README 的代码结构一节。
"""
