# Web应用问题记录

## 问题描述
尝试创建聊天界面时遇到服务无法启动的问题。

## 时间
2026-09-10

## 现象
1. 运行 `python web_final.py` 时出现端口占用错误
2. 即使切换到8888端口仍然无法启动
3. 测试代码显示 `ConnectionRefusedError: [WinError 10061] 由于目标计算机积极拒绝，无法连接。`

## 尝试的解决方案

### 1. 端口冲突
- 检测到端口8004被PID 36856占用
- 使用 `taskkill //F //PID 36856` 停止进程
- 尝试切换到8888端口
- 问题依旧存在

### 2. 路由冲突
- 发现健康检查路由 `/` 覆盖了主页路由
- 将健康检查移到 `/api/health`
- 更新聊天界面中的API请求路径
- 修改JavaScript从 `/qa/ask` 改为 `/api/qa/ask`

### 3. 服务格式问题
- 检查API响应格式是否符合前端期望
- 确保返回格式为 `{code, msg, data: {answer, tools_used}}`
- 添加了错误处理中的data字段

### 4. 尝试简化服务
- 创建了 `simple_server.py` 最小化版本
- 移除了所有复杂依赖
- 仅保留最基本的FastAPI服务

## 可能的原因
1. Python环境问题
2. 网络配置问题
3. FastAPI/Uvicorn安装问题
4. Windows系统权限问题
5. 防火墙或安全软件拦截

## 待检查项
1. 检查Python环境和依赖
2. 确认网络配置
3. 检查防火墙设置
4. 尝试在另一台机器上运行
5. 检查是否有杀毒软件拦截

## 代码位置
- `web_app.py` - 初始版本（端口8004）
- `web_app_improved.py` - 改进版本（编码修复）
- `web_final.py` - 最终版本（8888端口）
- `simple_server.py` - 简化版本
- `test_api.py` - 测试脚本
- `stop_port.ps1` - 停止端口脚本

## 清理计划
待问题解决后，删除所有测试文件：
- web_app.py
- web_app_improved.py
- web_final.py
- simple_server.py
- test_api.py
- test_access.py
- simple_test.py
- stop_port.ps1

## 相关文件
- `paper_agent/static/chat.html` - 聊天界面
- `web_app_improved.py` - 主要的服务代码
- `routers/health.py` - 健康检查路由
- `schemas.py` - API响应格式

## 下一步建议
1. 先确保基本的FastAPI服务能运行
2. 逐步添加功能
3. 检查系统环境
4. 尝试使用Docker运行