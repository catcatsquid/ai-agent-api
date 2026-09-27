@echo off
echo ===================================
echo  AI Agent Demo 启动脚本
echo ===================================

REM 检查虚拟环境
if not exist ".venv" (
    echo 创建虚拟环境...
    python -m venv .venv
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 安装依赖
echo 检查并安装依赖...
pip install -r requirements.txt

REM 启动Web应用
echo 启动Web应用...
echo 访问地址: http://127.0.0.1:8004
echo 按 Ctrl+C 停止服务
python web_app.py