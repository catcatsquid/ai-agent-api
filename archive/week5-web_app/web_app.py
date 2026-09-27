"""AI Agent Web应用 - 美观的聊天界面

支持：
1. 实时流式对话
2. 多轮对话历史
3. Agent执行过程可视化
4. 知识来源标注
5. 暗黑模式切换

启动命令：
python web_app.py
访问：http://127.0.0.1:8004
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
from datetime import datetime
import asyncio

# 导入你的Agent服务
from course_agent.server import app as agent_app
from course_agent.graph import graph
from schemas import ChatRequest, APIResponse

# 创建新的FastAPI应用
web_app = FastAPI(title="AI Agent Web应用", version="1.0.0")

# CORS配置
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
web_app.mount("/static", StaticFiles(directory="static"), name="static")

# 聊天历史存储（生产环境应该用数据库）
chat_history = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # thread_id: WebSocket

    async def connect(self, websocket: WebSocket, thread_id: str):
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        # 发送历史消息
        if thread_id in chat_history:
            await websocket.send_text(json.dumps({
                "type": "history",
                "data": chat_history[thread_id]
            }))

    def disconnect(self, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]

    async def broadcast(self, message: dict, thread_id: str):
        if thread_id in self.active_connections:
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# 静态文件目录
os.makedirs("static", exist_ok=True)

# 根路径 - 返回聊天界面
@web_app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Demo</title>
    <meta http-equiv="refresh" content="0; url=/static/simple_chat.html">
</head>
<body>
    <a href="/static/simple_chat.html">点击进入聊天界面</a>
</body>
</html>"""
    <style>
        :root {
            --primary-color: #4f46e5;
            --secondary-color: #7c3aed;
            --bg-primary: #ffffff;
            --bg-secondary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border-color: #e5e7eb;
        }

        [data-theme="dark"] {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --border-color: #374151;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .message-bubble {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            animation: fadeIn 0.3s ease;
        }

        .user-message {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            margin-left: 20%;
        }

        .agent-message {
            margin-right: 20%;
        }

        .tool-call {
            background-color: rgba(79, 70, 229, 0.1);
            border-left: 4px solid var(--primary-color);
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .source-tag {
            display: inline-block;
            background-color: rgba(124, 58, 237, 0.1);
            color: var(--secondary-color);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-top: 4px;
        }

        .typing-indicator {
            display: inline-flex;
            padding: 12px 20px;
            background-color: var(--bg-secondary);
            border-radius: 18px;
            margin: 10px auto;
            width: fit-content;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--primary-color);
            margin: 0 3px;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .sidebar {
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }

        .chat-container {
            height: calc(100vh - 120px);
        }

        .input-area {
            height: 80px;
            background-color: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
        }
    </style>
</head>
<body class="bg-gray-50">
    <div class="flex h-screen">
        <!-- 侧边栏 -->
        <div class="w-64 sidebar p-4">
            <h2 class="text-xl font-bold mb-4 flex items-center">
                <i class="fas fa-robot mr-2 text-indigo-600"></i>
                AI Agent
            </h2>

            <!-- 对话列表 -->
            <div class="mb-4">
                <h3 class="text-sm font-semibold text-gray-500 mb-2">历史对话</h3>
                <div id="chatHistory" class="space-y-2">
                    <div class="text-xs text-gray-400">暂无历史记录</div>
                </div>
            </div>

            <!-- 功能按钮 -->
            <div class="space-y-2">
                <button onclick="clearChat()" class="w-full py-2 px-3 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition">
                    <i class="fas fa-trash mr-1"></i> 清空对话
                </button>
                <button onclick="toggleTheme()" class="w-full py-2 px-3 text-sm bg-gray-700 text-white rounded hover:bg-gray-800 transition">
                    <i class="fas fa-moon mr-1"></i> 切换主题
                </button>
            </div>

            <!-- 功能说明 -->
            <div class="mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <h4 class="text-sm font-semibold mb-1">功能特点</h4>
                <ul class="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    <li><i class="fas fa-check mr-1"></i> Plan-and-Execute架构</li>
                    <li><i class="fas fa-check mr-1"></i> 多轮对话持久化</li>
                    <li><i class="fas fa-check mr-1"></i> 智能工具调用</li>
                    <li><i class="fas fa-check mr-1"></i> 知识来源标注</li>
                </ul>
            </div>
        </div>

        <!-- 主聊天区域 -->
        <div class="flex-1 flex flex-col">
            <!-- 聊天消息区域 -->
            <div class="chat-container p-4 overflow-y-auto" id="chatMessages">
                <div class="text-center text-gray-400 mt-20">
                    <i class="fas fa-comments text-6xl mb-4"></i>
                    <p class="text-lg">欢迎来到AI Agent智能助手</p>
                    <p class="text-sm mt-2">我将帮助您解答计算机网络和操作系统相关问题</p>
                </div>
            </div>

            <!-- 输入区域 -->
            <div class="input-area p-4">
                <form onsubmit="sendMessage(event)" class="flex items-center space-x-3">
                    <input
                        type="text"
                        id="messageInput"
                        placeholder="输入你的问题..."
                        class="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        autocomplete="off"
                    >
                    <button
                        type="submit"
                        class="px-6 py-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition flex items-center"
                    >
                        <i class="fas fa-paper-plane mr-2"></i>
                        发送
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8004/ws');
        let threadId = generateThreadId();
        let messageCount = 0;

        function generateThreadId() {
            return 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        function clearChat() {
            document.getElementById('chatMessages').innerHTML = `
                <div class="text-center text-gray-400 mt-20">
                    <i class="fas fa-comments text-6xl mb-4"></i>
                    <p class="text-lg">欢迎来到AI Agent智能助手</p>
                    <p class="text-sm mt-2">我将帮助您解答计算机网络和操作系统相关问题</p>
                </div>
            `;
            messageCount = 0;
        }

        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        }

        // 加载保存的主题
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);

        function addMessage(content, type = 'user', metadata = {}) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');

            if (type === 'user') {
                messageDiv.className = 'message-bubble user-message text-right';
                messageDiv.innerHTML = `
                    <div class="flex justify-end items-start">
                        <div>
                            <p class="font-semibold mb-1">你</p>
                            <p>${content}</p>
                        </div>
                        <img src="https://api.dicebear.com/7.x/personas/svg?seed=user" alt="User" class="w-8 h-8 rounded-full ml-2">
                    </div>
                `;
            } else {
                messageDiv.className = 'message-bubble agent-message';

                let messageContent = `
                    <div class="flex items-start">
                        <img src="https://api.dicebear.com/7.x/bots/svg?seed=ai" alt="AI" class="w-8 h-8 rounded-full mr-2">
                        <div class="flex-1">
                            <p class="font-semibold mb-1">AI Agent</p>
                            <p>${content}</p>
                `;

                if (metadata.sources && metadata.sources.length > 0) {
                    messageContent += '<div class="mt-2">';
                    metadata.sources.forEach(source => {
                        messageContent += `<span class="source-tag">${source}</span>`;
                    });
                    messageContent += '</div>';
                }

                messageContent += '</div></div>';
                messageDiv.innerHTML = messageContent;
            }

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addToolCall(toolName, query) {
            const chatMessages = document.getElementById('chatMessages');
            const toolDiv = document.createElement('div');
            toolDiv.className = 'message-bubble tool-call text-sm';
            toolDiv.innerHTML = `
                <i class="fas fa-tools mr-2"></i>
                <strong>工具调用:</strong> ${toolName}
                <br><em>查询: ${query}</em>
            `;
            chatMessages.appendChild(toolDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addTypingIndicator() {
            const chatMessages = document.getElementById('chatMessages');
            const typingDiv = document.createElement('div');
            typingDiv.id = 'typingIndicator';
            typingDiv.className = 'typing-indicator';
            typingDiv.innerHTML = `
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                AI Agent正在思考...
            `;
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }

        async function sendMessage(event) {
            event.preventDefault();
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            // 添加用户消息
            addMessage(message, 'user');
            input.value = '';

            // 显示思考中
            addTypingIndicator();

            try {
                const response = await fetch('/agent/qa/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: message,
                        mode: "plan",
                        thread_id: threadId
                    })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullResponse = '';
                let currentTool = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') {
                                removeTypingIndicator();
                                if (fullResponse) {
                                    addMessage(fullResponse, 'agent', { sources: extractSources(fullResponse) });
                                }
                                return;
                            }

                            try {
                                const parsed = JSON.parse(data);

                                if (parsed.type === 'tool_call') {
                                    addToolCall(parsed.tool, parsed.query);
                                } else if (parsed.type === 'content') {
                                    fullResponse += parsed.content;
                                    // 实时显示内容
                                    removeTypingIndicator();
                                    addMessage(fullResponse, 'agent', { sources: extractSources(fullResponse) });
                                } else if (parsed.type === 'source') {
                                    // 可以在这里更新来源标注
                                }
                            } catch (e) {
                                // 忽略解析错误
                            }
                        }
                    }
                }

            } catch (error) {
                removeTypingIndicator();
                addMessage('抱歉，发生了错误: ' + error.message, 'agent');
            }
        }

        function extractSources(response) {
            // 简单提取来源标签
            const sources = [];
            const sourceRegex = /来源：([^(）)]+)/g;
            let match;
            while ((match = sourceRegex.exec(response)) !== null) {
                sources.push(match[1].trim());
            }
            return sources;
        }
    </script>
</body>
</html>
    """

# 聊天API端点 - 简化版本，模拟AI回复
@web_app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    处理聊天请求，返回流式响应
    """
    import asyncio

    async def generate_response():
        # 模拟的AI回复
        responses = {
            "TCP": "TCP是面向连接的传输层协议。三次握手过程：1）客户端发送SYN；2）服务器回复SYN-ACK；3）客户端回复ACK。这样确保双方都准备好进行数据传输。",
            "进程": "进程是程序的一次执行，是系统资源分配的基本单位；线程是进程内的执行单元。主要区别：进程有独立内存空间，线程共享进程资源；进程切换开销大，线程切换开销小。",
            "死锁": "死锁的四个必要条件：1）互斥条件；2）请求与保持条件；3）不可剥夺条件；4）循环等待条件。避免方法：破坏其中任一条件，如资源排序、银行家算法等。",
            "虚拟内存": "虚拟内存是一种内存管理技术，通过页表机制将逻辑地址转换为物理地址。它允许程序使用比实际物理内存更大的地址空间，提供了内存保护和隔离。",
            "HTTP": "HTTP是超文本传输协议，基于TCP/IP，是无状态协议。HTTPS是HTTP的安全版本，通过SSL/TLS加密通信。主要区别：HTTPS更安全，需要证书，端口不同（443 vs 80）。"
        }

        # 简单匹配关键词
        response = "这是一个很好的问题！作为AI Agent，我会为你提供专业的解答。"
        for keyword, answer in responses.items():
            if keyword in request.query:
                response = answer
                break

        # 模拟逐字输出
        for char in response:
            yield f"data: {json.dumps({'type': 'content', 'content': char}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
    """
    处理聊天请求，返回流式响应
    """
    try:
        # 构建初始消息
        messages = [
            {"role": "system", "content": "你是一个专业的计算机网络和操作系统课程助教。"},
            {"role": "user", "content": request.query}
        ]

        # 获取对话历史（如果有）
        if request.thread_id and request.thread_id in chat_history:
            history = chat_history[request.thread_id]
            messages.extend(history)

        # 调用Agent
        config = {"configurable": {"thread_id": request.thread_id}}
        response_stream = graph.stream(
            {"messages": messages, "max_steps": request.max_steps},
            config
        )

        # 处理流式响应
        full_response = ""
        tool_calls = []

        for chunk in response_stream:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if hasattr(message, 'content') and message.content:
                        full_response += message.content
                        # 返回流式更新
                        yield f"data: {json.dumps({'type': 'content', 'content': message.content}, ensure_ascii=False)}\n\n"

        # 发送结束信号
        yield f"data: [DONE]\n\n"

        # 保存对话历史
        if request.thread_id:
            if request.thread_id not in chat_history:
                chat_history[request.thread_id] = []

            # 添加用户消息
            chat_history[request.thread_id].append({
                "role": "user",
                "content": request.query,
                "timestamp": datetime.now().isoformat()
            })

            # 添加AI回复
            chat_history[request.thread_id].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })

            # 限制历史记录长度（保留最近20条消息）
            if len(chat_history[request.thread_id]) > 20:
                chat_history[request.thread_id] = chat_history[request.thread_id][-20:]

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': '抱歉，发生了错误: ' + str(e)}, ensure_ascii=False)}\n\n"
        yield f"data: [DONE]\n\n"

# WebSocket端点
@web_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 这里可以处理WebSocket连接
    await websocket.close()

# 将原有的Agent应用挂载到子路径
web_app.mount("/agent", agent_app)

if __name__ == "__main__":
    print("Starting AI Agent Web App...")
    print("Access at: http://127.0.0.1:8004")
    print("Agent API docs: http://127.0.0.1:8004/agent/docs")
    uvicorn.run("web_app:web_app", host="0.0.0.0", port=8004, reload=True)