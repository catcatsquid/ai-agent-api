"""B3: 真实考试题测试 v2（修复 System Prompt 后）

与 v1 相同的 10 道题，对比修复效果。
"""

import json
import time
import requests

API_URL = "http://127.0.0.1:8003/qa/ask"
OUTPUT_FILE = "debug/003-exam-qa-results-v2.json"

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "进程和线程有什么区别？",
        "topic": "操作系统-进程线程",
        "reference": [
            "进程是资源分配的基本单位，线程是调度的基本单位",
            "进程有独立的地址空间，同一进程内线程共享地址空间",
            "线程上下文切换比进程切换开销小",
            "进程间通信需要 IPC，线程间可直接读写共享变量",
        ],
    },
    {
        "id": 2,
        "question": "进程有哪几种基本状态？它们之间是如何转换的？",
        "topic": "操作系统-进程状态",
        "reference": [
            "就绪态、运行态、阻塞态（三状态模型）",
            "就绪→运行：调度程序选中",
            "运行→就绪：时间片用完/被抢占",
            "运行→阻塞：等待I/O或事件",
            "阻塞→就绪：I/O完成或事件到达",
        ],
    },
    {
        "id": 3,
        "question": "死锁产生的四个必要条件是什么？",
        "topic": "操作系统-死锁",
        "reference": [
            "互斥条件：资源不能共享，只能一个进程使用",
            "占有并等待：已持有资源又请求新资源",
            "不可剥夺：资源不能被强行夺走",
            "循环等待：进程间形成资源等待环",
        ],
    },
    {
        "id": 4,
        "question": "常见的进程调度算法有哪些？各有什么特点？",
        "topic": "操作系统-调度算法",
        "reference": [
            "FCFS先来先服务：简单，平均等待时间长，对短作业不利",
            "SJF短作业优先：平均等待时间最短，可能饥饿",
            "时间片轮转：公平，响应快，时间片太小开销大",
            "优先级调度：按优先级调度，可能饥饿",
            "多级反馈队列：综合以上优点，动态调整优先级",
        ],
    },
    {
        "id": 5,
        "question": "什么是临界区？进入临界区要遵循哪些准则？",
        "topic": "操作系统-同步互斥",
        "reference": [
            "临界区是进程中访问临界资源的那段代码",
            "空闲让进：临界区空闲时允许进程进入",
            "忙则等待：已有进程在临界区时其他必须等待",
            "有限等待：不能无限等待，避免饥饿",
            "让权等待：不能进入时应让出CPU",
        ],
    },
    {
        "id": 6,
        "question": "系统调用的执行过程是怎样的？",
        "topic": "操作系统-系统调用",
        "reference": [
            "用户程序通过陷入指令（ecall/syscall）触发",
            "CPU从用户态切换到内核态",
            "保存用户上下文（寄存器等）",
            "根据系统调用号查找系统调用表，执行内核函数",
            "返回结果，恢复上下文，切回用户态",
        ],
    },
    {
        "id": 7,
        "question": "TCP 三次握手的过程是什么？为什么需要三次？",
        "topic": "计算机网络-TCP",
        "reference": [
            "第一次：客户端发SYN，seq=x",
            "第二次：服务端回SYN+ACK，ack=x+1，seq=y",
            "第三次：客户端回ACK，ack=y+1",
            "三次才能确认双方收发能力都正常",
            "防止已失效的连接请求到达服务端造成错误连接",
        ],
    },
    {
        "id": 8,
        "question": "TCP 和 UDP 有什么区别？",
        "topic": "计算机网络-TCP/UDP",
        "reference": [
            "TCP面向连接，UDP无连接",
            "TCP可靠（确认/重传/流量控制），UDP不可靠",
            "TCP面向字节流，UDP面向报文",
            "TCP有拥塞控制，UDP没有",
            "TCP首部开销大（20字节），UDP小（8字节）",
            "TCP适合文件传输等可靠场景，UDP适合实时应用",
        ],
    },
    {
        "id": 9,
        "question": "分页和分段有什么区别？",
        "topic": "操作系统-内存管理",
        "reference": [
            "分页是信息的物理单位，大小固定；分段是逻辑单位，大小不固定",
            "分页对用户透明，分段是用户可见的",
            "分页地址空间是一维的，分段是二维的（段号+段内偏移）",
            "分页容易产生内部碎片，分段容易产生外部碎片",
            "分段便于共享和保护，分页不便",
        ],
    },
    {
        "id": 10,
        "question": "什么是虚拟内存？它有什么作用？",
        "topic": "操作系统-虚拟内存",
        "reference": [
            "虚拟内存是操作系统提供的一种内存管理技术",
            "让每个进程以为自己拥有连续的完整地址空间",
            "实际物理内存可以不连续，通过页表映射",
            "作用：扩大地址空间、内存保护、内存共享、简化编程",
            "实现基础：局部性原理",
        ],
    },
]


def main():
    results = []
    print(f"共 {len(TEST_QUESTIONS)} 道题，开始测试（修复后版本）...\n")

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {q['question']}")
        print(f"  知识点: {q['topic']}")

        try:
            resp = requests.post(API_URL, json={"query": q["question"]}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            tool_names = [t.get("name", "") for t in data.get("tool_calls", [])]
            print(f"  调用工具: {tool_names if tool_names else '无'}")
            answer = data.get("answer", "")
            print(f"  回答预览: {answer[:120]}...")
            results.append({
                "question": q["question"],
                "topic": q["topic"],
                "reference": q["reference"],
                "answer": answer,
                "tool_calls": data.get("tool_calls", []),
                "error": None,
            })
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                "question": q["question"],
                "topic": q["topic"],
                "reference": q["reference"],
                "answer": "",
                "tool_calls": [],
                "error": str(e),
            })

        print()
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    used_tool = sum(1 for r in results if r["tool_calls"])
    no_error = sum(1 for r in results if not r["error"])
    print(f"测试完成，结果已保存到 {OUTPUT_FILE}")
    print(f"\n统计:")
    print(f"  总题数: {len(results)}")
    print(f"  成功回答: {no_error}")
    print(f"  调用了工具: {used_tool}")
    print(f"  未调用工具: {len(results) - used_tool}")


if __name__ == "__main__":
    main()
