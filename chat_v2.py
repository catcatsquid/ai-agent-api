import os
import sys
import json #json.dumps()把Python对象翻译成字符串，json.loads()把字符串翻译成Python对象
from datetime import datetime
import argparse #命令行参数解析工具库
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LOG_FILE = "chat_log.json" #文件名字符串,告诉程序对话记录保存在哪个文件夹里.


def chat(prompt, model="glm-4-flash", system_prompt=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages}

    try:
        resp = requests.post(URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        print(f"API错误: {resp.status_code} - {resp.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
        sys.exit(1)
    except (KeyError, IndexError):
        print(f"响应格式异常，原始数据: {resp.text}")
        sys.exit(1)

#把一轮问答记录追加保存到chat_log.json文件里,如果文件不存在就创建一个新的文件
#如果文件存在就读取原来的内容,然后把新的问答记录追加到原来的内容里,最后再写回到文件里。
def save_log(question, answer, model, output_file):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "question": question,
        "answer": answer,
    }
    try:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(entry)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"[已保存到 {output_file}]")
    except PermissionError:
        print(f"[警告: 无法写入 {output_file}，权限不足]")
    except Exception as e:
        print(f"[警告: 保存失败 - {e}]")


def read_log(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[暂无历史记录]")
        return []
    except json.JSONDecodeError:
        print("[日志文件损坏，已忽略]")
        return []


def main():
    #创建一个解析器对象,description是帮助信息的开头，epilog是帮助信息的结尾。
    #运行python chat_v2.py -h就能看到这些帮助信息。
    parser = argparse.ArgumentParser(
        description="智谱GLM命令行聊天工具",
        epilog="示例: python chat_v2.py -p \"什么是RAG\" --system \"你是Python老师\" --save",
    )
    #给解析器添加参数规则
    parser.add_argument("-p", "--prompt", help="你的问题")
    parser.add_argument("-m", "--model", default="glm-4-flash", help="模型名称 (默认: glm-4-flash)")
    parser.add_argument("-s", "--system", help="设定AI人设，如 \"你是毒舌mentor\"")
    parser.add_argument("--save", action="store_true", help="把本次对话保存到文件")
    parser.add_argument("--output", default="chat_log.json", help="保存到哪个文件 (默认: chat_log.json)")
    parser.add_argument("--history", action="store_true", help="查看历史对话记录")
    #执行解析命令行参数,把解析结果存到args对象里。args.prompt就是用户输入的-p参数，args.model就是-m参数，args.system就是-s参数。
    args = parser.parse_args()

    if not API_KEY:
        print("错误: 请在 .env 文件中设置 ZHIPU_API_KEY")
        sys.exit(1)

    #args.history是一个布尔值,如果用户输入了--history参数,就会显示历史记录,否则就会进行新的对话。
    if args.history:
        logs = read_log(args.output)
        for log in logs:
            print(f"\n[{log['time']}] 模型: {log['model']}")
            print(f"  问: {log['question']}")
            print(f"  答: {log['answer'][:80]}...")
        return

    #args.prompt是用户输入的-p参数,如果没有输入就会提示用户输入问题
    prompt = args.prompt or input("请输入你的问题: ")
    #.strip()方法去掉字符串首尾的空白字符
    if not prompt.strip():
        print("问题不能为空")
        sys.exit(1)

    print(f"\n用户: {prompt}")
    if args.system:
        print(f"人设: {args.system}")
    print(f"GLM: ", end="")

    reply = chat(prompt, model=args.model, system_prompt=args.system)
    print(reply)
    if args.save:
        save_log(prompt, reply, args.model, args.output)


if __name__ == "__main__":
    main()
