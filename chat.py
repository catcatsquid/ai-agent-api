import os #读取环境变量
import sys #获取命令行参数、程序退出
import requests #做http请求调用API
from dotenv import load_dotenv
#dotenv是Python库,读取项目里的.env文件，把文件里的密钥、密码这些配置加载到程序的环境变量里。

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def chat(prompt: str, model: str = "glm-4-flash") -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    } #请求头
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": "your name is Sam."},{"role": "user", "content": prompt}],
    } #请求体

    try:
        #向智谱接口发送 POST 网络请求。30秒没收到回复就判定超时，抛出异常
        resp = requests.post(URL, headers=headers, json=payload, timeout=30)
        #如果http状态码正常就继续跑，不正常就抛出HTTPError异常，跳到第一个except
        resp.raise_for_status()
        #把服务器返回的字符串翻译成Python 字典。
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    #如果捕获到这个类型的异常，**把这个错误实例，起一个别名，变量名叫e。
    except requests.exceptions.HTTPError as e:
        print(f"API返回错误: {resp.status_code} - {resp.text}")
        #sys.exit(0)=正常结束
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"网络请求失败: {e}")
        sys.exit(1)
        #KeyError：执行data["choices"]时，字典里没有这个 key。服务器返回的 json 缺少字段。
        #IndexError：执行data["choices"][0]，choices 列表是空的，没有第 0 号元素。
    except (KeyError, IndexError):
        print(f"响应格式异常，原始数据: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    if not API_KEY:
        print("错误: 请在 .env 文件中设置 ZHIPU_API_KEY")
        sys.exit(1)

    #sys.argv是一个列表，包含了命令行参数。sys.argv[0]是脚本名称，sys.argv[1:]是传入的参数。
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    #只运行脚本char.py，没传入参数，就让用户输入问题。
    else:
        prompt = input("请输入你的问题: ")

    print(f"\n用户问: {prompt}")
    print(f"GLM回答: ", end="")

    reply = chat(prompt)
    print(reply)
