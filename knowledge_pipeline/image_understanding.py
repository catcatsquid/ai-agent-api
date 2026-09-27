"""图片理解模块

从图片中提取文字和语义信息，支持:
- VLM: 调用视觉语言模型（智谱 GLM-4V）理解图片内容，生成文字描述
- OCR: （可选）本地 OCR 提取图片中的文字

对于 PPT 中的图表、流程图等，VLM 理解效果远好于纯 OCR，
因为它能理解结构关系（如"箭头从A指向B"、"状态转换"等）。
"""

import io
import base64
import time
import requests
from PIL import Image
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL

# 智谱视觉模型
VLM_MODEL = "glm-4v-flash"


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """PIL Image 转 base64"""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image_vlm(
    img: Image.Image,
    context: str = "",
    detail: str = "low",
    max_retries: int = 3,
) -> str:
    """用 VLM 理解图片内容，返回文字描述

    Args:
        img: PIL Image 对象
        context: 上下文提示（如"这是一页关于TCP状态转换的PPT"）
        detail: 图片质量 "low" | "high"，low 更快更便宜
        max_retries: 最大重试次数

    Returns:
        图片内容的文字描述
    """
    b64 = image_to_base64(img)

    system_prompt = """你是一个课件图片理解助手。请用简洁的中文描述这张图片的内容，重点提取:
1. 图表表达的核心概念和关系
2. 流程图/状态图的转换关系
3. 关键数据和标注
4. 如果是架构图/示意图，描述各组件的关系

请直接输出描述，不要多余的开场白。如果图片内容不清晰或无法识别，返回空字符串。"""

    user_text = context if context else "请描述这张图片中的知识点内容。"

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                time.sleep(wait)
                continue
            print(f"  [VLM 错误] {e}")
            return ""

    return ""


def extract_images_from_page(page) -> list[Image.Image]:
    """从 PyMuPDF 页面对象中提取所有图片

    Args:
        page: fitz.Page 对象

    Returns:
        PIL Image 对象列表
    """
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pix = page.parent.extract_image(xref)
            img_data = pix["image"]
            img = Image.open(io.BytesIO(img_data))
            # 过滤掉太小的图片（可能是图标/装饰）
            if img.width >= 100 and img.height >= 100:
                images.append(img)
        except Exception:
            continue
    return images


# 图片质量过滤：太小/太大的图跳过
def is_worth_processing(img: Image.Image, min_size: int = 150, max_bytes: int = 2 * 1024 * 1024) -> bool:
    """判断图片是否值得花 API 调用去理解"""
    if img.width < min_size or img.height < min_size:
        return False
    # 预估大小（粗略）
    if img.width * img.height * 3 > max_bytes * 10:
        return False
    return True
