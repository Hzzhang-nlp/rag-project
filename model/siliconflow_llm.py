"""
SiliconFlow LLM 模块
封装 SiliconFlow API 的大语言模型调用
"""

import requests
from typing import List
import yaml


def siliconflow_chat(messages: List[dict], config_path: str = "./config.yml") -> str:
    """
    调用 SiliconFlow LLM API 生成回复

    Args:
        messages: 对话消息列表 (OpenAI 格式)
        config_path: 配置文件路径

    Returns:
        LLM 生成的回复文本
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    api_key = config["siliconflow_api_key"]
    model = config.get("llm_model", "Qwen/Qwen3-8B")
    base_url = config.get("siliconflow_base_url", "https://api.siliconflow.cn/v1")
    url = f"{base_url}/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return str(data)
    except Exception as e:
        print(f"LLM API 请求失败: {e}")
        return f"请求失败: {e}"
