"""
SiliconFlow Rerank 模块
封装 SiliconFlow API 的文档重排序功能
"""

import requests
from typing import List
import yaml


class SiliconFlowReranker:
    """SiliconFlow Rerank API 封装"""

    def __init__(self, config_path: str = "./config.yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.api_key = config["siliconflow_api_key"]
        self.model = config.get("rerank_model", "Qwen/Qwen3-Reranker-0.6B")
        self.base_url = config.get(
            "siliconflow_base_url", "https://api.siliconflow.cn/v1"
        )
        self.url = f"{self.base_url}/rerank"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> dict:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回的Top N结果

        Returns:
            重排序结果
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        try:
            response = requests.post(
                self.url, json=payload, headers=self.headers, timeout=60
            )
            return response.json()
        except Exception as e:
            print(f"Rerank API 请求失败: {e}")
            return {"results": []}
