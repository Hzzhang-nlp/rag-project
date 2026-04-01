"""
SiliconFlow 向量化模块
封装 SiliconFlow API 的文本嵌入功能
"""

import requests
from typing import List, Optional
import yaml


class SiliconFlowEmbedder:
    """SiliconFlow Embedding API 封装"""

    def __init__(self, config_path: str = "./config.yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.api_key = config["siliconflow_api_key"]
        self.model = config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        self.base_url = config.get(
            "siliconflow_base_url", "https://api.siliconflow.cn/v1"
        )
        self.url = f"{self.base_url}/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_query(self, text: str) -> Optional[List[float]]:
        """单条查询文本嵌入"""
        payload = {"model": self.model, "input": text, "encoding_format": "float"}
        try:
            response = requests.post(
                self.url, json=payload, headers=self.headers, timeout=30
            )
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            print(f"Embedding 返回异常: {data}")
            return None
        except Exception as e:
            print(f"Embedding API 请求失败: {e}")
            return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档嵌入"""
        results = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            try:
                response = requests.post(
                    self.url, json=payload, headers=self.headers, timeout=60
                )
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        if "embedding" in item:
                            results.append(item["embedding"])
                        else:
                            results.append([0.0] * 1024)
                else:
                    results.extend([[0.0] * 1024] * len(batch))
            except Exception as e:
                print(f"批量 Embedding 请求失败: {e}")
                results.extend([[0.0] * 1024] * len(batch))
        return results
