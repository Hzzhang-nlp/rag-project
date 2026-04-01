"""
RAG (检索增强生成) 系统主程序
功能：文档向量化存储、向量检索、Rerank重排序、LLM问答
"""

import os
import yaml
import hashlib
from langchain_core.documents import Document
from pymilvus import MilvusClient
from embedding.siliconflow_embedder import SiliconFlowEmbedder
from rerank.siliconflow_reranker import SiliconFlowReranker
from model.siliconflow_llm import siliconflow_chat
from utils.splitter import split_docs_to_chunks
from utils.logger import setup_logger

logger = setup_logger("rag")


def load_multiple_format_files(folder_path: str) -> list[Document]:
    """加载指定目录下的文档文件，支持 txt/md 格式"""
    docs = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith((".txt", ".md", ".markdown")):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(
                    Document(page_content=content, metadata={"reference": filename})
                )
                logger.info(f"加载文档: {filename}, 长度: {len(content)} 字符")
    logger.info(f"共加载 {len(docs)} 个文档")
    return docs


def get_files_hash(
    folder_path: str, chunk_size: int, chunk_overlap: int, embedding_model: str
) -> str:
    """根据文件内容和切分参数生成唯一hash，用于判断是否需要重建索引"""
    m = hashlib.md5()
    m.update(str(chunk_size).encode())
    m.update(str(chunk_overlap).encode())
    m.update(embedding_model.encode())
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".txt", ".md", ".markdown")):
            with open(os.path.join(folder_path, filename), "rb") as f:
                m.update(f.read())
    return m.hexdigest()


def query_loop(milvus_client: MilvusClient, collection_name: str, config: str):
    """交互式问答循环"""
    embedder = SiliconFlowEmbedder(config_path=config)
    reranker = SiliconFlowReranker(config_path=config)

    print("\n" + "=" * 50)
    print("RAG 问答系统已启动，输入问题后按回车查询")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")
    logger.info("问答系统启动")

    while True:
        try:
            query = input("请输入您的问题: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                print("再见!")
                logger.info("用户退出问答系统")
                break

            logger.info(f"用户查询: {query}")

            # 1. 生成查询向量
            logger.info("开始生成查询向量...")
            query_vec = embedder.embed_query(query)
            if query_vec is None:
                print("查询向量生成失败，请检查 API 配置")
                logger.error("查询向量生成失败")
                continue
            logger.info(f"查询向量生成成功，维度: {len(query_vec)}")

            # 2. 向量检索
            logger.info("开始 Milvus 向量检索...")
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[query_vec],
                limit=10,
                output_fields=["text", "reference", "wider_text"],
            )
            logger.info(
                f"Milvus 检索返回结果数: {len(search_res[0]) if search_res else 0}"
            )

            # 3. 解析检索结果
            retrieved_texts = []
            retrieved_metadatas = []
            if search_res and len(search_res) > 0:
                hits = search_res[0]
                for i in range(len(hits)):
                    hit = hits[i]
                    text = hit.get("text", "") if hasattr(hit, "get") else str(hit)
                    if text and isinstance(text, str):
                        text_stripped = text.strip()
                        if text_stripped:
                            retrieved_texts.append(text_stripped)
                            retrieved_metadatas.append(
                                {
                                    "reference": hit.get("reference", ""),
                                    "wider_text": hit.get("wider_text", ""),
                                }
                            )
            logger.info(f"解析后有效文本数: {len(retrieved_texts)}")

            if not retrieved_texts:
                print("未检索到相关文档，请尝试其他问题")
                logger.warning("未检索到相关文档")
                continue

            # 4. Rerank 重排序
            logger.info("开始 Rerank 重排序...")
            rerank_result = reranker.rerank(query, retrieved_texts, top_n=5)
            logger.info(f"Rerank 返回结果数: {len(rerank_result.get('results', []))}")

            # 5. 整理上下文
            context_parts = []
            if "results" in rerank_result and isinstance(
                rerank_result["results"], list
            ):
                for idx, item in enumerate(rerank_result["results"][:5], 1):
                    doc = item.get("document", {})
                    text = doc.get("text", "")
                    ref_idx = item.get("index", 0)
                    reference = (
                        retrieved_metadatas[ref_idx].get("reference", "")
                        if ref_idx < len(retrieved_metadatas)
                        else ""
                    )
                    context_parts.append(f"[来源 {idx}: {reference}]\n{text}")
                    logger.debug(
                        f"Rerank {idx}: 分数={item.get('relevance_score', 0):.4f}, 来源={reference}"
                    )

            if not context_parts:
                context_parts = [retrieved_texts[0]]

            best_context = "\n\n---\n\n".join(context_parts)

            # 6. 调用 LLM 生成答案
            messages = [
                {
                    "role": "system",
                    "content": "你是一个有帮助的AI助手。请结合检索到的内容回答用户问题，直接输出详细合适的最终答案。",
                },
                {
                    "role": "user",
                    "content": f"检索到的内容：\n{best_context}\n\n问题：{query}",
                },
            ]

            logger.info("开始调用 LLM 生成答案...")
            print("\n正在生成答案...\n")
            answer = siliconflow_chat(messages, config_path=config)
            logger.info("LLM 答案生成成功")

            print("=" * 50)
            print("答案：")
            print(answer)
            print("=" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n再见!")
            logger.info("用户中断退出")
            break
        except Exception as e:
            print(f"处理查询时出错: {e}")
            logger.error(f"处理查询异常: {str(e)}", exc_info=True)


def main():
    """主函数：初始化向量库并启动问答"""
    folder_path = "./database/data/uploads"
    collection_name = "rag_collection"
    chunk_size = 600
    chunk_overlap = 100
    config_path = "./config.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    embedding_model = config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")

    logger.info("=" * 50)
    logger.info("RAG 系统启动")
    logger.info(f"文档目录: {folder_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Embedding 模型: {embedding_model}")
    logger.info("=" * 50)

    milvus_client = MilvusClient("./database/milvus_demo.db")

    # 检查是否需要重建索引
    current_hash = get_files_hash(
        folder_path, chunk_size, chunk_overlap, embedding_model
    )
    hash_file = "./.rag_collection.hash"
    need_rebuild = True

    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash and milvus_client.has_collection(collection_name):
            try:
                stats = milvus_client.get_collection_stats(collection_name)
                if stats.get("row_count", 0) > 0:
                    need_rebuild = False
                    logger.info(
                        f"使用缓存数据，集合已有 {stats.get('row_count', 0)} 条记录"
                    )
            except Exception as e:
                logger.error(f"获取集合状态失败: {e}")

    # 重建索引
    if need_rebuild:
        logger.info("开始重建索引...")

        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
            logger.info("已删除旧 collection")

        documents = load_multiple_format_files(folder_path)
        if not documents:
            print(f"未找到文档，请确保 {folder_path} 目录下有 .txt 或 .md 文件")
            logger.error("未找到文档文件")
            return

        chunks = split_docs_to_chunks(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info(f"文档切分完成，共 {len(chunks)} 个文本块")

        texts = [chunk.text for chunk in chunks]
        references = [chunk.reference for chunk in chunks]
        wider_texts = [chunk.wider_text for chunk in chunks]

        logger.info("开始生成文本向量...")
        embedder = SiliconFlowEmbedder(config_path=config_path)
        vectors = embedder.embed_documents(texts)
        logger.info(f"向量生成完成，有效向量数: {len([v for v in vectors if v])}")

        valid_vectors = [v for v in vectors if v is not None and len(v) > 0]
        if not valid_vectors:
            print("向量生成失败，请检查 API 配置")
            logger.error("向量生成失败")
            return

        dimension = len(valid_vectors[0])
        logger.info(f"创建 Milvus Collection，维度: {dimension}")
        milvus_client.create_collection(
            collection_name=collection_name, dimension=dimension
        )

        data = [
            {
                "id": i,
                "vector": vectors[i],
                "text": texts[i],
                "reference": references[i],
                "wider_text": wider_texts[i],
            }
            for i in range(len(texts))
        ]
        milvus_client.insert(collection_name=collection_name, data=data)
        logger.info(f"数据插入完成，共 {len(data)} 条")

        with open(hash_file, "w") as f:
            f.write(current_hash)
        print(f"已完成向量化，共插入 {len(data)} 条数据。")
    else:
        print("Milvus 已有数据，无需重复向量化。")

    query_loop(milvus_client, collection_name, config_path)


if __name__ == "__main__":
    main()
