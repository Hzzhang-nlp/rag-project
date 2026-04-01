"""
文档切分模块
基于 Sentence Window 策略的文档切分，为每个 chunk 添加上下文窗口
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk:
    """文本块数据结构"""

    def __init__(
        self,
        text: str,
        reference: str,
        metadata: Optional[dict] = None,
        embedding: Optional[List[float]] = None,
    ):
        self.text = text
        self.reference = reference
        self.metadata = metadata or {}
        self.embedding = embedding
        self.wider_text = metadata.get("wider_text", "") if metadata else ""


def _sentence_window_split(
    split_docs: List[Document], original_document: Document, offset: int = 200
) -> List[Chunk]:
    """
    为切分后的文档片段添加上下文窗口

    Args:
        split_docs: 已切分的文档片段
        original_document: 原始完整文档
        offset: 上下文窗口大小（字符数）

    Returns:
        包含上下文窗口的 Chunk 列表
    """
    chunks = []
    original_text = original_document.page_content
    for doc in split_docs:
        doc_text = doc.page_content
        start_index = original_text.index(doc_text)
        end_index = start_index + len(doc_text)
        wider_text = original_text[
            max(0, start_index - offset) : min(len(original_text), end_index + offset)
        ]
        reference = doc.metadata.pop("reference", "")
        doc.metadata["wider_text"] = wider_text
        chunks.append(Chunk(text=doc_text, reference=reference, metadata=doc.metadata))
    return chunks


def split_docs_to_chunks(
    documents: List[Document], chunk_size: int = 600, chunk_overlap: int = 100
) -> List[Chunk]:
    """
    将文档列表切分为带上下文窗口的文本块

    Args:
        documents: 待切分文档列表
        chunk_size: 文本块大小（字符数）
        chunk_overlap: 块间重叠大小

    Returns:
        切分后的 Chunk 列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        split_chunks = _sentence_window_split(split_docs, doc, offset=200)
        all_chunks.extend(split_chunks)
    return all_chunks
