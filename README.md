# RAG 检索增强生成系统

基于 Milvus + SiliconFlow 的轻量级 RAG 系统，支持文档向量化存储、向量检索、Rerank 重排序和 LLM 问答。

## 功能特性

- 📄 **多格式文档支持**：支持 txt、md 格式文档
- 🔍 **向量检索**：基于 Milvus 的语义检索
- 🔄 **Rerank 重排序**：提升检索精度
- 💬 **LLM 问答**：结合上下文生成答案
- 📝 **Sentence Window**：文档切分时保留上下文窗口

## 技术栈

- **向量数据库**：Milvus (本地 SQLite)
- **Embedding**：SiliconFlow Qwen3-Embedding
- **Rerank**：SiliconFlow Qwen3-Reranker
- **LLM**：SiliconFlow Qwen3-8B

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yml`，填入 SiliconFlow API Key：

```yaml
siliconflow_api_key: "your-api-key"
```

### 3. 运行

```bash
python main.py
```

### 4. 添加文档

将文档放入 `database/data/uploads/` 目录，支持 `.txt` 和 `.md` 格式。

## 项目结构

```
RAG/
├── main.py                    # 主程序
├── config.yml                # 配置文件
├── requirements.txt           # 依赖
├── embedding/                # 向量化模块
├── rerank/                   # 重排序模块
├── model/                    # LLM 模块
├── utils/                    # 工具模块
└── database/                 # 数据库和文档
```

## 使用说明

1. 首次运行会自动向量化文档并存入 Milvus
2. 程序会缓存向量结果，文档无变化时跳过向量化
3. 进入问答后输入问题，查看答案
4. 输入 `quit` 或 `exit` 退出

## 日志

运行时日志保存在 `./logs/` 目录

## License

MIT
