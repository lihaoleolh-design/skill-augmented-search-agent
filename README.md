# Skill-Augmented Search Agent

面向 BrowseComp-Plus 的检索增强问答项目。系统以提高召回率为当前重点，提供 BM25、Qdrant 向量检索以及二者的混合检索，并支持不接入 LLM 的独立检索评测。

## 功能

- **BM25 检索**：SQLite FTS5、多查询变体与 RRF 融合。
- **Dense 检索**：通过硅基流动 API 调用 `BAAI/bge-large-en-v1.5`，向量存储在 Qdrant。
- **Hybrid 检索**：融合 BM25 与 Dense 排名，返回去重文档和证据片段。
- **文档工具**：按 `docid` 分段读取文档、定位文档内关键词。
- **搜索 Agent**：基于 LangGraph，支持技能路由、搜索进度检测、证据管理和答案校验。
- **离线评测**：不调用 LLM，统计 Recall、Hit Rate、MRR、nDCG，以及答案是否出现在召回片段中的 `answer_hit@K`。
- **索引管理**：Dense 构建支持断点续传；Qdrant Snapshot 支持跨电脑迁移。

## 项目结构

```text
agent/          Agent、技能路由、证据与答案校验
retrieval/      BM25、Dense、Hybrid 检索
tools/          文档读取和文档内搜索工具
evaluation/     检索及 Agent 评测指标
scripts/        数据下载、Dense 索引构建、评测入口
skills/         多跳搜索、实体消歧、搜索恢复技能
docs/           评测与 Qdrant 迁移说明
```

## 快速开始

### 1. 安装依赖

推荐 Python 3.12：

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写硅基流动 API Key，并按需填写 LLM 配置：

```dotenv
LLM_API_KEY=<your-siliconflow-api-key>

RETRIEVAL_METHOD=hybrid
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=browsecomp_bge_large_en_v1_5
DENSE_MODEL_NAME=BAAI/bge-large-en-v1.5
DENSE_VECTOR_SIZE=1024
```

文档与查询必须使用相同的 Embedding 模型。真实 `.env` 已被 Git 忽略。

### 2. 启动 Qdrant

```powershell
docker volume create qdrant_storage
docker compose up -d qdrant
```

验证服务：

```powershell
Invoke-RestMethod http://127.0.0.1:6333/collections
```

### 3. 构建索引

BM25：

```powershell
python retrieval/build_bm25_full.py
python scripts/build_doc_locator.py
```

Dense/Qdrant 冒烟测试：

```powershell
python scripts/build_dense_index.py --limit 10
```

继续构建完整索引：

```powershell
python scripts/build_dense_index.py --batch-size 32
```

脚本每完成一篇文档就写入 checkpoint。中断后运行相同命令会继续构建；`--recreate` 会删除现有 collection 并从头开始。

## 检索评测（不调用 LLM）

完整 Dense 索引构建后运行：

```powershell
python scripts/evaluate_retrieval.py `
  --modes rrf dense hybrid `
  --shuffle `
  --seed 42 `
  --limit 100
```

重点关注：

- `gold_recall@K`：标准文档召回率；
- `evidence_recall@K`：证据文档召回率；
- `answer_hit@K`：前 K 条线索中是否直接包含标准答案；
- `MRR`、`nDCG@10`：排序质量。

少量 Dense 索引只适合验证链路，不能用于公平比较 BM25、Dense 和 Hybrid。

## 运行 Agent

```powershell
python main.py
```

运行固定样本 Agent 评测：

```powershell
python scripts/evaluate_agent.py --limit 10 --seed 42
```

Agent 评测会调用 LLM 和 Embedding API，可能产生费用。

## 测试

```powershell
python -m pytest -q
```

## 文档

- [检索与 Agent 评测](docs/evaluation.md)
- [Qdrant 向量索引导出与导入](docs/qdrant_migration.md)

## TODO

- [ ] 完成 100,195 篇文档的 Dense Embedding 和 Qdrant 索引。
- [ ] 在完整语料上建立 BM25、Dense、Hybrid 固定评测基线。
- [ ] 调优候选数量、RRF 参数和最终 `top_k`。
- [ ] 评估 Cross-Encoder reranker 对 Top-K 排序的提升。
- [ ] 对比固定 token、段落/标题和多粒度切块策略。
- [ ] 增加 Embedding API 限流、费用统计和批次级恢复。
- [ ] 自动校验 checkpoint、Qdrant point 数量和语料一致性。
- [ ] 自动化 Qdrant Snapshot 导出、导入与完整性校验。
- [ ] 完成全部 830 道 BrowseComp-Plus 检索及 Agent 评测。

## 数据与安全

数据集、索引、输出结果、IDE 文件及 `.env` 不进入 Git。请勿提交 API Key、完整语料或 Qdrant volume。
