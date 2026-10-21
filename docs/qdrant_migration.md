# Qdrant 向量索引导出与导入指南

本文用于将本项目已经完成的 Qdrant 向量索引迁移到另一台电脑，无需重新调用 Embedding API。

当前项目默认配置：

| 项目 | 值 |
|---|---|
| Collection | `browsecomp_bge_large_en_v1_5` |
| Embedding 模型 | `BAAI/bge-large-en-v1.5` |
| 向量维度 | `1024` |
| 距离算法 | `Cosine` |
| Qdrant 地址 | `http://127.0.0.1:6333` |

推荐使用 Qdrant Collection Snapshot 进行迁移，不要在 Qdrant 运行时直接复制 Docker volume。Snapshot 包含 collection 配置、向量以及 chunk 文本、`docid`、URL、offset 等 payload。

## 一、导出前检查

建议等待完整索引构建结束后再创建快照：

```powershell
$collection = "browsecomp_bge_large_en_v1_5"

$info = Invoke-RestMethod `
  -Uri "http://127.0.0.1:6333/collections/$collection"

$info.result |
  Select-Object status, points_count, indexed_vectors_count, segments_count
```

确认：

- `status` 为 `green`；
- `points_count` 大于 0；
- 索引构建脚本已经正常结束；
- 没有其他程序正在继续向该 collection 写入数据。

记录 `points_count`，导入后需要用它校验数据是否完整。

## 二、在原电脑创建并下载 Snapshot

在项目根目录执行：

```powershell
$collection = "browsecomp_bge_large_en_v1_5"
$exportDir = Join-Path $PWD "output\qdrant_snapshots"

New-Item -ItemType Directory -Force -Path $exportDir | Out-Null

$snapshot = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:6333/collections/$collection/snapshots"

$snapshotName = $snapshot.result.name
$snapshotPath = Join-Path $exportDir $snapshotName

Invoke-WebRequest `
  -Uri "http://127.0.0.1:6333/collections/$collection/snapshots/$snapshotName" `
  -OutFile $snapshotPath

Write-Output "Snapshot saved to: $snapshotPath"
```

生成 SHA-256 校验值：

```powershell
Get-FileHash $snapshotPath -Algorithm SHA256
```

保存输出中的哈希值。文件传输到另一台电脑后，需要再次计算并比较。

## 三、将文件复制到新电脑

至少复制以下内容：

```text
output/qdrant_snapshots/<snapshot-name>.snapshot
compose.yaml
项目代码
```

可以通过移动硬盘、局域网、NAS 或对象存储传输 Snapshot。

不要共享项目中的 `.env`，因为其中包含 API Key。新电脑应自行创建 `.env` 并填写自己的密钥。

如果新电脑还需要使用 BM25、Hybrid 检索以及 `open_document`，还需复制：

```text
indexes/bm25_full.sqlite
```

复制 SQLite 文件时应确保没有程序正在写入数据库。如果目录中存在对应的 `-wal` 文件，应先停止相关程序，再复制数据库文件。

## 四、在新电脑启动 Qdrant

进入项目目录。当前 `compose.yaml` 使用 external volume，因此第一次启动前先创建 volume：

```powershell
docker volume create qdrant_storage
docker compose up -d qdrant
```

检查 Qdrant：

```powershell
Invoke-RestMethod "http://127.0.0.1:6333/collections" |
  ConvertTo-Json -Depth 10
```

如果能看到 `status: ok`，说明 Qdrant 已经启动。

## 五、校验 Snapshot 文件

假设复制后的文件位于：

```text
C:\transfer\browsecomp.snapshot
```

计算哈希：

```powershell
Get-FileHash "C:\transfer\browsecomp.snapshot" -Algorithm SHA256
```

结果必须与原电脑一致。不一致时不要导入，应重新传输文件。

## 六、导入 Snapshot

目标 Qdrant 中最好不存在同名 collection。先查看：

```powershell
Invoke-RestMethod "http://127.0.0.1:6333/collections" |
  ConvertTo-Json -Depth 10
```

使用 `curl.exe` 上传并恢复：

```powershell
$snapshotPath = "C:\transfer\browsecomp.snapshot"
$collection = "browsecomp_bge_large_en_v1_5"

curl.exe -X POST `
  "http://127.0.0.1:6333/collections/$collection/snapshots/upload?priority=snapshot" `
  -H "Content-Type: multipart/form-data" `
  -F "snapshot=@$snapshotPath"
```

注意要使用 `curl.exe`，避免 Windows PowerShell 将 `curl` 解析成 `Invoke-WebRequest` 的别名。

上传大文件可能需要较长时间。命令运行期间不要关闭终端或停止 Qdrant。

### 保留现有同名 Collection

如果目标电脑已经存在同名 collection，不想影响它，可以导入为另一个名称：

```powershell
$snapshotPath = "C:\transfer\browsecomp.snapshot"
$targetCollection = "browsecomp_bge_large_en_v1_5_copy"

curl.exe -X POST `
  "http://127.0.0.1:6333/collections/$targetCollection/snapshots/upload?priority=snapshot" `
  -H "Content-Type: multipart/form-data" `
  -F "snapshot=@$snapshotPath"
```

然后把 `.env` 中的 `QDRANT_COLLECTION` 改成新名称。

## 七、配置新电脑

新电脑的 `.env` 至少应包含：

```dotenv
RETRIEVAL_METHOD=hybrid

QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=browsecomp_bge_large_en_v1_5

DENSE_MODEL_NAME=BAAI/bge-large-en-v1.5
DENSE_VECTOR_SIZE=1024
DENSE_API_URL=https://api.siliconflow.cn/v1/embeddings

LLM_API_KEY=新电脑自己的硅基流动API密钥
```

查询时仍需调用同一个 Embedding 模型把问题转换为向量，所以新电脑需要可用的硅基流动 API Key。文档向量已经包含在 Snapshot 中，不需要重新生成。

## 八、验证导入结果

查询 collection 状态：

```powershell
$collection = "browsecomp_bge_large_en_v1_5"

$result = Invoke-RestMethod `
  -Uri "http://127.0.0.1:6333/collections/$collection"

$result.result |
  Select-Object status, points_count, indexed_vectors_count, segments_count,
    @{Name="vector_size"; Expression={$_.config.params.vectors.size}},
    @{Name="distance"; Expression={$_.config.params.vectors.distance}}
```

检查：

- `status` 为 `green`；
- `points_count` 与原电脑一致；
- `vector_size` 为 `1024`；
- `distance` 为 `Cosine`。

随后运行一个小型检索测试：

```powershell
D:\envs\nlp_project\python.exe scripts\evaluate_retrieval.py `
  --modes dense hybrid `
  --shuffle `
  --seed 42 `
  --limit 5
```

## 九、是否需要复制其他文件

| 使用场景 | 需要复制的内容 |
|---|---|
| 仅 Dense 检索 | Qdrant Snapshot、项目代码、目标电脑的 API Key |
| Dense + BM25 Hybrid | 以上内容以及 `indexes/bm25_full.sqlite` |
| 继续构建未完成的 Dense 索引 | Snapshot、完整 `data/corpus`、checkpoint 文件 |
| 重新生成文档向量 | 完整 `data/corpus` 和硅基流动 API Key |

如果要在新电脑继续一个尚未完成的索引，还需要复制：

```text
indexes/qdrant/browsecomp_bge_large_en_v1_5.checkpoint.json
data/corpus/
```

两台电脑上的 corpus 必须完全相同且顺序一致，否则 checkpoint 中的文档位置不再可靠。更推荐完成全部索引后再迁移。

## 十、常见问题

### 上传时返回 502

确认访问的是：

```text
http://127.0.0.1:6333
```

并检查 Qdrant 容器是否运行：

```powershell
docker compose ps
```

### 返回 Collection already exists

不要直接覆盖未知数据。可以导入为新的 collection 名称，并修改 `.env` 中的 `QDRANT_COLLECTION`。

### 新电脑可以检索，但 Hybrid 报 BM25 文件不存在

Snapshot 只包含 Qdrant collection，不包含 SQLite BM25 索引。还需要复制：

```text
indexes/bm25_full.sqlite
```

### 导入后是否还会产生 Embedding 费用

不会重新生成文档 embedding，但每次 Dense 查询仍要为查询文本调用一次 Embedding API，因此会产生少量查询 embedding 费用。

### 能否直接复制 Docker volume

理论上可以在彻底停止 Qdrant 后复制，但更容易受到 Qdrant 版本、Docker 存储路径和未完成写入的影响。跨电脑迁移优先使用 Snapshot。

## 参考资料

- [Qdrant Snapshots 官方文档](https://qdrant.tech/documentation/operations/snapshots/)
- [Qdrant 上传并恢复 Snapshot API](https://api.qdrant.tech/master/api-reference/snapshots/recover-from-uploaded-snapshot)
