from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "corpus"
DB_PATH = ROOT / "indexes" / "bm25_full.sqlite"

BATCH_SIZE = 100


def build_index():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(DATA_PATH))
    total = len(dataset)

    conn = sqlite3.connect(str(DB_PATH), timeout=60)

    # 增量写入，避免出现大量未提交的数据
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents
            USING fts5(
                docid UNINDEXED,
                text,
                url UNINDEXED,
                tokenize='porter unicode61'
            )
        """)
    except sqlite3.OperationalError as e:
        conn.close()
        raise RuntimeError(
            "当前 Python SQLite 不支持 FTS5"
        ) from e

    # rowid 使用原始 Dataset 的位置 + 1
    # 因此可以通过 MAX(rowid) 恢复构建进度
    start = conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM documents"
    ).fetchone()[0]

    if start > total:
        conn.close()
        raise ValueError("索引文档数量超过数据集大小")

    print(f"数据集文档总数: {total}")
    print(f"已索引文档: {start}")
    print(f"剩余文档: {total - start}")

    try:
        for offset in tqdm(
            range(start, total, BATCH_SIZE),
            desc="Building BM25"
        ):
            end = min(offset + BATCH_SIZE, total)

            batch = dataset.select(range(offset, end))

            rows = []

            for index, doc in enumerate(batch, start=offset):
                rows.append((
                    index + 1,
                    doc["docid"],
                    doc["text"] or "",
                    doc["url"] or ""
                ))

            with conn:
                conn.executemany(
                    """
                    INSERT INTO documents
                    (rowid, docid, text, url)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows
                )

        count = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        print(f"\n索引构建完成: {count}/{total}")
        print("数据库路径:", DB_PATH)

    finally:
        conn.close()


if __name__ == "__main__":
    build_index()