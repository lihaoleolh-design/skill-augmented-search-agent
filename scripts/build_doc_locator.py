# 给所有 100,195 篇文档建立 docid → rowid 的快速映射，只需要运行一次
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "indexes" / "bm25_full.sqlite"

EXPECTED_COUNT = 100195


def build_doc_locator():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"BM25 database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(str(DB_PATH), timeout=60)

    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        unique_count = conn.execute(
            "SELECT COUNT(DISTINCT docid) FROM documents"
        ).fetchone()[0]

        if total != EXPECTED_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_COUNT} documents, got {total}"
            )

        if unique_count != total:
            raise ValueError(
                "Duplicate docid detected. "
                "Cannot build a one-to-one document locator."
            )

        print("Document count:", total)
        print("Unique docid:", unique_count)

        # Build and populate the mapping in one transaction.
        # Any failure will roll back the changes.
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_locator (
                    docid TEXT PRIMARY KEY,
                    document_rowid INTEGER NOT NULL UNIQUE
                )
            """)

            conn.execute("DELETE FROM doc_locator")

            conn.execute("""
                INSERT INTO doc_locator
                    (docid, document_rowid)
                SELECT docid, rowid
                FROM documents
            """)

            locator_count = conn.execute(
                "SELECT COUNT(*) FROM doc_locator"
            ).fetchone()[0]

            if locator_count != total:
                raise RuntimeError(
                    "Document locator count mismatch"
                )

        print("Locator count:", locator_count)
        print("Document locator built successfully!")

    finally:
        conn.close()


if __name__ == "__main__":
    build_doc_locator()