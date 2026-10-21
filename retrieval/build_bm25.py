import argparse
from pathlib import Path

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "corpus"
INDEX_PATH = ROOT / "indexes" / "bm25"


def build_index(limit=5000):
    dataset = load_from_disk(str(DATA_PATH))

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"开始构建 BM25 索引，文档数量: {len(dataset)}")

    stemmer = Stemmer.Stemmer("english")

    tokenizer = Tokenizer(
        stemmer=stemmer,
        stopwords="en"
    )

    # 先做可验证的基线：保持文档原始顺序
    texts = dataset["text"]

    print("正在分词...")
    corpus_tokens = tokenizer.tokenize(
        texts,
        return_as="tuple"
    )

    print("正在建立 BM25 索引...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    # 不重复保存完整文档，文档继续从 data/corpus 读取
    retriever.save(str(INDEX_PATH))
    tokenizer.save_vocab(save_dir=str(INDEX_PATH))
    tokenizer.save_stopwords(save_dir=str(INDEX_PATH))

    # 保存索引的文档数量，供检索时恢复位置映射
    (INDEX_PATH / "document_count.txt").write_text(
        str(len(dataset)),
        encoding="utf-8"
    )

    print("BM25 索引构建完成！")
    print("索引路径:", INDEX_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="文档数量；设置 0 使用全部文档"
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能为负数")

    build_index(
        limit=None if args.limit == 0 else args.limit
    )