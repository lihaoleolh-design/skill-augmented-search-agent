import re
from pathlib import Path

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]


class BM25Retriever:
    def __init__(
        self,
        index_path=ROOT / "indexes" / "bm25",
        data_path=ROOT / "data" / "corpus"
    ):
        index_path = Path(index_path)

        self.dataset = load_from_disk(str(data_path))

        self.retriever = bm25s.BM25.load(
            str(index_path),
            mmap=True,
            load_corpus=False
        )

        self.tokenizer = Tokenizer(
            stemmer=Stemmer.Stemmer("english"),
            stopwords=[]
        )
        self.tokenizer.load_vocab(str(index_path))
        self.tokenizer.load_stopwords(str(index_path))

        self.document_count = int(
            (index_path / "document_count.txt")
            .read_text(encoding="utf-8")
        )

    @staticmethod
    def extract_snippet(text, query, length=400):
        # 优先从包含查询词的位置附近提取
        words = re.findall(r"\w+", query.lower())

        position = 0
        lower_text = text.lower()

        for word in words:
            match = re.search(
                rf"\b{re.escape(word)}\b",
                lower_text
            )
            if match:
                position = match.start()
                break

        start = max(0, position - 100)
        end = min(len(text), start + length)

        return text[start:end].replace("\n", " ")

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            return []

        tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False
        )

        ids, scores = self.retriever.retrieve(
            tokens,
            k=min(top_k, self.document_count)
        )

        results = []

        for rank, (idx, score) in enumerate(
            zip(ids[0], scores[0]),
            start=1
        ):
            idx = int(idx)
            score = float(score)

            # 不返回零相关性的填充结果
            if score <= 0:
                continue

            doc = self.dataset[idx]

            results.append({
                "rank": rank,
                "docid": doc["docid"],
                "score": score,
                "snippet": self.extract_snippet(
                    doc["text"], query
                ),
                "url": doc["url"]
            })

        return results