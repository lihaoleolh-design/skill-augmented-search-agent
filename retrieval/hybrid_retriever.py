from __future__ import annotations

from collections import defaultdict
from typing import Any

from config import settings
from retrieval.bm25_full_retriever import BM25FullRetriever
from retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """Fuse BM25 and dense document rankings with reciprocal-rank fusion."""

    def __init__(
        self,
        sparse: Any | None = None,
        dense: Any | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> None:
        self.sparse = sparse or BM25FullRetriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k or settings.retrieval_rrf_k
        self.candidate_k = candidate_k or settings.retrieval_candidate_k

    def search(self, query: str, top_k: int = 8, candidate_k: int | None = None) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []
        pool_size = max(top_k, candidate_k or self.candidate_k)
        rankings = {
            "bm25": self.sparse.search(query, top_k=pool_size, candidate_k=pool_size),
            "dense": self.dense.search(query, top_k=pool_size),
        }

        scores: dict[str, float] = defaultdict(float)
        records: dict[str, dict] = {}
        source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for source, results in rankings.items():
            for rank, item in enumerate(results, 1):
                docid = str(item["docid"])
                scores[docid] += 1.0 / (self.rrf_k + rank)
                source_ranks[docid][source] = rank
                # Prefer dense chunks as evidence: unlike FTS snippets they are
                # contiguous passages and can contain a complete answer string.
                if docid not in records or source == "dense":
                    records[docid] = dict(item)

        ordered = sorted(
            records,
            key=lambda docid: (-scores[docid], min(source_ranks[docid].values()), docid),
        )[:top_k]
        return [
            {
                **records[docid],
                "rank": rank,
                "score": scores[docid],
                "source_ranks": source_ranks[docid],
            }
            for rank, docid in enumerate(ordered, 1)
        ]

    def close(self) -> None:
        for retriever in (self.sparse, self.dense):
            close = getattr(retriever, "close", None)
            if close:
                close()
