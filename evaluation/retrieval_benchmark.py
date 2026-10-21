from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

from evaluation.benchmark import normalize_answer


def answer_in_results(results: list[dict], answer: str, cutoffs: Iterable[int]) -> dict[str, float]:
    """Check whether normalized reference text occurs in a retrieved clue."""
    expected = normalize_answer(answer)
    hits = [
        bool(expected) and expected in normalize_answer(item.get("snippet", ""))
        for item in results
    ]
    return {
        f"answer_hit@{cutoff}": float(any(hits[:cutoff]))
        for cutoff in cutoffs
    }


def ranking_metrics(
    retrieved_docids: list[str],
    relevant_docids: Iterable[str],
    cutoffs: Iterable[int],
) -> dict[str, float]:
    relevant = {str(docid) for docid in relevant_docids}
    retrieved = [str(docid) for docid in retrieved_docids]
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        top = retrieved[:cutoff]
        matches = relevant.intersection(top)
        metrics[f"recall@{cutoff}"] = len(matches) / len(relevant) if relevant else 0.0
        metrics[f"hit_rate@{cutoff}"] = float(bool(matches))

    first_rank = next(
        (rank for rank, docid in enumerate(retrieved, 1) if docid in relevant),
        None,
    )
    metrics["reciprocal_rank"] = 1.0 / first_rank if first_rank else 0.0

    cutoff = 10
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, docid in enumerate(retrieved[:cutoff], 1)
        if docid in relevant
    )
    ideal_count = min(len(relevant), cutoff)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics["ndcg@10"] = dcg / ideal if ideal else 0.0
    return metrics


def summarize_retrieval(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)

    output: dict[str, Any] = {}
    for mode, rows in grouped.items():
        numeric_keys = [
            key for key, value in rows[0].items()
            if isinstance(value, (int, float)) and key != "query_id"
        ] if rows else []
        summary = {"cases": len(rows)}
        for key in numeric_keys:
            summary[f"avg_{key}"] = statistics.fmean(float(row[key]) for row in rows)
        latencies = sorted(float(row["elapsed_seconds"]) for row in rows)
        if latencies:
            summary["p50_elapsed_seconds"] = latencies[math.ceil(0.50 * len(latencies)) - 1]
            summary["p95_elapsed_seconds"] = latencies[math.ceil(0.95 * len(latencies)) - 1]
        output[mode] = summary
    return output
