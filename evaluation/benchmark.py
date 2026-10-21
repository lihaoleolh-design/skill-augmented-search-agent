from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, Iterable


ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(value: Any) -> str:
    """BrowseComp-style normalization for inexpensive local regression checks."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = NON_WORD.sub(" ", text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, reference: str) -> dict[str, float | bool]:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    exact = bool(expected) and predicted == expected
    contains = bool(expected and predicted) and (
        expected in predicted or predicted in expected
    )

    predicted_tokens = predicted.split()
    expected_tokens = expected.split()
    common = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if not predicted_tokens or not expected_tokens or not common:
        token_f1 = 0.0
    else:
        precision = common / len(predicted_tokens)
        recall = common / len(expected_tokens)
        token_f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact, "contains_match": contains, "token_f1": token_f1}


def _compact_case(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    question = row.get("query", row.get("question"))
    answer = row.get("answer")
    if question is None or answer is None:
        raise ValueError(f"Line {line_number} needs query/question and answer fields.")

    def docids(field: str) -> list[str]:
        return [
            str(item.get("docid") if isinstance(item, dict) else item)
            for item in (row.get(field, []) or [])
        ]

    # The official JSONL embeds full document text. Evaluation only needs IDs;
    # dropping text here keeps 830 cases small even though the source is >2 GB.
    return {
        "query_id": str(row.get("query_id", line_number)),
        "query": str(question),
        "answer": str(answer),
        "evidence_docs": docids("evidence_docs"),
        "gold_docs": docids("gold_docs"),
    }


def iter_cases(
    path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ValueError("Benchmark input must be a JSONL file.")

    total_bytes = source.stat().st_size
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield _compact_case(json.loads(line), line_number)
            if progress:
                progress(handle.tell(), total_bytes)


def load_cases(
    path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    return list(iter_cases(path, progress=progress))


def retrieved_docids(result: dict[str, Any]) -> list[str]:
    ids = {str(value) for value in result.get("visited_docs", [])}
    ids.update(
        str(item["docid"])
        for item in result.get("evidence", [])
        if item.get("docid") is not None
    )
    return sorted(ids)


def _gold_docids(case: dict[str, Any], field: str) -> set[str]:
    values = case.get(field, []) or []
    return {
        str(item.get("docid") if isinstance(item, dict) else item)
        for item in values
    }


def make_record(
    case: dict[str, Any], result: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    prediction = str(result.get("final_answer", ""))
    retrieved = retrieved_docids(result)
    evidence_ids = _gold_docids(case, "evidence_docs")
    gold_ids = _gold_docids(case, "gold_docs")

    def recall(expected: set[str]) -> float | None:
        return len(expected.intersection(retrieved)) / len(expected) if expected else None

    return {
        "query_id": case["query_id"],
        "query": case["query"],
        "reference_answer": case["answer"],
        "prediction": prediction,
        **answer_metrics(prediction, case["answer"]),
        "status": "completed" if result.get("termination_reason") == "completed" else "failed",
        "termination_reason": result.get("termination_reason", ""),
        "llm_rounds": int(result.get("search_round", 0)),
        "tool_calls": int(result.get("tool_calls", 0)),
        "total_tokens": int(result.get("total_tokens", 0)),
        "elapsed_seconds": elapsed_seconds,
        "retrieved_docids": retrieved,
        "evidence_recall": recall(evidence_ids),
        "gold_recall": recall(gold_ids),
        "error": None,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = len(rows)

    def mean(field: str) -> float:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return statistics.fmean(values) if values else 0.0

    timings = [float(row.get("elapsed_seconds", 0)) for row in rows]
    reasons = Counter(row.get("termination_reason") or "error" for row in rows)
    return {
        "cases": count,
        "accuracy": mean("exact_match"),
        "contains_accuracy": mean("contains_match"),
        "mean_token_f1": mean("token_f1"),
        "completion_rate": (
            sum(row.get("status") == "completed" for row in rows) / count if count else 0.0
        ),
        "unknown_rate": (
            sum(normalize_answer(row.get("prediction")) == "unknown" for row in rows) / count
            if count else 0.0
        ),
        "avg_llm_rounds": mean("llm_rounds"),
        "avg_tool_calls": mean("tool_calls"),
        "avg_total_tokens": mean("total_tokens"),
        "avg_elapsed_seconds": mean("elapsed_seconds"),
        "p50_elapsed_seconds": percentile(timings, 0.50),
        "p95_elapsed_seconds": percentile(timings, 0.95),
        "avg_evidence_recall": mean("evidence_recall"),
        "avg_gold_recall": mean("gold_recall"),
        "termination_reasons": dict(sorted(reasons.items())),
    }


def official_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one local result to BrowseComp-Plus' official run schema."""
    return {
        "query_id": record["query_id"],
        "tool_call_counts": {"search": record.get("tool_calls", 0)},
        "status": record.get("status", "failed"),
        "retrieved_docids": record.get("retrieved_docids", []),
        "result": [{"type": "output_text", "output": record.get("prediction", "")}],
    }
