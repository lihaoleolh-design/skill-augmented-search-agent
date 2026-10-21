import json

import pytest

from evaluation.benchmark import (
    answer_metrics,
    load_cases,
    make_record,
    normalize_answer,
    official_run_record,
    summarize,
)


def test_normalize_answer_ignores_case_punctuation_and_articles():
    assert normalize_answer("The Queen Arwa University.") == "queen arwa university"


def test_answer_metrics_include_exact_contains_and_token_f1():
    exact = answer_metrics("The Queen Arwa University", "Queen Arwa University")
    assert exact == {"exact_match": True, "contains_match": True, "token_f1": 1.0}
    partial = answer_metrics("Queen Arwa University, 1996", "Queen Arwa University")
    assert partial["exact_match"] is False
    assert partial["contains_match"] is True
    assert 0 < partial["token_f1"] < 1
    assert answer_metrics("", "Alpha")["contains_match"] is False


def test_load_cases_accepts_official_and_simple_field_names(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "Question?", "answer": "Answer"}) + "\n"
        + json.dumps({"question": "Q2?", "answer": "A2"}) + "\n",
        encoding="utf-8",
    )
    rows = load_cases(path)
    assert [row["query_id"] for row in rows] == ["q1", "2"]


def test_load_cases_discards_embedded_document_text_and_reports_progress(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({
        "query_id": "q1", "query": "Q?", "answer": "A",
        "evidence_docs": [{"docid": "10", "text": "large evidence"}],
        "gold_docs": [{"docid": "11", "text": "large gold"}],
        "negative_docs": [{"docid": "12", "text": "large negative"}],
    }) + "\n", encoding="utf-8")
    progress = []
    rows = load_cases(path, progress=lambda current, total: progress.append((current, total)))
    assert rows[0]["evidence_docs"] == ["10"]
    assert rows[0]["gold_docs"] == ["11"]
    assert "negative_docs" not in rows[0]
    assert progress[-1][0] == progress[-1][1]


def test_make_record_measures_answer_efficiency_and_retrieval():
    case = {
        "query_id": "q1", "query": "Q", "answer": "Alpha",
        "evidence_docs": [{"docid": "1"}, {"docid": "2"}],
        "gold_docs": [{"docid": "2"}],
    }
    result = {
        "final_answer": "Alpha", "termination_reason": "completed",
        "search_round": 2, "tool_calls": 3, "total_tokens": 100,
        "visited_docs": ["2"], "evidence": [{"docid": "1"}],
    }
    record = make_record(case, result, 1.25)
    assert record["exact_match"] is True
    assert record["evidence_recall"] == 1.0
    assert record["gold_recall"] == 1.0
    assert record["retrieved_docids"] == ["1", "2"]


def test_summary_and_official_export():
    rows = [
        {"query_id": "1", "prediction": "A", "exact_match": True, "contains_match": True,
         "token_f1": 1.0, "status": "completed", "termination_reason": "completed",
         "llm_rounds": 2, "tool_calls": 1, "total_tokens": 10, "elapsed_seconds": 1.0,
         "evidence_recall": 1.0, "gold_recall": 1.0, "retrieved_docids": ["9"]},
        {"query_id": "2", "prediction": "Unknown", "exact_match": False, "contains_match": False,
         "token_f1": 0.0, "status": "failed", "termination_reason": "search_stalled",
         "llm_rounds": 4, "tool_calls": 3, "total_tokens": 30, "elapsed_seconds": 3.0,
         "evidence_recall": None, "gold_recall": None, "retrieved_docids": []},
    ]
    summary = summarize(rows)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert summary["avg_total_tokens"] == pytest.approx(20)
    assert summary["unknown_rate"] == pytest.approx(0.5)
    exported = official_run_record(rows[0])
    assert exported["result"][0]["output"] == "A"
    assert exported["retrieved_docids"] == ["9"]
