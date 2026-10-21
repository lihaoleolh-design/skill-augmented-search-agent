import pytest

from evaluation.retrieval_benchmark import answer_in_results, ranking_metrics, summarize_retrieval


def test_ranking_metrics_compute_recall_hit_mrr_and_ndcg():
    metrics = ranking_metrics(["x", "b", "a", "z"], ["a", "b"], [1, 2, 4])
    assert metrics["recall@1"] == 0
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["recall@4"] == 1
    assert metrics["hit_rate@2"] == 1
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert 0 < metrics["ndcg@10"] <= 1


def test_retrieval_summary_keeps_modes_separate():
    rows = [
        {"query_id": "1", "mode": "legacy", "gold_recall@5": 0.0, "elapsed_seconds": 1.0},
        {"query_id": "2", "mode": "legacy", "gold_recall@5": 1.0, "elapsed_seconds": 3.0},
        {"query_id": "1", "mode": "rrf", "gold_recall@5": 1.0, "elapsed_seconds": 2.0},
    ]
    summary = summarize_retrieval(rows)
    assert summary["legacy"]["avg_gold_recall@5"] == pytest.approx(0.5)
    assert summary["legacy"]["avg_elapsed_seconds"] == pytest.approx(2.0)
    assert summary["rrf"]["avg_gold_recall@5"] == 1.0


def test_answer_hit_normalizes_case_articles_and_punctuation():
    results = [
        {"snippet": "unrelated"},
        {"snippet": "The institution was Queen Arwa University."},
    ]
    metrics = answer_in_results(results, "Queen Arwa University", [1, 2])
    assert metrics == {"answer_hit@1": 0.0, "answer_hit@2": 1.0}
