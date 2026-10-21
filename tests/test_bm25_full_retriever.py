from retrieval.bm25_full_retriever import (
    BM25FullRetriever,
    build_query_variants,
)
from agent.tool_registry import search_documents
from config import settings


def test_query_variants_remove_stopwords_and_keep_entity_phrases():
    variants = build_query_variants(
        "Which university held the Fifth Cultural Week in May 2002?"
    )
    assert any('"Fifth Cultural Week"' in value for value in variants)
    assert any('"2002"' in value for value in variants)
    focused_or = variants[-1]
    assert '"which"' not in focused_or.lower()
    assert '"the"' not in focused_or.lower()


def test_query_variants_have_strict_and_broad_forms():
    variants = build_query_variants("Queen Arwa University established year")
    assert any(" AND " in value for value in variants)
    assert any(" OR " in value for value in variants)


def test_rrf_search_returns_unique_limited_results_and_known_document():
    retriever = BM25FullRetriever()
    try:
        results = retriever.search(
            "Fifth Cultural Week May 2002 Queen Arwa University",
            top_k=5,
            candidate_k=20,
        )
    finally:
        retriever.close()

    assert len(results) == 5
    assert len({item["docid"] for item in results}) == 5
    assert "5412" in {item["docid"] for item in results}
    assert [item["rank"] for item in results] == [1, 2, 3, 4, 5]
    assert all(item["score"] > 0 for item in results)


def test_search_tool_uses_configured_result_count():
    assert search_documents.args["top_k"]["default"] == settings.retrieval_top_k
