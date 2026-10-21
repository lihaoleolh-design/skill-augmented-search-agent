from retrieval.hybrid_retriever import HybridRetriever


class FakeSparse:
    def search(self, query, top_k, candidate_k=None):
        return [
            {"docid": "shared", "snippet": "sparse shared"},
            {"docid": "sparse", "snippet": "sparse only"},
        ]


class FakeDense:
    def search(self, query, top_k):
        return [
            {"docid": "dense", "snippet": "dense only"},
            {"docid": "shared", "snippet": "dense shared"},
        ]


def test_hybrid_rrf_rewards_documents_found_by_both_retrievers():
    retriever = HybridRetriever(
        sparse=FakeSparse(), dense=FakeDense(), rrf_k=60, candidate_k=10,
    )
    results = retriever.search("question", top_k=3)
    assert [item["docid"] for item in results] == ["shared", "dense", "sparse"]
    assert results[0]["source_ranks"] == {"bm25": 1, "dense": 2}
    assert results[0]["snippet"] == "dense shared"
