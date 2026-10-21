from retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever()

query = "Arwa University cultural activities"

results = retriever.search(
    query=query,
    top_k=5
)

print("Query:", query)
print("Result Count:", len(results))
print("=" * 60)

for doc in results:
    print(f"Rank: {doc['rank']}")
    print(f"Doc ID: {doc['docid']}")
    print(f"Score: {doc['score']:.4f}")
    print(f"Snippet: {doc['snippet']}")
    print(f"URL: {doc['url']}")
    print("-" * 60)