from retrieval.bm25_full_retriever import BM25FullRetriever
import time

retriever = BM25FullRetriever()

print("Indexed documents:", retriever.count())

query = "Arwa University cultural activities"

start = time.perf_counter()

results = retriever.search(
    query=query,
    top_k=5
)

elapsed = time.perf_counter() - start

print("Query:", query)
print(f"Search Time: {elapsed:.4f}s")
print("=" * 60)

for result in results:
    print("Rank:", result["rank"])
    print("Doc ID:", result["docid"])
    print("Score:", result["score"])
    print("Snippet:", result["snippet"])
    print("URL:", result["url"])
    print("-" * 60)

retriever.close()