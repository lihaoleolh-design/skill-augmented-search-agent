from functools import lru_cache

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings
from retrieval.bm25_full_retriever import BM25FullRetriever
from tools.document_tools import DOCUMENT_TOOLS


@lru_cache(maxsize=1)
def get_dense_retriever():
    from retrieval.dense_retriever import DenseRetriever
    return DenseRetriever()


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever()


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="Focused keyword search query"
    )

    top_k: int = Field(
        default=settings.retrieval_top_k,
        ge=1,
        le=20,
        description=(
            "Number of fused ranked documents to return. Omit this unless "
            "there is a reason to override the configured default."
        )
    )


@tool(args_schema=SearchDocumentsInput)
def search_documents(
    query: str,
    top_k: int = settings.retrieval_top_k
) -> list[dict]:
    """
    Search the BrowseComp-Plus corpus.

    Returns ranked document IDs, BM25 scores,
    evidence snippets and source URLs.

    Use this tool to find documents relevant
    to a question or unresolved clue.
    """
    if settings.retrieval_method == "dense":
        return get_dense_retriever().search(query=query, top_k=top_k)
    if settings.retrieval_method == "hybrid":
        return get_hybrid_retriever().search(query=query, top_k=top_k)

    retriever = BM25FullRetriever(
        db_path=settings.resolve_path(
            settings.bm25_db_path
        )
    )

    try:
        results = retriever.search(
            query=query,
            top_k=top_k
        )
        return results
    finally:
        retriever.close()


ALL_TOOLS = [
    search_documents,
    *DOCUMENT_TOOLS
]
