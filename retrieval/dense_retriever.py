from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from config import settings
from retrieval.embedding_client import SiliconFlowEmbeddingClient


QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """SiliconFlow BGE query encoder backed by a Qdrant collection."""

    def __init__(
        self,
        qdrant_path: str | Path | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
        model: Any | None = None,
    ):
        self.collection_name = collection_name or settings.qdrant_collection
        path = Path(qdrant_path or settings.resolve_path(settings.qdrant_path))
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as error:
                raise RuntimeError(
                    "Dense retrieval requires qdrant-client. Run: pip install -r requirements.txt"
                ) from error
            if settings.qdrant_url.strip():
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    # A localhost Qdrant endpoint must not be routed through a
                    # Windows/system HTTP proxy, which otherwise returns 502.
                    trust_env=False,
                    check_compatibility=False,
                )
            else:
                client = QdrantClient(path=str(path))
        if model is None:
            model = SiliconFlowEmbeddingClient(model_name=model_name)
        self.client = client
        self.model = model

        if not self.client.collection_exists(self.collection_name):
            raise FileNotFoundError(
                f"Qdrant collection '{self.collection_name}' does not exist. "
                "Run scripts/build_dense_index.py first."
            )

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [QUERY_INSTRUCTION + query.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def search(self, query: str, top_k: int = 8, chunk_limit: int | None = None) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []
        limit = chunk_limit or max(top_k * 5, top_k)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.encode_query(query),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # A document can own many chunks. Preserve the highest-scoring chunk
        # per document so document-level evaluation does not contain duplicates.
        documents: OrderedDict[str, dict] = OrderedDict()
        for point in response.points:
            payload = point.payload or {}
            docid = str(payload.get("docid", ""))
            if not docid or docid in documents:
                continue
            documents[docid] = {
                "rank": len(documents) + 1,
                "docid": docid,
                "score": float(point.score),
                "snippet": str(payload.get("text", "")),
                "url": str(payload.get("url", "")),
                "chunk_id": str(payload.get("chunk_id", point.id)),
                "start_offset": int(payload.get("start_offset", 0)),
                "end_offset": int(payload.get("end_offset", 0)),
            }
            if len(documents) >= top_k:
                break
        return list(documents.values())

    def close(self) -> None:
        model_close = getattr(self.model, "close", None)
        if model_close:
            model_close()
        close = getattr(self.client, "close", None)
        if close:
            close()
