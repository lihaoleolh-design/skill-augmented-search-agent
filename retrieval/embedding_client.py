from __future__ import annotations

import time
from typing import Any

import httpx

from config import settings


class SiliconFlowEmbeddingClient:
    """SentenceTransformer-like adapter for SiliconFlow's embedding API."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None,
                 api_url: str | None = None, timeout: int | None = None,
                 max_retries: int | None = None, client: Any | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        if not self.api_key:
            raise ValueError("Missing LLM_API_KEY for SiliconFlow embeddings")
        self.model_name = model_name or settings.dense_model_name
        self.api_url = api_url or settings.dense_api_url
        self.max_retries = settings.dense_api_max_retries if max_retries is None else max_retries
        self.client = client or httpx.Client(timeout=timeout or settings.dense_api_timeout)
        self._owns_client = client is None

    def encode(self, values: list[str], **_: Any) -> list[list[float]]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("Embedding input must contain non-empty strings")
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model_name, "input": values,
                          "encoding_format": "float"},
                )
                response.raise_for_status()
                rows = sorted(response.json()["data"], key=lambda item: item["index"])
                embeddings = [row["embedding"] for row in rows]
                if len(embeddings) != len(values):
                    raise RuntimeError(
                        f"Embedding API returned {len(embeddings)} vectors for {len(values)} inputs"
                    )
                return embeddings
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Unreachable embedding retry state")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
