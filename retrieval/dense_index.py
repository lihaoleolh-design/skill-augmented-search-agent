from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


POINT_NAMESPACE = uuid.UUID("7cad4d9a-c2f0-4e47-8ba4-59191f767867")


@dataclass(frozen=True)
class TextChunk:
    docid: str
    url: str
    text: str
    start_offset: int
    end_offset: int
    chunk_index: int

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(
            POINT_NAMESPACE,
            f"{self.docid}:{self.start_offset}:{self.end_offset}",
        ))


def iter_text_chunks(
    docid: str,
    url: str,
    text: str,
    tokenizer: Any,
    chunk_tokens: int,
    overlap_tokens: int,
) -> Iterator[TextChunk]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    if not text or not text.strip():
        return

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        # Long source documents are intentionally tokenized in full and then
        # split below. Suppress the model-limit warning because only bounded
        # chunks, never this full sequence, are sent to the embedding API.
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    step = chunk_tokens - overlap_tokens
    chunk_index = 0
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + chunk_tokens, len(offsets))
        if token_start >= token_end:
            break
        start_offset = int(offsets[token_start][0])
        end_offset = int(offsets[token_end - 1][1])
        chunk_text = text[start_offset:end_offset].strip()
        if chunk_text:
            yield TextChunk(
                docid=str(docid), url=str(url or ""), text=chunk_text,
                start_offset=start_offset, end_offset=end_offset,
                chunk_index=chunk_index,
            )
            chunk_index += 1
        if token_end == len(offsets):
            break
