import hashlib
import re
from typing import Any


MAX_STORED_EVIDENCE_ITEMS = 50
MAX_STORED_EVIDENCE_CHARS = 2000


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def create_evidence(
    *,
    docid: str | None,
    url: str | None,
    text: Any,
    source_tool: str,
    query: str | None = None,
    start_offset: int | None = None,
    end_offset: int | None = None,
    score: float | None = None,
    rank: int | None = None,
) -> dict[str, Any] | None:
    """Build one normalized evidence item, or None for empty text."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    if not text:
        return None

    text = text[:MAX_STORED_EVIDENCE_CHARS]
    identity = "\x1f".join([
        str(docid or ""),
        str(start_offset if start_offset is not None else ""),
        str(end_offset if end_offset is not None else ""),
        _normalized_text(text),
    ])
    evidence_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return {
        "evidence_id": evidence_id,
        "docid": docid,
        "url": url,
        "text": text,
        "source_tool": source_tool,
        "query": query,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "score": score,
        "rank": rank,
    }


def merge_evidence(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Merge, deduplicate by stable ID, and bound stored evidence."""
    unique: dict[str, dict[str, Any]] = {}

    for item in [*existing, *additions]:
        if not item:
            continue

        evidence_id = item.get("evidence_id")
        if not evidence_id:
            normalized = create_evidence(
                docid=item.get("docid"),
                url=item.get("url"),
                text=item.get("text", ""),
                source_tool=item.get("source_tool", "unknown"),
                query=item.get("query"),
                start_offset=item.get("start_offset"),
                end_offset=item.get("end_offset"),
                score=item.get("score"),
                rank=item.get("rank"),
            )
            if normalized is None:
                continue
            item = normalized
            evidence_id = item["evidence_id"]

        unique[evidence_id] = item

    return list(unique.values())[-MAX_STORED_EVIDENCE_ITEMS:]
