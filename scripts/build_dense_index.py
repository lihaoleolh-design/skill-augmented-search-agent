from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from retrieval.dense_index import iter_text_chunks
from retrieval.embedding_client import SiliconFlowEmbeddingClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable BGE/Qdrant dense index.")
    parser.add_argument("--start", type=int, default=None, help="Override checkpoint start document.")
    parser.add_argument("--limit", type=int, default=None, help="Index only this many documents.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection.")
    parser.add_argument("--batch-size", type=int, default=settings.dense_batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if settings.dense_chunk_overlap >= settings.dense_chunk_tokens:
        raise ValueError("DENSE_CHUNK_OVERLAP must be smaller than DENSE_CHUNK_TOKENS")

    from datasets import load_from_disk
    from qdrant_client import QdrantClient, models
    from transformers import AutoTokenizer

    qdrant_path = settings.resolve_path(settings.qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = qdrant_path / f"{settings.qdrant_collection}.checkpoint.json"
    if settings.qdrant_url.strip():
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            # Avoid sending localhost traffic through a system HTTP proxy.
            trust_env=False,
            check_compatibility=False,
        )
        qdrant_display = settings.qdrant_url
    else:
        client = QdrantClient(path=str(qdrant_path))
        qdrant_display = str(qdrant_path)

    if args.recreate and client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
        checkpoint_path.unlink(missing_ok=True)
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.dense_vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        )

    checkpoint = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    start = args.start if args.start is not None else int(checkpoint.get("next_document", 0))

    dataset = load_from_disk(str(settings.resolve_path(settings.corpus_path)))
    stop = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    print(f"Model: {settings.dense_model_name}")
    print(f"Embedding API: {settings.dense_api_url}")
    print(f"Documents: {start}..{stop - 1} of {len(dataset)}")
    print(f"Qdrant: {qdrant_display} / {settings.qdrant_collection}")
    model = SiliconFlowEmbeddingClient()
    # Only the small tokenizer files are loaded locally; model weights and
    # inference remain on SiliconFlow.
    tokenizer = AutoTokenizer.from_pretrained(settings.dense_model_name)
    total_chunks = int(checkpoint.get("total_chunks", 0)) if start else 0

    try:
        for document_index in range(start, stop):
            document = dataset[document_index]
            chunks = list(iter_text_chunks(
                docid=document["docid"], url=document.get("url", ""),
                text=document.get("text", ""), tokenizer=tokenizer,
                chunk_tokens=settings.dense_chunk_tokens,
                overlap_tokens=settings.dense_chunk_overlap,
            ))
            for batch_start in range(0, len(chunks), args.batch_size):
                batch = chunks[batch_start:batch_start + args.batch_size]
                vectors = model.encode(
                    [chunk.text for chunk in batch],
                )
                points = [
                    models.PointStruct(
                        id=chunk.point_id,
                        vector=vector,
                        payload={
                            "docid": chunk.docid, "url": chunk.url,
                            "text": chunk.text,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "chunk_index": chunk.chunk_index,
                            "chunk_id": chunk.point_id,
                        },
                    )
                    for chunk, vector in zip(batch, vectors)
                ]
                client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points,
                    wait=True,
                )
            total_chunks += len(chunks)
            checkpoint_path.write_text(json.dumps({
                "next_document": document_index + 1,
                "total_chunks": total_chunks,
                "model": settings.dense_model_name,
                "chunk_tokens": settings.dense_chunk_tokens,
                "chunk_overlap": settings.dense_chunk_overlap,
            }, indent=2), encoding="utf-8")
            print(
                f"[{document_index + 1}/{stop}] docid={document['docid']} "
                f"chunks={len(chunks)} total_chunks={total_chunks}",
                flush=True,
            )
    finally:
        model.close()
        client.close()
    print("Dense index build complete.")


if __name__ == "__main__":
    main()
