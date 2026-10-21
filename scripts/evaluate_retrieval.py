from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import load_cases
from evaluation.retrieval_benchmark import answer_in_results, ranking_metrics, summarize_retrieval
from retrieval.bm25_full_retriever import BM25FullRetriever


# PyCharm: open this file and click Run. No LLM/API is used.
DEFAULT_DATASET = ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_LIMIT = 10
DEFAULT_SEED = 42
DEFAULT_CUTOFFS = (5, 10, 20, 50)
DEFAULT_MODES = ("hybrid",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval only; never calls the LLM.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--modes", nargs="+", choices=["legacy", "rrf", "dense", "hybrid"],
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(DEFAULT_CUTOFFS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"Test dataset not found: {dataset}")
    if args.limit < 1 or any(value < 1 for value in args.cutoffs):
        raise ValueError("limit and cutoffs must be positive")

    print("=" * 60)
    print("BrowseComp-Plus Retrieval-Only Evaluation (no LLM)")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Cutoffs: {args.cutoffs}")

    cases = load_cases(dataset)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    cases = cases[:args.limit]
    output = Path(args.output) if args.output else (
        ROOT / "output" / "retrieval_evaluations" /
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"

    max_k = max(args.cutoffs)
    records = []
    retriever = BM25FullRetriever()
    dense_retriever = None
    hybrid_retriever = None
    try:
        with results_path.open("w", encoding="utf-8") as handle:
            for index, case in enumerate(cases, 1):
                for mode in args.modes:
                    started = time.perf_counter()
                    if mode == "legacy":
                        results = retriever.search_legacy(case["query"], top_k=max_k)
                    elif mode == "rrf":
                        results = retriever.search(case["query"], top_k=max_k, candidate_k=max_k)
                    elif mode == "dense":
                        if dense_retriever is None:
                            from retrieval.dense_retriever import DenseRetriever
                            dense_retriever = DenseRetriever()
                        results = dense_retriever.search(case["query"], top_k=max_k)
                    else:
                        if hybrid_retriever is None:
                            from retrieval.hybrid_retriever import HybridRetriever
                            hybrid_retriever = HybridRetriever(sparse=retriever)
                        results = hybrid_retriever.search(case["query"], top_k=max_k)
                    elapsed = time.perf_counter() - started
                    docids = [str(item["docid"]) for item in results]
                    gold = ranking_metrics(docids, case["gold_docs"], args.cutoffs)
                    evidence = ranking_metrics(docids, case["evidence_docs"], args.cutoffs)
                    answer_hits = answer_in_results(results, case["answer"], args.cutoffs)
                    record = {
                        "query_id": case["query_id"],
                        "mode": mode,
                        "elapsed_seconds": elapsed,
                        **{f"gold_{key}": value for key, value in gold.items()},
                        **{f"evidence_{key}": value for key, value in evidence.items()},
                        **answer_hits,
                        "retrieved_docids": docids,
                        "clues": [
                            {
                                "rank": item["rank"],
                                "docid": str(item["docid"]),
                                "snippet": item.get("snippet", ""),
                                "source_ranks": item.get("source_ranks", {}),
                            }
                            for item in results
                        ],
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"[{index}/{len(cases)}] {case['query_id']} {mode} "
                        f"gold_hit@{max_k}={int(gold[f'hit_rate@{max_k}'])} "
                        f"gold_recall@{max_k}={gold[f'recall@{max_k}']:.3f} "
                        f"answer_hit@{max_k}={int(answer_hits[f'answer_hit@{max_k}'])} "
                        f"time={elapsed:.2f}s",
                        flush=True,
                    )
    finally:
        retriever.close()
        if dense_retriever is not None:
            dense_retriever.close()
        # Hybrid owns the shared sparse retriever, so closing it here would
        # close sparse twice. Its dense child still needs releasing.
        if hybrid_retriever is not None:
            hybrid_retriever.dense.close()

    summary = summarize_retrieval(records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved retrieval evaluation to {output}")


if __name__ == "__main__":
    main()
