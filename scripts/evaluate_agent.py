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

from evaluation.benchmark import load_cases, make_record, official_run_record, summarize

# PyCharm: open this file and click Run. These defaults are deliberately small
# so an accidental click does not start the full 830-case paid evaluation.
DEFAULT_DATASET = ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_LIMIT = 10
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeatable BrowseComp-Plus agent benchmark.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Decrypted BrowseComp-Plus JSONL file (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--output", help="Output directory; defaults to output/evaluations/<timestamp>.")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Maximum cases to run (default: {DEFAULT_LIMIT}; use 830 for the full set).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True,
        help="Deterministically shuffle before selecting cases (default: enabled).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip query IDs already in results.jsonl.")
    parser.add_argument(
        "--trace", action=argparse.BooleanOptionalAction, default=True,
        help="Show normal gray agent trace output (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {dataset_path}\n"
            "Run scripts/download_browsecomp_plus.py first."
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    print("=" * 60)
    print("BrowseComp-Plus Agent Evaluation")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Sample:  limit={args.limit}, shuffle={args.shuffle}, seed={args.seed}")
    print(f"Trace:   {args.trace}")

    last_percent = -1

    def loading_progress(read_bytes: int, total_bytes: int) -> None:
        nonlocal last_percent
        percent = int(read_bytes * 100 / total_bytes) if total_bytes else 100
        if percent >= last_percent + 5 or percent == 100:
            last_percent = percent
            print(f"[dataset] loading {percent}%", flush=True)

    print(f"[dataset] reading {dataset_path}", flush=True)
    cases = load_cases(dataset_path, progress=loading_progress)
    print(f"[dataset] loaded {len(cases)} cases", flush=True)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    cases = cases[args.offset:]
    if args.limit is not None:
        cases = cases[:args.limit]

    output = Path(args.output) if args.output else (
        ROOT / "output" / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    existing = []
    if args.resume and results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    completed_ids = {row["query_id"] for row in existing}

    from agent.graph import run_agent
    from agent.hooks import ConsoleTraceHook

    hooks = ConsoleTraceHook() if args.trace else None
    records = list(existing)
    pending = [case for case in cases if case["query_id"] not in completed_ids]
    for index, case in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {case['query_id']}", flush=True)
        started = time.perf_counter()
        try:
            result = run_agent(case["query"], hooks=hooks)
            record = make_record(case, result, time.perf_counter() - started)
        except Exception as error:
            record = {
                "query_id": case["query_id"], "query": case["query"],
                "reference_answer": case["answer"], "prediction": "",
                "exact_match": False, "contains_match": False, "token_f1": 0.0,
                "status": "failed", "termination_reason": "exception",
                "llm_rounds": 0, "tool_calls": 0, "total_tokens": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "retrieved_docids": [], "evidence_recall": None, "gold_recall": None,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"  exact={record['exact_match']} rounds={record['llm_rounds']} "
            f"tools={record['tool_calls']} tokens={record['total_tokens']} "
            f"time={record['elapsed_seconds']:.2f}s",
            flush=True,
        )

    summary = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    official_dir = output / "official_run"
    official_dir.mkdir(exist_ok=True)
    for record in records:
        (official_dir / f"{record['query_id']}.json").write_text(
            json.dumps(official_run_record(record), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
