# Agent benchmark

## Run directly in PyCharm

Open `scripts/evaluate_agent.py` and click **Run**. No parameters are required.
The safe defaults load the project dataset, select the same 10 shuffled cases
with seed 42, show Agent traces, and create a timestamped directory under
`output/evaluations/`. Change `DEFAULT_LIMIT` near the top of the script if a
larger click-to-run baseline is desired.

The run calls the configured real LLM API. The default is intentionally 10
cases so clicking Run cannot accidentally launch all 830 paid cases.

`data/corpus` contains retrieval documents only. Download the official 830
BrowseComp-Plus test questions and answers once (the decrypted file is excluded
from git):

```powershell
D:\envs\nlp_project\python.exe scripts\download_browsecomp_plus.py
```

Create a low-cost, repeatable baseline with a fixed sample and seed:

```powershell
D:\envs\nlp_project\python.exe scripts\evaluate_agent.py `
  --dataset data\browsecomp_plus_decrypted.jsonl `
  --shuffle --seed 42 --limit 20 `
  --output output\evaluations\baseline-20
```

Run the same command after an Agent change with a different output directory.
The report includes exact/contains match, token F1, completion and Unknown rates,
retrieval recall, LLM rounds, tool calls, tokens, latency, and termination reasons.
Use `--resume` to continue an interrupted run. Add `--trace` when node-level logs
are useful.

The deterministic answer metrics are intended for fast regression. Formal
BrowseComp-Plus reporting should additionally run the official Qwen3-32B judge
against the generated `official_run` directory.

## Retrieval-only evaluation

Open `scripts/evaluate_retrieval.py` in PyCharm and click **Run**. It never
imports or calls the LLM. By default it evaluates BM25 + dense hybrid retrieval on
the fixed seed-42 sample and reports Gold/Evidence Recall and Hit Rate at
5/10/20/50, MRR, nDCG@10, latency, and `answer_hit@K`. The last metric directly
checks whether a normalized standard answer occurs in any returned clue snippet.

Start Qdrant and build the dense index before the first hybrid run:

```powershell
docker compose up -d qdrant
python scripts/build_dense_index.py
python scripts/evaluate_retrieval.py --modes rrf dense hybrid --limit 20
```

Dense indexing and query embedding use SiliconFlow's
`BAAI/bge-large-en-v1.5` API. The API credential is read from `LLM_API_KEY`;
the key is never stored in Qdrant payloads. Only the matching tokenizer is
downloaded locally so chunks stay below the model's 512-token input limit.

The old long-OR retriever is retained for occasional ablation runs with
`--modes legacy rrf`, but it is intentionally not enabled by default because
Top-50 queries can take tens of seconds per case on the current SQLite index.
