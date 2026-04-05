# RESULTS_INDEX.md

Results index for RECITE benchmark experiments (KDD 2026 rebuttal).

---

## Metrics Reference

| Metric | Range | Meaning |
|--------|-------|---------|
| `binary_equiv` | 0–1 | Fraction of predictions judged acceptable (binary LLM judge). Primary acceptability metric. |
| `mean_ordinal` | 0–4 | Mean ordinal quality score (0=no match, 1=poor, 2=partial, 3=good, 4=excellent). |
| `edit_similarity` | 0–1 | 1 − normalized edit distance; higher = closer to reference. |
| `rouge_l` | 0–1 | ROUGE-L F1; higher = more overlap with reference. |
| `bleu` | 0–1 | BLEU n-gram overlap. |
| `binary_correct` | 0 or 1 | Exact string match (almost always 0 for revision tasks). |

See `data/benchmark_predictions/BENCHMARK_SUMMARY.md` for full metric definitions.

---

## Original Paper Results (from paper Table, LLM-judge binary equivalence)

These runs are in `data/benchmark_predictions/` and are **gitignored** (reproduced locally).

| Model ID | HF / API Model | Params | Run Directory | Runs | `binary_equiv` (paper) | `edit_similarity` | `rouge_l` |
|----------|---------------|--------|---------------|------|------------------------|-------------------|-----------|
| versa-4o | GPT-4o via Versa | — | `run_2026-04-05T01-04-36Z_no_rag` | 2 | **0.858** | 0.545 | 0.560 |
| versa-4o-mini | GPT-4o-mini via Versa | ~8B | `run_2026-04-05T01-09-46Z_no_rag` | 2 | **0.840** | 0.560 | 0.573 |
| local-gemma2-9b | google/gemma-2-9b-it | 9B | `run_2026-04-05T01-41-42Z_no_rag` | 2 | **0.847** | — | — |
| local-qwen-7b | Qwen/Qwen2.5-7B-Instruct | 7B | `run_2026-04-05T01-34-31Z_no_rag` | 2 | **0.500** | 0.598 | 0.474 |
| local-qwen-3b | Qwen/Qwen2.5-3B-Instruct | 3B | `run_2026-04-05T01-25-03Z_no_rag` | 2 | 0.367 | 0.494 | 0.511 |
| local-gemma2-2b | google/gemma-2-2b-it | 2B | `run_2026-04-05T01-34-52Z_no_rag` | 2 | 0.233 | 0.305 | 0.253 |
| local-qwen-0_5b | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | `run_2026-04-05T01-15-53Z_no_rag` | 2 | 0.067 | 0.212 | 0.167 |
| gemma2-27b-endpoint | google/gemma-2-27b-it (endpoint) | 27B | `run_2026-04-04T16-49-58Z_no_rag` | 2 | — (no judge) | 0.729 | 0.776 |
| local-longctx-7b | (long-context 7B local) | 7B | `run_2026-04-05T01-41-48Z_no_rag` | 2 | — | — | — |
| vllm-llama-70b | meta-llama/Llama-3.1-70B-Instruct | 70B | `run_2026-04-04T10-26-18Z_no_rag` | 1 | — (no judge) | 0.671 | 0.659 |
| vllm-qwen3-32b | Qwen/Qwen3-32B | 32B | `run_2026-04-04T13-19-01Z_no_rag` | 1 | — (no judge) | 0.597 | 0.603 |

Notes:
- All runs are `no_rag` (evidence truncated, no retrieval) on the `train` split (n=30 for small models, n=2 for some, n=3116 full for rebuttal).
- `gemma2-27b-endpoint` has a second RAG run: `run_2026-04-04T16-51-58Z_topk10`.
- `—` in `binary_equiv` means LLM judge was not run for that benchmark run.
- First run directory per model is typically an aborted/checkpoint run; second is the completed run.

---

## Rebuttal Results (large open-weight models, full benchmark, n=3116)

Stored in `data/rebuttal/` as consolidated JSON files. These **are tracked in git** (rebuttal evidence).

| File | HF Model | Params | n_samples | n_errors | `binary_equiv` | `mean_ordinal` | n_judged | Timestamp |
|------|----------|--------|-----------|----------|----------------|----------------|----------|-----------|
| `gemma2-27b_no_rag.json` | google/gemma-2-27b-it | 27B | 3116 | 0 | — (pending) | — | — | 2026-04-05T12:08Z |
| `llama31_70b_no_rag.json` | meta-llama/Llama-3.1-70B-Instruct | 70B | 3116 | 296 | **0.422** | 3.056 | 2820 | 2026-04-04T11:32Z |
| `qwen25-72b_no_rag.json` | Qwen/Qwen2.5-72B-Instruct | 72B | 3116 | 0 | — (pending) | — | — | 2026-04-05T14:03Z |
| `qwen3_32b_no_rag.json` | Qwen/Qwen3-32B | 32B | 3116 | 577 | **0.432** | 3.153 | 2539 | 2026-04-04T16:22Z |

Notes:
- All runs: `no_rag=True`, full benchmark split (3116 samples).
- `gemma2-27b` and `qwen25-72b` contain predictions only; LLM judge scoring not yet run.
- `llama31_70b` and `qwen3_32b` have full judge scores (`binary_equiv`, `mean_ordinal`).
- Errors in `llama31_70b` (296) and `qwen3_32b` (577) are inference errors; `n_judged` = n_samples − n_errors.
- Result format: top-level keys `model`, `no_rag`, `n_samples`, `n_errors`, `timestamp`, `results` (list); judged files also have `binary_equiv`, `mean_ordinal`, `n_judged`.

### Rebuttal JSON schema (per result item)

```json
{
  "id": "<int>",
  "instance_id": "<NCT...>",
  "prediction": "<model output>",
  "predicted_at": "<ISO timestamp>",
  "ground_truth": "<reference EC>",
  "is_error": false,
  // Present only in judged files:
  "judge_binary": 0,
  "judge_ordinal": 3,
  "judge_raw": "<raw judge response>"
}
```

---

## Gitignore Status

| Path | Tracked |
|------|---------|
| `data/benchmark_predictions/` | Gitignored (reproduced locally) |
| `data/rebuttal/*.json` | Tracked in git (rebuttal evidence, ~19–20 MB each) |
| `data/benchmark_splits/benchmark.parquet` | Tracked via Git LFS (`.gitattributes`) |
| `data/dev/`, `data/prod/` | Gitignored |
