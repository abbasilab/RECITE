# SCRIPT_CATALOG.md

Scripts and CLI entry points for RECITE.

---

## CLI: `recite` (installed package)

Entry point: `recite.cli:app` (see `pyproject.toml`). Registered via `uv run recite <command>`.

### `recite benchmark` subcommands

Source: `recite/cli/benchmark.py`

| Command | Description |
|---------|-------------|
| `recite benchmark run-benchmark` | Run benchmark predictions over a config YAML. Handles multi-run configs, threading, checkpointing, and optional RAG. |
| `recite benchmark summarize` | Print or export a summary table of all benchmark results from the results DB. |
| `recite benchmark export-splits` | Export benchmark splits from the DB to Parquet files. |
| `recite benchmark init-benchmark` | Initialize benchmark DB from ClinicalTrials.gov data (download versions, EC, protocols, detect amendments). |
| `recite benchmark ready` | Run readiness checks before a benchmark run (config validation, DB state). |
| `recite benchmark build-rag-index` | Build or rebuild the LlamaIndex RAG vector index for evidence retrieval. |
| `recite benchmark benchmark-summary` | Alias/alternate summary output (tabular, rich-formatted). |
| `recite benchmark merge-duplicate-configs` | Merge duplicate run configs in the results DB. |
| `recite benchmark clear-local-gpu-results` | Clear results for a given config (for re-runs). |
| `recite benchmark verify-local-gpu` | Verify GPU inference setup (model loading, tokenizer, single-sample test). |
| `recite benchmark migrate` | Migrate old results DB schema to current. |

### `recite crawl` subcommands

Source: `recite/cli/crawl.py`

| Command | Description |
|---------|-------------|
| `recite crawl run` | Crawl literature (PubMed, Semantic Scholar) using LLM-generated queries with seed-based expansion. |
| `recite crawl stats` | Print crawl DB statistics (documents, queries, relevance distribution). |
| `recite crawl review-sync-papers` | Sync reviewed papers from the literature DB into the RECITE DB. |
| `recite crawl review-sync-paper-trials` | Sync paper-trial matches from the crawl DB. |
| `recite crawl paper-trial-match` | Run LLM-based paper-to-trial matching for top relevant papers. |

---

## Legacy Orchestrators

### `recite.py` (legacy, repo root)

Legacy monolithic Typer app with direct orchestration of benchmark runs. Predates the CLI refactor. Commands:

| Command | Description |
|---------|-------------|
| `run` | Run benchmark predictions (legacy, pre-refactor). Supports Versa API and local GPU backends. |
| `ready` | Readiness checks. |
| `export-splits` | Export splits to Parquet. |
| `benchmark-summary` | Print benchmark summary table. |
| `merge-duplicate-configs` | Merge duplicate configs in results DB. |
| `build-rag-index` | Build RAG index. |
| `clear-local-gpu-results` | Clear local GPU results. |
| `verify-local-gpu` | Verify GPU setup. |
| `migrate` | Migrate DB schema. |

**Note:** Prefer `recite benchmark` CLI over `recite.py` for new runs.

### `accrual.py` (repo root)

Typer app for the accrual impact pipeline. Orchestrates:
- Phase 1: Screen documents via LLM (EC directives + impact extraction) → `paper_answers`.
- Match phase: LLM-based paper-to-trial matching → `paper_trial_matches_<preset>`.
- Phase 2: Compute enrollment gains → `paper_trial_gains`.
- Phase 3: Summary statistics.

Usage: `uv run python accrual.py [options]`

---

## Utility Scripts

### `init_recite_benchmark.py` (repo root)

Multi-worker benchmark initialization. Distributes ClinicalTrials.gov pipeline across workers using chunked NCT ID batches. Handles rate-limited API calls with a shared limiter.

Usage: `uv run init_recite_benchmark.py --workers 4 --num-chunks 10 --chunks 0,1,2,3`

---

## Reproduce Scripts (`reproduce/`)

| Script | Description |
|--------|-------------|
| `reproduce/validate_1to1.py` | 1:1 validation: re-runs original `clintriaLM` code on original DB inputs and compares outputs character-by-character. Must be run from `/home/rro/projects/clintriaLM` using its `.venv`. |
| `reproduce/compare_predictions.py` | Joins reproduced JSONL predictions with original DB results on `(instance_id, source_version, target_version)`, reports exact-match rates and character-level similarity. |

---

## Rebuttal Scripts (`scripts/rebuttal/`)

KDD 2026 rebuttal tooling (Reviewer Yns9: ≥10B open-weight model results).
Used alongside the standard `recite benchmark run-benchmark` CLI for specialized eval workflows.

| Script | Lines | Purpose |
|--------|-------|---------|
| `rebuttal_eval_endpoint.py` | 492 | Async concurrent eval for vLLM-served endpoints (Llama-3.1-70B, Qwen3-32B) |
| `run_hf_eval.py` | 399 | Standalone HF Transformers eval for large models (multi-GPU, no vLLM) |
| `rebuttal_compile_results.py` | 81 | Compile rebuttal JSON results into scaling trend markdown table |
| `equiv_test_hf.py` | 107 | Equivalence test: compare vLLM endpoint vs HF Transformers outputs at temperature=0 |

### rebuttal_eval_endpoint.py

Sends parallel async requests to an OpenAI-compatible vLLM endpoint.
Runs predictions first, then optional LLM-as-judge scoring (batch mode, 10 samples/batch).

Key features:
- Concurrent dispatch (default 8 requests, configurable)
- Checkpoint/resume: saves every 50 predictions
- Gemma system-role workaround: merges system+user prompts (Gemma rejects system role)
- Qwen3 thinking-token stripping: strips `<think>...</think>` blocks post-generation
- Evidence truncated to 20,000 chars at prompt-build time

```bash
python3 scripts/rebuttal/rebuttal_eval_endpoint.py \
  --endpoint http://localhost:8100/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --parquet data/benchmark_splits/final_test.parquet \
  --output data/rebuttal/llama31_70b_no_rag.json \
  --max-concurrent 8 --timeout 300
```

### run_hf_eval.py

Runs inference directly via HuggingFace `transformers` (`device_map="auto"`, no vLLM server).
Computes edit_similarity, BLEU, and ROUGE-L automatically (no paid judge).
Sends Telegram milestone notifications. Designed for Gemma 2 27B on abbasi-gpu-2.

```bash
python3 scripts/rebuttal/run_hf_eval.py \
  --model google/gemma-2-27b-it \
  --parquet data/benchmark_splits/final_test.parquet \
  --output data/rebuttal/gemma2_27b_no_rag.json \
  --checkpoint-dir /tmp/gemma27b_ckpt
```

### rebuttal_compile_results.py

Loads all `data/rebuttal/*_no_rag.json`, merges with paper baselines, prints a markdown
scaling trend table (model × params × binary_equiv × ordinal × errors).

```bash
python3 scripts/rebuttal/rebuttal_compile_results.py
```

### equiv_test_hf.py

Runs 2 samples via HF Transformers and compares outputs to a vLLM endpoint run,
verifying temperature=0 determinism across backends.

```bash
python3 scripts/rebuttal/equiv_test_hf.py \
  --data samples.json --prompts config/benchmark_prompts.json --output hf_results.json
```

---

## Analysis Scripts (`scripts/`)

| Script | Lines | Description |
|--------|-------|-------------|
| `truncation_analysis.py` | 322 | Per-model truncation/coverage stats for all 3,116 benchmark samples (rebuttal Reviewer DVUB W1). Outputs `data/truncation_analysis.json`. |

Computes: fraction of documents exceeding each model's effective evidence token budget,
mean/median/p95 document lengths per model tokenizer, length-controlled re-analysis.

```bash
uv run python scripts/truncation_analysis.py
```

---

## Config Files (`config/`)

### Core Configs

| File | Description |
|------|-------------|
| `benchmarks.yaml` | Main benchmark run config (model specs, RAG settings, split definitions) |
| `model_presets.yaml` | Per-model context window, API type, and endpoint URL presets |
| `benchmark_prompts.json` | System + user prompt templates for model predictor and LLM-as-judge |
| `paths.yaml` | Data root and DB path resolution |
| `accrual.yaml` | Accrual impact scoring pipeline config |
| `accrual_prompts.json` | Prompt templates for EC directive extraction |
| `crawler_prompts.json` | Prompt templates for literature crawler relevance scoring |
| `recite_quick.yaml` | Quick eval config (small sample, fast iteration) |
| `smoke/e2e.yaml` | Smoke test: 3-sample end-to-end eval |

### Rebuttal Configs (`config/rebuttal/`)

| Config | Model | TP | max-model-len | Notes |
|--------|-------|----|---------------|-------|
| `rebuttal_gemma27b.yaml` | `google/gemma-2-27b-it` | 4 | 8192 | Gemma scaling story (2B→9B→27B) |
| `rebuttal_qwen72b.yaml` | `Qwen/Qwen2.5-72B-Instruct` | 8 | 8192 | Qwen scaling story (0.5B→…→72B) |
| `rebuttal_llama70b.yaml` | `meta-llama/Llama-3.1-70B-Instruct` | 8 | 32768 | AWQ-INT4, enforce-eager |
| `rebuttal_qwen32b.yaml` | `Qwen/Qwen3-32B` (FP8) | 8 | 32768 | enable_thinking: false |
| `rebuttal_large_models.yaml` | Llama-3.1-70B (pilot) | — | 131072 | First pilot; OOM, superseded |
| `rebuttal_scaling_models.yaml` | Gemma27B + Qwen72B | 4/8 | 32768 | Combined scaling config |
| `equiv_test_endpoint.yaml` | `google/gemma-2-27b-it` | — | 8192 | Endpoint equivalence test |
| `equiv_test_python.yaml` | `google/gemma-2-27b-it` | — | 8192 | python_gpu equivalence test |

vLLM server launch commands are embedded as comments in each config. General pattern:

```bash
# Gemma 2 27B (4 GPUs, TP=4)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/gemma-2-27b-it \
  --tensor-parallel-size 4 --max-model-len 8192 \
  --dtype bfloat16 --port 8200 --enforce-eager

# Qwen2.5-72B (8 GPUs, TP=8)
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 --max-model-len 8192 \
  --dtype bfloat16 --port 8300 --enforce-eager

# Run eval via CLI
uv run python -m recite.cli.benchmark run-benchmark \
  --config config/rebuttal/rebuttal_gemma27b.yaml
```
