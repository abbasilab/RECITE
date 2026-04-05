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

| Script | Description |
|--------|-------------|
| `scripts/rebuttal/run_hf_eval.py` | Standalone HuggingFace transformers evaluation for large models (multi-GPU, no vLLM). Outputs consolidated JSON to `data/rebuttal/`. |
| `scripts/rebuttal/rebuttal_eval_endpoint.py` | Concurrent async evaluation for vLLM-served endpoints (Llama-3.1-70B, Qwen3-32B). Runs predictions then batch-judges via UCSF Versa. |
| `scripts/rebuttal/equiv_test_hf.py` | Equivalence test: verify HF transformers and vLLM produce identical outputs at temperature=0 on a small sample. |
| `scripts/rebuttal/rebuttal_compile_results.py` | Compile results from all rebuttal JSONs into a scaling trend table. Reads `data/rebuttal/*.json`. |

---

## Analysis Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `scripts/truncation_analysis.py` | Computes per-model truncation/coverage statistics for all 3,116 benchmark samples (rebuttal reviewer DVUB W1 response). Outputs `data/truncation_analysis.json`. |

---

## Config Files (`config/`)

| File | Description |
|------|-------------|
| `config/benchmarks.yaml` | Main benchmark run config (model specs, RAG settings, split definitions). |
| `config/model_presets.yaml` | Model preset definitions (endpoint URLs, model IDs, token limits). |
| `config/benchmark_prompts.json` | System + user prompt templates for predictor and LLM judge. |
| `config/paths.yaml` | Data root and DB path resolution. |
| `config/rebuttal/rebuttal_*.yaml` | Per-model rebuttal run configs (gemma2-27b, llama-70b, qwen3-32b, qwen2.5-72b). |
| `config/rebuttal/equiv_test_*.yaml` | Equivalence test configs (endpoint vs HF). |
| `config/rebuttal/rebuttal_large_models.yaml` | Combined large-model scaling config. |
| `config/rebuttal/rebuttal_scaling_models.yaml` | Scaling ablation config. |
| `config/smoke/e2e.yaml` | Smoke test end-to-end config. |
