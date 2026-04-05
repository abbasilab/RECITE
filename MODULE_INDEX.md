# MODULE_INDEX.md

Package structure for `recite/` (installed as `recite` via `pyproject.toml`).

---

## Entry Point

| Path | Description |
|------|-------------|
| `recite/cli/__init__.py` | Root Typer app. Registers `benchmark` and `crawl` sub-apps. Invoked as `recite` CLI. |

---

## `recite/benchmark/`

Core benchmark pipeline: data ingestion, inference, evaluation.

| Module | Description |
|--------|-------------|
| `pipeline.py` | E2E pipeline orchestrator. Calls discovery → download → process → build stages. `run_e2e_pipeline()` with progressive filtering and optional chunked parallelism. |
| `evaluator.py` | Benchmark runner and LLM judge. `run_single_sample()`: calls model API, computes edit metrics. `batched_scorer()`: batch judge scoring via Versa. `load_python_gpu_model()` / `clear_python_gpu_cache()` for local HF inference. |
| `ec_detector.py` | EC change detection. `detect_ec_changes()`: difflib-based change detection with configurable thresholds. |
| `config_loader.py` | Parses `config/benchmarks.yaml` → list of experiment specs. `load_benchmark_config()`, `get_experiment_specs()`. |
| `dataloader.py` | Streams benchmark samples from SQLite DB or Parquet splits. `stream_from_db()`, `stream_parquet_splits()`, `validate_train_split()`. |
| `results_db.py` | Results persistence: stores predictions + metrics in SQLite. `insert_result()`, `update_judge_scores()`, `get_benchmark_summary_rows()`, `has_sample()`. |
| `db.py` | Main RECITE benchmark DB (trials, EC versions). `get_connection()`, `init_database()`, `get_db_path()`. |
| `discovery.py` | Trial discovery from ClinicalTrials.gov. `discover_all_instance_ids()` (bulk XML or API), `check_trial_versions_batch()`. |
| `downloaders.py` | Download trial versions, EC text, protocol PDFs. `download_versions()`, `download_ecs()`, `download_protocols()`. |
| `processors.py` | Identify EC amendments and extract evidence from protocol PDFs. `identify_amendments()`, `extract_evidence()`. |
| `builders.py` | Build RECITE instances (train/test splits) from processed trials. `create_recite_instances()`. |
| `parquet_exporter.py` | Export splits to Parquet. `export_to_parquet_splits()`, `export_final_test_to_parquet()`. |
| `summary_table.py` | Build markdown/rich summary tables from results DB. |
| `rate_limiter.py` | Shared token-bucket rate limiter for ClinicalTrials.gov API calls. |
| `api_client.py` | ClinicalTrials.gov REST API client. `fetch_version_history()`. |
| `evidence_downloader.py` | Download and parse protocol PDF evidence for RAG. |
| `protocol_parser.py` | Extract structured text from protocol PDFs (PyMuPDF). |
| `module_labels.py` | EC module label constants (inclusion/exclusion categories). |
| `utils.py` | DB query helpers: `get_trials_ready_for_recite()`, `get_trials_with_ec_changes()`, etc. |

---

## `recite/cli/`

CLI entry points (Typer apps).

| Module | Description |
|--------|-------------|
| `__init__.py` | Root app; registers `benchmark` and `crawl` sub-apps. |
| `benchmark.py` | `recite benchmark` subcommands: `run-benchmark`, `summarize`, `export-splits`, `init-benchmark`, `ready`, `build-rag-index`, `benchmark-summary`, `merge-duplicate-configs`, `clear-local-gpu-results`, `verify-local-gpu`, `migrate`. |
| `crawl.py` | `recite crawl` subcommands: `run`, `stats`, `review-sync-papers`, `review-sync-paper-trials`, `paper-trial-match`. |
| `common.py` | Shared CLI utilities: `MODEL_PRESETS` dict, common option types. |

---

## `recite/crawler/`

Literature discovery pipeline.

| Module | Description |
|--------|-------------|
| `adapters.py` | Search adapters: `PubMedAdapter`, `SemanticScholarAdapter`, `ClinicalTrialsGovAdapter`. `generate_queries()`, `generate_seed_queries()`, `evaluate_paper()`. |
| `llm.py` | LLM client for crawler relevance scoring. `LLMClient` wraps OpenAI-compatible endpoint. |
| `paper_trial_matcher.py` | LLM-based paper-to-trial matching. |
| `db.py` | Crawler SQLite DB: documents, queries, relevance scores. `init_db()`, `save_document()`, `has_query()`, etc. |

---

## `recite/accrual/`

Accrual impact pipeline: quantifies enrollment impact of EC amendments.

| Module | Description |
|--------|-------------|
| `__init__.py` | Public API: re-exports DB helpers, parsing, and prompt loading. |
| `db.py` | Accrual DB: `paper_answers`, `paper_trial_matches`, `paper_trial_gains`. `init_accrual_db()`, `insert_paper_answer()`, `insert_paper_trial_gain()`. |
| `llm.py` | LLM calls for directive extraction and impact scoring. `call_accrual_llm()`. |
| `prompts.py` | Load accrual prompts from `config/accrual_prompts.json`. `load_accrual_prompts()`. |
| `parsing.py` | Parse LLM responses for directives and impact. `parse_directives_response()`, `parse_impact_response()`. |
| `match_phase.py` | Trial matching phase: find matching trials for papers via LLM. |
| `summary_phase.py` | Compute and format accrual summary statistics. |
| `backfill_impact.py` | Backfill `impact_evidence` for rows with null values (re-calls impact LLM). |

---

## `recite/rag/`

RAG (Retrieval-Augmented Generation) support using LlamaIndex.

| Module | Description |
|--------|-------------|
| `query.py` | LlamaIndex-backed RAG. `build_index_for_document()`: builds or loads a cached vector index from evidence text. Supports HuggingFace and OpenAI-compatible embeddings. Falls back to direct LLM call when document is empty. |

---

## `recite/utils/`

Shared utilities.

| Module | Description |
|--------|-------------|
| `logging_config.py` | Central loguru configuration. `configure_logging()`: sets up file sink under `logs/` and optional stderr. |
| `path_loader.py` | Path resolution from `config/paths.yaml` and `.env`. `get_project_root()`, `get_data_root()`, `get_local_db_dir()`, `resolve_path()`. |

---

## `recite/llmapis.py`

Lightweight OpenAI API wrapper (legacy, from original clintriaLM codebase).

- `AbstractLLMAPI`: base class with `__call__` interface.
- Subclasses for GPT-4.1 family models via UCSF Versa endpoint.
- Used by early pipeline stages; newer code uses `httpx` directly or the evaluator's inline API client.

---

## Package Install

```bash
uv pip install -e .          # Standard install
uv pip install -e ".[rag]"   # With RAG dependencies (LlamaIndex, sentence-transformers, torch)
uv pip install -e ".[dev]"   # With test dependencies
```

Runtime: Python ≥ 3.12. Key dependencies: `typer`, `loguru`, `pyarrow`, `pandas`, `openai`, `httpx`, `pymupdf`, `tiktoken`.
