# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence** — public-facing benchmark package. Automates discovery, extraction, and implementation of evidence-based eligibility criteria revisions for clinical trials.

Note: This is the clean public release repo.

## Quick Start

```bash
uv sync                        # install dependencies
uv sync --extra rag             # with RAG support
uv run pytest -q                # tests (synthetic fixtures, no API keys)
```

## Project Structure

```
recite/                     # Core package
  benchmark/                # EC diff detection, triplet building, evaluation
  crawler/                  # Literature crawl (PubMed, Semantic Scholar)
  accrual/                  # Directive extraction, impact scoring
  rag/                      # RAG-based evidence retrieval (optional)
  cli/                      # Typer CLI (benchmark, crawl)
  utils/                    # Logging, path resolution
  llmapis.py                # LLM API wrappers

config/                     # YAML/JSON configs and prompt templates
tests/fixtures/             # Synthetic trial, protocol, paper data
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `recite.py` | RECITE benchmark runner (`run`, `ready`, `build-rag-index`) |
| `init_recite_benchmark.py` | Benchmark construction from ClinicalTrials.gov |
| `accrual.py` | Accrual impact scoring pipeline |
| CLI: `recite crawl` | Literature discovery and relevance scoring |

## Key Numbers

- 11,913 benchmark instances across 5,735 trials
- Best model: GPT-4o-mini (85.8% binary equivalence)
- Best open-weight: Gemma 2 9B (84.7%)

## Environment

- `.env` — `OPENAI_API_KEY` (or UCSF Versa credentials)
- Python 3.12+, managed with `uv`

## Conventions

- Always use `uv run` for Python execution
- Test markers: `@pytest.mark.e2e`, `@pytest.mark.slow`
- CC BY-NC-SA 4.0 licensed
