# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence**

Overly restrictive eligibility criteria contribute to widespread enrollment failure in clinical trials, yet published recommendations for broadening specific criteria remain disconnected from trial design at scale. RECITE automates the discovery, extraction, and implementation of evidence-based eligibility criteria revisions. It constructs an 11,913-instance dataset across 5,735 trials from ClinicalTrials.gov protocol amendments, benchmarks 8 LLM configurations (best: 85.8% binary equivalence with GPT-4o-mini; best open-weight: 84.7% with Gemma 2 9B), and includes an agentic literature discovery system that identifies 44 paper–trial matches with estimated enrollment gains of 25–246%.

## Pipeline

```
1. Benchmark Construction   →  EC diff detection across trial versions
2. Literature Crawl         →  Agentic paper discovery + relevance scoring
3. Directive Extraction     →  Extract modification directives from papers
4. Accrual Scoring          →  Match papers to trials, estimate enrollment gains
```

## Quick Start

```bash
# Install
uv sync

# With RAG support (optional)
uv sync --extra rag

# Configure
cp .env.example .env   # then add your OPENAI_API_KEY

# Run pipeline steps
uv run python init_recite_benchmark.py --help   # Step 1: benchmark construction
uv run recite crawl --help                      # Step 2: literature crawl
uv run python accrual.py --help                 # Step 3: accrual scoring

# Tests (synthetic fixtures, no API keys needed)
uv run pytest -q
```

## Structure

```
recite/                      # Core package
├── benchmark/               # EC diff, triplet building, evaluation
├── crawler/                 # Literature crawl (PubMed, Semantic Scholar)
├── accrual/                 # Directive extraction, impact scoring
├── rag/                     # RAG-based evidence retrieval (optional)
├── cli/                     # Typer CLI (benchmark, crawl)
├── utils/                   # Logging, path resolution
└── llmapis.py               # LLM API wrappers
config/                      # YAML/JSON configs and prompt templates
tests/fixtures/              # Synthetic trial, protocol, paper data
```

## Citation

```bibtex
@article{ro2026recite,
  title  = {{RECITE}: Revising Eligibility Criteria Incorporating Textual Evidence},
  author = {Ro, Russell and Abbasi-Asl, Reza},
  year   = {2026}
}
```
