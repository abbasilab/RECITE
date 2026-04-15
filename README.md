# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence**

Overly restrictive eligibility criteria contribute to widespread enrollment failure in clinical trials, yet published recommendations for broadening specific criteria remain disconnected from trial design at scale. RECITE automates the discovery, extraction, and implementation of evidence-based eligibility criteria revisions. It constructs an 11,913-instance dataset across 5,735 trials from ClinicalTrials.gov protocol amendments, benchmarks 8 LLM configurations (best: 85.8% binary equivalence with GPT-4o; best open-weight: 84.7% with Gemma 2 9B), and includes an agentic literature discovery system that identifies 44 paper-trial matches with estimated enrollment gains of 25-246%.

## Pipeline

```
1. Benchmark Construction   ->  EC diff detection across trial versions
2. Literature Crawl         ->  Agentic paper discovery + relevance scoring
3. Directive Extraction     ->  Extract modification directives from papers
4. Accrual Scoring          ->  Match papers to trials, estimate enrollment gains
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key (for LLM-as-judge evaluation and model inference)
- (Optional) GPU with CUDA support for local open-weight models

### Installation

```bash
git clone https://github.com/russro/RECITE.git
cd RECITE

# Install dependencies
uv sync

# With RAG support (optional, for retrieval-augmented generation)
uv sync --extra rag

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Tests

```bash
# Unit tests use synthetic fixtures - no API keys needed
uv run pytest -q
```

## Benchmark Data

The `data/benchmark_splits/benchmark.parquet` file contains the **3,116-sample evaluation set** used in the paper (2,492 train / 311 val / 313 test). Each row contains:

| Column | Description |
|--------|-------------|
| `id` | Unique instance ID |
| `instance_id` | ClinicalTrials.gov trial identifier |
| `source_version` | Source version number (0-indexed) |
| `target_version` | Target version number |
| `source_text` | Original eligibility criteria text |
| `evidence` | Protocol amendment document text (evidence for the revision) |
| `reference_text` | Ground-truth revised eligibility criteria |
| `quality_score` | Evidence extraction quality score |
| `split` | Dataset split (train/val/test) |

### Regenerating the Benchmark from ClinicalTrials.gov

To reconstruct the dataset from scratch (downloads amendment documents from ClinicalTrials.gov):

```bash
# Initialize database and discover trials with eligibility criteria amendments
uv run python init_recite_benchmark.py --workers 4 --num-chunks 10 --chunks 0,1,2,3,4,5,6,7,8,9

# Check pipeline status
uv run python init_recite_benchmark.py --status

# Export to parquet
uv run python recite.py export-splits --db-path data/dev/recite.db --output-dir data/benchmark_splits
```

Note: Full reconstruction requires downloading protocol PDFs from ClinicalTrials.gov and takes several hours depending on network speed and rate limits.

## Reproducing the Benchmark Evaluation

### Original Paper Models (single GPU, HuggingFace Transformers)

The 8 models in the paper (Qwen 0.5B–7B, Gemma 2B–9B, Mistral 7B, DeepSeek-R1 7B) each fit on a single GPU and use the `python_gpu` backend (HuggingFace Transformers `generate()`):

```bash
# Run all original models (requires 1x GPU with ≥32GB VRAM)
uv run python -m recite.cli.benchmark run-benchmark \
  --config config/benchmarks.yaml

# Quick smoke test (10 samples, 2 models)
uv run python -m recite.cli.benchmark run-benchmark \
  --config config/recite_quick.yaml
```

API models (GPT-4o, GPT-4o-mini) require a UCSF Versa or OpenAI API key in `.env`.

### Rebuttal Models (multi-GPU, vLLM)

Larger rebuttal models (Gemma-2 27B, Qwen2.5 72B, Llama-3.1 70B, Qwen3 32B) require multi-GPU inference via [vLLM](https://docs.vllm.ai/) tensor parallelism:

```bash
# 1. Install vLLM (requires CUDA 12.1+)
pip install vllm

# 2. Start a vLLM server (example: Gemma-2 27B on 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/gemma-2-27b-it \
  --tensor-parallel-size 4 --max-model-len 8192 \
  --dtype bfloat16 --port 8200

# 3. Run the benchmark against the vLLM endpoint
uv run python -m recite.cli.benchmark run-benchmark \
  --config config/rebuttal/rebuttal_gemma27b.yaml
```

Each rebuttal model has its own config with vLLM server instructions:

| Config | Model | GPUs | vLLM Port |
|--------|-------|------|-----------|
| `config/rebuttal/rebuttal_gemma27b.yaml` | Gemma-2 27B | 4 | 8200 |
| `config/rebuttal/rebuttal_qwen72b.yaml` | Qwen2.5 72B | 8 | 8300 |
| `config/rebuttal/rebuttal_llama70b.yaml` | Llama-3.1 70B | 4-8 | 8100 |
| `config/rebuttal/rebuttal_qwen32b.yaml` | Qwen3 32B | 4 | 8100 |

### View Results

```bash
# Generate markdown summary from prediction outputs
uv run python -m recite.cli.benchmark summarize \
  --predictions-dir data/benchmark_predictions
```

### Reproduction Validation

Tools in `reproduce/` verify that reproduced results match original outputs:

```bash
# Run a small reproduction check (10 samples, 2 models)
uv run python -m recite.cli.benchmark run-benchmark \
  --config reproduce/reproduce_originals.yaml --num-samples 10

# Compare reproduced predictions against original database
uv run python reproduce/compare_predictions.py \
  --original-db data/results/benchmark_results_cluster1.db \
  --predictions-dir data/benchmark_predictions
```

## LLM-as-Judge Evaluation

The judge prompt template is in `config/benchmark_prompts.json` under the `judge_prompt` key. The judge evaluates each prediction against the ground truth on two scales:

- **Binary score** (0 or 1): Is the prediction correct?
- **Ordinal score** (0-4): Quality of match
  - 0 = No match (unrelated or contradicts target)
  - 1 = Poor match (minimal overlap, major errors)
  - 2 = Partial match (some key elements correct, notable gaps)
  - 3 = Good match (most elements correct, minor differences)
  - 4 = Excellent match (essentially identical)

A batched variant (`judge_prompt_batched`) scores multiple pairs per API call for efficiency.

## Model Configurations

Models evaluated in the paper are defined in `config/benchmarks.yaml`:

| Model ID | Type | Model |
|----------|------|-------|
| `versa-4o` | API | GPT-4o (2024-08-06) |
| `versa-4o-mini` | API | GPT-4o-mini (2024-07-18) |
| `local-qwen-0.5b` | Local GPU | Qwen2.5-0.5B-Instruct |
| `local-qwen-3b` | Local GPU | Qwen2.5-3B-Instruct |
| `local-qwen-7b` | Local GPU | Qwen2.5-7B-Instruct |
| `local-gemma2-2b` | Local GPU | Gemma-2-2B-IT |
| `local-gemma2-9b` | Local GPU | Gemma-2-9B-IT |
| `local-longctx-7b` | Local GPU | DeepSeek-R1-Distill-Qwen-7B |

Additional model presets for vLLM serving are in `config/model_presets.yaml`.

## Project Structure

```
recite/                      # Core package
  benchmark/                 # EC diff, triplet building, evaluation
  crawler/                   # Literature crawl (PubMed, Semantic Scholar)
  accrual/                   # Directive extraction, impact scoring
  rag/                       # RAG-based evidence retrieval (optional)
  cli/                       # Typer CLI (benchmark, crawl)
  utils/                     # Logging, path resolution
  llmapis.py                 # LLM API wrappers
config/                      # YAML/JSON configs and prompt templates
  benchmarks.yaml            # Full benchmark config (all models)
  benchmark_prompts.json     # Prompt templates including LLM judge
  model_presets.yaml         # vLLM model serving presets
data/
  benchmark_splits/          # Parquet evaluation data (3,116 instances)
tests/fixtures/              # Synthetic trial, protocol, paper data
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `recite.py` | Benchmark runner: `run`, `ready`, `export-splits`, `benchmark-summary` |
| `init_recite_benchmark.py` | Benchmark construction from ClinicalTrials.gov |
| `accrual.py` | Accrual impact scoring pipeline |
| CLI: `recite crawl` | Literature discovery and relevance scoring |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls and judge |
| `LOCAL_DB_DIR` | No | Database directory (default: `data/dev`) |
| `HF_HOME` | No | HuggingFace model cache directory |
| `UCSF_API_KEY` | No | UCSF Versa institutional API key |
| `UCSF_API_VER` | No | Azure API version for UCSF Versa |
| `UCSF_RESOURCE_ENDPOINT` | No | UCSF Versa endpoint URL |

## License

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

```bibtex
@article{ro2026recite,
  title  = {{RECITE}: Revising Eligibility Criteria Incorporating Textual Evidence},
  author = {Ro, Russell and Abbasi-Asl, Reza},
  year   = {2026}
}
```
