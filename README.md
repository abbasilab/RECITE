# RECITE

**Revising Eligibility Criteria Incorporating Textual Evidence**

A benchmark for evaluating LLMs on eligibility criteria revision using protocol amendment evidence from ClinicalTrials.gov. 3,116 instances across 5,735 trials, with LLM-as-judge evaluation.

## Benchmark Results

| Model | Binary Equiv. | Ordinal (0-4) | ≥3 Rate | ≥4 Rate |
|-------|:---:|:---:|:---:|:---:|
| GPT-4o | 85.8% | 3.4±0.7 | 91.3% | 49.2% |
| GPT-4o-mini | 84.2% | 3.4±0.7 | 90.1% | 52.3% |
| Qwen2.5-72B | 82.1% | 3.3±0.7 | 90.3% | 44.9% |
| Gemma-2-9B | 84.7% | 3.3±0.7 | 86.9% | 45.0% |
| Qwen2.5-7B | 80.3% | 3.3±0.8 | 81.4% | 46.8% |
| Qwen3-32B | 81.5% | 3.2±0.7 | 90.1% | 34.4% |
| Llama-3.1-70B | 76.2% | 3.2±0.7 | 86.2% | 31.5% |
| Gemma-2-27B | 83.2% | 3.3±0.7 | 89.5% | 43.1% |
| Gemma-2-2B | 78.6% | 3.1±0.8 | 78.2% | 38.4% |
| Qwen2.5-3B | 79.1% | 3.2±0.8 | 80.5% | 40.2% |
| Qwen2.5-0.5B | 68.3% | 2.8±0.9 | 65.4% | 28.1% |
| DeepSeek-R1-7B | 79.8% | 3.2±0.8 | 82.1% | 41.3% |
| Mistral-7B | 77.4% | 3.1±0.8 | 79.8% | 37.6% |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI-compatible API key (for LLM inference and judge evaluation)
- (Optional) GPU with CUDA support for local open-weight models

### Installation

```bash
git clone https://github.com/russro/RECITE.git
cd RECITE
uv sync
cp .env.example .env
# Edit .env and add your API key
```

### Run Tests

```bash
# Unit tests use synthetic fixtures — no API keys needed
uv run pytest -q
```

## Loading the Benchmark

The benchmark dataset is at `data/benchmark_splits/benchmark.parquet` (3,116 samples: 2,492 train / 311 val / 313 test).

| Column | Description |
|--------|-------------|
| `id` | Unique instance ID |
| `instance_id` | ClinicalTrials.gov trial identifier |
| `source_text` | Original eligibility criteria text |
| `evidence` | Protocol amendment document text |
| `reference_text` | Ground-truth revised eligibility criteria |
| `split` | Dataset split (train/val/test) |

### Regenerating from ClinicalTrials.gov

```bash
uv run python init_recite_benchmark.py --workers 4 --num-chunks 10 --chunks 0,1,2,3,4,5,6,7,8,9
uv run python init_recite_benchmark.py --status
uv run python recite.py export-splits --db-path data/dev/recite.db --output-dir data/benchmark_splits
```

## Running the Benchmark

### Single-GPU Models (HuggingFace Transformers)

```bash
# All original paper models (requires 1x GPU ≥32GB)
uv run python -m recite.cli.benchmark run-benchmark --config config/benchmarks.yaml

# Quick smoke test (10 samples, 2 models)
uv run python -m recite.cli.benchmark run-benchmark --config config/recite_quick.yaml
```

### Multi-GPU Models (vLLM)

Larger models require [vLLM](https://docs.vllm.ai/) tensor parallelism:

```bash
# Start vLLM server (example: Gemma-2 27B on 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/gemma-2-27b-it \
  --tensor-parallel-size 4 --max-model-len 8192 --dtype bfloat16 --port 8200

# Run benchmark against vLLM endpoint
uv run python -m recite.cli.benchmark run-benchmark --config config/rebuttal/rebuttal_gemma27b.yaml
```

### View Results

```bash
uv run python -m recite.cli.benchmark summarize --predictions-dir data/benchmark_predictions
```

## LLM-as-Judge Evaluation

Judge prompt template: `config/benchmark_prompts.json` (`judge_prompt` key).

- **Binary score** (0/1): Is the prediction correct?
- **Ordinal score** (0-4): Quality of match (0=no match, 4=excellent)

## Fine-tuning Benchmarks

QLoRA fine-tuned models (0.5B, 3B) are included in the benchmark. Fine-tuning configs and scripts are in `config/finetune/`.

## Project Structure

```
recite/                      # Core package
  benchmark/                 # EC diff, triplet building, evaluation
  crawler/                   # Literature crawl (PubMed, Semantic Scholar)
  accrual/                   # Directive extraction, impact scoring
  rag/                       # RAG-based evidence retrieval (optional)
  cli/                       # CLI (benchmark, crawl)
config/                      # YAML/JSON configs and prompt templates
data/benchmark_splits/       # Parquet evaluation data (3,116 instances)
tests/                       # Tests (synthetic fixtures, no API deps)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls and judge |
| `AZURE_API_KEY` | No | Azure OpenAI API key |
| `AZURE_API_VER` | No | Azure API version |
| `AZURE_RESOURCE_ENDPOINT` | No | Azure endpoint URL |
| `LOCAL_DB_DIR` | No | Database directory (default: `data/dev`) |
| `HF_HOME` | No | HuggingFace model cache directory |

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

```bibtex
@article{ro2026recite,
  title  = {{RECITE}: Revising Eligibility Criteria Incorporating Textual Evidence},
  author = {Ro, Russell and Abbasi-Asl, Reza},
  year   = {2026}
}
```
