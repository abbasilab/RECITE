# Changelog: vllm-merge session — 2026-04-06

## Summary
Added `api_type: vllm_endpoint` to the CLI benchmark evaluator, enabling all vLLM-served models to use the same pipeline as local `python_gpu` models. This eliminates the divergence with the standalone `scripts/rebuttal/rebuttal_eval_endpoint.py` script.

## Changes

### `recite/benchmark/evaluator.py`
- Added `asyncio` import
- Added `_strip_thinking_tags()` — strips `<think>` tags from Qwen3 responses
- Added `_vllm_endpoint_call()` — async HTTP call to OpenAI-compatible vLLM endpoint with:
  - Gemma handling (merge system prompt into user message)
  - Qwen3 handling (disable thinking via `chat_template_kwargs`)
  - Retry with exponential backoff (3 attempts)
  - 429/503 rate limit handling
  - Context-length error auto-truncation
- Added `_vllm_endpoint_predict_sync()` — synchronous wrapper for single-sample path
- Added `_vllm_endpoint_predict_batch()` — async concurrent batch predictions with semaphore throttling and checkpoint callbacks
- Added `vllm_endpoint` code path in `run_single_sample()` (lines ~1533-1583):
  - Same prompt construction as `python_gpu` (`_format_model_prompt`)
  - Same evidence budget formula: `max(0, ctx - 4096)` clamped by `min(budget, max(256, ctx - 2048 - 512))`
  - Tokenizer-based truncation with char-based fallback
  - `prompt_suffix` support
- Added `vllm_endpoint` code path in `run_benchmark()` (lines ~1901-1964):
  - Same evidence/prompt logic
  - Logs tokenizer status, evidence budget, concurrency settings
  - Sets `is_endpoint = False` (no RAG needed)

### `recite/cli/benchmark.py`
- Added `vllm_endpoint` dispatch in multi-model loop (between `python_gpu` and generic `endpoint`):
  - Server health check before run
  - Passes all vllm-specific config fields: `max_concurrent`, `save_every`, `max_tokens`, `timeout`, `prompt_suffix`

### Config files updated (`api_type: endpoint` -> `api_type: vllm_endpoint`)
- `config/rebuttal/rebuttal_gemma27b.yaml` — added `max_concurrent: 16`, `save_every: 50`
- `config/rebuttal/rebuttal_qwen32b.yaml` — added `max_concurrent: 16`, `save_every: 50`
- `config/rebuttal/rebuttal_llama70b.yaml` — added `max_concurrent: 16`, `save_every: 50`
- `config/rebuttal/rebuttal_qwen72b.yaml` — added `max_concurrent: 16`, `save_every: 50`
- `config/rebuttal/rebuttal_large_models.yaml` — added `max_concurrent: 16`, `save_every: 50`

## Not modified
- `scripts/rebuttal/rebuttal_eval_endpoint.py` — kept as reference (per task briefing)
- `config/rebuttal/rebuttal_scaling_models.yaml` — uses `python_gpu`, not vLLM endpoint
- `config/rebuttal/equiv_test_*.yaml` — test configs, left unchanged
- Existing `python_gpu`, `ucsf_versa`, and generic `endpoint` code paths — untouched

## Testing
- All 48 existing tests pass (`uv run python -m pytest -q`)
- All new functions importable from `recite.benchmark.evaluator`

## YAML config format for vllm_endpoint
```yaml
models:
  - id: my-model
    api_type: vllm_endpoint
    model: org/model-name
    endpoint: http://localhost:8200/v1
    context_window: 8192
    max_concurrent: 16      # async concurrency (default 16)
    save_every: 50           # checkpoint interval (default 50)
    max_tokens: 2048         # generation max tokens (default 2048)
    timeout: 300.0           # per-request timeout (default 300s)
    prompt_suffix: ""        # optional model-specific suffix
```
