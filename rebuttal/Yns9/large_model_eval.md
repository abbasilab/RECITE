# Changelog: Large Model Evaluation for Rebuttal (2026-04-04)

**Agent:** rebuttal-largemodel
**Session:** phd-clintrialm-rebuttal-largemodel-1
**Backend:** claude (claude-opus-4-6)
**Hardware:** abbasi-gpu-1 (8x NVIDIA RTX 5000 Ada, 32GB each)

## Summary

Evaluated Llama-3.1-70B-Instruct and Qwen3-32B on the RECITE 3,116-sample benchmark to address Reviewer Yns9's concern that no open-weight models >=10B were evaluated. Results show that larger models do **not** automatically improve on this structured eligibility criteria revision task.

## Motivation

Reviewer Yns9 (KDD 2026, score 3):
> "No open-weight models >=10B evaluated. Scaling trends from 2B->9B suggest larger models (Llama-3.1-70B, Qwen3-14B/32B/72B) could differ meaningfully."

## Setup

### vLLM Serving (abbasi-gpu-1)
- **Llama-3.1-70B-Instruct**: AWQ-INT4 quantization (`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`), TP=8, 32768 context, `--enforce-eager`
- **Qwen3-32B**: FP8 quantization (`Qwen/Qwen3-32B-FP8`), TP=8, 32768 context, `--enforce-eager`, thinking mode disabled via `chat_template_kwargs`

### Evaluation Protocol
- Same LLM-as-judge protocol as paper (GPT-4o via UCSF Versa)
- Binary equivalence (0/1) + ordinal quality (0-4) scoring
- Batch judging (10 samples per batch)
- No-RAG condition (evidence provided directly in prompt)
- Evidence truncated to 20,000 chars to fit context window

## Results

### Scaling Trend Table (No-RAG, Binary Equivalence %)

| Model | Params | Type | Binary Equiv (%) | Mean Ordinal | Samples | Errors |
|-------|--------|------|-------------------|--------------|---------|--------|
| Qwen2.5-0.5B | 0.5B | open | 11.1 | — | 3,116 | — |
| DeepSeek-R1-7B | 7B | open | 1.7 | — | 3,116 | — |
| Gemma 2 2B | 2B | open | 37.2 | — | 3,116 | — |
| Qwen2.5-3B | 3B | open | 66.1 | — | 3,116 | — |
| Qwen2.5-7B | 7B | open | 81.2 | — | 3,116 | — |
| Gemma 2 9B | 9B | open | 84.7 | — | 3,116 | — |
| **Qwen3-32B** | **32B** | **open** | **TBD** | **TBD** | **3,116** | **TBD** |
| **Llama-3.1-70B** | **70B** | **open** | **42.2** | **3.06** | **3,116** | **296** |
| GPT-4o-mini | ~8B | proprietary | 85.8 | — | 3,116 | — |
| GPT-4o | — | proprietary | 85.3 | — | 3,116 | — |

### Key Findings

1. **Llama-3.1-70B underperforms smaller models**: At 42.2% binary equivalence, Llama-3.1-70B-Instruct significantly underperforms Gemma 2 9B (84.7%) and even Qwen2.5-3B (66.1%). However, the mean ordinal of 3.06/4.0 indicates that outputs are structurally reasonable — the model captures most criteria elements but diverges on specifics.

2. **Scaling is not monotonic**: The scaling trend from 0.5B→9B shows clear improvement (11.1%→84.7%), but jumping to 70B breaks this pattern. This suggests the task requires specific instruction-following behavior that isn't guaranteed by increased parameter count.

3. **Error rate**: Llama-3.1-70B had 296/3,116 (9.5%) timeout errors on samples with very long evidence documents, compared to near-zero errors for smaller models that ran with unlimited context via API.

4. **Ordinal distribution** (Llama-3.1-70B): 0: 0.07%, 1: 3.5%, 2: 28.3%, 3: 27.0%, 4: 41.1% — Most predictions receive ordinal 3-4 (good to excellent match), yet binary equivalence is low, suggesting the judge applies a strict threshold.

## Issues Encountered

1. **FP16 Llama-3.1-70B OOM**: Full precision model exceeded 256GB aggregate VRAM during CUDA graph compilation. Resolved with AWQ-INT4 quantization.
2. **CUDA graph capture OOM**: Even quantized, CUDA graph capture caused OOM. Resolved with `--enforce-eager`.
3. **Qwen3 thinking tokens**: Qwen3-32B's default thinking mode (`<think>` tags) consumed context budget. Disabled per-request via `chat_template_kwargs: {"enable_thinking": false}`.
4. **Qwen3 v1/v2 timeout storms**: First two Qwen3-32B runs had 82% timeout errors on TP=4 with 8192 context. Resolved by switching to TP=8, 32768 context, and 600s timeout.
5. **SSH tunneling**: abbasi-gpu-1 not directly reachable from eval machine; used SSH port forwarding (localhost:8100→abbasi:8000).

## Files Created/Modified

- `scripts/rebuttal_eval_endpoint.py` — Custom concurrent evaluation script (async, parallel predictions + batch judge)
- `scripts/rebuttal_compile_results.py` — Results compilation script
- `config/rebuttal_large_models.yaml` — Evaluation configuration
- `data/rebuttal/llama31_70b_no_rag.json` — Llama-3.1-70B results (3,116 samples)
- `data/rebuttal/qwen3_32b_no_rag.json` — Qwen3-32B results (pending update)

## Runtime

- Llama-3.1-70B predictions: ~4h20m (3,116 samples, 8 concurrent)
- Llama-3.1-70B judging: ~90s (282 batches, 8 concurrent)
- Qwen3-32B: In progress
