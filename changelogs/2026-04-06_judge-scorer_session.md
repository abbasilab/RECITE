# Judge-Scorer Session — 2026-04-06

**Agent:** judge-scorer
**Session:** phd-RECITE-judge-scorer-1
**Backend:** claude (claude-opus-4-6)
**Status:** completed

## Summary

Ran LLM-as-judge scoring (GPT-4o via UCSF Versa) on two rebuttal prediction files: Gemma-27B and Qwen-72B (no RAG). Both files had 3,116 predictions with no prior judge scores.

## Results

| Model | Binary Equivalence | Mean Ordinal (/4.0) | Samples | Time |
|-------|-------------------|---------------------|---------|------|
| Gemma 2 27B (no RAG) | 46.3% | 2.964 | 3,116 | 107.4s |
| Qwen 2.5 72B (no RAG) | 50.5% | 3.216 | 3,116 | 94.8s |

## Bug Fix

- Fixed `scripts/rebuttal/judge_only.py` line 22: `ROOT` path resolution was wrong (`parent.parent` only reached `scripts/`, not project root). Changed to `parent.parent.parent`.

## Files Modified

- `scripts/rebuttal/judge_only.py` — fixed ROOT path (line 22)
- `data/rebuttal/gemma2-27b_no_rag.json` — added judge_binary, judge_ordinal, judge_raw fields + summary stats
- `data/rebuttal/qwen25-72b_no_rag.json` — added judge_binary, judge_ordinal, judge_raw fields + summary stats

## Cost

Estimated ~$16-17 total for both models (312 batches x 2 models, GPT-4o rates). Well within $50 budget.
