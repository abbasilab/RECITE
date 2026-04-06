# RECITE KDD 2026 Rebuttal Materials

**Submission:** #736, KDD 2026 AI4Sciences Track
**Rebuttal window:** April 11-17, 2026
**Scores:** DVUB=2, ECp3=3, Yns9=3, aiEs=4 (avg 3.0, borderline)

## Directory Structure

```
rebuttal/
  README.md                              # This file
  REBUTTAL_PLAN.md                       # Full rebuttal strategy and review analysis
  DAG_PLAN.md                            # Agent DAG plan for rebuttal work
  general_gap_analysis.md                # Cross-reviewer gap analysis
  response_drafts_ECp3_Yns9_aiEs.md      # Draft responses for 3 reviewers

  paper/
    RECITE_KDD2026.pdf                   # Submitted paper

  DVUB/                                  # Reviewer DVUB (score 2 — MUST flip to 3)
    review.md                            # Full review with scores and upgrade conditions
    W1_truncation_analysis.md            # Truncation/coverage stats (addresses W1)
    W3_c2c6_analysis.md                  # C2/C6 logical inconsistency analysis (W3)
    critic_review_1.md                   # First DVUB-perspective stress test
    critic_review_2.md                   # Second DVUB-perspective stress test

  ECp3/                                  # Reviewer ECp3 (score 3 — maintain/bump)
    review.md                            # Full review

  Yns9/                                  # Reviewer Yns9 (score 3 — maintain/bump)
    review.md                            # Full review
    judge_prompt_protocol.md             # LLM-as-judge prompt extraction (W3)
    large_model_eval.md                  # Large model (70B, 32B) results (W4)

  aiEs/                                  # Reviewer aiEs (score 4 — reinforce)
    review.md                            # Full review

  response/                              # Final response letters (to be drafted)
```

## Status

| Deliverable | Status |
|-------------|--------|
| Paper PDF in repo | Done |
| Reviewer comments extracted | Done |
| DVUB W1 (truncation/coverage) | Done - full analysis with length-controlled re-analysis |
| DVUB W2 (accrual framing) | Draft ready - needs corrected numbers |
| DVUB W3 (C2/C6 examples) | Done - root cause identified, replacement examples verified |
| ECp3 response draft | Done |
| Yns9 response draft | Done |
| aiEs response draft | Done |
| Large model results (Gemma-27B, Qwen-72B) | Complete with judge scores (46.3% / 50.5% binary) |
| Large model results (Llama-70B, Qwen3-32B) | Complete with judge scores |
| Final response letters | Not started |

## Judge Prompt

The complete LLM-as-judge evaluation prompt is at `config/benchmark_prompts.json` in this repo. See `Yns9/judge_prompt_protocol.md` for the full protocol documentation.

## Rebuttal Evaluation Results

Results from large model evaluations (addressing Yns9 W4) are in `data/rebuttal/`:
- `gemma2-27b_no_rag.json` — complete with judge scores (46.3% binary, 2.964 ordinal)
- `qwen25-72b_no_rag.json` — complete with judge scores (50.5% binary, 3.216 ordinal)
- `llama31_70b_no_rag.json` — complete with judge scores (42.2% binary)
- `qwen3_32b_no_rag.json` — complete with judge scores (43.2% binary)
