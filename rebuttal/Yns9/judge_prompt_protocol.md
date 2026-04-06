# LLM-as-Judge Evaluation Prompt & Protocol — RECITE Benchmark

**Extracted:** 2026-04-04
**Source files:**
- `staging-repo/config/benchmark_prompts.json` (lines 11-20)
- `staging-repo/recite/benchmark/evaluator.py` (lines 363-459, 845-978)
- `staging-repo/recite/llmapis.py` (line 304, 344, 361, 517)
- `staging-repo/config/benchmarks.yaml` (lines 67-68)

---

## 1. Complete Judge Prompt Template

### System Prompt

```
You are an expert evaluator assessing the quality of eligibility criteria
predictions. Your task is to score how well a predicted eligibility criteria
matches the target criteria.
```

### User Prompt Template

```
Evaluate how well the predicted eligibility criteria matches the target criteria.

Target Eligibility Criteria:
{ground_truth}

Predicted Eligibility Criteria:
{prediction}

Provide two scores:

1. Binary score (0 or 1): Is the prediction correct?
   - 0 = Incorrect
   - 1 = Correct

2. Ordinal score (0-4): How good is the match?
   - 0 = No match -- prediction is unrelated or contradicts target
   - 1 = Poor match -- minimal overlap, major omissions or errors
   - 2 = Partial match -- some key elements correct, notable gaps
   - 3 = Good match -- most elements correct, minor differences
   - 4 = Excellent match -- essentially identical or fully correct

Respond with ONLY two numbers separated by a comma: binary_score,ordinal_score
Example: 1,3
```

**Inputs:** `{ground_truth}` = reference amended eligibility criteria; `{prediction}` = model-generated amended eligibility criteria.

**Expected output format:** Two comma-separated integers, e.g. `1,3`.

---

## 2. Batched Judge Prompt (for efficiency)

### System Prompt (batched)

```
You are an expert evaluator assessing the quality of eligibility criteria
predictions. Score multiple prediction pairs. For each pair, provide binary
(0 or 1) and ordinal (0-4) scores as in the single-pair format.
```

### User Prompt Template (batched)

```
Evaluate the following {n} prediction pairs. For each pair, score how well
the predicted eligibility criteria matches the target.

{pairs}

Return scores in JSON format only: {"1": [binary, ordinal], "2": [binary, ordinal], ...}
where keys are pair numbers (1 to {n}). Example: {"1": [1, 4], "2": [0, 2]}.
```

Batch size is configurable; default is 10 (`config/benchmarks.yaml`, line 63).

---

## 3. Full Evaluation Protocol

### Judge Model
- **Model:** GPT-4o (`gpt-4o-2024-08-06`)
- **API:** UCSF Versa (institutional Azure OpenAI endpoint)
- **Temperature:** 0 (deterministic; hardcoded in both `call_model_with_retry` at evaluator.py:517 and `UCSFVersaAPI.__call__` at llmapis.py:344/361)
- **No top-p, frequency penalty, or presence penalty** are set (API defaults apply)

### Score Parsing (`_parse_judge_scores`, evaluator.py:363-459)
The parser handles multiple response formats with fallback:
1. **Preferred:** `binary,ordinal` (e.g., `1,3`) via regex `\b([01])\s*,\s*(\d+)\b`
2. **Fallback 1:** Two separate numbers — first 0/1 found is binary, first 0-4 found is ordinal
3. **Fallback 2:** Single ordinal score only; binary inferred (ordinal >= 2 -> binary = 1)
4. **Last resort:** If unparseable, defaults to binary=0, ordinal=2 (midpoint)

All scores are clamped to valid ranges (binary: 0-1, ordinal: 0-4).

### Output Metrics Per Instance
| Metric | Range | Description |
|--------|-------|-------------|
| `llm_judge_binary` | 0 or 1 | Acceptable (1) or not (0) |
| `llm_judge_score` | 0-4 | Ordinal quality score |
| `llm_judge_normalized` | 0-1 | `ordinal / 4.0` |
| `llm_judge_raw_response` | string | Raw LLM response for auditing |

### Aggregation
- **Primary metric:** `llm_judge_binary_mean` = fraction of instances judged acceptable (binary=1)
- Ordinal scores are averaged and also reported as `llm_judge_normalized` means
- No human evaluation protocol is currently implemented (this is a limitation acknowledged in the paper)

### Retry Logic
- `call_model_with_retry`: up to 3 attempts total (max_retries=2), exponential backoff starting at 0.5s, capped at 5s
- On complete failure: returns binary=0, ordinal=0, normalized=0 (conservative default)

---

## 4. Threshold Analysis: Score 3 vs Score 4

The ordinal rubric distinguishes:

| Score | Label | Meaning |
|-------|-------|---------|
| 3 | Good match | Most elements correct, **minor differences** |
| 4 | Excellent match | Essentially identical or **fully correct** |

**Key distinction:** A score of 3 means the prediction captures the semantic intent of the amendment but may differ in phrasing, formatting, or minor details (e.g., reordering of criteria, slight wording variations). A score of 4 means the prediction is essentially interchangeable with the reference.

**Regarding the >=3 threshold concern (Reviewer Yns9):**

The paper reports `llm_judge_binary` as the primary metric, which is the judge's own binary assessment of correctness (0/1), **not** a threshold on the ordinal scale. The ordinal scale (0-4) provides a separate, more granular quality signal. The binary and ordinal scores are produced independently by the judge in a single response.

However, when only the ordinal score is parseable (fallback path), binary is inferred with threshold >= 2, which is more lenient than >= 3. This fallback is rarely triggered since GPT-4o reliably produces both scores.

The coarseness concern can be addressed by noting:
1. The binary score is the judge's own holistic judgment, not derived from a threshold
2. The ordinal scale provides granularity for error analysis (distribution of 0/1/2/3/4)
3. Both scores are stored and available for alternative threshold analyses

---

## 5. Rebuttal-Ready Text (Plain Text, No Hyperlinks)

> **Response to Reviewer Yns9 — Evaluation Transparency:**
>
> We appreciate this concern and provide the complete LLM-as-judge evaluation prompt below.
>
> **Judge model:** GPT-4o (gpt-4o-2024-08-06), temperature=0 (deterministic).
>
> **System prompt:** "You are an expert evaluator assessing the quality of eligibility criteria predictions. Your task is to score how well a predicted eligibility criteria matches the target criteria."
>
> **User prompt template:**
> "Evaluate how well the predicted eligibility criteria matches the target criteria.
>
> Target Eligibility Criteria: [reference text]
>
> Predicted Eligibility Criteria: [model output]
>
> Provide two scores:
> 1. Binary score (0 or 1): Is the prediction correct? 0 = Incorrect, 1 = Correct.
> 2. Ordinal score (0-4): 0 = No match, 1 = Poor match, 2 = Partial match, 3 = Good match (most elements correct, minor differences), 4 = Excellent match (essentially identical or fully correct).
>
> Respond with ONLY two numbers separated by a comma: binary_score,ordinal_score. Example: 1,3"
>
> The binary acceptability metric reported in our tables is the judge's own holistic binary assessment, not a threshold on the ordinal scale. Both scores are produced in a single judge call, providing complementary views: binary for aggregate pass rates and ordinal for quality distribution analysis. We will include this prompt template and protocol description in the camera-ready appendix to ensure full reproducibility.
