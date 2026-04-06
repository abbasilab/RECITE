# RECITE KDD 2026 AI4Sciences Rebuttal Plan

## Paper Info
- **Title:** RECITE: Revising Eligibility Criteria Incorporating Textual Evidence
- **Authors:** Russell Ro, Reza Abbasi-Asl
- **Track:** KDD 2026 AI4Sciences Track February Submission
- **Submission Number:** 736
- **Type:** 8-page full paper (Student Paper)
- **Keywords:** Clinical trials, Eligibility criteria, Large language models, Retrieval-augmented generation

## Abstract
Overly restrictive eligibility criteria contribute to widespread enrollment failure in clinical trials, yet systematic methods for identifying and applying clinical expert recommendations on broadening such criteria remain lacking. We introduce RECITE (Revising Eligibility Criteria Incorporating Textual Evidence), a framework for automated discovery, extraction, and implementation of clinical evidence-based eligibility criteria revisions. The framework utilizes an agentic literature discovery system that crawls indexed biomedical literature (PubMed and Semantic Scholar) to identify expert-authored recommendations, matches them to clinical trials with amendable criteria, and extracts actionable modification directives. To benchmark approaches to this task, a dataset is constructed from ClinicalTrials.gov protocol amendment documents containing 11,913 curated instances across 5,735 trials, each pairing original criteria with documentary evidence and target revisions. Eight LLM configurations, with and without retrieval-augmented generation, are evaluated on a 3,116-instance sample under an LLM-as-judge protocol. The best open-weight model, Gemma 2 9B, achieves 84.7% binary equivalence (normalized score 0.774), while the best closed-weight model, GPT-4o-mini via UCSF Versa, reaches 85.8%. Applied to real-world literature and clinical trials, the system identified 44 paper-trial matches across 6 domain expert publications with estimated accrual gains of 25-246%, demonstrating that existing clinical knowledge can be operationalized into actionable trial design recommendations at scale.

---

## Review Scores Summary

| Reviewer | Overall Rec | Scientific Contribution | Usability Contribution | Confidence | Oral? |
|----------|------------|------------------------|----------------------|------------|-------|
| DVUB     | **2** (Reject) | 3 (Top 50%) | 2 (Bottom 50%) | 4 (High) | No |
| ECp3     | **3** (Weak Accept) | 3 (Top 50%) | 4 (Top 10%) | 3 (Moderate) | No |
| Yns9     | **3** (Weak Accept) | 4 (Top 10%) | 2 (Bottom 50%) | 4 (High) | Yes |
| aiEs     | **4** (Accept) | 3 (Top 50%) | 3 (Top 50%) | 4 (High) | Yes |

**Average Overall Recommendation: 3.0 (borderline)**

### Scoring Rubric Reference
- 5: Strong Accept / Top 1%
- 4: Accept / Top 10%
- 3: Weak Accept / Top 50%
- 2: Reject / Bottom 50%
- 1: Strong Reject / Bottom 25%

---

## Criterion-Level Scores (0–3 scale)

| Criterion | DVUB | ECp3 | Yns9 | aiEs |
|-----------|------|------|------|------|
| 1-1: Novel data clearly described | 3 | 3 | 2 | 2 |
| 1-2: Novel/under-explored problem | 2 | 3 | 3 | 2 |
| 1-3: Novel insights/findings | 2 | 3 | 2 | 2 |
| 2-1: Scientific rigor | 1 | 3 | 2 | 2 |
| 2-2: AI methods innovative/appropriate | 2 | 2 | 2 | 2 |
| 2-3: Comprehensive analysis | 2 | 3 | 2 | 2 |
| 3-1: Domain expert co-authors | 2 | 2 | 2 | 2 |
| 3-2: Discusses AI challenges in domain | 2 | 3 | 2 | 2 |
| 3-3: Key contributions to domain | 2 | 3 | 2 | 2 |
| 4-1: Limitations & Ethics section | 1 | 1 | 1 | 1 |
| 4-2: (Second ethics criterion) | 1 | 1 | 1 | 1 |

**Key observation:** DVUB gave a 1 (Disagree) on scientific rigor (Criterion 2-1) — this is the most critical score to address.

---

## Detailed Review Breakdown

### Reviewer DVUB — Score: 2 (Reject) — HIGH PRIORITY
**Confidence: 4 (High)**

**Strengths cited:**
- S1: Important and practical problem; clinically meaningful and timely
- S2: Strong dataset contribution; the triplet design fills a major gap in medical NLP
- S3: Provenance emphasis is appropriate for high-stakes clinical use

**Weaknesses:**
- **W1: Full-document vs. RAG comparison is uninterpretable.** Average evidence length is 134,180 characters (~30K+ tokens). Many models have 8K/32K context limits. What does "full-document" actually mean? Requests: exact preprocessing/inference pipeline, truncation/coverage statistics for all 3,116 samples, length-controlled re-analysis.
- **W2: Accrual gains (25–246%) are likely overstated.** The scalar multiplication approach doesn't account for trial-specific interactions. Requests: reframe as hypothesis-generating projections, add uncertainty bounds/sensitivity analysis, validate on retrospective data if possible.
- **W3: Representative examples C2 and C6 have logical inconsistencies.** C2 tightens ECOG (≤2 → ≤1) yet reports +233% gain. C6's trial already uses ECOG ≤1, making the change non-actionable, yet +246% gain is propagated. Requests: separate patient-level matching from trial-level accrual metrics, enforce directionality checks, revise/remove these examples.

**Explicit upgrade conditions stated:**
> "I am open to upgrading my recommendation at rebuttal if the authors (1) clarify the exact full-document preprocessing pipeline with truncation/coverage statistics, (2) correct the accrual-impact framing by separating patient-level matching from trial-level enrollment, and (3) revise or remove logically inconsistent representative examples (e.g., C2/C6) with a corrected analysis."

---

### Reviewer ECp3 — Score: 3 (Weak Accept)
**Confidence: 3 (Moderate)**

**Strengths cited:**
- Targets pain-point of clinical trial eligibility criteria
- Defines a novel medical question with curated benchmark dataset
- Demonstrates significant real-world impact

**Weaknesses:**
- **No training/finetuning explored.** For real-world use, finetuning SOTA models could maximize performance.
- **No ablation studies** for alternative framework designs.
- **Data leakage concern.** Current LLMs may have been pretrained on data overlapping with the benchmark, leading to unfair evaluation.

**Suggestion:** Splitting dataset and doing some finetuning would strengthen the work's impact.

---

### Reviewer Yns9 — Score: 3 (Weak Accept)
**Confidence: 4 (High)**

**Strengths cited:**
- Clinical trial accrual failure is a genuine crisis; RECITE operationalizes expert knowledge
- Clean, scalable benchmark construction pipeline
- 44 concrete matches with peer-reviewed accrual estimates show deployability

**Weaknesses:**
- **Only oncology** — limits generalizability (acknowledged as good starting point)
- **Accrual gains inherited, not prospectively measured**
- **Evaluation transparency:** LLM-as-judge prompt template not included; no human evaluation protocol for clinicians to replicate

**Detailed concerns:**
- No open-weight models ≥10B evaluated. Scaling trends from 2B→9B suggest larger models (Llama-3.1-70B, Qwen3-14B/32B/72B) could differ meaningfully.
- Binary threshold (≥3 = "largely equivalent") is coarse; clinically meaningful differences could hide in the 3 vs. 4 bucket.
- No comparison to human expert judgments or baseline systems (rule-based extraction, non-agentic LLMs).
- **The repo contains no actual data or instructions to run the code.**

---

### Reviewer aiEs — Score: 4 (Accept) — CHAMPION REVIEWER
**Confidence: 4 (High)**

**Strengths cited:**
- High clinical relevance (80% of trials fail enrollment timelines)
- Valuable benchmark without manual annotation
- End-to-end system design with strong real-world utility
- Smaller open-source models competitive (Gemma 2 9B within 2 points of best commercial)

**Weaknesses (mild):**
- **No human validation** at scale; LLM-as-judge confidence is limited
- **RAG provides negligible improvement** over full-context for capable models — questions whether RAG is needed in the pipeline

---

## Strategic Assessment

### Acceptance Likelihood
- Average score of 3.0 is borderline. KDD 2025 accepted papers had a mean reviewer average around 3.57, with the minimum around 3.1.
- You need at least one score bump (ideally DVUB 2→3) to be competitive.
- KDD 2025 Research Track acceptance rate was ~19% overall, ~43% for resubmissions.
- AI4Sciences is a brand new track — no historical data, which could mean either more or less selective.
- Even if not accepted, a "Resubmit" decision is a strong outcome — resubmissions have much higher acceptance rates.

### Key Timeline
- **Author rebuttals:** April 11–17, 2026
- **Reviewer engagement with rebuttals:** April 11–17, 2026
- **Reviewer-AC discussion:** April 18–25, 2026
- **No hyperlinks allowed** in rebuttals — all evidence must be in text

---

## Rebuttal Strategy

### Priority 1: Flip Reviewer DVUB (2 → 3+)
This reviewer gave explicit conditions for upgrading. Address all three precisely:

**Response to W1 (full-document pipeline):**
- Document the exact preprocessing pipeline: was text truncated? Chunked? Summarized?
- Provide truncation/coverage statistics across all 3,116 samples (e.g., "X% of documents exceeded model context, truncated to Y tokens")
- If possible, include a length-controlled re-analysis showing performance by document length bucket

**Response to W2 (accrual gains framing):**
- Reframe numbers explicitly as hypothesis-generating projections, not causal estimates
- Acknowledge the scalar multiplication limitation directly
- Add uncertainty language and describe what additional validation would look like

**Response to W3 (C2/C6 inconsistencies):**
- Explain the distinction between patient-level matching metrics and trial-level accrual
- For C2: explain why ECOG tightening with +233% gain occurs (is this a patient-pool composition effect?)
- For C6: acknowledge the non-actionable nature and either explain the nuance or state you will remove/replace the example
- Commit to adding directionality checks in the camera-ready

### Priority 2: Strengthen for Reviewer ECp3 (maintain 3, ideally bump)
- **Finetuning:** Argue this is a benchmark/framework paper, not a model paper. Finetuning is valuable future work but the contribution is the task formulation and dataset.
- **Ablation studies:** If you can describe any ablation results (e.g., with/without RAG is already an ablation), frame them as such.
- **Data leakage:** Argue that the dataset comes from structured protocol amendment documents on ClinicalTrials.gov, which differ significantly from typical pretraining web text. The triplet format (original + evidence + revision) is novel and unlikely to appear in pretraining data verbatim. Offer to add a contamination analysis in camera-ready.

### Priority 3: Address Reviewer Yns9 (maintain 3, ideally bump)
- **Repo with no code/data:** Commit to releasing code and data. Describe what will be available and when.
- **Larger models:** If you can run Llama-3.1-70B or similar before the rebuttal deadline, include results as text. If not, acknowledge the scaling trend and commit to including in camera-ready.
- **LLM-as-judge prompt:** Include the exact prompt template in your rebuttal text (since you can't link to supplementary).
- **Binary threshold:** Acknowledge the coarseness and describe how a finer-grained analysis would look; commit to adding in camera-ready.
- **Oncology only:** Acknowledge as a limitation and frame as a strong starting point given oncology's outsized share of clinical trials.

### Priority 4: Reinforce Reviewer aiEs (maintain 4)
- Thank for the thorough and positive review.
- On RAG utility: reframe as a finding — showing when RAG helps and doesn't is itself a contribution. For smaller/weaker models, RAG may still be crucial.
- On human validation: acknowledge and describe plans for clinical expert evaluation as future work.

---

## Cross-Cutting Concerns to Address

| Concern | Raised by | Response Approach |
|---------|-----------|-------------------|
| Full-doc preprocessing unclear | DVUB | Provide exact pipeline details + stats |
| Accrual gains overstated | DVUB, Yns9 | Reframe as projections, separate metrics |
| C2/C6 logical inconsistencies | DVUB | Explain or remove; add directionality checks |
| No finetuning | ECp3 | Frame as benchmark paper; future work |
| No ablations | ECp3 | Reframe existing comparisons as ablations |
| Data leakage | ECp3 | Argue dataset novelty; offer contamination check |
| No larger models (≥10B) | Yns9 | Run if possible; else commit to camera-ready |
| No human evaluation | Yns9, aiEs | Acknowledge; describe future plans |
| LLM-judge prompt not shared | Yns9 | Include in rebuttal text |
| Repo empty | Yns9 | Commit to release; describe contents |
| RAG marginal utility | aiEs | Reframe as a finding/contribution |
| Oncology only | Yns9 | Acknowledge; frame as strong starting point |

---

## Work to Do Before Rebuttal Deadline

### Must-do (required for DVUB upgrade)
- [ ] Document exact full-document preprocessing pipeline
- [ ] Compute truncation/coverage statistics for all 3,116 samples
- [ ] Prepare corrected analysis of C2 and C6 examples
- [ ] Prepare clear framing separating patient-level matching from trial-level accrual
- [ ] Draft reframed accrual impact language (hypothesis-generating, not causal)

### Should-do (strengthens overall case)
- [ ] Run at least one larger model (e.g., Llama-3.1-70B) if feasible
- [ ] Prepare LLM-as-judge prompt template text for inclusion in rebuttal
- [ ] Draft data contamination argument
- [ ] Prepare repo release plan with timeline

### Nice-to-have
- [ ] Length-controlled re-analysis (performance by document length bucket)
- [ ] Sensitivity analysis on accrual estimates
- [ ] Finer-grained equivalence threshold analysis (3 vs. 4 bucket breakdown)

---

## Rebuttal Format Reminders
- Respond to **each reviewer separately** on OpenReview
- **No hyperlinks allowed** — describe everything in text
- Be concise, direct, and numbered to match reviewer concerns (W1, W2, W3)
- Tone: respectful, substantive, not defensive
- For changes you can't fully execute before rebuttal: "We will incorporate X in the camera-ready" — but pair this with as much evidence as possible now
- One comment/ping per reviewer is enough — don't badger
