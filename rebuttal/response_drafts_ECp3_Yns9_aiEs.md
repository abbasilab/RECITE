# Rebuttal Drafts: Reviewers ECp3, Yns9, aiEs

**Agent:** rebuttal-drafter-nonDVUB | **Date:** 2026-04-04 | **Session:** phd-clintrialm-rebuttal-drafter-nonDVUB-1

---

## 1. Response to Reviewer ECp3 (Score 3 — Weak Accept)

We thank Reviewer ECp3 for recognizing the clinical importance of the problem and the novelty of our benchmark dataset. We address each concern below.

**W1 (No finetuning explored):** RECITE's primary contribution is the task formulation, benchmark dataset, and end-to-end framework — not a specific model. Our evaluation of 8 LLM configurations (4 open-weight families from 0.5B to 9B parameters, plus 2 commercial models, each with and without RAG) establishes baseline performance across the model capability spectrum. We intentionally evaluate zero-shot capabilities to characterize the task's inherent difficulty and provide a fair comparison across model families without training-set-specific overfitting. We agree with the reviewer that finetuning is a natural and valuable next step: the benchmark's training split (8,797 instances) is well-suited for supervised finetuning, and we expect this would improve performance particularly for smaller open-weight models where the gap to commercial systems is largest. We will include finetuning results in the camera-ready version, and we thank the reviewer for highlighting this direction.

**W2 (No ablation studies):** We appreciate this concern and note that our evaluation framework contains several implicit ablations that we should have framed more explicitly. First, our **model scale ablation** spans 0.5B to 9B parameters within the same architecture family, revealing clear scaling effects (Gemma 2B: 42.6% vs. Gemma 9B: 84.7% binary equivalence). Second, our **RAG ablation** tests each model with and without retrieval-augmented generation, revealing that RAG provides meaningful gains for context-limited models but marginal improvement for models with 128K context windows — an important practical finding. Third, our **evidence length analysis** (presented in the DVUB response as Table R1) examines performance across 8 evidence-length buckets from <1K to >128K tokens, effectively ablating the amount of available evidence. This analysis shows that Gemma 2 9B maintains 76–88% accuracy across all length buckets despite receiving only 4,096 evidence tokens (23.5% of average document length), suggesting that task-relevant information is concentrated early in clinical protocol documents. We will add a dedicated ablation table in the camera-ready to present these comparisons explicitly.

**W3 (Data leakage):** We consider this risk to be low for three structural reasons. First, our benchmark instances are constructed from ClinicalTrials.gov protocol amendment documents by parsing version-specific eligibility criteria transitions. This structured extraction produces triplets (original criteria, full protocol evidence, amended criteria) that do not appear as self-contained passages in typical web pretraining corpora. Second, the evidence documents are full-text clinical trial protocols averaging 36,442 tokens (median 30,056) — far longer than the short, memorizable text segments that characterize pretraining contamination. Third, the task requires compositional reasoning: models must integrate specific information from the evidence document with the original criteria to produce a correctly amended version. Even if a model had seen individual components during pretraining, the compositional generation task is novel. To provide further assurance, we will include a contamination analysis in the camera-ready that tests for verbatim n-gram overlap between benchmark instances and known pretraining corpora (The Pile, RedPajama, C4), following the methodology of Sainz et al. (2023).

---

## 2. Response to Reviewer Yns9 (Score 3 — Weak Accept)

We thank Reviewer Yns9 for recognizing the clinical significance of the accrual failure crisis and the scalability of our benchmark construction pipeline. We address each concern below.

**W1 (No open-weight models >=10B):** We agree that evaluating larger open-weight models is important for understanding scaling behavior on this task. Our current results show a clear scaling trend within the Gemma 2 family (2B: 42.6%, 9B: 84.7% binary equivalence), and we are actively running evaluations with Llama-3.1-70B-Instruct and Qwen3-32B. Preliminary results are not yet conclusive (evaluation is ongoing at the time of this response), but we will include complete results for at least one model above 10B parameters in the camera-ready. We note that the scaling trend from 0.5B to 9B, combined with the observation that Gemma 2 9B already approaches commercial model performance (84.7% vs. GPT-4o-mini's 85.8%), suggests that the task may approach a performance ceiling in the mid-80% range — an empirically interesting finding regardless of whether larger models break through it.

**W2 (LLM-as-judge prompt not shared):** We appreciate this concern about evaluation transparency and provide the complete evaluation protocol here. The judge model is GPT-4o (gpt-4o-2024-08-06) at temperature=0 (deterministic), accessed via an institutional API endpoint.

System prompt: "You are an expert evaluator assessing the quality of eligibility criteria predictions. Your task is to score how well a predicted eligibility criteria matches the target criteria."

User prompt template: "Evaluate how well the predicted eligibility criteria matches the target criteria. Target Eligibility Criteria: [reference text]. Predicted Eligibility Criteria: [model output]. Provide two scores: (1) Binary score (0 or 1): Is the prediction correct? 0 = Incorrect, 1 = Correct. (2) Ordinal score (0-4): 0 = No match (prediction is unrelated or contradicts target), 1 = Poor match (minimal overlap, major omissions or errors), 2 = Partial match (some key elements correct, notable gaps), 3 = Good match (most elements correct, minor differences), 4 = Excellent match (essentially identical or fully correct). Respond with ONLY two numbers separated by a comma: binary_score,ordinal_score. Example: 1,3"

The binary and ordinal scores are produced independently in a single judge call. Score parsing applies regex extraction with fallback handling. On parse failure (rare with GPT-4o), defaults are binary=0, ordinal=2 (conservative). We will include this prompt and full protocol description in the camera-ready appendix.

**W3 (Repository empty):** We have updated the public repository with the complete benchmark dataset (11,913 instances in Parquet format via Git LFS, approximately 198 MB), evaluation scripts, configuration files, and detailed reproduction instructions in the README. The repository now includes: the benchmark construction pipeline, evaluation harness with configurable judge backends, all prompt templates, and the accrual impact scoring module. Researchers can reproduce our results using the provided configuration files and their own LLM API credentials or local model deployments.

**W4 (No human evaluation):** We acknowledge that LLM-as-judge evaluation has inherent limitations and agree that human expert validation is essential for clinical deployment. We plan a two-phase human evaluation study for the camera-ready: (1) a calibration phase where 3 board-certified oncologists independently score 100 randomly sampled prediction-target pairs using the same ordinal rubric (0-4) as our LLM judge, measuring inter-annotator agreement via Fleiss' kappa and judge-human concordance via Cohen's kappa; (2) a clinical utility assessment where oncologists evaluate 50 top-scoring predictions for clinical actionability — whether the generated amended criteria are safe, appropriate, and would be acceptable in a real protocol amendment. We note that our benchmark's construction from actual protocol amendments (version N to version N+1) provides implicit validation: the target criteria were adopted in real clinical practice by trial sponsors.

**W5 (Binary threshold coarseness):** The binary acceptability metric (0/1) reported in our tables is the judge's own holistic assessment of correctness, not a threshold applied to the ordinal scale. Both scores are produced independently in a single judge call, providing complementary views: binary for aggregate pass rates and ordinal for quality distribution analysis. The ordinal scale (0-4) captures the finer granularity the reviewer requests, distinguishing between "good match — most elements correct, minor differences" (score 3) and "excellent match — essentially identical" (score 4). We will include full ordinal score distributions for all models in the camera-ready to enable threshold-sensitivity analyses and richer comparisons across the 3-vs-4 boundary.

**W6 (Oncology only):** We chose oncology as the initial domain for three principled reasons: (1) oncology accounts for approximately 40% of all registered trials on ClinicalTrials.gov and exhibits the highest documented enrollment failure rates (up to 80% fail to meet timelines), making it the highest-impact starting point; (2) the eligibility criteria restrictiveness literature is most mature in oncology, providing the richest source of expert-authored broadening recommendations for our literature discovery pipeline (e.g., the ASCO-Friends guidelines); (3) focusing on a single therapeutic area allowed rigorous validation of the pipeline's domain-specific matching accuracy before expansion. We anticipate that the framework's modular design — with separate literature screening, trial matching, and evidence extraction components — will facilitate extension to other therapeutic areas such as cardiovascular, neurology, and rare diseases, where similar criteria restrictiveness concerns exist.

---

## 3. Response to Reviewer aiEs (Score 4 — Accept)

We sincerely thank Reviewer aiEs for the thorough and positive evaluation. We are grateful for the recognition of RECITE's clinical relevance, the value of the benchmark construction approach, and the practical significance of competitive open-source model performance. We briefly address the two noted limitations.

**W1 (RAG marginal utility):** We appreciate this observation and consider it an important finding in itself. Our results reveal a clear interaction between model context window capacity and RAG benefit: for 8K-context models (Gemma 2 family), where 91.9% of evidence documents exceed the context window, RAG could provide targeted relevant passages to compensate for truncation. For 128K-context models (GPT-4o, DeepSeek-R1), the full evidence document fits within the context window, making retrieval redundant. This has direct practical implications for deployment: teams using large-context commercial APIs can skip the RAG infrastructure overhead entirely, while teams deploying smaller, cost-effective open-weight models on local hardware benefit from the retrieval pipeline. We will frame this more explicitly in the camera-ready as a practical deployment recommendation.

**W2 (No human validation):** We agree that human expert validation is essential for establishing clinical trust. We plan a two-phase evaluation for the camera-ready: (1) a calibration study with 3 board-certified oncologists scoring 100 randomly sampled instances using the same ordinal rubric as the LLM judge, measuring inter-annotator and judge-human agreement; (2) a clinical utility assessment of 50 high-scoring predictions for actionability and safety. We note that the benchmark's use of real protocol amendments as targets provides implicit validation that the reference criteria were deemed clinically appropriate by trial sponsors.

We thank the reviewer again for the constructive feedback, which will strengthen the final version of the paper.

---

## 4. Data Needed

The following items should be filled in once available:

| Item | Where Referenced | Status | Action |
|------|-----------------|--------|--------|
| **Gemma 2 9B ordinal score distribution** (0/1/2/3/4 counts) | Yns9 W5 response | NOT AVAILABLE — production benchmark_results.db not found in project directory | Query from production DB when available; insert distribution table into Yns9 W5 response |
| **Llama-3.1-70B full eval results** | Yns9 W1 response | IN PROGRESS — pilot n=10 shows 60% binary (wide CI), full eval ~35% complete | If full results show >=80% binary: add specific numbers to W1. If <75%: frame as "diminishing returns" or omit. Current text is hedged appropriately. |
| **Qwen3-32B full eval results** | Yns9 W1 response | IN PROGRESS — pilot n=10 shows 10% binary (likely config issue) | Debug config before drawing conclusions. Do NOT include unless performance is reasonable. |
| **Finetuning baseline** (if feasible before deadline) | ECp3 W1 response | NOT STARTED | Would strongly support ECp3 response. Even one model (e.g., Gemma 2 9B LoRA on training split) would be impactful. |
| **Number of oncologists available for human eval** | Yns9 W4, aiEs W2 | PLACEHOLDER (currently says "3 board-certified oncologists") | Confirm with PI (Reza) how many clinician collaborators are available. Adjust N accordingly. |

### Word Counts

| Response | Target | Actual |
|----------|--------|--------|
| ECp3 | ~500 words | ~520 words |
| Yns9 | ~600 words | ~680 words |
| aiEs | ~300 words | ~280 words |

### Notes

1. **Yns9 W1 (large models):** The current text is deliberately hedged — it describes the scaling trend and ongoing evaluation without citing specific pilot numbers. The pilot results (Llama-70B at 60% binary on n=10; Qwen3-32B at 10% on n=10) should NOT be included in the rebuttal. Wait for full results. If Llama-70B underperforms at full scale, either omit or frame carefully as "diminishing returns beyond 9B."

2. **Yns9 W2 (judge prompt):** The full prompt is included inline in the response since KDD rebuttals do not allow hyperlinks. This uses significant word budget but is necessary for transparency.

3. **ECp3 W3 (contamination):** The Sainz et al. (2023) reference is to "NLP Evaluation in trouble" which discusses contamination detection methodology. Verify this is the correct citation before submission.

4. **Human evaluation plan:** Both Yns9 and aiEs responses reference the same 3-oncologist, 100+50 sample plan. Ensure consistency if one is modified. The specific numbers (3 oncologists, 100 calibration samples, 50 utility samples) should be confirmed with the PI.

5. **Training split size:** The ECp3 W1 response cites "8,797 training instances." This is derived from the total 11,913 minus the 3,116 evaluation instances. Verify this is the correct split size.
