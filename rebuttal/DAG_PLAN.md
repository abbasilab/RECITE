# RECITE KDD 2026 Rebuttal — Agent DAG Plan

**Created:** 2026-04-04
**Rebuttal deadline:** April 11-17, 2026
**Reference:** See `RECITE_rebuttal_plan.md` (cached in phdmanager/mcp/telegram-mcp/dl/)

## Scores: 2/3/3/4 (avg 3.0, borderline)
- **DVUB (2, Reject):** Must flip to 3+. Gave explicit upgrade conditions.
- **ECp3 (3, Weak Accept):** Maintain/bump.
- **Yns9 (3, Weak Accept):** Maintain/bump.
- **aiEs (4, Accept):** Reinforce.

## Agent DAG

### Phase 1: Parallel Computation (no dependencies)

#### Agent A: `phd-clintrialm-rebuttal-truncation`
- **Task:** Compute truncation/coverage statistics for all 3,116 benchmark samples
- **Details:** For each model's context window (8K, 32K, 128K), compute % of documents that exceeded it, truncation coverage stats, mean/median/p95 doc lengths
- **Output:** Stats table + length distribution for rebuttal text
- **Addresses:** DVUB W1

#### Agent B: `phd-clintrialm-rebuttal-c2c6`
- **Task:** Investigate C2 and C6 representative examples
- **Details:**
  - C2: ECOG ≤2→≤1 (tightening) yet reports +233% gain — explain the logic
  - C6: Trial already uses ECOG ≤1, making change non-actionable, yet +246% gain
  - Separate patient-level matching metrics from trial-level accrual
  - Check the actual data/code to explain or correct
- **Output:** Corrected analysis + explanation for rebuttal
- **Addresses:** DVUB W3

#### Agent C: `phd-clintrialm-rebuttal-prompt`
- **Task:** Extract the exact LLM-as-judge prompt template from code
- **Details:** Find the prompt, format as plain text (no hyperlinks allowed in rebuttal)
- **Output:** Formatted prompt text ready for rebuttal inclusion
- **Addresses:** Yns9

#### Agent D: `phd-clintrialm-rebuttal-largemodel` [Abbasi GPU]
- **Task:** Run larger open-weight models (≥10B params) on the benchmark
- **Infra:** abbasi-gpu-1 (8x RTX 5000 Ada, 256GB VRAM) — SSH access confirmed
- **Models to run (in priority order):**
  1. **Llama-3.1-70B** (full precision, ~140GB across 8 GPUs) — strongest answer to reviewer
  2. **Qwen3-32B** (already available locally, can also run on Abbasi)
- **Setup:** Install vLLM on abbasi-gpu-1, serve model via OpenAI-compatible API, run eval from local
- **Details:** Run on same 3,116-sample eval set, compare to Gemma 2 9B baseline (84.7%)
- **Output:** Results table showing scaling trend (2B→9B→32B→70B)
- **Addresses:** Yns9 (should-do, strengthens case significantly with Abbasi hardware)

#### Agent F: `phd-clintrialm-rebuttal-repo`
- **Task:** Prepare RECITE repo for reviewer transparency (code + data + instructions)
- **Repos:**
  - **Staging:** `russro/RECITE` (private, all work happens here first)
  - **Public:** `abbasilab/RECITE` (DO NOT push directly — only after PI approval)
- **Details:**
  - Add clear README with setup instructions, dependencies, how to reproduce results
  - Include benchmark data (3,116 samples) or instructions to regenerate
  - Add evaluation scripts with usage examples
  - Include model configs and prompt templates used in paper
  - Add LICENSE file if missing
  - Ensure no hardcoded paths, API keys, or private data leak
- **Output:** Staging repo ready for reviewer access; PR or release notes for PI review before public push
- **Addresses:** Yns9 ("The repo contains no actual data or instructions to run the code")

### Phase 1.5: Critical DVUB Review (depends on A + B)

#### Agent G: `phd-clintrialm-rebuttal-dvub-critic`
- **Task:** Adopt DVUB's perspective and critically stress-test the A+B outputs
- **Depends on:** Agents A and B (waits for both to complete)
- **Details:**
  - Read DVUB's full review (score 2, explicit upgrade conditions)
  - Read Agent A output (truncation stats) and Agent B output (C2/C6 analysis)
  - Act as a hostile reviewer: Are the truncation stats convincing? Do they fully address W1? Are there gaps?
  - For C2/C6: Does the explanation hold up? Would DVUB find it satisfactory or poke holes?
  - Check for logical inconsistencies, missing data, weak arguments
  - Flag anything that needs strengthening before it goes into the rebuttal
  - If the analysis reveals problems, propose specific fixes or additional analysis needed
- **Output:** Critical review memo with: (1) strengths, (2) remaining weaknesses, (3) specific fixes needed
- **Addresses:** Ensures DVUB response is airtight — this is the reviewer who must flip from 2→3+

### Phase 2: Synthesis (depends on Phase 1 + 1.5)

#### Agent E: `phd-clintrialm-rebuttal-drafter`
- **Task:** Draft per-reviewer rebuttal responses
- **Inputs:** Results from Agents A-D, F, G + rebuttal plan
- **Details:**
  - DVUB: Address W1 (truncation), W2 (accrual framing), W3 (C2/C6) — **incorporate Agent G's critical review to ensure airtight response**
  - ECp3: Finetuning defense, ablation framing, data leakage argument
  - Yns9: LLM-judge prompt, larger model results, repo release plan (link to staging repo)
  - aiEs: Thank + reframe RAG finding
- **Output:** Draft rebuttal text per reviewer

## Large Open-Weight Model Candidates (for Agent D)
| Model | Params | VRAM (FP16) | Fits Abbasi? | Fits Local 2x4090? | Priority |
|-------|--------|-------------|--------------|---------------------|----------|
| Llama-3.1-70B | 70B | ~140GB | Yes (8x32GB) | No | **P1** |
| Qwen3-32B | 32B | ~64GB | Yes | Yes (already loaded) | **P2** |

## GPU Resources
- **Local:** 2x RTX 4090 (24GB each, 48GB total) — currently running qwen3:32b for clincrawl
- **Abbasi GPU-1:** 8x RTX 5000 Ada (32GB each, 256GB total), 128 CPU, 1TB RAM, 384GB disk — SSH confirmed, idle
- **Abbasi GPU-2:** Same spec, 1.1TB disk free — SSH confirmed, idle (reserve)

## Agent D Setup Plan (abbasi-gpu-1)
1. SSH into abbasi-gpu-1
2. Install vLLM: `pip install vllm`
3. Download model: `huggingface-cli download meta-llama/Llama-3.1-70B-Instruct`
4. Serve: `vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 8 --port 8000`
5. Run eval from local machine pointing to `http://abbasi-gpu-1:8000/v1`

## Repo Transparency (Agent F)
- **Staging repo:** `russro/RECITE` — all changes go here first
- **Public repo:** `abbasilab/RECITE` — DO NOT push directly; only after PI (Reza) approval
- **Workflow:** Agent F prepares staging repo → Russ/Reza review → merge/push to public
- **Reviewer Yns9 noted:** "The repo contains no actual data or instructions to run the code"
- **Goal:** Repo should be self-contained enough for a reviewer to clone and reproduce key results

## Status
- [x] Phase 0: Cache plan + rebuttal reference
- [x] SSH access to Abbasi GPUs confirmed
- [x] Model selection finalized (Llama-3.1-70B + Qwen3-32B)
- [x] Phase 1: Deploy agents A-D, F (all parallel) — deployed 2026-04-04 06:36 UTC
- [ ] Phase 1.5: Deploy agent G (after A+B complete — critical DVUB reviewer)
- [ ] Phase 2: Deploy agent E (after all Phase 1 + 1.5 complete)
