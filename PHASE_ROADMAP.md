# RECITE Phase Roadmap

Tracks what needs to happen in the RECITE repo across rebuttal, camera-ready, and public release.

## Phase 1: Pre-Rebuttal Script Transfer (NOW)

Transfer rebuttal-supporting scripts from clintriaLM to RECITE so the repo is self-contained for reviewers.

**Status: DONE**

| Script | Source | Destination | Notes |
|--------|--------|-------------|-------|
| `truncation_evidence_check.py` | clintriaLM/scripts/ | scripts/rebuttal/ | DVUB W1 — classifies 3,116 samples for EC in truncated window |
| `truncation_stratified_verify.py` | clintriaLM/scripts/ | scripts/rebuttal/ | DVUB W1 — stratified re-verification (150 samples, gpt-4.1 + mini) |
| `recite_rebuttal_analysis.py` | clintriaLM/scripts/ | scripts/rebuttal/ | Table 1 recreation + truncation subset analysis |
| `judge_score.py` | clintriaLM/scripts/ | scripts/rebuttal/ | Unified judge scorer (452 LOC, concurrent, resumable) |
| `judge_only.py` | clintriaLM/scripts/ | scripts/rebuttal/ | Judge-only mode (225 LOC, score existing predictions) |

**Data symlinks created:**
- `data/prod/benchmark_results.db` -> clintriaLM
- `config/judge_variants/` -> clintriaLM

**Path adaptations:** ROOT updated from `parent.parent` to `parent.parent.parent` (scripts now in `scripts/rebuttal/` instead of `scripts/`).

## Phase 2: Human Eval Infra (Post-Rebuttal)

Transfer human evaluation infrastructure for Yns9 W4 response and camera-ready.

| Script | Source | Purpose |
|--------|--------|---------|
| `human_eval_sampling.py` | clintriaLM/scripts/ | Stratified sampling for human evaluation (needs DB schema reconciliation) |
| `generate_human_eval_4panel.py` | clintriaLM/scripts/ | 4-panel HTML diff viewer generation |
| HTML diff assets | clintriaLM/data/human_eval/ | Diff pages for evaluators |

**Complexity:** Medium — DB schema differences between clintriaLM's dev DB and RECITE's data layout need reconciliation.

## Phase 3: Camera-Ready (If Accepted)

| Item | Purpose | Effort |
|------|---------|--------|
| `recite_finetune.py` | QLoRA finetuning script (ECp3 weakness) | Medium |
| Judge calibration scripts | Supplementary material for judge methodology | Low |
| Directionality checks | DVUB W3 — verify proposed changes broaden criteria | Medium |
| Applicability gate | DVUB W3 — suppress gains when trial already implements recommendation | Medium |
| Preprocessing Pipeline subsection | DVUB W1 — exact 6-step pipeline + truncation table in paper | Low (text only) |

## Phase 4: Public Release (Upon Acceptance)

| Item | Purpose | Effort |
|------|---------|--------|
| Clean README for external users | Remove internal references, add setup instructions | Low |
| Single-command reproduction | `recite benchmark run-benchmark` already works; verify end-to-end | Low |
| LICENSE file | Already CC-BY-NC-SA-4.0 | Done |
| Data release | Benchmark splits (11,913 instances) already in Git LFS | Done |
| CI/CD | GitHub Actions for pytest + linting | Low |
| Remove internal symlinks | Replace clintriaLM symlinks with standalone data or instructions | Medium |
