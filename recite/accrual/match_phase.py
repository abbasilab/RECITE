"""
Match phase: find and score paper–trial matches for accrual top papers that have none.

Uses UCSF Versa (or configurable LLM) for query generation and match evaluation.
Candidates come from ClinicalTrials.gov via the crawler adapter; scores are saved
to paper_trial_matches_<preset> so Phase 2 can compute accrual gains.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from recite.accrual.db import get_top_papers
from recite.crawler.adapters import ClinicalTrialsGovAdapter, CTG_INSTRUCTIONS, Document
from recite.crawler.db import (
    get_document_by_source_id,
    get_paper_trial_matches_for_paper,
    get_used_queries,
    has_paper_trial_match,
    save_paper_trial_match,
    save_query,
)
from recite.crawler.paper_trial_matcher import (
    evaluate_match,
    find_candidate_trials,
    generate_trial_queries,
)


# Default path for crawler match prompts (query gen + match eval)
DEFAULT_CRAWLER_PROMPTS_PATH = Path(__file__).parent.parent.parent / "config" / "crawler_prompts.json"


def _load_match_prompts(path: Path) -> dict[str, str]:
    """Load match-related prompts from crawler_prompts.json."""
    with open(path) as f:
        data = json.load(f)
    return {
        "match_query_system": data["query_generation_for_matching"]["system"],
        "match_query_user": data["query_generation_for_matching"]["user"],
        "match_eval_system": data["match_evaluation"]["system"],
        "match_eval_user": data["match_evaluation"]["user"],
    }


def _parse_json_from_llm(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON from LLM output (strip markdown, trailing commas)."""
    if not text or not text.strip():
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


class VersaMatchLLM:
    """Thin LLM wrapper for paper–trial matching using UCSF Versa. Exposes .prompts and .complete_json for crawler matcher."""

    def __init__(self, ucsf_versa_model: str, prompts_path: Path = DEFAULT_CRAWLER_PROMPTS_PATH):
        from recite.llmapis import UCSFVersaAPI

        prompts = _load_match_prompts(prompts_path)
        self._api = UCSFVersaAPI(model=ucsf_versa_model, system_prompt=prompts["match_eval_system"])
        self.prompts = type("Prompts", (), {
            "match_query_system": prompts["match_query_system"],
            "match_query_user": prompts["match_query_user"],
            "match_eval_system": prompts["match_eval_system"],
            "match_eval_user": prompts["match_eval_user"],
        })()

    def complete_json(self, prompt: str, system: str = "") -> Optional[dict[str, Any]]:
        """Call Versa and return parsed JSON."""
        raw = self._api(prompt, system or self.prompts.match_eval_system, temperature=0)
        return _parse_json_from_llm(raw)


def run_match_phase(
    accrual_db_path: Path,
    model_preset: str,
    ucsf_versa_model: str,
    match_trial_limit: int = 20,
    min_match_score: int = 3,
    crawler_prompts_path: Optional[Path] = None,
) -> int:
    """Run paper–trial matching for top papers that have no matches.

    For each top paper (directives + impact) with zero matches in paper_trial_matches_<preset>:
    1. Load paper (title, abstract) from clintrialm documents_<preset>.
    2. Generate CTG search queries via Versa; find candidate trials (CTG adapter).
    3. Score up to match_trial_limit trials with Versa (match_eval prompt).
    4. Save matches with score >= min_match_score to paper_trial_matches_<preset>.

    Requires DATABASE_PATH set to clintrialm.db path (accrual.py sets this).

    Returns:
        Number of new matches saved across all papers.
    """
    prompts_path = crawler_prompts_path or DEFAULT_CRAWLER_PROMPTS_PATH
    if not prompts_path.exists():
        logger.warning(f"Match phase: crawler prompts not found at {prompts_path}, skipping")
        return 0

    top_papers = get_top_papers(accrual_db_path, limit=None)
    papers_without_matches = []
    for p in top_papers:
        psrc, pid = p["paper_source"], p["paper_source_id"]
        existing = get_paper_trial_matches_for_paper(psrc, pid, model_preset, min_match_score=1)
        if not existing:
            papers_without_matches.append(p)

    if not papers_without_matches:
        logger.info("Match phase: all top papers already have matches, nothing to do")
        return 0

    logger.info(f"Match phase: {len(papers_without_matches)} top papers without matches, finding trials (limit {match_trial_limit} per paper)")
    llm = VersaMatchLLM(ucsf_versa_model, prompts_path)
    adapter = ClinicalTrialsGovAdapter(requests_per_second=3.0)
    total_saved = 0

    for paper_row in papers_without_matches:
        psrc = paper_row["paper_source"]
        pid = paper_row["paper_source_id"]
        doc_dict = get_document_by_source_id(psrc, pid, model_preset)
        if doc_dict:
            title = doc_dict.get("title") or ""
            abstract = doc_dict.get("abstract") or ""
        else:
            # Fallback: use raw LLM responses from paper_answers so match phase can still run
            # (e.g. when documents_local was cleared or papers came from a different preset)
            logger.warning(f"Match phase: no document in clintrialm for {psrc}|{pid}, using raw responses as context")
            title = paper_row.get("directives_exact_text") or f"Paper {pid}"
            if isinstance(title, str) and len(title) > 200:
                title = title[:197] + "..."
            raw_dir = (paper_row.get("raw_response_directives") or "")[:2500]
            raw_imp = (paper_row.get("raw_response_impact") or "")[:1500]
            abstract = (raw_dir + "\n\n" + raw_imp).strip() or "(No abstract available)"
        paper = Document(source=psrc, source_id=pid, title=title, abstract=abstract)

        used_queries = get_used_queries("ctg", model_preset)
        queries = generate_trial_queries(llm, paper, used_queries, CTG_INSTRUCTIONS)
        if not queries:
            logger.warning(f"Match phase: no queries generated for {pid}, skipping")
            continue
        for q in queries:
            save_query("ctg", q, model_preset)

        trials = list(find_candidate_trials(adapter, queries, max_per_query=10))
        trials = trials[:match_trial_limit]
        if not trials:
            logger.warning(f"Match phase: no candidate trials for {pid}, skipping")
            continue

        saved_this_paper = 0
        for trial in trials:
            if has_paper_trial_match(psrc, pid, trial.source_id, model_preset):
                continue
            score = evaluate_match(llm, paper, trial, max_paper_chars=2000, max_trial_chars=4000)
            if score is None:
                continue
            match_score = max(1, min(5, score.match_score))
            applicability_score = max(1, min(5, score.applicability_score))
            if match_score < min_match_score:
                continue
            saved = save_paper_trial_match(
                paper_source=psrc,
                paper_source_id=pid,
                paper_title=paper.title,
                trial_instance_id=trial.source_id,
                trial_title=trial.title or "",
                match_score=match_score,
                match_reasoning=score.match_reasoning or "",
                applicability_score=applicability_score,
                change_directive_quote=score.change_directive_quote or "",
                change_rationale_quote=score.change_rationale_quote or "",
                model_preset=model_preset,
            )
            if saved:
                saved_this_paper += 1
                total_saved += 1
                logger.info(f"Match phase: saved {psrc}|{pid} <-> {trial.source_id} (match={match_score}, applicability={applicability_score})")

        logger.info(f"Match phase: paper {pid} -> {saved_this_paper} new matches (evaluated {len(trials)} trials)")

    logger.info(f"Match phase: done, {total_saved} new matches saved")
    return total_saved
