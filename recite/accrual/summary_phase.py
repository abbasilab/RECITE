"""
Summary phase: generate evidenced summaries for paper_trial_gains rows.

For each paper-trial pair, calls the LLM to produce a short evidenced summary
(how ECs are amended with quotes; how accrual is computed with quotes) and
stores it in paper_trial_gains.evidenced_summary.
"""

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from recite.accrual.db import (
    get_paper_answers,
    get_paper_trial_gains,
    update_paper_trial_evidenced_summary,
)
from recite.accrual.llm import call_accrual_llm
from recite.accrual.prompts import load_accrual_prompts
from recite.crawler.db import get_paper_trial_matches_for_paper


def _str(val: Any) -> str:
    """Coerce value for prompt placeholder; None -> empty string."""
    if val is None:
        return ""
    return str(val).strip()


def run_summary_phase(
    accrual_db_path: Path,
    prompts_path: Path,
    only_null_evidenced_summary: bool = True,
    model_preset: str = "local",
    endpoint: str = "http://localhost:8000/v1",
    model: str = "local-model",
    ucsf_versa_model: Optional[str] = None,
) -> int:
    """
    Generate evidenced summaries for paper_trial_gains rows (or only those with
    evidenced_summary IS NULL). Returns the number of rows updated.
    """
    prompts = load_accrual_prompts(prompts_path)
    cfg = prompts.get("evidenced_summary") or {}
    system = cfg.get("system", "")
    user_tpl = cfg.get("user_template", "")
    if not user_tpl:
        logger.warning("evidenced_summary user_template missing in prompts; skipping summary phase")
        return 0

    rows = get_paper_trial_gains(accrual_db_path, only_null_evidenced_summary=only_null_evidenced_summary)
    if not rows:
        logger.info("Summary phase: no paper_trial_gains rows to process")
        return 0

    # Build paper lookup: (paper_source, paper_source_id) -> paper_answers row
    paper_keys = {(r["paper_source"], r["paper_source_id"]) for r in rows}
    paper_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for psrc, pid in paper_keys:
        answers = get_paper_answers(accrual_db_path, paper_source=psrc, paper_source_id=pid)
        if answers:
            paper_by_key[(psrc, pid)] = answers[0]

    updated = 0
    for row in rows:
        psrc = row["paper_source"]
        pid = row["paper_source_id"]
        instance_id = row["trial_instance_id"]
        paper = paper_by_key.get((psrc, pid)) or {}
        directive_quote = row.get("change_directive_quote")
        rationale_quote = row.get("change_rationale_quote")
        if (directive_quote is None or rationale_quote is None) or (
            (isinstance(directive_quote, str) and not directive_quote.strip())
            or (isinstance(rationale_quote, str) and not rationale_quote.strip())
        ):
            # Fetch match from clintrialm to get quotes
            matches = get_paper_trial_matches_for_paper(
                paper_source=psrc,
                paper_source_id=pid,
                model_preset=model_preset,
                min_match_score=0,
            )
            for m in matches:
                if m.get("trial_instance_id") == instance_id:
                    directive_quote = directive_quote if directive_quote else m.get("change_directive_quote")
                    rationale_quote = rationale_quote if rationale_quote else m.get("change_rationale_quote")
                    break

        prompt = user_tpl.format(
            directives_exact_text=_str(paper.get("directives_exact_text")),
            impact_percent=_str(paper.get("impact_percent")),
            impact_absolute=_str(paper.get("impact_absolute")),
            impact_evidence=_str(paper.get("impact_evidence")),
            trial_instance_id=_str(instance_id),
            enrollment=_str(row.get("enrollment")),
            change_directive_quote=_str(directive_quote),
            change_rationale_quote=_str(rationale_quote),
            paper_pct_gain=_str(row.get("paper_pct_gain")),
            scalar_gain=_str(row.get("scalar_gain")),
        )
        try:
            raw = call_accrual_llm(
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                system_prompt=system,
                ucsf_versa_model=ucsf_versa_model,
            )
            summary = (raw or "").strip()[:10000]
            if summary:
                update_paper_trial_evidenced_summary(
                    accrual_db_path, psrc, pid, instance_id, summary
                )
                updated += 1
        except Exception as e:
            logger.warning(f"Summary phase failed for {psrc}/{pid} / {instance_id}: {e}")
    logger.info(f"Summary phase: updated {updated} evidenced_summary rows")
    return updated
