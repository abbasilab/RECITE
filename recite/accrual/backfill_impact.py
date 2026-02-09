"""
Backfill impact_evidence for paper_answers rows that have null/empty impact_evidence.

Re-calls the impact LLM using title/abstract from clintrialm documents_<preset>
(direct DB query so config path is used), then updates paper_answers.
Used by the pipeline (--backfill-impact / config) and by scripts/re_extract_accrual_impact_evidence.py.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from loguru import logger

from recite.accrual.db import (
    get_paper_answers,
    update_paper_answer_impact,
)
from recite.accrual.llm import call_accrual_llm
from recite.accrual.parsing import parse_impact_response
from recite.accrual.prompts import load_accrual_prompts


def _get_doc_from_clintrialm(
    clintrialm_db_path: Path,
    paper_source: str,
    paper_source_id: str,
    model_preset: str,
) -> Optional[dict]:
    """Get document from clintrialm DB (direct query so we use config path)."""
    table = f"documents_{model_preset.replace('-', '_')}"
    conn = sqlite3.connect(str(clintrialm_db_path))
    try:
        row = conn.execute(
            f"SELECT source, source_id, title, abstract FROM {table} WHERE source=? AND source_id=?",
            (paper_source, paper_source_id),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {"source": row[0], "source_id": row[1], "title": row[2] or "", "abstract": row[3] or ""}


def run_backfill_impact_evidence(
    accrual_db_path: Path,
    clintrialm_db_path: Path,
    model_preset: str,
    prompts_path: Path,
    endpoint: str,
    model: str,
    ucsf_versa_model: Optional[str],
    limit: Optional[int] = None,
) -> int:
    """
    Re-call the impact LLM for paper_answers rows with null/empty impact_evidence,
    then update those rows. Returns the number of rows updated.
    """
    prompts = load_accrual_prompts(prompts_path)
    impact_cfg = prompts.get("impact") or {}
    system_impact = impact_cfg.get("system", "")
    user_tpl_impact = impact_cfg.get("user_template", "Title: {title}\n\nAbstract:\n{abstract}")
    if not user_tpl_impact:
        logger.warning("impact user_template missing in prompts; skipping backfill")
        return 0

    all_answers = get_paper_answers(accrual_db_path)
    to_process = [r for r in all_answers if not (r.get("impact_evidence") or "").strip()]
    if limit is not None:
        to_process = to_process[:limit]
    if not to_process:
        logger.info("Backfill impact_evidence: no rows with null/empty impact_evidence")
        return 0

    logger.info(f"Backfill impact_evidence: re-extracting for {len(to_process)} rows")
    updated = 0
    for row in to_process:
        psrc = row["paper_source"]
        pid = row["paper_source_id"]
        doc = _get_doc_from_clintrialm(clintrialm_db_path, psrc, pid, model_preset)
        if not doc:
            logger.warning(f"No document in clintrialm for {psrc}/{pid[:16]}...; skip")
            continue
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        try:
            raw_impact = call_accrual_llm(
                endpoint=endpoint,
                model=model,
                prompt=user_tpl_impact.format(title=title, abstract=abstract),
                system_prompt=system_impact,
                ucsf_versa_model=ucsf_versa_model,
            )
        except Exception as e:
            logger.warning(f"LLM impact failed for {psrc}/{pid[:16]}...: {e}")
            continue
        i_parsed = parse_impact_response(raw_impact or "")
        update_paper_answer_impact(
            accrual_db_path=accrual_db_path,
            paper_source=psrc,
            paper_source_id=pid,
            raw_response_impact=raw_impact or None,
            impact_answer_location=i_parsed["impact_answer_location"],
            impact_has_answer=i_parsed["impact_has_answer"],
            impact_percent=i_parsed.get("impact_percent"),
            impact_absolute=i_parsed.get("impact_absolute"),
            impact_unit=i_parsed.get("impact_unit"),
            impact_qualitative=i_parsed.get("impact_qualitative"),
            impact_evidence=i_parsed.get("impact_evidence"),
        )
        updated += 1
    logger.info(f"Backfill impact_evidence: updated {updated} rows")
    return updated
