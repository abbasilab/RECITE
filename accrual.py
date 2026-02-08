"""
accrual.py — Orchestrate accrual pipeline.

Phase 1: Screen documents_local via LLM (directives + impact) → paper_answers.
Match phase (optional): For top papers with no paper_trial_matches, use LLM (Versa) to find and score trials → paper_trial_matches_<preset>.
Phase 2: Top papers + paper_trial_matches_local → enrollment from recite.db → paper_trial_gains.
Phase 3: Summary statistics (per-paper and distribution).
"""

import os
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

from recite.utils.path_loader import get_project_root, resolve_path

app = typer.Typer()


def _load_config(config_path: Path, root: Path) -> dict:
    path = resolve_path(config_path, root)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command()
def run(
    config: Path = typer.Option(Path("config/accrual.yaml"), "--config", "-c", help="Path to accrual YAML config"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max documents to process in Phase 1 (default: all)"),
    skip_phase2: bool = typer.Option(False, "--skip-phase2", help="Only run Phase 1 (paper screening)"),
    skip_phase3: bool = typer.Option(False, "--skip-phase3", help="Skip summary statistics"),
    skip_match_phase: bool = typer.Option(False, "--skip-match-phase", help="Skip paper–trial match phase (find/score trials for top papers without matches)"),
    skip_evidence_phase: bool = typer.Option(False, "--skip-evidence-phase", help="Skip evidenced-summary phase even when enabled in config"),
    backfill_impact: bool = typer.Option(False, "--backfill-impact", help="After Phase 1, re-extract impact (incl. impact_evidence) for rows with null impact_evidence"),
) -> None:
    """Run the accrual pipeline: Phase 1 (paper Q&A), Phase 2 (matches + gains), Phase 3 (summary)."""
    root = get_project_root()
    cfg = _load_config(config, root)

    clintrialm_db_path = resolve_path(Path(cfg.get("clintrialm_db_path", "data/dev/clintrialm.db")), root)
    recite_db_path = resolve_path(Path(cfg.get("recite_db_path", "data/dev/recite.db")), root)
    accrual_db_path = resolve_path(Path(cfg.get("accrual_db_path", "data/dev/accrual.db")), root)
    model_preset = cfg.get("model_preset", "local")
    min_relevance = int(cfg.get("min_relevance", 3))
    min_match_score = int(cfg.get("min_match_score", 3))
    top_matches_per_paper = int(cfg.get("top_matches_per_paper", 10))
    # Phase 1 document cap: CLI --limit overrides config limit
    phase1_limit = limit if limit is not None else cfg.get("limit", 10000)
    llm_cfg = cfg.get("llm") or {}
    endpoint = llm_cfg.get("endpoint", "http://localhost:8000/v1")
    model = llm_cfg.get("model", "local-model")
    api_type = llm_cfg.get("api_type")  # "ucsf_versa" => use UCSF Versa API; else endpoint
    ucsf_versa_model = model if api_type == "ucsf_versa" else None
    prompts_file = cfg.get("prompts_file", "config/accrual_prompts.json")
    prompts_path = resolve_path(Path(prompts_file), root)
    match_trials_phase = cfg.get("match_trials_phase", False)
    match_trial_limit = int(cfg.get("match_trial_limit", 20))
    evidence_summary_phase = cfg.get("evidence_summary_phase", False)
    backfill_impact_evidence = cfg.get("backfill_impact_evidence", False) or backfill_impact

    os.environ["DATABASE_PATH"] = str(clintrialm_db_path)

    from recite.accrual.db import (
        get_paper_answers,
        get_top_papers,
        get_trial_metadata_enrollment,
        init_accrual_db,
        insert_paper_answer,
        insert_paper_trial_gain,
    )
    from recite.accrual.llm import call_accrual_llm
    from recite.accrual.parsing import parse_directives_response, parse_impact_response
    from recite.accrual.prompts import load_accrual_prompts
    from recite.crawler.db import get_relevant_documents, get_paper_trial_matches_for_paper

    init_accrual_db(accrual_db_path)
    prompts = load_accrual_prompts(prompts_path)
    directives_cfg = prompts.get("directives") or {}
    impact_cfg = prompts.get("impact") or {}
    system_directives = directives_cfg.get("system", "")
    user_tpl_directives = directives_cfg.get("user_template", "Title: {title}\n\nAbstract:\n{abstract}")
    system_impact = impact_cfg.get("system", "")
    user_tpl_impact = impact_cfg.get("user_template", "Title: {title}\n\nAbstract:\n{abstract}")

    # Phase 1: documents → LLM (directives + impact) → paper_answers
    docs = get_relevant_documents(
        model_preset=model_preset,
        min_relevance=min_relevance,
        limit=phase1_limit,
    )
    existing = { (r["paper_source"], r["paper_source_id"]) for r in get_paper_answers(accrual_db_path) }
    to_skip = sum(1 for d in docs if (d.get("source"), d.get("source_id")) in existing)
    to_process = len(docs) - to_skip
    logger.info(f"Phase 1: {len(docs)} documents (skip {to_skip} already in DB, process {to_process})")
    processed = 0
    for doc in docs:
        key = (doc.get("source"), doc.get("source_id"))
        if key in existing:
            continue
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        try:
            raw_directives = call_accrual_llm(
                endpoint=endpoint,
                model=model,
                prompt=user_tpl_directives.format(title=title, abstract=abstract),
                system_prompt=system_directives,
                ucsf_versa_model=ucsf_versa_model,
            )
        except Exception as e:
            logger.warning(f"LLM directives failed for {key}: {e}")
            raw_directives = ""
        try:
            raw_impact = call_accrual_llm(
                endpoint=endpoint,
                model=model,
                prompt=user_tpl_impact.format(title=title, abstract=abstract),
                system_prompt=system_impact,
                ucsf_versa_model=ucsf_versa_model,
            )
        except Exception as e:
            logger.warning(f"LLM impact failed for {key}: {e}")
            raw_impact = ""

        try:
            d_parsed = parse_directives_response(raw_directives)
            i_parsed = parse_impact_response(raw_impact)
            insert_paper_answer(
                accrual_db_path=accrual_db_path,
                paper_source=key[0],
                paper_source_id=key[1],
                raw_response_directives=raw_directives or None,
                raw_response_impact=raw_impact or None,
                raw_response_population=None,
                raw_response_eligibility=None,
                directives_answer_location=d_parsed["directives_answer_location"],
                directives_has_answer=d_parsed["directives_has_answer"],
                directives_exact_text=d_parsed.get("directives_exact_text"),
                directives_count=d_parsed.get("directives_count"),
                impact_answer_location=i_parsed["impact_answer_location"],
                impact_has_answer=i_parsed["impact_has_answer"],
                impact_percent=i_parsed.get("impact_percent"),
                impact_absolute=i_parsed.get("impact_absolute"),
                impact_unit=i_parsed.get("impact_unit"),
                impact_qualitative=i_parsed.get("impact_qualitative"),
                impact_evidence=i_parsed.get("impact_evidence"),
            )
            processed += 1
            existing.add(key)
        except Exception as e:
            logger.warning(f"Phase 1: skip {key} (parse/insert failed): {e}. Re-run will retry this document.")
    logger.info(f"Phase 1: inserted/updated {processed} paper_answers")

    # Backfill impact_evidence for existing rows with null impact_evidence (e.g. from before prompt asked for it)
    if backfill_impact_evidence and ucsf_versa_model:
        from recite.accrual.backfill_impact import run_backfill_impact_evidence
        run_backfill_impact_evidence(
            accrual_db_path=accrual_db_path,
            clintrialm_db_path=clintrialm_db_path,
            model_preset=model_preset,
            prompts_path=prompts_path,
            endpoint=endpoint,
            model=model,
            ucsf_versa_model=ucsf_versa_model,
            limit=None,
        )
    elif backfill_impact_evidence and not ucsf_versa_model:
        logger.info("Backfill impact enabled but LLM is not UCSF Versa; skipping backfill")

    if skip_phase2:
        logger.info("Skipping Phase 2 (--skip-phase2)")
        return

    # Match phase (optional): find and score paper–trial matches for top papers that have none
    if not skip_match_phase and match_trials_phase and ucsf_versa_model:
        from recite.accrual.match_phase import run_match_phase
        run_match_phase(
            accrual_db_path=accrual_db_path,
            model_preset=model_preset,
            ucsf_versa_model=ucsf_versa_model,
            match_trial_limit=match_trial_limit,
            min_match_score=min_match_score,
        )
    elif match_trials_phase and not ucsf_versa_model:
        logger.info("Match phase enabled in config but LLM is not UCSF Versa; skipping match phase")

    # Phase 2: top papers → matches → enrollment → scalar_gain → paper_trial_gains
    top_papers = get_top_papers(accrual_db_path, limit=None)
    logger.info(f"Phase 2: {len(top_papers)} top papers")
    gains_count = 0
    for paper in top_papers:
        psrc, pid = paper["paper_source"], paper["paper_source_id"]
        impact_pct = paper.get("impact_percent")
        impact_abs = paper.get("impact_absolute")
        matches = get_paper_trial_matches_for_paper(
            paper_source=psrc,
            paper_source_id=pid,
            model_preset=model_preset,
            min_match_score=min_match_score,
        )
        for m in matches[:top_matches_per_paper]:
            instance_id = m["trial_instance_id"]
            enrollment = get_trial_metadata_enrollment(recite_db_path, instance_id)
            if impact_pct is not None:
                paper_pct_gain = float(impact_pct)
            elif impact_abs is not None and enrollment is not None and enrollment > 0:
                paper_pct_gain = 100.0 * float(impact_abs) / enrollment
            else:
                paper_pct_gain = None
            scalar_gain = None
            if enrollment is not None and paper_pct_gain is not None:
                scalar_gain = enrollment * (paper_pct_gain / 100.0)
            insert_paper_trial_gain(
                accrual_db_path=accrual_db_path,
                paper_source=psrc,
                paper_source_id=pid,
                trial_instance_id=instance_id,
                match_score=m.get("match_score"),
                applicability_score=m.get("applicability_score"),
                enrollment=enrollment,
                paper_pct_gain=paper_pct_gain,
                scalar_gain=scalar_gain,
                amended_ec_text=None,
                change_directive_quote=m.get("change_directive_quote"),
                change_rationale_quote=m.get("change_rationale_quote"),
                evidenced_summary=None,
            )
            gains_count += 1
    logger.info(f"Phase 2: inserted/updated {gains_count} paper_trial_gains")

    # Evidence summary phase (optional): LLM-generated evidenced summary per paper-trial
    if not skip_evidence_phase and evidence_summary_phase and ucsf_versa_model:
        from recite.accrual.summary_phase import run_summary_phase
        run_summary_phase(
            accrual_db_path=accrual_db_path,
            prompts_path=prompts_path,
            only_null_evidenced_summary=True,
            model_preset=model_preset,
            endpoint=endpoint,
            model=model,
            ucsf_versa_model=ucsf_versa_model,
        )
    elif evidence_summary_phase and not ucsf_versa_model:
        logger.info("Evidence summary phase enabled in config but LLM is not UCSF Versa; skipping")

    if skip_phase3:
        logger.info("Skipping Phase 3 (--skip-phase3)")
        return

    # Phase 3: summary statistics
    all_answers = get_paper_answers(accrual_db_path)
    top_again = get_top_papers(accrual_db_path, limit=None)
    pct_gains = [p["impact_percent"] for p in top_again if p.get("impact_percent") is not None]
    if pct_gains:
        mean_pct = sum(pct_gains) / len(pct_gains)
        logger.info(f"Phase 3: papers with impact_percent: n={len(pct_gains)}, mean_pct={mean_pct:.2f}")
    # Scalar gain sum from paper_trial_gains would require a small query; we keep Phase 3 minimal here
    logger.info("Phase 3: summary complete (see paper_answers + paper_trial_gains for details)")


if __name__ == "__main__":
    app()
