"""
Crawl sub-commands: run, stats, review-sync-papers, review-sync-paper-trials, paper-trial-match.
"""
import typer
from loguru import logger
from recite.cli.common import MODEL_PRESETS

app = typer.Typer(help="Literature crawl, stats, review-sync, paper-trial matching.")


@app.command("run")
def crawl_run(
    endpoint: str = typer.Option("http://localhost:8001/v1", help="LLM endpoint (default: wrapper API)"),
    sources: str = typer.Option("pubmed,s2", help="Comma-separated adapters: pubmed, s2"),
    max_papers: int = typer.Option(100),
    relevance_threshold: int = typer.Option(4, help="Min relevance score to trigger seed expansion"),
    max_stale_rounds: int = typer.Option(5, help="Stop after N rounds with no relevant papers"),
    model_preset: str = typer.Option("local", help=f"Model preset: {', '.join(MODEL_PRESETS.keys())}"),
):
    """Crawl literature using LLM-generated queries with seed-based expansion.
    
    Uses multiple adapters (PubMed, Semantic Scholar) and expands search based on
    highly relevant papers found. Continues until max_papers or search exhaustion.
    """
    from recite.crawler.db import (
        init_db, save_document, has_query, save_query, 
        get_used_queries, has_document, get_relevant_documents
    )
    from recite.crawler.adapters import (
        ADAPTERS, generate_queries, generate_seed_queries, evaluate_paper
    )
    from recite.crawler.llm import LLMClient
    
    # Validate model preset
    if model_preset not in MODEL_PRESETS:
        typer.echo(f"Unknown preset: {model_preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
        raise typer.Exit(1)
    
    init_db(model_preset)
    from recite.crawler.llm import _get_cache_dir
    llm = LLMClient(endpoint, cache_dir=_get_cache_dir())
    logger.info(f"Using model preset: {model_preset}")
    
    # Parse adapters
    adapter_names = [s.strip() for s in sources.split(",")]
    adapters = {name: ADAPTERS[name] for name in adapter_names if name in ADAPTERS}
    if not adapters:
        typer.echo(f"No valid adapters. Available: {list(ADAPTERS.keys())}")
        raise typer.Exit(1)
    logger.info(f"Using adapters: {list(adapters.keys())}")
    
    count = 0
    stale_rounds = 0
    seed_queue: list[tuple] = []  # (doc, source_name) pairs for seed expansion
    
    while count < max_papers:
        found_relevant = False
        
        # Process each adapter
        for source_name, adapter in adapters.items():
            if count >= max_papers:
                break
            
            # Generate queries: either from seed or general
            if seed_queue:
                # Seed-based expansion from a relevant paper
                seed_doc, _ = seed_queue.pop(0)
                logger.info(f"[SEED] Expanding from: {seed_doc.title[:50]}...")
                queries = generate_seed_queries(llm, seed_doc, adapter.instructions)
            else:
                # General query generation
                used = get_used_queries(source_name, model_preset)
                queries = generate_queries(llm, adapter.instructions, used)
            
            if not queries:
                logger.warning(f"No queries generated for {source_name} - LLM call may have failed")
                continue
            
            # Filter already-used queries
            new_queries = [q for q in queries if not has_query(source_name, q, model_preset)]
            if not new_queries:
                logger.debug(f"All queries already used for {source_name}")
                continue
            
            # Process queries
            for query in new_queries:
                if count >= max_papers:
                    break
                    
                save_query(source_name, query, model_preset)
                logger.info(f"[{source_name}] Searching: {query}")
                
                for doc in adapter.search(query):
                    if count >= max_papers:
                        break
                    
                    # Skip if already have this document (by source+id)
                    if has_document(doc.source, doc.source_id, model_preset):
                        continue
                    
                    # Evaluate paper
                    scores = evaluate_paper(llm, doc)
                    if scores.reasoning == "evaluation failed":
                        logger.warning(f"Failed to evaluate paper: {doc.title[:50]}... - LLM call may have failed")
                    save_document(doc, scores, model_preset)
                    count += 1
                    
                    rel_marker = "***" if scores.relevance >= relevance_threshold else ""
                    logger.info(f"[{count}] {doc.title[:50]}... rel={scores.relevance} {rel_marker}")
                    
                    # Queue highly relevant papers for seed expansion
                    if scores.relevance >= relevance_threshold:
                        found_relevant = True
                        seed_queue.append((doc, source_name))
                        logger.info(f"  -> Queued for seed expansion ({len(seed_queue)} in queue)")
        
        # Check for stale rounds (no relevant papers found)
        if found_relevant:
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= max_stale_rounds and not seed_queue:
                logger.info(f"Stopping: {stale_rounds} rounds with no relevant papers and no seeds queued")
                break
    
    logger.info(f"Crawl complete: {count} papers, {len(seed_queue)} seeds remaining")


@app.command("stats")
def stats_cmd(
    model_preset: str = typer.Option(None, help=f"Show stats for specific model preset, or omit for aggregate across all models"),
):
    """Show database statistics.
    
    If model_preset is provided, shows stats for that model only.
    Otherwise, shows aggregate stats across all models with per-model breakdown.
    """
    from recite.crawler.db import get_stats, get_all_model_presets
    
    if model_preset:
        # Single model stats
        if model_preset not in MODEL_PRESETS:
            typer.echo(f"Unknown preset: {model_preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
            raise typer.Exit(1)
        s = get_stats(model_preset)
        typer.echo(f"Model: {model_preset}")
        typer.echo(f"Documents: {s['documents']}")
        typer.echo(f"Relevant (≥4): {s['relevant']}")
        typer.echo(f"Queries used: {s['prompts']}")
        typer.echo(f"Avg relevance: {s['avg_relevance']:.1f}")
    else:
        # Aggregate stats with per-model breakdown
        s = get_stats()
        typer.echo("=== Aggregate Stats (All Models) ===")
        typer.echo(f"Total Documents: {s['documents']}")
        typer.echo(f"Total Relevant (≥4): {s['relevant']}")
        typer.echo(f"Total Queries: {s['prompts']}")
        typer.echo(f"Overall Avg Relevance: {s['avg_relevance']:.1f}")
        
        # Per-model breakdown
        presets = get_all_model_presets()
        if presets:
            typer.echo("\n=== Per-Model Breakdown ===")
            for preset in presets:
                ps = get_stats(preset)
                typer.echo(f"\n{preset}:")
                typer.echo(f"  Documents: {ps['documents']}")
                typer.echo(f"  Relevant (≥4): {ps['relevant']}")
                typer.echo(f"  Avg relevance: {ps['avg_relevance']:.1f}")


@app.command("review-sync-papers")
def review_sync_papers(
    model_preset: str = typer.Option(..., help=f"Model preset to use for reviewing: {', '.join(MODEL_PRESETS.keys())}"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="LLM endpoint (default: wrapper API)"),
    limit: int = typer.Option(None, help="Max documents to review (optional)"),
    min_relevance: int = typer.Option(3, help="Only review documents with relevance >= this from other models"),
    skip_recent_hours: float = typer.Option(24.0, help="Skip completed documents updated within last N hours (0 = re-run all)"),
    re_review_own: bool = typer.Option(False, "--re-review-own", help="Archive and re-review this model's own documents along with other models"),
):
    """Review papers/documents from other model tables with the specified model.
    
    Collects all documents from other model tables, deduplicates them, then
    re-evaluates with the specified model preset. Saves results to the
    model's own table.
    
    When --re-review-own is used:
    1. Archives the live documents table to archive.db
    2. Merges all backup tables from archive.db
    3. Filters by completion and time window (skip_recent_hours)
    4. Restores completed recent entries to live table
    5. Re-evaluates entries outside window or incomplete
    """
    from recite.crawler.db import (
        init_db, save_document, has_document,
        get_all_documents_from_other_models, get_all_model_presets,
        archive_model_table, _get_table_name, get_document_by_source_id,
        merge_documents_from_archive, filter_completed_within_hours
    )
    from recite.crawler.adapters import evaluate_paper, Document, Scores
    from recite.crawler.llm import LLMClient, _get_cache_dir
    import sqlite3
    
    # Validate model preset
    if model_preset not in MODEL_PRESETS:
        typer.echo(f"Unknown preset: {model_preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
        raise typer.Exit(1)
    
    # Archive own table if flag is set
    if re_review_own:
        from recite.crawler.db import _get_conn
        table_name = _get_table_name(model_preset)
        with _get_conn() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            if not cursor.fetchone():
                typer.echo(f"Error: Table {table_name} does not exist. Cannot archive.")
                raise typer.Exit(1)
        
        archive_name = archive_model_table(model_preset)
        typer.echo(f"Archived {table_name} to {archive_name}")
        logger.info(f"Archived {table_name} to {archive_name} for re-review")
    
    init_db(model_preset)
    from recite.crawler.llm import _get_cache_dir
    llm = LLMClient(endpoint, cache_dir=_get_cache_dir())
    
    # Merge-based resume logic
    if re_review_own:
        # Merge all backup tables from archive.db
        all_docs = merge_documents_from_archive(model_preset)
        
        # Filter by completion and time window
        completed_recent, to_review = filter_completed_within_hours(
            all_docs, skip_recent_hours, "documents"
        )
        
        # Filter by min_relevance
        completed_recent = [d for d in completed_recent if d.get("relevance", 0) >= min_relevance]
        to_review = [d for d in to_review if d.get("relevance", 0) >= min_relevance]
        
        # Restore completed recent entries to live table (no re-evaluation)
        restored = 0
        for doc_dict in completed_recent:
            try:
                doc = Document(
                    source=doc_dict["source"],
                    source_id=doc_dict["source_id"],
                    title=doc_dict["title"],
                    abstract=doc_dict.get("abstract"),
                    doi=doc_dict.get("doi"),
                    pmid=doc_dict.get("pmid"),
                )
                scores = Scores(
                    relevance=doc_dict["relevance"],
                    extraction_confidence=doc_dict["extraction_confidence"],
                    accrual_ease=doc_dict["accrual_ease"],
                    reasoning=doc_dict.get("reasoning", ""),
                )
                save_document(doc, scores, model_preset)
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to restore document {doc_dict.get('source_id')}: {e}")
        
        if restored > 0:
            typer.echo(f"Restored {restored} completed documents from last {skip_recent_hours} hours (skipped re-review)")
        
        # Re-evaluate entries outside window or incomplete
        all_docs = to_review
        if limit:
            all_docs = all_docs[:limit]
    else:
        # Standard mode: get documents from other models
        all_docs = get_all_documents_from_other_models(model_preset, min_relevance)
        
        # Apply skip_recent_hours filter: check if documents exist in current model's table
        if skip_recent_hours > 0:
            # Get existing documents from current model's table
            existing_docs = []
            for doc_data in all_docs:
                existing = get_document_by_source_id(
                    model_preset, doc_data["source"], doc_data["source_id"]
                )
                if existing:
                    existing_docs.append(existing)
            
            # Filter existing documents by completion and time window
            completed_recent_existing, _ = filter_completed_within_hours(
                existing_docs, skip_recent_hours, "documents"
            )
            
            # Create set of (source, source_id) to skip
            skip_set = {(d["source"], d["source_id"]) for d in completed_recent_existing}
            
            # Filter all_docs to exclude skipped ones
            all_docs = [d for d in all_docs if (d["source"], d["source_id"]) not in skip_set]
    
    if not all_docs:
        if re_review_own:
            typer.echo(f"No documents found to review after filtering (min_relevance={min_relevance}, skip_recent_hours={skip_recent_hours})")
        else:
            typer.echo(f"No documents found to review (min_relevance={min_relevance}, skip_recent_hours={skip_recent_hours})")
        raise typer.Exit(1)
    
    typer.echo(f"Reviewing with model: {model_preset}")
    if re_review_own:
        typer.echo(f"Including archived own table in review")
    typer.echo(f"Found {len(all_docs)} unique documents to review")
    
    if limit:
        all_docs = all_docs[:limit]
        typer.echo(f"Limited to {limit} documents")
    
    reviewed = 0
    skipped = 0
    
    for doc_data in all_docs:
        # Create Document object
        doc = Document(
            source=doc_data["source"],
            source_id=doc_data["source_id"],
            title=doc_data["title"],
            abstract=doc_data.get("abstract"),
            doi=doc_data.get("doi"),
            pmid=doc_data.get("pmid"),
        )
        
        # Re-evaluate with current model
        scores = evaluate_paper(llm, doc)
        save_document(doc, scores, model_preset)
        reviewed += 1
        
        rel_marker = "***" if scores.relevance >= 4 else ""
        logger.info(f"[{reviewed}] {doc.title[:50]}... rel={scores.relevance} {rel_marker}")
    
    typer.echo(f"\nReview complete:")
    typer.echo(f"  Reviewed: {reviewed}")
    typer.echo(f"  Skipped (already reviewed): {skipped}")


@app.command("review-sync-paper-trials")
def review_sync_paper_trials(
    model_preset: str = typer.Option(..., help=f"Model preset to use for reviewing: {', '.join(MODEL_PRESETS.keys())}"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="LLM endpoint (default: wrapper API)"),
    limit: int = typer.Option(None, help="Max matches to review (optional)"),
    min_match_score: int = typer.Option(1, help="Only review matches with score >= this"),
    skip_recent_hours: float = typer.Option(24.0, help="Skip completed matches updated within last N hours (0 = re-run all)"),
    re_review_own: bool = typer.Option(False, "--re-review-own", help="Archive and re-review this model's own matches"),
):
    """Review and re-evaluate existing paper-trial matches with the specified model.
    
    When --re-review-own is used:
    1. Archives the live matches table to archive.db
    2. Merges all backup tables from archive.db
    3. Filters by completion and time window (skip_recent_hours)
    4. Restores completed recent entries to live table
    5. Re-evaluates entries outside window or incomplete
    """
    from recite.crawler.db import (
        init_db, init_paper_trial_matches_table,
        get_all_paper_trial_matches, update_paper_trial_match, save_paper_trial_match,
        get_document_by_source_id, get_all_model_presets,
        archive_paper_trial_matches_table, merge_matches_from_archive, filter_completed_within_hours,
        _get_paper_trial_matches_table_name, _get_conn,
    )
    from recite.crawler.adapters import Document, ClinicalTrialsGovAdapter
    from recite.crawler.llm import LLMClient, _get_cache_dir
    from recite.crawler.paper_trial_matcher import evaluate_match
    import sqlite3
    
    # Validate model preset
    if model_preset not in MODEL_PRESETS:
        typer.echo(f"Unknown preset: {model_preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
        raise typer.Exit(1)
    
    # Archive own table if flag is set
    if re_review_own:
        table_name = _get_paper_trial_matches_table_name(model_preset)
        with _get_conn() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            if not cursor.fetchone():
                typer.echo(f"Error: Table {table_name} does not exist. Cannot archive.")
                raise typer.Exit(1)
        
        archive_name = archive_paper_trial_matches_table(model_preset)
        typer.echo(f"Archived {table_name} to {archive_name} in archive.db")
        logger.info(f"Archived {table_name} to {archive_name} for re-review")
    
    init_db(model_preset)
    init_paper_trial_matches_table(model_preset)
    from recite.crawler.llm import _get_cache_dir
    llm = LLMClient(endpoint, cache_dir=_get_cache_dir())
    adapter = ClinicalTrialsGovAdapter()
    
    # Merge-based resume logic
    if re_review_own:
        # Merge all backup tables from archive.db
        all_matches = merge_matches_from_archive(model_preset)
        
        # Filter by completion and time window
        completed_recent, to_review = filter_completed_within_hours(
            all_matches, skip_recent_hours, "matches"
        )
        
        # Filter by min_match_score
        completed_recent = [m for m in completed_recent if m.get("match_score", 0) >= min_match_score]
        to_review = [m for m in to_review if m.get("match_score", 0) >= min_match_score]
        
        # Restore completed recent entries to live table (no re-evaluation)
        restored = 0
        for match in completed_recent:
            try:
                save_paper_trial_match(
                    paper_source=match["paper_source"],
                    paper_source_id=match["paper_source_id"],
                    paper_title=match["paper_title"],
                    trial_nct_id=match["trial_nct_id"],
                    trial_title=match["trial_title"],
                    match_score=match["match_score"],
                    match_reasoning=match.get("match_reasoning", ""),
                    applicability_score=match["applicability_score"],
                    change_directive_quote=match.get("change_directive_quote", ""),
                    change_rationale_quote=match.get("change_rationale_quote", ""),
                    model_preset=model_preset,
                )
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to restore match {match.get('paper_source_id')} <-> {match.get('trial_nct_id')}: {e}")
        
        if restored > 0:
            typer.echo(f"Restored {restored} completed matches from last {skip_recent_hours} hours (skipped re-review)")
        
        # Re-evaluate entries outside window or incomplete
        matches = to_review
        if limit:
            matches = matches[:limit]
    else:
        # Standard mode: get matches from live table only
        matches = get_all_paper_trial_matches(model_preset, min_match_score=min_match_score, limit=limit)
    
    if not matches:
        if re_review_own:
            typer.echo(f"No matches found to review after filtering (min_match_score={min_match_score}, skip_recent_hours={skip_recent_hours})")
        else:
            typer.echo(f"No matches found to review (min_match_score={min_match_score})")
        raise typer.Exit(1)
    
    typer.echo(f"Reviewing with model: {model_preset}")
    typer.echo(f"Found {len(matches)} matches to review")
    
    updated = 0
    skipped = 0
    failed = 0
    
    for match in matches:
        try:
            # Fetch paper document (searches current preset first, then other presets)
            paper_data = get_document_by_source_id(
                model_preset,
                match["paper_source"],
                match["paper_source_id"],
            )
            
            if not paper_data:
                logger.warning(f"Paper not found: {match['paper_source']}/{match['paper_source_id']}")
                skipped += 1
                continue
            
            # Create Document object for paper
            paper = Document(
                source=paper_data["source"],
                source_id=paper_data["source_id"],
                title=paper_data["title"],
                abstract=paper_data.get("abstract"),
                doi=paper_data.get("doi"),
                pmid=paper_data.get("pmid"),
            )
            
            # Fetch trial document from CTG
            trial = adapter.fetch_by_nct_id(match["trial_nct_id"])
            if not trial:
                logger.warning(f"Trial not found: {match['trial_nct_id']}")
                skipped += 1
                continue
            
            # Re-evaluate match
            score = evaluate_match(llm, paper, trial)
            if score is None:
                logger.warning(f"Failed to evaluate match: {match['paper_source_id']} <-> {match['trial_nct_id']}")
                failed += 1
                continue
            
            # Update match in database
            updated_success = update_paper_trial_match(
                paper_source=match["paper_source"],
                paper_source_id=match["paper_source_id"],
                trial_nct_id=match["trial_nct_id"],
                match_score=score.match_score,
                match_reasoning=score.match_reasoning,
                applicability_score=score.applicability_score,
                change_directive_quote=score.change_directive_quote,
                change_rationale_quote=score.change_rationale_quote,
                model_preset=model_preset,
                paper_title=match.get("paper_title", paper.title),
                trial_title=match.get("trial_title", trial.title),
            )
            
            if updated_success:
                updated += 1
                logger.info(
                    f"[{updated}] Updated match: {match['paper_source_id']} <-> {match['trial_nct_id']} "
                    f"(score={score.match_score}, applicability={score.applicability_score})"
                )
            else:
                logger.warning(f"Failed to update match: {match['paper_source_id']} <-> {match['trial_nct_id']}")
                failed += 1
        
        except Exception as e:
            logger.error(f"Error reviewing match {match.get('paper_source_id', 'unknown')} <-> {match.get('trial_nct_id', 'unknown')}: {e}")
            failed += 1
            continue
    
    typer.echo(f"\nReview complete:")
    typer.echo(f"  Updated: {updated}")
    typer.echo(f"  Skipped (paper/trial not found): {skipped}")
    typer.echo(f"  Failed (evaluation/update error): {failed}")


@app.command("paper-trial-match")
def paper_trial_match(
    model_preset: str = typer.Option(..., help=f"Model preset: {', '.join(MODEL_PRESETS.keys())}"),
    endpoint: str = typer.Option("http://localhost:8001/v1", help="LLM endpoint (default: wrapper API)"),
    min_relevance: int = typer.Option(4, help="Only match papers with relevance >= this"),
    presets: str = typer.Option(None, help="Comma-separated list of presets to query (e.g., 'local,cluster-72b'). If not provided, uses model_preset only."),
    min_extraction_confidence: int = typer.Option(None, help="Min extraction_confidence to match (1-5)"),
    min_accrual_ease: int = typer.Option(None, help="Min accrual_ease to match (1-5)"),
    min_total_score: int = typer.Option(None, help="Min total_score to match (3-15)"),
    sort_by: str = typer.Option("total_score", help="Sort papers by: relevance, extraction_confidence, accrual_ease, total_score"),
    top_k_trials: int = typer.Option(10, help="Max trials to evaluate per paper (each evaluation is an LLM call; not all evaluated trials will match; increase if you want more matches)"),
    batch_size: int = typer.Option(10, help="Papers per batch"),
    limit: int = typer.Option(None, help="Max papers to process"),
    resume: bool = typer.Option(True, help="Skip papers that already have matches"),
    seed_expansion: bool = typer.Option(True, help="Use high-quality matches to generate more queries"),
    seed_threshold_match: int = typer.Option(4, help="Min match_score for seed expansion"),
    seed_threshold_applicability: int = typer.Option(4, help="Min applicability_score for seed expansion"),
    show_queries: bool = typer.Option(False, help="Display previously used CTG queries"),
    min_match_score: int = typer.Option(1, help="Min match_score to save (1=save all)"),
):
    """Match research papers to relevant clinical trials.
    
    For each paper in the database, generates search queries, finds candidate trials
    from ClinicalTrials.gov, and evaluates matches using LLM-based scoring.
    
    Papers are filtered and sorted by quality scores (relevance, extraction_confidence,
    accrual_ease, total_score) to prioritize the most actionable papers.
    
    High-quality matches (meeting seed thresholds) can trigger seed expansion to
    find additional related trials.
    """
    from recite.crawler.db import (
        init_db, get_relevant_documents, get_used_queries,
        init_paper_trial_matches_table, get_papers_with_matches,
        get_paper_trial_matches_stats,
    )
    from recite.crawler.adapters import Document, ClinicalTrialsGovAdapter
    from recite.crawler.llm import LLMClient
    from recite.crawler.paper_trial_matcher import (
        match_paper_to_trials, generate_seed_trial_queries,
        is_high_quality_match,
    )
    
    # Determine which presets to query
    if presets:
        # Parse comma-separated presets
        preset_list = [p.strip() for p in presets.split(",") if p.strip()]
        if not preset_list:
            # Empty string means query all active presets
            preset_list = None
        else:
            # Validate all presets
            for preset in preset_list:
                if preset not in MODEL_PRESETS:
                    typer.echo(f"Unknown preset in --presets: {preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
                    raise typer.Exit(1)
    else:
        # Use single model_preset
        preset_list = model_preset
        if model_preset not in MODEL_PRESETS:
            typer.echo(f"Unknown preset: {model_preset}. Available: {', '.join(MODEL_PRESETS.keys())}")
            raise typer.Exit(1)
    
    # Initialize (always initialize the primary model_preset for matches table)
    init_db(model_preset)
    init_paper_trial_matches_table(model_preset)
    from recite.crawler.llm import _get_cache_dir
    llm = LLMClient(endpoint, cache_dir=_get_cache_dir())
    adapter = ClinicalTrialsGovAdapter()
    
    # Display which presets are being queried
    if isinstance(preset_list, list):
        logger.info(f"Querying presets: {', '.join(preset_list)}")
        typer.echo(f"Querying presets: {', '.join(preset_list)}")
    elif preset_list is None:
        logger.info("Querying all active presets")
        typer.echo("Querying all active presets")
    else:
        logger.info(f"Using model preset: {model_preset}")
        typer.echo(f"Using model preset: {model_preset}")
    
    # Show query history if requested
    if show_queries:
        used_queries = get_used_queries("ctg", model_preset)
        if used_queries:
            typer.echo(f"\n=== Previously used CTG queries ({len(used_queries)}) ===")
            for i, q in enumerate(used_queries[-20:], 1):  # Show last 20
                typer.echo(f"  {i}. {q}")
            if len(used_queries) > 20:
                typer.echo(f"  ... and {len(used_queries) - 20} more")
            typer.echo()
        else:
            typer.echo("No CTG queries used yet.\n")
    
    # Build filter criteria message
    filter_parts = [f"relevance>={min_relevance}"]
    if min_extraction_confidence is not None:
        filter_parts.append(f"extraction_confidence>={min_extraction_confidence}")
    if min_accrual_ease is not None:
        filter_parts.append(f"accrual_ease>={min_accrual_ease}")
    if min_total_score is not None:
        filter_parts.append(f"total_score>={min_total_score}")
    filter_msg = ", ".join(filter_parts)
    
    # Get papers to match with filtering and sorting
    papers_data = get_relevant_documents(
        min_relevance=min_relevance,
        limit=limit or 1000,
        model_preset=preset_list,
        min_extraction_confidence=min_extraction_confidence,
        min_accrual_ease=min_accrual_ease,
        min_total_score=min_total_score,
        sort_by=sort_by,
        sort_desc=True,
    )
    
    if not papers_data:
        typer.echo(f"No papers found with {filter_msg}")
        raise typer.Exit(0)
    
    typer.echo(f"Found {len(papers_data)} papers with {filter_msg} (sorted by {sort_by})")
    
    # Filter out already-matched papers if resuming
    papers_with_matches = set()
    if resume:
        papers_with_matches = get_papers_with_matches(model_preset)
        original_count = len(papers_data)
        papers_data = [
            p for p in papers_data 
            if (p["source"], p["source_id"]) not in papers_with_matches
        ]
        skipped = original_count - len(papers_data)
        if skipped > 0:
            typer.echo(f"Resuming: skipping {skipped} papers that already have matches")
    
    if not papers_data:
        typer.echo("All papers already have matches. Nothing to do.")
        raise typer.Exit(0)
    
    if limit and len(papers_data) > limit:
        papers_data = papers_data[:limit]
    
    typer.echo(f"Processing {len(papers_data)} papers...")
    
    # Statistics tracking
    stats = {
        "papers_processed": 0,
        "papers_failed": 0,
        "total_matches": 0,
        "high_quality_matches": 0,
        "seed_expansions": 0,
    }
    
    # Seed queue for expansion
    seed_queue: list = []
    
    # Process papers in batches
    for batch_start in range(0, len(papers_data), batch_size):
        batch_end = min(batch_start + batch_size, len(papers_data))
        batch = papers_data[batch_start:batch_end]
        
        typer.echo(f"\n=== Batch {batch_start // batch_size + 1}: papers {batch_start + 1}-{batch_end} ===")
        
        for paper_data in batch:
            # Create Document object
            paper = Document(
                source=paper_data["source"],
                source_id=paper_data["source_id"],
                title=paper_data["title"],
                abstract=paper_data.get("abstract"),
                doi=paper_data.get("doi"),
                pmid=paper_data.get("pmid"),
            )
            
            typer.echo(f"\nMatching: {paper.title[:60]}...")
            
            try:
                # Run matching pipeline
                matches_for_paper = 0
                for match in match_paper_to_trials(
                    paper=paper,
                    adapter=adapter,
                    llm=llm,
                    model_preset=model_preset,
                    top_k=top_k_trials,
                    min_match_score=min_match_score,
                ):
                    matches_for_paper += 1
                    stats["total_matches"] += 1
                    
                    # Check for high-quality match
                    if is_high_quality_match(match, seed_threshold_match, seed_threshold_applicability):
                        stats["high_quality_matches"] += 1
                        
                        # Queue for seed expansion if enabled
                        if seed_expansion:
                            seed_queue.append(match)
                            logger.debug(f"Queued match for seed expansion: {match.trial_nct_id}")
                
                stats["papers_processed"] += 1
                typer.echo(f"  -> {matches_for_paper} matches found")
                
            except Exception as e:
                stats["papers_failed"] += 1
                logger.error(f"Failed to match paper {paper.source_id}: {e}")
                typer.echo(f"  -> Error: {e}")
                continue
        
        # Process seed queue after each batch (if enabled)
        if seed_expansion and seed_queue:
            typer.echo(f"\n--- Seed expansion: {len(seed_queue)} high-quality matches ---")
            
            used_queries = get_used_queries("ctg", model_preset)
            
            for seed_match in seed_queue:
                try:
                    # Generate additional queries from seed
                    new_queries = generate_seed_trial_queries(
                        llm=llm,
                        seed_match=seed_match,
                        used_queries=used_queries,
                    )
                    
                    if new_queries:
                        stats["seed_expansions"] += 1
                        logger.info(f"Seed expansion generated {len(new_queries)} queries from match {seed_match.trial_nct_id}")
                        
                        # Save new queries (they'll be used in next paper's matching)
                        from recite.crawler.db import save_query, has_query
                        for query in new_queries:
                            if not has_query("ctg", query, model_preset):
                                save_query("ctg", query, model_preset)
                                used_queries.append(query)
                
                except Exception as e:
                    logger.warning(f"Seed expansion failed for match {seed_match.trial_nct_id}: {e}")
            
            seed_queue.clear()
        
        # Progress update
        typer.echo(f"\nProgress: {stats['papers_processed']}/{len(papers_data)} papers, {stats['total_matches']} matches")
    
    # Final statistics
    typer.echo("\n" + "=" * 50)
    typer.echo("=== Matching Complete ===")
    typer.echo(f"Papers processed: {stats['papers_processed']}")
    typer.echo(f"Papers failed: {stats['papers_failed']}")
    typer.echo(f"Total matches saved: {stats['total_matches']}")
    typer.echo(f"High-quality matches: {stats['high_quality_matches']}")
    if seed_expansion:
        typer.echo(f"Seed expansions: {stats['seed_expansions']}")
    
    # Show database stats
    match_stats = get_paper_trial_matches_stats(model_preset)
    typer.echo("\n=== Database Stats ===")
    typer.echo(f"Total matches in DB: {match_stats['total_matches']}")
    typer.echo(f"Unique papers matched: {match_stats['unique_papers']}")
    typer.echo(f"Unique trials matched: {match_stats['unique_trials']}")
    typer.echo(f"Avg match score: {match_stats['avg_match_score']}")
    typer.echo(f"Avg applicability score: {match_stats['avg_applicability_score']}")
    typer.echo(f"High-quality matches (score>=4, applicability>=4): {match_stats['high_quality_matches']}")

