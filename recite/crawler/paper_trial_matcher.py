"""
paper_trial_matcher.py

Core logic for matching research papers to clinical trials.
Uses LLM-based evaluation to assess match quality and applicability.
"""
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from recite.crawler.adapters import (
    Document,
    AdapterInstructions,
    ClinicalTrialsGovAdapter,
    CTG_INSTRUCTIONS,
    _summarize_used_queries,
)
from recite.crawler.llm import LLMClient
from recite.crawler.db import (
    has_query,
    save_query,
    get_used_queries,
    has_paper_trial_match,
    save_paper_trial_match,
)


# Default max characters for context truncation (~4 chars/token)
DEFAULT_MAX_CHARS = 6000  # ~1500 tokens


@dataclass
class MatchScore:
    """Scores and reasoning for a paper-trial match."""
    match_score: int  # 1-5 overall relevance
    match_reasoning: str  # Explanation of match quality
    applicability_score: int  # 1-5 how easy to apply the change
    change_directive_quote: str  # Verbatim quote: HOW/WHAT to change
    change_rationale_quote: str  # Verbatim quote: WHY to change


@dataclass
class Match:
    """A matched paper-trial pair with scores."""
    paper_source: str
    paper_source_id: str
    paper_title: str
    trial_instance_id: str
    trial_title: str
    match_score: int
    match_reasoning: str
    applicability_score: int
    change_directive_quote: str
    change_rationale_quote: str


def truncate_text_smart(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Truncate text at sentence boundaries to fit within limits.
    
    Uses 'smart' strategy: tries to cut at sentence/paragraph boundaries
    while preserving at least 70% of the allowed content.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters (rough proxy for tokens, ~4 chars/token)
    
    Returns:
        Truncated text with ellipsis marker if truncated
    """
    if not text or len(text) <= max_chars:
        return text or ""
    
    truncated = text[:max_chars]
    
    # Find last sentence ending (period + space or newline)
    last_period = truncated.rfind('. ')
    last_newline = truncated.rfind('\n')
    cut_point = max(last_period, last_newline)
    
    # Only use smart cut if we keep at least 70% of content
    if cut_point > max_chars * 0.7:
        truncated = truncated[:cut_point + 1]
    
    return truncated.rstrip() + "\n\n[... content truncated ...]"


def generate_trial_queries(
    llm: LLMClient,
    paper: Document,
    used_queries: list[str],
    instructions: AdapterInstructions = CTG_INSTRUCTIONS,
) -> list[str]:
    """Generate CTG search queries from a paper to find relevant trials.
    
    Args:
        llm: LLM client for query generation
        paper: Paper document to generate queries from
        used_queries: Previously used queries to avoid duplicates
        instructions: CTG adapter instructions (default: CTG_INSTRUCTIONS)
    
    Returns:
        List of generated query strings
    """
    used_text = _summarize_used_queries(used_queries)
    
    # Truncate abstract if needed
    abstract = paper.abstract or "(no abstract)"
    if len(abstract) > 1500:
        abstract = abstract[:1500] + "..."
    
    prompt = llm.prompts.match_query_user.format(
        paper_title=paper.title,
        paper_abstract=abstract,
        used_queries=used_text,
    )
    
    result = llm.complete_json(prompt, llm.prompts.match_query_system)
    queries = result.get("queries", []) if result else []
    
    # Filter out duplicates
    new_queries = [q for q in queries if q not in used_queries]
    logger.debug(f"Generated {len(new_queries)} new queries for paper: {paper.title[:50]}...")
    
    return new_queries


def generate_seed_trial_queries(
    llm: LLMClient,
    seed_match: Match,
    used_queries: list[str],
    trial_condition: str = "",
    trial_intervention: str = "",
    instructions: AdapterInstructions = CTG_INSTRUCTIONS,
) -> list[str]:
    """Generate additional queries based on a high-quality match (seed expansion).
    
    When a match has high match_score and applicability_score, use that context
    to explore similar trials, related conditions, and adjacent research areas.
    
    Args:
        llm: LLM client for query generation
        seed_match: High-quality match to use as seed
        used_queries: Previously used queries to avoid duplicates
        trial_condition: Condition from the matched trial (if available)
        trial_intervention: Intervention from the matched trial (if available)
        instructions: CTG adapter instructions
    
    Returns:
        List of generated query strings
    """
    used_text = _summarize_used_queries(used_queries)
    
    # Truncate abstract for prompt
    # Note: We only have paper_title in Match, so we construct what we can
    paper_abstract = ""  # We don't store abstract in Match, so skip it
    
    prompt = llm.prompts.match_seed_query_user.format(
        paper_title=seed_match.paper_title,
        paper_abstract=paper_abstract if paper_abstract else "(see paper for details)",
        trial_condition=trial_condition or "(not specified)",
        trial_intervention=trial_intervention or "(not specified)",
        used_queries=used_text,
    )
    
    result = llm.complete_json(prompt, llm.prompts.match_seed_query_system)
    queries = result.get("queries", []) if result else []
    
    # Filter out duplicates
    new_queries = [q for q in queries if q not in used_queries]
    logger.debug(f"Generated {len(new_queries)} seed expansion queries from match")
    
    return new_queries


def find_candidate_trials(
    adapter: ClinicalTrialsGovAdapter,
    queries: list[str],
    max_per_query: int = 10,
) -> list[Document]:
    """Search CTG for candidate trials using multiple queries.
    
    Deduplicates results by NCT ID.
    
    Args:
        adapter: CTG adapter for searching
        queries: List of search queries
        max_per_query: Maximum results per query
    
    Returns:
        List of unique trial documents
    """
    seen_instance_ids: set[str] = set()
    trials: list[Document] = []
    
    for query in queries:
        try:
            for trial in adapter.search(query, max_results=max_per_query):
                if trial.source_id not in seen_instance_ids:
                    seen_instance_ids.add(trial.source_id)
                    trials.append(trial)
        except Exception as e:
            logger.warning(f"Failed to search CTG with query '{query}': {e}")
            continue
    
    logger.debug(f"Found {len(trials)} unique candidate trials from {len(queries)} queries")
    return trials


def evaluate_match(
    llm: LLMClient,
    paper: Document,
    trial: Document,
    max_paper_chars: int = 2000,
    max_trial_chars: int = 4000,
) -> MatchScore | None:
    """Evaluate the match between a paper and a trial using LLM.
    
    Truncates inputs to fit within context window while preserving
    the most important information (eligibility criteria, abstract).
    
    Args:
        llm: LLM client for evaluation
        paper: Paper document
        trial: Trial document (from CTG)
        max_paper_chars: Max characters for paper abstract
        max_trial_chars: Max characters for trial description
    
    Returns:
        MatchScore with scores and reasoning, or None if evaluation fails
    """
    # Prepare paper text (truncate if needed)
    paper_abstract = truncate_text_smart(
        paper.abstract or "(no abstract)",
        max_chars=max_paper_chars
    )
    
    # Prepare trial text (truncate if needed, prioritize eligibility criteria)
    trial_description = truncate_text_smart(
        trial.abstract or "(no description)",
        max_chars=max_trial_chars
    )
    
    prompt = llm.prompts.match_eval_user.format(
        paper_title=paper.title,
        paper_abstract=paper_abstract,
        trial_instance_id=trial.source_id,
        trial_title=trial.title,
        trial_description=trial_description,
    )
    
    result = llm.complete_json(prompt, llm.prompts.match_eval_system)
    
    if not result:
        logger.warning(f"Failed to evaluate match: {paper.source_id} <-> {trial.source_id}")
        return None
    
    # Validate and extract scores
    try:
        match_score = int(result.get("match_score", 0))
        applicability_score = int(result.get("applicability_score", 0))
        
        # Clamp scores to valid range
        match_score = max(1, min(5, match_score))
        applicability_score = max(1, min(5, applicability_score))
        
        return MatchScore(
            match_score=match_score,
            match_reasoning=result.get("match_reasoning", ""),
            applicability_score=applicability_score,
            change_directive_quote=result.get("change_directive_quote", ""),
            change_rationale_quote=result.get("change_rationale_quote", ""),
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid score format in LLM response: {e}")
        return None


def match_paper_to_trials(
    paper: Document,
    adapter: ClinicalTrialsGovAdapter,
    llm: LLMClient,
    model_preset: str,
    top_k: int = 10,
    min_match_score: int = 1,
    save_queries: bool = True,
) -> Iterator[Match]:
    """Full pipeline: match a single paper to relevant clinical trials.
    
    1. Generate search queries from the paper
    2. Search CTG for candidate trials
    3. Evaluate each paper-trial pair
    4. Save and yield matches
    
    Args:
        paper: Paper document to match
        adapter: CTG adapter for searching
        llm: LLM client for query generation and evaluation
        model_preset: Model preset for database operations
        top_k: Maximum number of trials to evaluate (cost control; not all evaluated trials will match; increase for more matches)
        min_match_score: Minimum score to save (default: 1, save all)
        save_queries: Whether to save queries to prevent reuse
    
    Yields:
        Match objects for each paper-trial pair evaluated
    """
    # Get previously used queries
    used_queries = get_used_queries("ctg", model_preset)
    
    # Generate search queries
    queries = generate_trial_queries(llm, paper, used_queries)
    
    if not queries:
        logger.warning(f"No queries generated for paper: {paper.title[:50]}...")
        return
    
    # Save queries if requested
    if save_queries:
        for query in queries:
            if not has_query("ctg", query, model_preset):
                save_query("ctg", query, model_preset)
    
    # Find candidate trials
    # Fetch more per query to get diversity, then limit total evaluations
    # If we have multiple queries, fetch more per query to ensure we have enough candidates
    max_per_query = max(top_k, 50) if len(queries) > 1 else top_k
    trials = find_candidate_trials(adapter, queries, max_per_query=max_per_query)
    
    if not trials:
        logger.warning(f"No candidate trials found for paper: {paper.title[:50]}...")
        return
    
    # Limit to top_k trials for evaluation (cost control)
    trials = trials[:top_k]
    
    # Evaluate each trial
    matches_count = 0
    for trial in trials:
        # Skip if already matched
        if has_paper_trial_match(paper.source, paper.source_id, trial.source_id, model_preset):
            logger.debug(f"Skipping existing match: {paper.source_id} <-> {trial.source_id}")
            continue
        
        # Evaluate the match
        score = evaluate_match(llm, paper, trial)
        
        if score is None:
            continue
        
        # Create match object
        match = Match(
            paper_source=paper.source,
            paper_source_id=paper.source_id,
            paper_title=paper.title,
            trial_instance_id=trial.source_id,
            trial_title=trial.title,
            match_score=score.match_score,
            match_reasoning=score.match_reasoning,
            applicability_score=score.applicability_score,
            change_directive_quote=score.change_directive_quote,
            change_rationale_quote=score.change_rationale_quote,
        )
        
        # Save match if above threshold
        if score.match_score >= min_match_score:
            saved = save_paper_trial_match(
                paper_source=match.paper_source,
                paper_source_id=match.paper_source_id,
                paper_title=match.paper_title,
                trial_instance_id=match.trial_instance_id,
                trial_title=match.trial_title,
                match_score=match.match_score,
                match_reasoning=match.match_reasoning,
                applicability_score=match.applicability_score,
                change_directive_quote=match.change_directive_quote,
                change_rationale_quote=match.change_rationale_quote,
                model_preset=model_preset,
            )
            if saved:
                matches_count += 1
                logger.info(
                    f"Match saved: {paper.source_id} <-> {trial.source_id} "
                    f"(score={score.match_score}, applicability={score.applicability_score})"
                )
        
        yield match
    
    logger.info(f"Matched paper {paper.source_id}: {matches_count} new matches saved")


def is_high_quality_match(match: Match, min_match: int = 4, min_applicability: int = 4) -> bool:
    """Check if a match qualifies as high-quality for seed expansion.
    
    Args:
        match: Match to evaluate
        min_match: Minimum match_score threshold (default: 4)
        min_applicability: Minimum applicability_score threshold (default: 4)
    
    Returns:
        True if match exceeds both thresholds
    """
    return match.match_score >= min_match and match.applicability_score >= min_applicability
