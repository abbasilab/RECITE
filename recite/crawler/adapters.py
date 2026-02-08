"""
adapters.py

Data source adapters (PubMed, Semantic Scholar, ClinicalTrials.gov) with LLM query instructions.
"""
import os
import time
import httpx
import requests

from dataclasses import dataclass
from typing import Iterator, Optional
from loguru import logger


@dataclass
class Document:
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    doi: str | None = None
    pmid: int | None = None


@dataclass
class Scores:
    relevance: int
    extraction_confidence: int
    accrual_ease: int
    reasoning: str


@dataclass
class AdapterInstructions:
    """Instructions for LLM on how to query this source."""
    name: str
    format: str
    examples: list[str]
    
    def to_prompt(self) -> str:
        ex = "\n".join(f"  - {e}" for e in self.examples)
        return f"Source: {self.name}\nFormat: {self.format}\nExamples:\n{ex}"


def _request_with_backoff(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """Make HTTP request with exponential backoff on rate limit or server errors."""
    for attempt in range(max_retries):
        try:
            resp = client.request(method, url, **kwargs)
            
            # Success or client error (4xx except 429) - don't retry
            if resp.status_code < 500 and resp.status_code != 429:
                return resp
            
            # Rate limited or server error - retry with backoff
            status = resp.status_code
        except httpx.RequestError as e:
            # Network error - retry with backoff
            status = f"network error: {e}"
        
        delay = base_delay * (2 ** attempt)
        logger.warning(f"Request failed ({status}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)
    
    # Final attempt - let it raise if it fails
    return client.request(method, url, **kwargs)


# --- PubMed Adapter ---

PUBMED_INSTRUCTIONS = AdapterInstructions(
    name="PubMed",
    format="Boolean queries with [tiab], [mh], [dp], [pt] tags. Vary query strategies: disease-specific, methodology-focused, population-based, regulatory/policy, and temporal.",
    examples=[
        # Eligibility criteria focused
        '"eligibility criteria"[tiab] AND (broadening[tiab] OR loosening[tiab])',
        '"inclusion criteria"[tiab] AND "clinical trial"[pt] AND modernization[tiab]',
        # Disease-specific domains
        'oncology[mh] AND "trial eligibility"[tiab] AND relaxed[tiab]',
        'cardiovascular diseases[mh] AND "enrollment criteria"[tiab]',
        'diabetes mellitus[mh] AND "exclusion criteria"[tiab] AND barriers[tiab]',
        'mental disorders[mh] AND clinical trial[pt] AND "patient selection"[tiab]',
        # Population-focused
        '"elderly"[tiab] AND "clinical trial"[pt] AND "age criteria"[tiab]',
        'pediatric[tiab] AND "trial enrollment"[tiab] AND eligibility[tiab]',
        '"pregnant women"[tiab] AND "exclusion criteria"[tiab] AND clinical trial[pt]',
        'comorbidity[mh] AND "trial participation"[tiab]',
        # Methodology and design
        'pragmatic trial[tiab] AND "inclusive criteria"[tiab]',
        '"real-world"[tiab] AND "eligibility"[tiab] AND "clinical trial"[pt]',
        'adaptive trial design[tiab] AND enrollment[tiab]',
        # Regulatory and policy
        'FDA[tiab] AND "eligibility criteria"[tiab] AND guidance[tiab]',
        '"drug approval"[tiab] AND "trial population"[tiab] AND representativeness[tiab]',
        # Accrual and enrollment
        '"patient accrual"[tiab] AND barriers[tiab] AND "clinical trial"[pt]',
        '"enrollment rate"[tiab] AND "eligibility"[tiab]',
        'underrepresentation[tiab] AND "clinical trials"[tiab]',
        # Temporal
        '"eligibility criteria"[tiab] AND 2020:2025[dp]',
    ],
)


class PubMedAdapter:
    instructions = PUBMED_INSTRUCTIONS
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, requests_per_second: float = 3.0):
        self.api_key = os.getenv("NCBI_API_KEY")
        self.client = httpx.Client(timeout=30)
        self.min_interval = 1.0 / requests_per_second
        self._last_request = 0.0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()
    
    def search(self, query: str, max_results: int = 50) -> Iterator[Document]:
        self._rate_limit()
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        
        resp = _request_with_backoff(self.client, "GET", f"{self.BASE}/esearch.fcgi", params=params)
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        
        for pmid in pmids:
            doc = self._fetch(pmid)
            if doc:
                yield doc
    
    def _fetch(self, pmid: str) -> Document | None:
        self._rate_limit()
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        resp = _request_with_backoff(self.client, "GET", f"{self.BASE}/efetch.fcgi", params=params)
        # Parse XML... (simplified)
        return Document(source="pubmed", source_id=pmid, title=f"Paper {pmid}", pmid=int(pmid))


# --- Semantic Scholar Adapter ---

S2_INSTRUCTIONS = AdapterInstructions(
    name="Semantic Scholar",
    format="Natural language queries. Be specific and descriptive.",
    examples=[
        "clinical trial eligibility criteria broadening modernization",
        "age restrictions exclusion criteria oncology trials",
        "comorbidity exclusion clinical trial enrollment barriers",
        "pragmatic trials inclusive eligibility real-world patients",
        "FDA guidance broadening trial eligibility underrepresented populations",
        "pediatric clinical trial enrollment age criteria",
        "pregnant women exclusion clinical trials safety",
        "elderly patients trial participation age limits",
        "racial diversity clinical trial eligibility criteria",
        "rare disease trial enrollment relaxed criteria",
    ],
)


class SemanticScholarAdapter:
    """Semantic Scholar API adapter - supports natural language queries."""
    instructions = S2_INSTRUCTIONS
    BASE = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, requests_per_second: float = 10.0):
        self.api_key = os.getenv("S2_API_KEY")  # Optional but recommended
        self.client = httpx.Client(timeout=30)
        self.min_interval = 1.0 / requests_per_second
        self._last_request = 0.0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()
    
    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    def search(self, query: str, max_results: int = 50) -> Iterator[Document]:
        """Search Semantic Scholar with natural language query."""
        self._rate_limit()
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API max is 100
            "fields": "paperId,title,abstract,externalIds,authors,year",
        }
        
        resp = _request_with_backoff(
            self.client, "GET", f"{self.BASE}/paper/search",
            params=params, headers=self._headers()
        )
        
        if resp.status_code != 200:
            logger.warning(f"S2 search failed: {resp.status_code}")
            return
        
        data = resp.json()
        for paper in data.get("data", []):
            doc = self._parse_paper(paper)
            if doc:
                yield doc
    
    def _parse_paper(self, paper: dict) -> Document | None:
        """Parse S2 paper response into Document."""
        paper_id = paper.get("paperId")
        title = paper.get("title")
        if not paper_id or not title:
            return None
        
        external_ids = paper.get("externalIds") or {}
        return Document(
            source="s2",
            source_id=paper_id,
            title=title,
            abstract=paper.get("abstract"),
            doi=external_ids.get("DOI"),
            pmid=int(external_ids["PubMed"]) if external_ids.get("PubMed") else None,
        )
    
    def get_references(self, paper_id: str, max_results: int = 20) -> Iterator[Document]:
        """Get papers cited by this paper."""
        self._rate_limit()
        params = {"fields": "paperId,title,abstract,externalIds", "limit": max_results}
        resp = _request_with_backoff(
            self.client, "GET", f"{self.BASE}/paper/{paper_id}/references",
            params=params, headers=self._headers()
        )
        
        if resp.status_code != 200:
            return
        
        for ref in resp.json().get("data", []):
            cited = ref.get("citedPaper")
            if cited:
                doc = self._parse_paper(cited)
                if doc:
                    yield doc
    
    def get_citations(self, paper_id: str, max_results: int = 20) -> Iterator[Document]:
        """Get papers that cite this paper."""
        self._rate_limit()
        params = {"fields": "paperId,title,abstract,externalIds", "limit": max_results}
        resp = _request_with_backoff(
            self.client, "GET", f"{self.BASE}/paper/{paper_id}/citations",
            params=params, headers=self._headers()
        )
        
        if resp.status_code != 200:
            return
        
        for cit in resp.json().get("data", []):
            citing = cit.get("citingPaper")
            if citing:
                doc = self._parse_paper(citing)
                if doc:
                    yield doc


# --- ClinicalTrials.gov Adapter ---

CTG_INSTRUCTIONS = AdapterInstructions(
    name="ClinicalTrials.gov",
    format="Natural language queries for conditions, interventions, populations, or combined terms. The API supports free-text search across trial titles, conditions, interventions, and eligibility criteria.",
    examples=[
        # Condition-based queries
        "breast cancer HER2 positive",
        "type 2 diabetes mellitus",
        "non-small cell lung cancer",
        "chronic kidney disease",
        "hypertension",
        # Intervention-based queries
        "metformin",
        "chemotherapy",
        "immunotherapy checkpoint inhibitor",
        # Population-based queries
        "elderly patients",
        "pediatric",
        "pregnant women",
        # Combined queries (condition + intervention/population)
        "breast cancer trastuzumab",
        "diabetes metformin elderly",
        "lung cancer pembrolizumab",
        # Eligibility-focused queries (when looking for trials with specific EC patterns)
        "platelet count eligibility",
        "renal function creatinine clearance",
        "ECOG performance status",
    ],
)


class ClinicalTrialsGovAdapter:
    """ClinicalTrials.gov API v2 adapter - supports search and fetch by NCT ID.
    
    Uses requests library instead of httpx because the CTG API blocks httpx requests.
    """
    instructions = CTG_INSTRUCTIONS
    BASE = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self, requests_per_second: float = 5.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "recite/1.0 (research tool)",
            "Accept": "application/json",
        })
        self.min_interval = 1.0 / requests_per_second
        self._last_request = 0.0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()
    
    def _request_with_backoff(self, method: str, url: str, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
        """Make HTTP request with exponential backoff on rate limit or server errors."""
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                
                # Success or client error (4xx except 429) - don't retry
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                
                # Rate limited or server error - retry with backoff
                status = resp.status_code
            except requests.RequestException as e:
                # Network error - retry with backoff
                status = f"network error: {e}"
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Request failed ({status}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        
        # Final attempt - let it raise if it fails
        return self.session.request(method, url, timeout=30, **kwargs)
    
    def search(self, query: str, max_results: int = 50) -> Iterator[Document]:
        """Search ClinicalTrials.gov with natural language query."""
        self._rate_limit()
        params = {
            "query.term": query,
            "pageSize": min(max_results, 1000),  # API max is 1000
            "format": "json",
        }
        
        resp = self._request_with_backoff("GET", self.BASE, params=params)
        
        if resp.status_code != 200:
            logger.warning(f"CTG search failed: {resp.status_code}")
            return
        
        data = resp.json()
        studies = data.get("studies", [])
        
        for study in studies:
            doc = self._parse_study(study)
            if doc:
                yield doc
    
    def search_all_pages(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> Iterator[Document]:
        """
        Search ClinicalTrials.gov with pagination support.
        
        Args:
            query: Search query (use "*" for all trials)
            max_results: Maximum number of results to return (None for all)
            
        Yields:
            Document objects
        """
        page_size = 1000  # API maximum
        page_token = None
        total_yielded = 0
        
        while True:
            self._rate_limit()
            params = {
                "query.term": query,
                "pageSize": page_size,
                "format": "json",
            }
            
            if page_token:
                params["pageToken"] = page_token
            
            resp = self._request_with_backoff("GET", self.BASE, params=params)
            
            if resp.status_code != 200:
                logger.warning(f"CTG search failed: {resp.status_code}")
                break
            
            data = resp.json()
            studies = data.get("studies", [])
            
            if not studies:
                break
            
            for study in studies:
                doc = self._parse_study(study)
                if doc:
                    yield doc
                    total_yielded += 1
                    if max_results and total_yielded >= max_results:
                        return
            
            # Check for next page
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
            
            page_token = next_page_token
    
    def fetch_by_instance_id(self, instance_id: str) -> Document | None:
        """Fetch a specific trial by NCT ID."""
        self._rate_limit()
        url = f"{self.BASE}/{instance_id}"
        params = {"format": "json"}
        
        resp = self._request_with_backoff("GET", url, params=params)
        
        if resp.status_code != 200:
            logger.warning(f"CTG fetch failed for {instance_id}: {resp.status_code}")
            return None
        
        data = resp.json()
        # API returns single study in different format
        if "protocolSection" in data:
            return self._parse_study(data)
        return None
    
    def _parse_study(self, study: dict) -> Document | None:
        """Parse CTG study response into Document."""
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        instance_id = ident.get("nctId")
        title = ident.get("briefTitle") or ident.get("officialTitle")
        
        if not instance_id or not title:
            return None
        
        # Extract eligibility criteria if available
        elig = protocol.get("eligibilityModule", {})
        criteria_text = elig.get("eligibilityCriteria")
        
        # Build abstract from available fields
        desc = protocol.get("descriptionModule", {})
        brief_summary = desc.get("briefSummary")
        detailed_description = desc.get("detailedDescription")
        
        abstract_parts = []
        if brief_summary:
            abstract_parts.append(brief_summary)
        if detailed_description:
            abstract_parts.append(detailed_description)
        if criteria_text:
            abstract_parts.append(f"Eligibility Criteria: {criteria_text}")
        
        abstract = "\n\n".join(abstract_parts) if abstract_parts else None
        
        return Document(
            source="ctg",
            source_id=instance_id,
            title=title,
            abstract=abstract,
            doi=None,  # CTG doesn't have DOI
            pmid=None,  # CTG doesn't have PMID
        )


ADAPTERS = {
    "pubmed": PubMedAdapter(),
    "s2": SemanticScholarAdapter(),
    "ctg": ClinicalTrialsGovAdapter(),
}


# --- LLM Functions (use crawler/llm.py) ---

def _summarize_used_queries(used: list[str], max_recent: int = 20, max_chars: int = 2000) -> str:
    """Summarize used queries to fit in context while showing diversity."""
    if not used:
        return "(none yet - you have full freedom to explore any angle)"
    
    # Show count and recent queries
    lines = [f"Total queries used so far: {len(used)}"]
    
    # Show most recent queries (these are most important to avoid)
    recent = used[-max_recent:]
    lines.append(f"\nMost recent {len(recent)} queries (DO NOT repeat these or similar):")
    for q in recent:
        lines.append(f"  - {q}")
    
    # If there are more, show a sample of older ones
    if len(used) > max_recent:
        older = used[:-max_recent]
        # Sample evenly from older queries
        sample_size = min(10, len(older))
        step = max(1, len(older) // sample_size)
        sample = older[::step][:sample_size]
        lines.append(f"\nSample of {len(sample)} older queries (also avoid):")
        for q in sample:
            lines.append(f"  - {q}")
    
    result = "\n".join(lines)
    # Truncate if too long
    if len(result) > max_chars:
        result = result[:max_chars] + "\n  ... (truncated)"
    return result


def generate_queries(llm, instructions: AdapterInstructions, used: list[str]) -> list[str]:
    """Ask LLM to generate search queries."""
    used_text = _summarize_used_queries(used)
    
    prompt = llm.prompts.query_user.format(
        source_name=instructions.name,
        adapter_instructions=instructions.to_prompt(),
        used_queries=used_text,
    )
    
    result = llm.complete_json(prompt, llm.prompts.query_system)
    return result.get("queries", []) if result else []


def generate_seed_queries(llm, seed_doc: Document, instructions: AdapterInstructions) -> list[str]:
    """Generate new queries based on a relevant seed paper.
    
    This enables the multiplicative effect: a good find leads to more good finds.
    """
    seed_text = f"Title: {seed_doc.title}"
    if seed_doc.abstract:
        # Truncate abstract if too long
        abstract = seed_doc.abstract[:500] + "..." if len(seed_doc.abstract) > 500 else seed_doc.abstract
        seed_text += f"\nAbstract: {abstract}"
    
    prompt = f"""Based on this HIGHLY RELEVANT paper about clinical trial eligibility criteria:

{seed_text}

Generate 5 NEW search queries for {instructions.name} that explore:
1. Similar concepts from different angles
2. Related disease domains or populations
3. Cited methodologies or frameworks mentioned
4. Adjacent research areas that might have relevant findings

{instructions.to_prompt()}

Return ONLY valid JSON. Escape quotes with backslash.
Example: {{"queries": ["query1", "query2", ...]}}"""

    system = "You are expanding a literature search based on a relevant seed paper. Generate diverse queries that will find related but different papers."
    
    result = llm.complete_json(prompt, system)
    return result.get("queries", []) if result else []


def evaluate_paper(llm, doc: Document) -> Scores:
    """Ask LLM to score a paper."""
    prompt = llm.prompts.eval_user.format(
        title=doc.title,
        abstract=doc.abstract or "(no abstract)",
    )
    
    result = llm.complete_json(prompt, llm.prompts.eval_system)
    if result:
        return Scores(
            relevance=result.get("relevance", 1),
            extraction_confidence=result.get("extraction_confidence", 1),
            accrual_ease=result.get("accrual_ease", 1),
            reasoning=result.get("reasoning", ""),
        )
    return Scores(1, 1, 1, "evaluation failed")
