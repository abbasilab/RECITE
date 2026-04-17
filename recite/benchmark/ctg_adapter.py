"""ClinicalTrials.gov API adapter for benchmark data retrieval."""
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests
from loguru import logger


@dataclass
class Document:
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    doi: str | None = None
    pmid: int | None = None


class ClinicalTrialsGovAdapter:
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
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def _request_with_backoff(self, method: str, url: str, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                status = resp.status_code
            except requests.RequestException as e:
                status = f"network error: {e}"

            delay = base_delay * (2 ** attempt)
            logger.warning(f"Request failed ({status}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

        return self.session.request(method, url, timeout=30, **kwargs)

    def search(self, query: str, max_results: int = 50) -> Iterator[Document]:
        self._rate_limit()
        params = {
            "query.term": query,
            "pageSize": min(max_results, 1000),
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
        page_size = 1000
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

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            page_token = next_page_token

    def fetch_by_nct_id(self, nct_id: str) -> Document | None:
        self._rate_limit()
        url = f"{self.BASE}/{nct_id}"
        params = {"format": "json"}

        resp = self._request_with_backoff("GET", url, params=params)

        if resp.status_code != 200:
            logger.warning(f"CTG fetch failed for {nct_id}: {resp.status_code}")
            return None

        data = resp.json()
        if "protocolSection" in data:
            return self._parse_study(data)
        return None

    def _parse_study(self, study: dict) -> Document | None:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        nct_id = ident.get("nctId")
        title = ident.get("briefTitle") or ident.get("officialTitle")

        if not nct_id or not title:
            return None

        elig = protocol.get("eligibilityModule", {})
        criteria_text = elig.get("eligibilityCriteria")

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
            source_id=nct_id,
            title=title,
            abstract=abstract,
            doi=None,
            pmid=None,
        )
