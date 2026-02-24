#!/usr/bin/env python3
"""BibTeX metadata verifier and auto-corrector CLI."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.bwriter import BibTexWriter
except ImportError:  # pragma: no cover - handled at runtime
    bibtexparser = None
    BibTexParser = None
    BibTexWriter = None

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - handled at runtime
    BeautifulSoup = None

try:
    from googlesearch import search as google_search
except ImportError:  # pragma: no cover - handled at runtime
    google_search = None

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback below
    fuzz = None


RESERVED_FIELDS = {"ID", "ENTRYTYPE"}
CONFIDENCE_THRESHOLD = 0.78
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 2
RATE_LIMIT_SECONDS = 0.2

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
DASH_CHARS_RE = re.compile(r"[‐‑‒–—−]")


REQUIRED_RULES: dict[str, dict[str, list[Any]]] = {
    "article": {"all": ["author", "title", "journal", "year"], "any": []},
    "book": {
        "all": ["title", "publisher", "year"],
        "any": [["author", "editor"]],
    },
    "inproceedings": {"all": ["author", "title", "booktitle", "year"], "any": []},
    "inbook": {
        "all": ["title", "publisher", "year"],
        "any": [["author", "editor"], ["chapter", "pages"]],
    },
    "incollection": {
        "all": ["author", "title", "booktitle", "publisher", "year"],
        "any": [],
    },
    "phdthesis": {"all": ["author", "title", "school", "year"], "any": []},
    "mastersthesis": {"all": ["author", "title", "school", "year"], "any": []},
    "techreport": {"all": ["author", "title", "institution", "year"], "any": []},
    "misc": {"all": [], "any": []},
    "unpublished": {"all": ["author", "title", "note"], "any": []},
    "proceedings": {"all": ["title", "year"], "any": []},
}


@dataclass
class Change:
    field: str
    old: str
    new: str
    reason: str


@dataclass
class Candidate:
    metadata: dict[str, str]
    source: str
    confidence: float = 0.0
    details: str = ""


@dataclass
class EntryResult:
    key: str
    entry_type: str
    changes: list[Change] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    source_trace: list[str] = field(default_factory=list)
    used_source: Optional[str] = None


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_doi(value: str) -> str:
    doi = normalize_whitespace(value)
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def normalize_url(value: str) -> str:
    url = normalize_whitespace(value)
    if not url:
        return ""
    if url.startswith("www."):
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        rebuilt = parsed._replace(scheme=scheme, netloc=netloc)
        return rebuilt.geturl()
    return url


def normalize_pages(value: str) -> str:
    pages = normalize_whitespace(value)
    if not pages:
        return ""
    pages = DASH_CHARS_RE.sub("--", pages)
    pages = re.sub(r"\s*-\s*", "--", pages)
    pages = re.sub(r"--+", "--", pages)
    return pages


def normalize_year(value: str) -> str:
    year = normalize_whitespace(value)
    if re.fullmatch(r"\d{4}", year):
        return year
    found = re.search(r"\b(\d{4})\b", year)
    return found.group(1) if found else year


def normalize_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], list[Change]]:
    normalized: dict[str, Any] = {}
    changes: list[Change] = []

    for key, raw_value in entry.items():
        if key in RESERVED_FIELDS:
            normalized[key] = raw_value
            continue

        original_key = key
        target_key = key.strip().lower()
        original_value = "" if raw_value is None else str(raw_value)
        target_value = normalize_whitespace(original_value)

        if target_key == "doi":
            target_value = normalize_doi(target_value)
        elif target_key == "url":
            target_value = normalize_url(target_value)
        elif target_key == "pages":
            target_value = normalize_pages(target_value)
        elif target_key == "year":
            target_value = normalize_year(target_value)

        if target_key in normalized and normalized[target_key] and not target_value:
            continue
        normalized[target_key] = target_value

        if original_key != target_key or original_value != target_value:
            changes.append(
                Change(
                    field=target_key,
                    old=original_value,
                    new=target_value,
                    reason="normalized local metadata",
                )
            )

    if "ID" in entry and "ID" not in normalized:
        normalized["ID"] = entry["ID"]
    if "ENTRYTYPE" in entry and "ENTRYTYPE" not in normalized:
        normalized["ENTRYTYPE"] = entry["ENTRYTYPE"]

    return normalized, dedupe_changes(changes)


def split_authors(author_text: str) -> list[str]:
    if not author_text:
        return []
    return [part.strip() for part in re.split(r"\s+and\s+", author_text) if part.strip()]


def author_last_names(author_text: str) -> set[str]:
    last_names: set[str] = set()
    for author in split_authors(author_text):
        if "," in author:
            last = author.split(",", 1)[0].strip()
        else:
            bits = author.strip().split()
            last = bits[-1] if bits else ""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", last).lower()
        if cleaned:
            last_names.add(cleaned)
    return last_names


def valid_url(url_value: str) -> bool:
    parsed = urlparse(url_value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_entry(entry: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    entry_type = str(entry.get("ENTRYTYPE", "")).lower()
    rules = REQUIRED_RULES.get(entry_type, {"all": [], "any": []})

    for field_name in rules.get("all", []):
        if not normalize_whitespace(str(entry.get(field_name, ""))):
            issues.append(f"Missing required field '{field_name}' for type '{entry_type}'.")

    for any_group in rules.get("any", []):
        if not any(normalize_whitespace(str(entry.get(name, ""))) for name in any_group):
            options = " or ".join(any_group)
            issues.append(f"Missing one of required fields ({options}) for type '{entry_type}'.")

    author_value = normalize_whitespace(str(entry.get("author", "")))
    if author_value and not split_authors(author_value):
        issues.append("Author field is present but not parseable.")

    title_value = normalize_whitespace(str(entry.get("title", "")))
    if "title" in entry and not title_value:
        issues.append("Title field is empty.")

    year_value = normalize_whitespace(str(entry.get("year", "")))
    if year_value and not re.fullmatch(r"\d{4}", year_value):
        issues.append("Year should be a 4-digit number.")

    pages_value = normalize_whitespace(str(entry.get("pages", "")))
    if pages_value and "-" in pages_value and "--" not in pages_value:
        issues.append("Pages should use BibTeX range format with '--'.")

    doi_value = normalize_whitespace(str(entry.get("doi", "")))
    if doi_value and not DOI_RE.fullmatch(normalize_doi(doi_value)):
        issues.append("DOI format appears invalid.")

    url_value = normalize_whitespace(str(entry.get("url", "")))
    if url_value and not valid_url(normalize_url(url_value)):
        issues.append("URL should be absolute HTTP(S).")

    for numeric_field in ("volume", "number"):
        value = normalize_whitespace(str(entry.get(numeric_field, "")))
        if value and not re.search(r"\d", value):
            issues.append(f"Field '{numeric_field}' should include a numeric component.")

    return list(dict.fromkeys(issues))


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_whitespace(a).lower()
    b_norm = normalize_whitespace(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(a_norm, b_norm)) / 100.0

    # Fallback when rapidfuzz is unavailable.
    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def author_overlap_score(local_authors: str, candidate_authors: str) -> float:
    local_set = author_last_names(local_authors)
    candidate_set = author_last_names(candidate_authors)
    if not local_set or not candidate_set:
        return 0.0
    overlap = len(local_set & candidate_set)
    denom = max(len(local_set), len(candidate_set), 1)
    return overlap / denom


def compute_confidence(local_entry: dict[str, Any], candidate_metadata: dict[str, str]) -> float:
    components: list[tuple[float, float]] = []

    local_title = normalize_whitespace(str(local_entry.get("title", "")))
    candidate_title = normalize_whitespace(candidate_metadata.get("title", ""))
    if local_title and candidate_title:
        components.append((0.6, text_similarity(local_title, candidate_title)))
    elif candidate_title:
        components.append((0.6, 0.7))

    local_year = normalize_whitespace(str(local_entry.get("year", "")))
    candidate_year = normalize_whitespace(candidate_metadata.get("year", ""))
    if local_year and candidate_year:
        components.append((0.2, 1.0 if local_year == candidate_year else 0.0))
    elif candidate_year:
        components.append((0.2, 0.6))

    local_authors = normalize_whitespace(str(local_entry.get("author", "")))
    candidate_authors = normalize_whitespace(candidate_metadata.get("author", ""))
    if local_authors and candidate_authors:
        components.append((0.2, author_overlap_score(local_authors, candidate_authors)))
    elif candidate_authors:
        components.append((0.2, 0.6))

    if not components:
        return 0.0

    weighted_sum = sum(weight * value for weight, value in components)
    total_weight = sum(weight for weight, _ in components)
    return weighted_sum / total_weight


def normalize_external_metadata(entry_type: str, metadata: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field_name, field_value in metadata.items():
        if field_value is None:
            continue
        value = normalize_whitespace(str(field_value))
        if not value:
            continue

        key = field_name.lower()
        if key == "doi":
            value = normalize_doi(value)
        elif key == "url":
            value = normalize_url(value)
        elif key == "pages":
            value = normalize_pages(value)
        elif key == "year":
            value = normalize_year(value)

        normalized[key] = value

    # Harmonize venue field by entry type.
    conference_like = {"inproceedings", "incollection", "inbook", "proceedings"}
    if entry_type in conference_like and "booktitle" not in normalized and "journal" in normalized:
        normalized["booktitle"] = normalized["journal"]
    if entry_type == "article" and "journal" not in normalized and "booktitle" in normalized:
        normalized["journal"] = normalized["booktitle"]

    return normalized


def dedupe_changes(changes: list[Change]) -> list[Change]:
    unique: dict[tuple[str, str, str, str], Change] = {}
    for change in changes:
        unique[(change.field, change.old, change.new, change.reason)] = change
    return list(unique.values())


def apply_candidate(entry: dict[str, Any], candidate: Candidate) -> list[Change]:
    entry_type = str(entry.get("ENTRYTYPE", "")).lower()
    source_meta = normalize_external_metadata(entry_type, candidate.metadata)
    changes: list[Change] = []

    fields = [
        "author",
        "title",
        "journal",
        "booktitle",
        "year",
        "volume",
        "number",
        "pages",
        "doi",
        "url",
        "school",
        "institution",
        "note",
        "editor",
        "chapter",
    ]

    for field_name in fields:
        if field_name not in source_meta:
            continue

        old_value = normalize_whitespace(str(entry.get(field_name, "")))
        new_value = source_meta[field_name]
        if not new_value or old_value == new_value:
            continue

        entry[field_name] = new_value
        changes.append(
            Change(
                field=field_name,
                old=old_value,
                new=new_value,
                reason=f"overwritten from {candidate.source}",
            )
        )

    return dedupe_changes(changes)


def format_authors(author_items: list[dict[str, Any]] | list[str]) -> str:
    names: list[str] = []
    for item in author_items:
        if isinstance(item, str):
            text = normalize_whitespace(item)
            if text:
                names.append(text)
            continue

        given = normalize_whitespace(str(item.get("given") or item.get("firstName") or ""))
        family = normalize_whitespace(str(item.get("family") or item.get("lastName") or ""))
        name = normalize_whitespace(str(item.get("name") or ""))

        if family and given:
            names.append(f"{family}, {given}")
        elif family:
            names.append(family)
        elif name:
            names.append(name)

    return " and ".join(names)


def find_year_in_text(value: str) -> str:
    found = re.search(r"\b(19|20)\d{2}\b", value or "")
    return found.group(0) if found else ""


def extract_doi_from_text(value: str) -> str:
    found = re.search(r"10\.\d{4,9}/\S+", value or "", flags=re.IGNORECASE)
    return normalize_doi(found.group(0)) if found else ""


class SourceClient:
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        verbose: bool = False,
        threshold: float = CONFIDENCE_THRESHOLD,
        sleep_seconds: float = RATE_LIMIT_SECONDS,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.threshold = threshold
        self.sleep_seconds = sleep_seconds
        self._request_count = 0
        self.session = requests.Session() if requests is not None else None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[debug] {message}")

    def _sleep_for_rate_limit(self) -> None:
        if self._request_count > 0:
            time.sleep(self.sleep_seconds)

    def _request_json(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if self.session is None:
            return None, "requests dependency missing"

        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            self._sleep_for_rate_limit()
            self._request_count += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    backoff = min(2**attempt, 5)
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
                return response.json(), None
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                if attempt < self.retries:
                    backoff = min(2**attempt, 5)
                    time.sleep(backoff)

        return None, last_error or "request failed"

    def _request_text(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if self.session is None:
            return None, "requests dependency missing"

        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            self._sleep_for_rate_limit()
            self._request_count += 1
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    backoff = min(2**attempt, 5)
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
                return response.text, None
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                if attempt < self.retries:
                    backoff = min(2**attempt, 5)
                    time.sleep(backoff)

        return None, last_error or "request failed"

    def lookup_doi_candidate(self, doi: str) -> tuple[Optional[Candidate], str]:
        if not doi:
            return None, "DOI lookup skipped (no doi field)."

        normalized_doi = normalize_doi(doi)
        if not DOI_RE.fullmatch(normalized_doi):
            return None, "DOI lookup skipped (invalid DOI syntax)."

        url = f"https://api.crossref.org/works/{normalized_doi}"
        data, error = self._request_json(url)
        if error:
            return None, f"DOI lookup failed: {error}"

        message = (data or {}).get("message")
        if not isinstance(message, dict):
            return None, "DOI lookup returned no metadata."

        metadata = metadata_from_crossref_item(message)
        if not metadata:
            return None, "DOI metadata was empty."

        return Candidate(metadata=metadata, source="doi", confidence=1.0), "DOI metadata accepted."

    def lookup_url_candidate(self, url: str) -> tuple[Optional[Candidate], str]:
        if not url:
            return None, "URL lookup skipped (no url field)."

        metadata, error = self.fetch_metadata_from_url(url)
        if error:
            return None, f"URL lookup failed: {error}"
        if not metadata:
            return None, "URL lookup returned no parsable metadata."

        return Candidate(metadata=metadata, source="url"), "URL metadata extracted."

    def lookup_crossref_candidate(self, entry: dict[str, Any]) -> tuple[Optional[Candidate], str]:
        title = normalize_whitespace(str(entry.get("title", "")))
        author = normalize_whitespace(str(entry.get("author", "")))
        year = normalize_whitespace(str(entry.get("year", "")))
        query = " ".join(part for part in [title, author, year] if part)
        if not query:
            return None, "Crossref lookup skipped (insufficient query fields)."

        data, error = self._request_json(
            "https://api.crossref.org/works",
            params={"query.bibliographic": query, "rows": 5},
        )
        if error:
            return None, f"Crossref lookup failed: {error}"

        items = ((data or {}).get("message") or {}).get("items") or []
        if not items:
            return None, "Crossref returned no candidates."

        best_candidate: Optional[Candidate] = None
        for item in items:
            metadata = metadata_from_crossref_item(item)
            if not metadata:
                continue
            score = compute_confidence(entry, metadata)
            if best_candidate is None or score > best_candidate.confidence:
                best_candidate = Candidate(metadata=metadata, source="crossref", confidence=score)

        if best_candidate is None:
            return None, "Crossref candidates were unusable."

        return best_candidate, f"Crossref best score: {best_candidate.confidence:.2f}"

    def lookup_semantic_scholar_candidate(
        self, entry: dict[str, Any]
    ) -> tuple[Optional[Candidate], str]:
        title = normalize_whitespace(str(entry.get("title", "")))
        author = normalize_whitespace(str(entry.get("author", "")))
        year = normalize_whitespace(str(entry.get("year", "")))
        query = " ".join(part for part in [title, author, year] if part)
        if not query:
            return None, "Semantic Scholar lookup skipped (insufficient query fields)."

        data, error = self._request_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": 5,
                "fields": "title,year,authors,venue,journal,externalIds,url",
            },
        )
        if error:
            return None, f"Semantic Scholar lookup failed: {error}"

        papers = (data or {}).get("data") or []
        if not papers:
            return None, "Semantic Scholar returned no candidates."

        best_candidate: Optional[Candidate] = None
        for paper in papers:
            metadata = metadata_from_semantic_scholar(paper)
            if not metadata:
                continue
            score = compute_confidence(entry, metadata)
            if best_candidate is None or score > best_candidate.confidence:
                best_candidate = Candidate(
                    metadata=metadata,
                    source="semantic_scholar",
                    confidence=score,
                )

        if best_candidate is None:
            return None, "Semantic Scholar candidates were unusable."

        return best_candidate, f"Semantic Scholar best score: {best_candidate.confidence:.2f}"

    def lookup_google_candidate(self, entry: dict[str, Any]) -> tuple[Optional[Candidate], str]:
        if google_search is None:
            return None, "Google fallback unavailable (googlesearch-python missing)."

        title = normalize_whitespace(str(entry.get("title", "")))
        author = normalize_whitespace(str(entry.get("author", "")))
        year = normalize_whitespace(str(entry.get("year", "")))
        query = " ".join(part for part in [title, author, year] if part)
        if not query:
            return None, "Google fallback skipped (insufficient query fields)."

        try:
            urls = list(islice(google_search(query, num_results=5), 5))
        except Exception as exc:  # pylint: disable=broad-except
            return None, f"Google fallback failed: {exc}"

        if not urls:
            return None, "Google fallback returned no results."

        best_candidate: Optional[Candidate] = None
        for result_url in urls:
            metadata, error = self.fetch_metadata_from_url(result_url)
            if error or not metadata:
                continue

            metadata.setdefault("url", normalize_url(result_url))
            score = compute_confidence(entry, metadata)
            if best_candidate is None or score > best_candidate.confidence:
                best_candidate = Candidate(metadata=metadata, source="google", confidence=score)

        if best_candidate is None:
            return None, "Google results did not provide usable metadata."

        return best_candidate, f"Google best score: {best_candidate.confidence:.2f}"

    def fetch_metadata_from_url(self, url: str) -> tuple[dict[str, str], Optional[str]]:
        if BeautifulSoup is None:
            return {}, "beautifulsoup4 dependency missing"

        normalized_url = normalize_url(url)
        if not valid_url(normalized_url):
            return {}, "URL is not a valid absolute HTTP(S) link"

        html, error = self._request_text(
            normalized_url,
            headers={"User-Agent": "bibtex-checker/1.0"},
        )
        if error:
            return {}, error
        if not html:
            return {}, "empty HTML response"

        soup = BeautifulSoup(html, "html.parser")

        meta_values: dict[str, list[str]] = defaultdict(list)
        for tag in soup.find_all("meta"):
            key = (tag.get("name") or tag.get("property") or "").strip().lower()
            content = (tag.get("content") or "").strip()
            if key and content:
                meta_values[key].append(content)

        def first(*keys: str) -> str:
            for key in keys:
                values = meta_values.get(key.lower())
                if values:
                    return values[0]
            return ""

        def all_values(*keys: str) -> list[str]:
            values: list[str] = []
            for key in keys:
                values.extend(meta_values.get(key.lower(), []))
            return values

        metadata: dict[str, str] = {}
        title = first("citation_title", "dc.title", "og:title")
        if not title and soup.title and soup.title.string:
            title = soup.title.string
        if title:
            metadata["title"] = title

        authors = all_values("citation_author", "dc.creator", "article:author")
        if authors:
            metadata["author"] = " and ".join(normalize_whitespace(a) for a in authors if a)

        metadata["journal"] = first("citation_journal_title", "prism.publicationname")
        metadata["booktitle"] = first("citation_conference_title", "citation_book_title")

        year = first("citation_year")
        if not year:
            pub_date = first("citation_publication_date", "dc.date", "article:published_time")
            year = find_year_in_text(pub_date)
        if year:
            metadata["year"] = year

        first_page = first("citation_firstpage")
        last_page = first("citation_lastpage")
        pages = first("citation_pages")
        if first_page and last_page:
            pages = f"{first_page}--{last_page}"
        if pages:
            metadata["pages"] = pages

        volume = first("citation_volume")
        if volume:
            metadata["volume"] = volume

        number = first("citation_issue")
        if number:
            metadata["number"] = number

        doi = first("citation_doi", "dc.identifier")
        if not doi:
            doi = extract_doi_from_text(html)
        if doi:
            metadata["doi"] = doi

        metadata["url"] = normalized_url

        # JSON-LD fallback extraction.
        for script in soup.find_all("script"):
            type_attr = (script.get("type") or "").lower()
            if "ld+json" not in type_attr:
                continue
            text = script.string or script.get_text()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:  # pylint: disable=broad-except
                continue

            objects: list[dict[str, Any]] = []
            if isinstance(payload, dict):
                if "@graph" in payload and isinstance(payload["@graph"], list):
                    objects.extend(obj for obj in payload["@graph"] if isinstance(obj, dict))
                objects.append(payload)
            elif isinstance(payload, list):
                objects.extend(obj for obj in payload if isinstance(obj, dict))

            for obj in objects:
                candidate_type = str(obj.get("@type", "")).lower()
                if not any(token in candidate_type for token in ["article", "creativework", "scholarly"]):
                    continue

                if "title" not in metadata:
                    headline = obj.get("headline") or obj.get("name")
                    if isinstance(headline, str):
                        metadata["title"] = headline

                if "author" not in metadata:
                    authors_obj = obj.get("author")
                    author_names: list[str] = []
                    if isinstance(authors_obj, list):
                        for author_obj in authors_obj:
                            if isinstance(author_obj, dict):
                                name = author_obj.get("name")
                                if isinstance(name, str):
                                    author_names.append(name)
                            elif isinstance(author_obj, str):
                                author_names.append(author_obj)
                    elif isinstance(authors_obj, dict):
                        name = authors_obj.get("name")
                        if isinstance(name, str):
                            author_names.append(name)
                    if author_names:
                        metadata["author"] = " and ".join(author_names)

                if "year" not in metadata:
                    date_published = obj.get("datePublished")
                    if isinstance(date_published, str):
                        year_from_ld = find_year_in_text(date_published)
                        if year_from_ld:
                            metadata["year"] = year_from_ld

                if "doi" not in metadata:
                    identifier = obj.get("identifier")
                    identifier_text = ""
                    if isinstance(identifier, str):
                        identifier_text = identifier
                    elif isinstance(identifier, dict):
                        identifier_text = str(identifier.get("value") or "")
                    elif isinstance(identifier, list):
                        identifier_text = " ".join(str(i) for i in identifier)
                    doi_from_ld = extract_doi_from_text(identifier_text)
                    if doi_from_ld:
                        metadata["doi"] = doi_from_ld

                if "journal" not in metadata and isinstance(obj.get("isPartOf"), dict):
                    part_of = obj.get("isPartOf") or {}
                    journal_name = part_of.get("name")
                    if isinstance(journal_name, str):
                        metadata["journal"] = journal_name

        cleaned = normalize_external_metadata("", metadata)
        return cleaned, None

    def _decision(self, entry: dict[str, Any], candidate: Candidate, direct: bool = False) -> tuple[bool, str]:
        if direct:
            candidate.confidence = 1.0
            return True, "accepted direct DOI metadata"

        score = candidate.confidence if candidate.confidence else compute_confidence(entry, candidate.metadata)
        candidate.confidence = score
        if score >= self.threshold:
            return True, f"accepted score {score:.2f}"

        return False, f"rejected score {score:.2f} < {self.threshold:.2f}"

    def verify_metadata(
        self,
        entry: dict[str, Any],
    ) -> tuple[Optional[Candidate], list[str], list[str]]:
        trace: list[str] = []
        notes: list[str] = []

        doi_value = normalize_whitespace(str(entry.get("doi", "")))
        if doi_value:
            doi_candidate, message = self.lookup_doi_candidate(doi_value)
            if doi_candidate:
                accepted, decision = self._decision(entry, doi_candidate, direct=True)
                trace.append(f"doi: {decision}")
                if accepted:
                    return doi_candidate, trace, notes
            else:
                trace.append(f"doi: {message}")

        url_value = normalize_whitespace(str(entry.get("url", "")))
        if url_value:
            url_candidate, message = self.lookup_url_candidate(url_value)
            if url_candidate:
                accepted, decision = self._decision(entry, url_candidate)
                trace.append(f"url: {decision}")
                if accepted:
                    return url_candidate, trace, notes
                notes.append(f"URL metadata low confidence: {decision}")
            else:
                trace.append(f"url: {message}")

        crossref_candidate, crossref_message = self.lookup_crossref_candidate(entry)
        if crossref_candidate:
            accepted, decision = self._decision(entry, crossref_candidate)
            trace.append(f"crossref: {decision}")
            if accepted:
                return crossref_candidate, trace, notes
            notes.append(f"Crossref low confidence: {decision}")
        else:
            trace.append(f"crossref: {crossref_message}")

        semantic_candidate, semantic_message = self.lookup_semantic_scholar_candidate(entry)
        if semantic_candidate:
            accepted, decision = self._decision(entry, semantic_candidate)
            trace.append(f"semantic_scholar: {decision}")
            if accepted:
                return semantic_candidate, trace, notes
            notes.append(f"Semantic Scholar low confidence: {decision}")
        else:
            trace.append(f"semantic_scholar: {semantic_message}")

        google_candidate, google_message = self.lookup_google_candidate(entry)
        if google_candidate:
            accepted, decision = self._decision(entry, google_candidate)
            trace.append(f"google: {decision}")
            if accepted:
                return google_candidate, trace, notes
            notes.append(f"Google low confidence: {decision}")
        else:
            trace.append(f"google: {google_message}")

        notes.append(f"No high-confidence external match (>={self.threshold:.2f})")
        return None, trace, list(dict.fromkeys(notes))


def metadata_from_crossref_item(item: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}

    titles = item.get("title") or []
    if isinstance(titles, list) and titles:
        metadata["title"] = str(titles[0])
    elif isinstance(titles, str):
        metadata["title"] = titles

    authors = item.get("author") or []
    if authors:
        metadata["author"] = format_authors(authors)

    container = item.get("container-title") or []
    if isinstance(container, list) and container:
        metadata["journal"] = str(container[0])

    event = item.get("event") or {}
    if isinstance(event, dict):
        event_name = event.get("name")
        if isinstance(event_name, str) and event_name:
            metadata["booktitle"] = event_name

    for date_key in ["issued", "published-print", "published-online", "created"]:
        date_obj = item.get(date_key) or {}
        date_parts = date_obj.get("date-parts") if isinstance(date_obj, dict) else None
        if date_parts and isinstance(date_parts, list) and date_parts[0]:
            metadata["year"] = str(date_parts[0][0])
            break

    if item.get("volume"):
        metadata["volume"] = str(item["volume"])
    if item.get("issue"):
        metadata["number"] = str(item["issue"])
    if item.get("page"):
        metadata["pages"] = str(item["page"])
    if item.get("DOI"):
        metadata["doi"] = str(item["DOI"])
    if item.get("URL"):
        metadata["url"] = str(item["URL"])
    if item.get("publisher"):
        metadata["publisher"] = str(item["publisher"])

    return normalize_external_metadata("", metadata)


def metadata_from_semantic_scholar(item: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}

    title = item.get("title")
    if isinstance(title, str):
        metadata["title"] = title

    year = item.get("year")
    if year is not None:
        metadata["year"] = str(year)

    authors = item.get("authors") or []
    if isinstance(authors, list) and authors:
        author_names = [a.get("name") for a in authors if isinstance(a, dict) and a.get("name")]
        if author_names:
            metadata["author"] = " and ".join(author_names)

    journal = item.get("journal")
    if isinstance(journal, dict) and journal.get("name"):
        metadata["journal"] = str(journal["name"])

    venue = item.get("venue")
    if isinstance(venue, str) and venue:
        metadata.setdefault("booktitle", venue)

    external_ids = item.get("externalIds")
    if isinstance(external_ids, dict) and external_ids.get("DOI"):
        metadata["doi"] = str(external_ids["DOI"])

    url = item.get("url")
    if isinstance(url, str):
        metadata["url"] = url

    return normalize_external_metadata("", metadata)


def process_entry(entry: dict[str, Any], client: SourceClient) -> tuple[dict[str, Any], EntryResult]:
    working = copy.deepcopy(entry)
    working, normalize_changes = normalize_entry(working)

    key = str(working.get("ID", "<unknown>"))
    entry_type = str(working.get("ENTRYTYPE", "")).lower()

    candidate, source_trace, verification_notes = client.verify_metadata(working)

    changes: list[Change] = list(normalize_changes)
    used_source: Optional[str] = None
    if candidate:
        changes.extend(apply_candidate(working, candidate))
        used_source = candidate.source

    post_issues = validate_entry(working)

    unresolved = list(dict.fromkeys(post_issues + ([] if candidate else verification_notes)))
    result = EntryResult(
        key=key,
        entry_type=entry_type,
        changes=dedupe_changes(changes),
        unresolved=unresolved,
        source_trace=source_trace,
        used_source=used_source,
    )

    return working, result


def load_bibtex(path: Path):
    if bibtexparser is None or BibTexParser is None:
        raise RuntimeError("Missing dependency: bibtexparser")

    parser = BibTexParser(common_strings=True)
    with path.open("r", encoding="utf-8") as handle:
        return bibtexparser.load(handle, parser=parser)


def write_bibtex(path: Path, database: Any) -> None:
    if BibTexWriter is None:
        raise RuntimeError("Missing dependency: bibtexparser")

    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None

    with path.open("w", encoding="utf-8") as handle:
        handle.write(writer.write(database))


def markdown_report(
    input_path: Path,
    results: list[EntryResult],
    total_entries: int,
) -> str:
    unresolved_entries = [r for r in results if r.unresolved]
    fixed_entries = [r for r in results if r.changes and not r.unresolved]
    clean_count = total_entries - len(unresolved_entries) - len(fixed_entries)

    lines: list[str] = []
    lines.append("# BibTeX Metadata Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Input file: `{input_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total entries: {total_entries}")
    lines.append(f"- Clean: {clean_count}")
    lines.append(f"- Fixed: {len(fixed_entries)}")
    lines.append(f"- Unresolved: {len(unresolved_entries)}")
    lines.append("")

    lines.append("## Fixed entries")
    if not fixed_entries:
        lines.append("- None")
    else:
        for result in fixed_entries:
            lines.append(f"### `{result.key}` (`{result.entry_type}`)")
            if result.used_source:
                lines.append(f"- Source used: `{result.used_source}`")
            if result.source_trace:
                lines.append("- Source trace:")
                for trace_item in result.source_trace:
                    lines.append(f"  - {trace_item}")
            lines.append("- Changes:")
            for change in result.changes:
                lines.append(
                    f"  - `{change.field}`: `{change.old}` -> `{change.new}` ({change.reason})"
                )
            lines.append("")

    lines.append("")
    lines.append("## Unresolved/problem entries")
    if not unresolved_entries:
        lines.append("- None")
    else:
        for result in unresolved_entries:
            lines.append(f"### `{result.key}` (`{result.entry_type}`)")
            if result.used_source:
                lines.append(f"- Source used: `{result.used_source}`")
            if result.changes:
                lines.append("- Changes applied:")
                for change in result.changes:
                    lines.append(
                        f"  - `{change.field}`: `{change.old}` -> `{change.new}` ({change.reason})"
                    )
            lines.append("- Remaining issues:")
            for issue in result.unresolved:
                lines.append(f"  - {issue}")
            if result.source_trace:
                lines.append("- Source trace:")
                for trace_item in result.source_trace:
                    lines.append(f"  - {trace_item}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def terminal_unresolved_summary(results: list[EntryResult]) -> str:
    unresolved_entries = [r for r in results if r.unresolved]
    if not unresolved_entries:
        return "Unresolved entries: 0"

    lines = [f"Unresolved entries: {len(unresolved_entries)}"]
    for result in unresolved_entries:
        reason = "; ".join(result.unresolved)
        lines.append(f"- {result.key}: {reason}")
    return "\n".join(lines)


def default_report_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_report.md")


def default_corrected_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_corrected.bib")


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input_bib).expanduser().resolve()
    if not input_path.exists():
        print(f"fatal: input file does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        database = load_bibtex(input_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"fatal: could not parse bib file: {exc}", file=sys.stderr)
        return 2

    client = SourceClient(
        timeout=args.timeout,
        retries=args.retries,
        verbose=args.verbose,
        threshold=CONFIDENCE_THRESHOLD,
        sleep_seconds=RATE_LIMIT_SECONDS,
    )

    processed_entries: list[dict[str, Any]] = []
    results: list[EntryResult] = []

    for entry in database.entries:
        try:
            updated_entry, result = process_entry(entry, client)
        except Exception as exc:  # pylint: disable=broad-except
            key = str(entry.get("ID", "<unknown>"))
            entry_type = str(entry.get("ENTRYTYPE", "")).lower()
            processed_entries.append(entry)
            results.append(
                EntryResult(
                    key=key,
                    entry_type=entry_type,
                    unresolved=[f"Entry processing error: {exc}"],
                    source_trace=[],
                    used_source=None,
                )
            )
            continue

        processed_entries.append(updated_entry)
        results.append(result)

    database.entries = processed_entries

    report_path = Path(args.report).expanduser().resolve() if args.report else default_report_path(input_path)
    report_contents = markdown_report(input_path, results, total_entries=len(database.entries))
    report_path.write_text(report_contents, encoding="utf-8")

    if args.fix:
        out_bib = Path(args.out_bib).expanduser().resolve() if args.out_bib else default_corrected_path(input_path)
        write_bibtex(out_bib, database)
        print(f"Wrote corrected BibTeX: {out_bib}")

    print(f"Wrote markdown report: {report_path}")

    unresolved_summary = terminal_unresolved_summary(results)
    print(unresolved_summary)

    unresolved_count = sum(1 for result in results if result.unresolved)
    return 1 if unresolved_count else 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BibTeX metadata checker and auto-corrector")
    parser.add_argument("input_bib", help="Path to input .bib file")
    parser.add_argument("--fix", action="store_true", help="Write a corrected _corrected.bib output")
    parser.add_argument("--report", help="Path to markdown report output")
    parser.add_argument("--out-bib", help="Path to corrected bib output (only with --fix)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="HTTP retry count")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
