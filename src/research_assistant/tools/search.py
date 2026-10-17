import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from research_assistant.tools.base import Tool

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15
MIN_SECONDS_BETWEEN_REQUESTS = 3

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
# arXiv's front end rejects requests without an Accept header (HTTP 406).
REQUEST_HEADERS = {"User-Agent": "research-assistant/0.1", "Accept": "*/*"}
SORT_OPTIONS = {"relevance": "relevance", "recent": "submittedDate"}

MAX_FETCH_ATTEMPTS = 3
# Errors worth retrying: arXiv's intermittent 406, rate limiting, and server-side failures.
TRANSIENT_HTTP_CODES = {406, 429, 500, 502, 503, 504}
MAX_AUTHORS_SHOWN = 3
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10

# Fallback index when arXiv's search API fails (it rate-limits heavy use): OpenAlex, limited to arXiv-hosted papers.
OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_ARXIV_SOURCE_ID = "S4306400194"  # OpenAlex's ID for the arXiv repository
OPENALEX_SORT_OPTIONS = {"relevance": "relevance_score:desc", "recent": "publication_date:desc"}
# Matches arXiv IDs in both OpenAlex URL forms: arxiv.org/abs/2604.26622 and doi.org/10.48550/arxiv.2604.26622
ARXIV_ID_PATTERN = re.compile(r"arxiv[./](?:org/abs/)?(\d{4}\.\d{4,5})", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and line breaks into single spaces."""
    return " ".join(text.split())


def _extract_arxiv_id(entry_url: str) -> str:
    """
    Extract ID from an arXiv web address.
    e.g. 'http://arxiv.org/abs/2603.02888v1' into '2603.02888'.
    """
    return re.sub(r"v\d+$", "", entry_url.split("/abs/")[-1])


def _is_transient(error: Exception) -> bool:
    """True if retrying the same request might succeed."""
    if isinstance(error, HTTPError):
        return error.code in TRANSIENT_HTTP_CODES
    return True


def _rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Abstract text rebuilt from OpenAlex's index of each word's positions."""
    if not inverted_index:
        return ""
    words_by_position = {position: word for word, positions in inverted_index.items() for position in positions}
    return " ".join(words_by_position[position] for position in sorted(words_by_position))


def _arxiv_id_from_openalex(work: dict) -> str:
    """arXiv ID of an OpenAlex work from its arXiv location URL, or "" if it has none."""
    for location in work.get("locations") or []:
        match = ARXIV_ID_PATTERN.search(location.get("landing_page_url") or "")
        if match:
            return match.group(1)
    return ""


def _read_tag(element: ET.Element, tag: str) -> str:
    """Return the text of the first matching child tag, or "" if it's missing."""
    return element.findtext(tag, default="", namespaces=ATOM_NAMESPACE)


class ArxivQueryError(Exception):
    """arXiv rejected the search query itself."""


class SearchTool(Tool):
    name = "search_arxiv"
    description = (
        "Search arXiv for academic papers on a topic. "
        f"Returns up to {MAX_RESULTS_LIMIT} papers with title, authors, date, arXiv ID, and abstract. "
        "Use this whenever the user asks about research, papers, or the state of a field. "
        "Search first rather than answering from memory. "
        "If results are empty or off-topic, search again with fewer or broader terms."
    )
    parameters = {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "description": (
                    "1-5 search terms. Each term is a single word or an exact multi-word phrase. "
                    "Keep multi-word concepts together as one term: for 'world model atlas', "
                    'use ["world model", "atlas"]. Every term must appear in a matching paper.'
                ),
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RESULTS_LIMIT,
                "description": f"Number of papers to return. Defaults to {DEFAULT_MAX_RESULTS}.",
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "recent"],
                "description": (
                    "'relevance' (default) ranks by match to the query; "
                    "'recent' returns newest papers first. "
                    "Use 'recent' when the user asks for the latest or newest work."
                ),
            },
        },
        "required": ["terms"],
    }

    _last_request_time = 0.0
    _rate_limit_lock = threading.Lock()

    def run(
        self,
        terms: list[str],
        max_results: int = DEFAULT_MAX_RESULTS,
        sort_by: str = "relevance",
    ) -> str:
        """Search arXiv and return the results as text for the model."""
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            return (
                "Error: terms must be a list of strings, with each item representing "
                f"a single word or phrase, but received {terms!r}."
            )
        terms = [term for term in map(_normalize_whitespace, terms) if term]
        if not terms:
            return "Error: no search terms given. Provide 1-5 terms."

        if sort_by not in SORT_OPTIONS:
            return f"Error: sort_by must be one of {list(SORT_OPTIONS)}."

        if max_results < 1:
            return f"Error: max_results must be at least 1, but received {max_results!r}."
        max_results = min(max_results, MAX_RESULTS_LIMIT)
        search_label = ", ".join(f'"{term}"' for term in terms)
        logger.info(f"Searching arXiv: terms={search_label}, max_results={max_results}, sort_by={sort_by}")
        try:
            url = self._build_url(terms, max_results, sort_by)
            papers = self._parse(self._fetch_with_retries(url))
        except (URLError, TimeoutError, ET.ParseError) as error:
            logger.warning(f"arXiv search failed for terms {search_label}: {error}. Falling back to OpenAlex.")
            try:
                papers = self._search_openalex(terms, max_results, sort_by)
            except (URLError, TimeoutError, ValueError) as fallback_error:
                logger.error(f"OpenAlex fallback failed for terms {search_label}: {fallback_error}")
                return f"Error: arXiv search failed ({error}), and so did the fallback search ({fallback_error})."
            logger.info(f"OpenAlex returned {len(papers)} arXiv papers for {search_label}")
            return self._format(papers, search_label)
        except ArxivQueryError as error:
            logger.warning(f"arXiv rejected terms {search_label}: {error}")
            return f"Error: arXiv rejected the query ({error}). Use plain words or phrases as terms."

        logger.info(f"arXiv returned {len(papers)} papers for {search_label}")
        return self._format(papers, search_label)

    @staticmethod
    def _build_url(terms: list[str], max_results: int, sort_by: str) -> str:
        """Build the arXiv API URL that performs the given search."""
        search_query = " AND ".join(f'all:"{term}"' for term in terms)
        params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": SORT_OPTIONS[sort_by],
            "sortOrder": "descending",
        }
        return f"{ARXIV_API_URL}?{urlencode(params)}"

    @classmethod
    def _wait_for_rate_limit(cls) -> None:
        """Sleep if needed so arXiv requests are at least 3 seconds apart, one caller at a time."""
        with cls._rate_limit_lock:
            elapsed = time.monotonic() - cls._last_request_time
            remaining = MIN_SECONDS_BETWEEN_REQUESTS - elapsed

            if remaining > 0:
                time.sleep(remaining)

            cls._last_request_time = time.monotonic()

    @staticmethod
    def _fetch(url: str) -> bytes:
        """Download the raw arXiv response for the given URL."""
        request = Request(url, headers=REQUEST_HEADERS)
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as error:
            # arXiv reports an invalid query as HTTP 400 with an Atom error feed as the body.
            if error.code == 400:
                return error.read()
            raise

    def _fetch_with_retries(self, url: str) -> bytes:
        """Fetch url, retrying transient failures up to MAX_FETCH_ATTEMPTS times."""
        for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
            self._wait_for_rate_limit()
            try:
                return self._fetch(url)
            except (URLError, TimeoutError) as error:
                if not _is_transient(error) or attempt == MAX_FETCH_ATTEMPTS:
                    raise
                logger.warning(f"arXiv request failed (attempt {attempt}/{MAX_FETCH_ATTEMPTS}): {error}. Retrying.")

    @staticmethod
    def _parse(xml_data: bytes) -> list[dict]:
        """Parse each paper's details from an arXiv API response."""
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", ATOM_NAMESPACE)
        papers = []
        for entry in entries:
            if "/api/errors" in _read_tag(entry, "atom:id"):
                raise ArxivQueryError(_read_tag(entry, "atom:summary"))
            paper = {
                "id": _extract_arxiv_id(_read_tag(entry, "atom:id")),
                "title": _normalize_whitespace(_read_tag(entry, "atom:title")),
                "authors": [_read_tag(author, "atom:name") for author in entry.findall("atom:author", ATOM_NAMESPACE)],
                "date": _read_tag(entry, "atom:published")[:10],
                "abstract": _normalize_whitespace(_read_tag(entry, "atom:summary")),
            }
            papers.append(paper)
        return papers

    @staticmethod
    def _search_openalex(terms: list[str], max_results: int, sort_by: str) -> list[dict]:
        """Search OpenAlex for arXiv-hosted papers, returned in the same shape as _parse's papers."""
        params = {
            "search": " ".join(terms),
            "filter": f"locations.source.id:{OPENALEX_ARXIV_SOURCE_ID}",
            "sort": OPENALEX_SORT_OPTIONS[sort_by],
            # OpenAlex can list one arXiv paper twice, so fetch extra and drop repeats below.
            "per-page": max_results * 2,
            "select": "title,publication_date,authorships,locations,abstract_inverted_index",
        }
        request = Request(f"{OPENALEX_API_URL}?{urlencode(params)}", headers=REQUEST_HEADERS)
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            works = json.loads(response.read())["results"]

        papers_by_id = {}
        for work in works:
            arxiv_id = _arxiv_id_from_openalex(work)
            if not arxiv_id or arxiv_id in papers_by_id:
                continue
            papers_by_id[arxiv_id] = {
                "id": arxiv_id,
                "title": _normalize_whitespace(work.get("title") or ""),
                # OpenAlex sometimes appends a disambiguation number to names, e.g. "Xin Yang (104842)".
                "authors": [
                    re.sub(r"\s*\(\d+\)$", "", authorship["author"]["display_name"])
                    for authorship in work.get("authorships") or []
                ],
                "date": work.get("publication_date") or "",
                "abstract": _normalize_whitespace(_rebuild_abstract(work.get("abstract_inverted_index"))),
            }
        return list(papers_by_id.values())[:max_results]

    @staticmethod
    def _format(papers: list[dict], search_label: str) -> str:
        """Render papers as numbered, citable text for the model."""
        if not papers:
            return f"No papers found for {search_label}. Try fewer or broader terms."

        blocks = []
        for number, paper in enumerate(papers, start=1):
            authors = ", ".join(paper["authors"][:MAX_AUTHORS_SHOWN])
            if len(paper["authors"]) > MAX_AUTHORS_SHOWN:
                authors += " et al."

            block = "\n".join(
                [
                    f"[{number}] {paper['title']}",
                    f"    Authors: {authors} | Date: {paper['date']} | arXiv: {paper['id']}",
                    f"    Abstract: {paper['abstract']}",
                ]
            )
            blocks.append(block)

        header = f"Found {len(papers)} papers for {search_label}:"
        return header + "\n\n" + "\n\n".join(blocks)
