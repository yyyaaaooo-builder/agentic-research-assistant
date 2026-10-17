import io
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from research_assistant.tools.search import (
    MAX_FETCH_ATTEMPTS,
    MIN_SECONDS_BETWEEN_REQUESTS,
    REQUEST_TIMEOUT_SECONDS,
    ArxivQueryError,
    SearchTool,
    _extract_arxiv_id,
    _is_transient,
    _normalize_whitespace,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_XML = (FIXTURES / "arxiv_sample.xml").read_bytes()
ERROR_XML = (FIXTURES / "arxiv_error.xml").read_bytes()


class FakeClock:
    """Stands in for the `time` module: time only moves when a test moves it."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake_clock = FakeClock()
    monkeypatch.setattr("research_assistant.tools.search.time", fake_clock)
    monkeypatch.setattr(SearchTool, "_last_request_time", 0.0)
    return fake_clock


@pytest.fixture
def tool(monkeypatch):
    search_tool = SearchTool()
    monkeypatch.setattr(search_tool, "_wait_for_rate_limit", lambda: None)
    # Keep tests offline: the OpenAlex fallback fails unless a test replaces it.
    monkeypatch.setattr(search_tool, "_search_openalex", fail_openalex)
    return search_tool


def fail_openalex(terms, max_results, sort_by):
    raise URLError("fallback unavailable")


def make_paper(**fields) -> dict:
    defaults = {
        "id": "1111.0001",
        "title": "A Paper",
        "authors": ["Ana"],
        "date": "2026-01-01",
        "abstract": "About things.",
    }
    return defaults | fields


def parse_query_params(url: str) -> dict:
    return parse_qs(urlsplit(url).query)


def http_error(code: int) -> HTTPError:
    return HTTPError("https://x", code, "test error", None, None)


class FakeFetch:
    """Stands in for SearchTool._fetch: plays queued outcomes in order and records each requested URL."""

    def __init__(self):
        self.outcomes = []
        self.urls = []

    def respond_with(self, *outcomes):
        self.outcomes = list(outcomes)

    def __call__(self, url):
        self.urls.append(url)
        if not self.outcomes:
            raise AssertionError(f"unexpected request to arXiv: {url}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fetch(tool, monkeypatch):
    fake_fetch = FakeFetch()
    monkeypatch.setattr(tool, "_fetch", fake_fetch)
    return fake_fetch


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Retrieval  Augmented Generation    ", "Retrieval Augmented Generation"),
        ("Retrieval\n   augmented\nagents", "Retrieval augmented agents"),
        ("  padded  ", "padded"),
        ("", ""),
    ],
)
def test_normalize_whitespace(text, expected):
    assert _normalize_whitespace(text) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://arxiv.org/abs/2603.02888v1", "2603.02888"),
        ("http://arxiv.org/abs/2603.02888v123", "2603.02888"),
        ("http://arxiv.org/abs/hep-th/9901001v2", "hep-th/9901001"),
    ],
)
def test_extract_arxiv_id(url, expected):
    assert _extract_arxiv_id(url) == expected


@pytest.mark.parametrize("code, expected", [(406, True), (429, True), (503, True), (400, False), (404, False)])
def test_is_transient_code(code, expected):
    assert _is_transient(http_error(code)) == expected


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError()], ids=["network", "timeout"])
def test_is_transient_error(error):
    assert _is_transient(error)


@pytest.mark.parametrize(
    "terms, max_results, sort_by, expected",
    [
        pytest.param(
            ["agents"],
            5,
            "relevance",
            {
                "search_query": ['all:"agents"'],
                "max_results": ["5"],
                "sortBy": ["relevance"],
                "sortOrder": ["descending"],
            },
            id="one-term",
        ),
        pytest.param(
            ["world model", "atlas"],
            3,
            "recent",
            {
                "search_query": ['all:"world model" AND all:"atlas"'],
                "max_results": ["3"],
                "sortBy": ["submittedDate"],
                "sortOrder": ["descending"],
            },
            id="phrase-and-word-recent",
        ),
        pytest.param(
            ["retrieval augmented generation", "agents", "survey"],
            10,
            "relevance",
            {
                "search_query": ['all:"retrieval augmented generation" AND all:"agents" AND all:"survey"'],
                "max_results": ["10"],
                "sortBy": ["relevance"],
                "sortOrder": ["descending"],
            },
            id="three-terms",
        ),
    ],
)
def test_build_url(terms, max_results, sort_by, expected):
    url = SearchTool._build_url(terms, max_results, sort_by)
    assert parse_query_params(url) == expected


def test_rate_limit_waits_when_needed(clock):
    SearchTool._wait_for_rate_limit()  # first request: long since "time 0", so no wait

    clock.now += 1  # 1 second passes
    SearchTool._wait_for_rate_limit()  # wait the remaining 2s

    clock.now += 5  # 5 more seconds pass
    SearchTool._wait_for_rate_limit()  # no wait

    assert clock.slept == [2.0]


def test_rate_limit_is_shared(clock):
    SearchTool()._wait_for_rate_limit()
    SearchTool()._wait_for_rate_limit()  # a different object, immediately after

    assert clock.slept == [MIN_SECONDS_BETWEEN_REQUESTS]


def test_fetch_and_returns_body(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["accept"] = request.get_header("Accept")
        sent["timeout"] = timeout
        return io.BytesIO(SAMPLE_XML)

    monkeypatch.setattr("research_assistant.tools.search.urlopen", fake_urlopen)

    assert SearchTool._fetch("https://x") == SAMPLE_XML
    assert sent == {"accept": "*/*", "timeout": REQUEST_TIMEOUT_SECONDS}


def test_fetch_returns_body_on_400(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 400, "Bad Request", None, io.BytesIO(ERROR_XML))

    monkeypatch.setattr("research_assistant.tools.search.urlopen", fake_urlopen)

    assert SearchTool._fetch("https://x") == ERROR_XML


def test_fetch_raises_on_other_http_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", None, io.BytesIO(b""))

    monkeypatch.setattr("research_assistant.tools.search.urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        SearchTool._fetch("https://x")


def test_parse_cleans_every_paper():
    for paper in SearchTool._parse(SAMPLE_XML):
        assert set(paper) == {"id", "title", "authors", "date", "abstract"}
        assert not re.search(r"v\d+$", paper["id"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", paper["date"])
        assert "\n" not in paper["title"] and "  " not in paper["title"]
        assert paper["authors"]
        assert paper["abstract"]


def test_parse_raises_on_arxiv_error():
    with pytest.raises(ArxivQueryError, match="Invalid query string"):
        SearchTool._parse(ERROR_XML)


def test_parse_empty_feed():
    empty_feed = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    assert SearchTool._parse(empty_feed) == []


def test_format_no_papers():
    result = SearchTool._format([], '"world model"')

    assert result == 'No papers found for "world model". Try fewer or broader terms.'


def test_format_layout():
    papers = [
        make_paper(id="1111.0001", title="Two Authors", authors=["Ana", "Bo"]),
        make_paper(id="2222.0002", title="Three Authors", authors=["Ana", "Bo", "Cy"]),
        make_paper(id="3333.0003", title="Five Authors", authors=["Ana", "Bo", "Cy", "Di", "Ed"]),
    ]

    result = SearchTool._format(papers, '"agents"')

    assert result == (
        'Found 3 papers for "agents":\n'
        "\n"
        "[1] Two Authors\n"
        "    Authors: Ana, Bo | Date: 2026-01-01 | arXiv: 1111.0001\n"
        "    Abstract: About things.\n"
        "\n"
        "[2] Three Authors\n"
        "    Authors: Ana, Bo, Cy | Date: 2026-01-01 | arXiv: 2222.0002\n"
        "    Abstract: About things.\n"
        "\n"
        "[3] Five Authors\n"
        "    Authors: Ana, Bo, Cy et al. | Date: 2026-01-01 | arXiv: 3333.0003\n"
        "    Abstract: About things."
    )


def test_run_returns_formatted_papers(tool, fetch):
    fetch.respond_with(SAMPLE_XML)

    result = tool.run(terms=["retrieval agents"])

    assert result.startswith('Found 3 papers for "retrieval agents":\n\n[1] LLandMark')
    assert len(fetch.urls) == 1


def test_run_applies_defaults(tool, fetch):
    fetch.respond_with(SAMPLE_XML)

    tool.run(terms=["world model"])

    assert parse_query_params(fetch.urls[0]) == {
        "search_query": ['all:"world model"'],
        "max_results": ["5"],
        "sortBy": ["relevance"],
        "sortOrder": ["descending"],
    }


@pytest.mark.parametrize("requested, sent", [(50, "10"), (5, "5")])
def test_run_caps_max_results(tool, fetch, requested, sent):
    fetch.respond_with(SAMPLE_XML)

    tool.run(terms=["agents"], max_results=requested)

    assert parse_query_params(fetch.urls[0])["max_results"] == [sent]


@pytest.mark.parametrize(
    "arguments, expected",
    [
        pytest.param(
            {"terms": "world model"},
            "Error: terms must be a list of strings, with each item representing a single word or phrase, "
            "but received 'world model'.",
            id="plain-string",
        ),
        pytest.param(
            {"terms": [42]},
            "Error: terms must be a list of strings, with each item representing a single word or phrase, "
            "but received [42].",
            id="non-string-item",
        ),
        pytest.param(
            {"terms": ["  ", ""]},
            "Error: no search terms given. Provide 1-5 terms.",
            id="blank-terms",
        ),
        pytest.param(
            {"terms": ["agents"], "sort_by": "popular"},
            "Error: sort_by must be one of ['relevance', 'recent'].",
            id="invalid-sort",
        ),
        pytest.param(
            {"terms": ["agents"], "max_results": 0},
            "Error: max_results must be at least 1, but received 0.",
            id="max-results-below-1",
        ),
    ],
)
def test_run_rejects_bad_input(tool, arguments, expected):
    assert tool.run(**arguments) == expected


@pytest.mark.parametrize(
    "outcomes, expected_start, expected_calls",
    [
        pytest.param(
            [http_error(406), http_error(406), SAMPLE_XML],
            "Found 3 papers",
            3,
            id="recovers",
        ),
        pytest.param(
            [http_error(404)],
            "Error: arXiv search failed (HTTP Error 404: test error), and so did the fallback search",
            1,
            id="permanent-error",
        ),
        pytest.param(
            [http_error(503)] * MAX_FETCH_ATTEMPTS,
            "Error: arXiv search failed (HTTP Error 503: test error), and so did the fallback search",
            MAX_FETCH_ATTEMPTS,
            id="never-recovers",
        ),
        pytest.param(
            [ERROR_XML],
            "Error: arXiv rejected the query (Invalid query string: '(all:retrieval'). "
            "Use plain words or phrases as terms.",
            1,
            id="query-rejected",
        ),
    ],
)
def test_run_handles_fetch_outcomes(tool, fetch, outcomes, expected_start, expected_calls):
    fetch.respond_with(*outcomes)

    result = tool.run(terms=["retrieval agents"])

    assert result.startswith(expected_start)
    assert len(fetch.urls) == expected_calls


def test_run_falls_back_to_openalex_when_arxiv_fails(tool, fetch, monkeypatch):
    fetch.respond_with(*[http_error(429)] * MAX_FETCH_ATTEMPTS)
    monkeypatch.setattr(tool, "_search_openalex", lambda terms, max_results, sort_by: [make_paper(id="2604.26622")])

    result = tool.run(terms=["agent memory"])

    assert result.startswith("Found 1 papers")
    assert "arXiv: 2604.26622" in result


def test_search_openalex_returns_arxiv_papers_without_repeats(monkeypatch):
    def work(url, **fields):
        return {"locations": [{"landing_page_url": url}], **fields}

    works = [
        work(
            "https://arxiv.org/abs/2604.26622",
            title="OCR-Memory",
            publication_date="2026-04-29",
            authorships=[{"author": {"display_name": "Xin Yang (104842)"}}],
            abstract_inverted_index={"agents": [1], "Autonomous": [0]},
        ),
        work("https://doi.org/10.48550/arxiv.2604.26622", title="OCR-Memory (second record)"),
        work("https://example.org/not-arxiv", title="Not on arXiv"),
    ]
    payload = io.BytesIO(json.dumps({"results": works}).encode())
    monkeypatch.setattr("research_assistant.tools.search.urlopen", lambda request, timeout: payload)

    papers = SearchTool._search_openalex(["agent memory"], max_results=5, sort_by="relevance")

    assert papers == [
        {
            "id": "2604.26622",
            "title": "OCR-Memory",
            "authors": ["Xin Yang"],
            "date": "2026-04-29",
            "abstract": "Autonomous agents",
        }
    ]
