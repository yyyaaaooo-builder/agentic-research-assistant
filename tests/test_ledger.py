import pytest

from research_assistant.core.ledger import cited_finding_ids


@pytest.mark.parametrize(
    "answer, expected",
    [
        pytest.param("EWM wins [1]. The gap widens [2].", [1, 2], id="single-ids"),
        pytest.param("Both hold [1, 3]. Again [1].", [1, 3], id="list-without-repeats"),
        pytest.param("Errors include search loops [46–48].", [46, 47, 48], id="en-dash-range"),
        pytest.param("Mixed [2, 5-6] and [10].", [2, 5, 6, 10], id="list-with-hyphen-range"),
        pytest.param("See [arXiv:2210.03629 §4] and [2023 survey].", [], id="non-footnote-brackets"),
    ],
)
def test_cited_finding_ids(answer, expected):
    assert cited_finding_ids(answer) == expected
