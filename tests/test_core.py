"""Offline unit tests (no models, no network)."""
from rag.generate import check_citations, format_sources
from rag.retrieve import rrf, tokenize


def test_tokenize_drops_stopwords_and_keeps_hyphenated_terms():
    assert tokenize("How does the Multi-Hop retrieval work?") == ["multi-hop", "retrieval", "work"]


def test_rrf_rewards_agreement_between_lists():
    # 2 is second in both lists; 1 and 4 are first in only one. Agreement wins.
    fused = [i for i, _ in rrf([[1, 2, 3], [4, 2, 5]])]
    assert fused[0] == 2
    assert set(fused) == {1, 2, 3, 4, 5}


def test_rrf_uses_ranks_not_scores():
    a = rrf([[10, 20]])
    assert a[0][1] == 1 / 61 and a[1][1] == 1 / 62


def test_check_citations_flags_numbers_outside_sources():
    out = check_citations("Fact one [1]. Fact two [2][7].", n_sources=5)
    assert out == {"cited": [1, 2], "invalid": [7]}


def test_format_sources_numbers_and_labels_passages():
    s = format_sources([
        {"title": "Paper A", "paper": "2401.00001", "page": 0, "text": "abs"},
        {"title": "Paper B", "paper": "2401.00002", "page": 4, "text": "body"},
    ])
    assert s.startswith("[1] Paper A (arXiv:2401.00001, abstract)")
    assert "[2] Paper B (arXiv:2401.00002, page 4)" in s
