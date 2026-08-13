import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieval import HybridRetriever


def test_top_hit_is_relevant_for_specific_query():
    r = HybridRetriever()
    hits = r.search("token refresh race condition", top_k=1)
    assert hits[0].doc_id == "doc-3"


def test_search_returns_requested_count():
    r = HybridRetriever()
    hits = r.search("service overview", top_k=3)
    assert len(hits) == 3


def test_scores_are_descending():
    r = HybridRetriever()
    hits = r.search("billing payment invoice", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
