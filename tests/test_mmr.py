import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieval import HybridRetriever
from mmr import mmr_rerank


def test_mmr_returns_requested_count():
    r = HybridRetriever()
    candidates = r.search("service overview dependencies", top_k=6)
    reranked = mmr_rerank("service overview dependencies", candidates, top_k=3)
    assert len(reranked) == 3


def test_mmr_keeps_top_relevance_hit_first():
    r = HybridRetriever()
    candidates = r.search("token refresh race condition", top_k=5)
    reranked = mmr_rerank("token refresh race condition", candidates, top_k=3, lambda_param=0.9)
    assert reranked[0].doc_id == candidates[0].doc_id
