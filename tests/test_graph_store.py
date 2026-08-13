import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_store import GraphStore


def test_direct_neighbor_found():
    store = GraphStore()
    neighbors = store.neighbors("doc-1", max_hops=1)
    ids = {n["doc_id"] for n in neighbors}
    assert "doc-2" in ids  # doc-1 DEPENDS_ON doc-2


def test_unknown_doc_returns_empty():
    store = GraphStore()
    assert store.neighbors("doc-does-not-exist") == []


def test_two_hop_expands_further():
    store = GraphStore()
    one_hop = {n["doc_id"] for n in store.neighbors("doc-8", max_hops=1)}
    two_hop = {n["doc_id"] for n in store.neighbors("doc-8", max_hops=2)}
    assert one_hop.issubset(two_hop)
    assert len(two_hop) >= len(one_hop)
