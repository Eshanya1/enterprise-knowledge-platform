import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent_pipeline import answer_query


def test_pipeline_returns_citations_for_known_topic():
    result = answer_query("why does token refresh fail")
    assert len(result["citations"]) > 0
    assert "doc-3" in result["citations"]


def test_pipeline_expands_graph_for_relational_query():
    result = answer_query("what depends on the auth service")
    assert result["plan"]["expand_graph"] is True


def test_pipeline_handles_narrow_query_without_graph_expansion():
    result = answer_query("billing service payments")
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_expand_graph_node_is_never_invoked_for_narrow_queries():
    """Real conditional-edge routing, not an if-statement hiding inside a
    node: for a narrow query, expand_graph_node should never run at all, so
    its output key is entirely absent from the final state -- not present
    and empty."""
    result = answer_query("billing service payments")
    assert result["plan"]["expand_graph"] is False
    assert "expanded" not in result


def test_expand_graph_node_is_invoked_for_broad_queries():
    result = answer_query("what depends on the auth service")
    assert result["plan"]["expand_graph"] is True
    assert "expanded" in result
