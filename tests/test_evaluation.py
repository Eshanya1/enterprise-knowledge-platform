import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation import evaluate, precision_at_k, reciprocal_rank


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_partial():
    assert precision_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 1 / 3


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_second_hit():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_evaluate_runs_and_returns_reasonable_scores():
    result = evaluate(verbose=False)
    assert 0.0 <= result["precision_at_k"] <= 1.0
    assert 0.0 <= result["mrr"] <= 1.0
    assert result["mrr"] > 0.5  # sanity: retriever should do reasonably well on its own labeled set
