"""
Retrieval evaluation harness. Most portfolio RAG projects skip this
entirely; a labeled query set + measured precision/MRR is what turns
"I built a retriever" into "I can show it actually works."

Labels below were assigned by inspecting sample_corpus.py by hand -- each
query's `relevant` set is every doc that a competent engineer would
consider a correct/useful hit for that query, not just the single best
match.
"""
from dataclasses import dataclass
from typing import List, Set
from retrieval import HybridRetriever

EVAL_SET = [
    {"query": "why does token refresh fail intermittently", "relevant": {"doc-3", "doc-1"}},
    {"query": "what does the auth service depend on", "relevant": {"doc-1", "doc-2"}},
    {"query": "billing service payment processing", "relevant": {"doc-4"}},
    {"query": "notification retry storm webhook failures", "relevant": {"doc-6", "doc-7", "doc-8"}},
    {"query": "session storage migration to redis", "relevant": {"doc-5", "doc-1"}},
    {"query": "circuit breaker for outbound calls", "relevant": {"doc-8", "doc-6"}},
]


def precision_at_k(retrieved_ids: List[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def reciprocal_rank(retrieved_ids: List[str], relevant: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: HybridRetriever = None, k: int = 3, verbose: bool = True):
    retriever = retriever or HybridRetriever()
    precisions, rrs = [], []

    for case in EVAL_SET:
        hits = retriever.search(case["query"], top_k=max(k, 5))
        retrieved_ids = [h.doc_id for h in hits]
        p = precision_at_k(retrieved_ids, case["relevant"], k)
        rr = reciprocal_rank(retrieved_ids, case["relevant"])
        precisions.append(p)
        rrs.append(rr)
        if verbose:
            print(f"[P@{k}={p:.2f} RR={rr:.2f}] {case['query']!r} -> {retrieved_ids[:k]}")

    mean_precision = sum(precisions) / len(precisions)
    mrr = sum(rrs) / len(rrs)
    if verbose:
        print(f"\nMean Precision@{k}: {mean_precision:.3f}")
        print(f"Mean Reciprocal Rank: {mrr:.3f}")
    return {"precision_at_k": mean_precision, "mrr": mrr, "k": k}


if __name__ == "__main__":
    evaluate()
