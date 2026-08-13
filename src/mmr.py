"""
Maximal Marginal Relevance reranking: reduces redundancy in the top-k
retrieved set by penalizing candidates that are too textually similar to
what's already been selected. Without this, hybrid retrieval can return
3 near-duplicate "overview" docs instead of a diverse, useful set --
exactly the kind of thing that looks fine in a demo query but falls apart
under real usage.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import RetrievedDoc


def mmr_rerank(query: str, candidates: List[RetrievedDoc], top_k: int = 5, lambda_param: float = 0.7) -> List[RetrievedDoc]:
    """lambda_param closer to 1 favors relevance, closer to 0 favors diversity."""
    if len(candidates) <= 1:
        return candidates

    texts = [f"{c.title} {c.text}" for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english").fit(texts + [query])
    doc_vectors = vectorizer.transform(texts)
    query_vector = vectorizer.transform([query])

    relevance = cosine_similarity(query_vector, doc_vectors).flatten()
    similarity_matrix = cosine_similarity(doc_vectors)

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < top_k:
        if not selected_idx:
            best = max(remaining_idx, key=lambda i: relevance[i])
        else:
            def mmr_score(i):
                redundancy = max(similarity_matrix[i][j] for j in selected_idx)
                return lambda_param * relevance[i] - (1 - lambda_param) * redundancy
            best = max(remaining_idx, key=mmr_score)
        selected_idx.append(best)
        remaining_idx.remove(best)

    return [candidates[i] for i in selected_idx]
