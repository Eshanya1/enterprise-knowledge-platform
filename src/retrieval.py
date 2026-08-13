"""
Hybrid retrieval: TF-IDF cosine similarity (semantic-ish vector search
stand-in) combined with BM25 (keyword search), reciprocal-rank-fused into
a single ranked list.

Production note: swap `TfidfVectorizer` for real sentence embeddings
(e.g. sentence-transformers + FAISS, as named in the README) without
changing the `HybridRetriever` interface -- `_semantic_scores` is the only
method that would need to change.
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from sample_corpus import DOCUMENTS


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    repo: str
    score: float


class HybridRetriever:
    def __init__(self, documents: List[dict] = None):
        self.documents = documents or DOCUMENTS
        self.by_id = {d["id"]: d for d in self.documents}
        corpus_texts = [f"{d['title']} {d['text']}" for d in self.documents]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        tokenized = [t.lower().split() for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _semantic_scores(self, query: str) -> np.ndarray:
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.tfidf_matrix).flatten()

    def _keyword_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.lower().split()))

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        sem = self._semantic_scores(query)
        kw = self._keyword_scores(query)

        def normalize(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

        fused = 0.5 * normalize(sem) + 0.5 * normalize(kw)
        top_idx = np.argsort(-fused)[:top_k]

        results = []
        for i in top_idx:
            d = self.documents[i]
            results.append(RetrievedDoc(d["id"], d["title"], d["text"], d["repo"], float(fused[i])))
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    for r in retriever.search("why does token refresh fail intermittently", top_k=3):
        print(f"{r.score:.3f}  {r.doc_id:<8} {r.title}")
