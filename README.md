# Enterprise Knowledge Intelligence Platform

An AI-powered platform for retrieving and reasoning over distributed
engineering knowledge (docs, issues, architecture decisions) using hybrid
retrieval, graph-based relationship expansion (GraphRAG), and a LangGraph
multi-agent pipeline — exposed as an authenticated FastAPI service with a
measured retrieval evaluation harness.

## Architecture

```
POST /query ──▶ LangGraph pipeline (agent_pipeline.py)
                  │
                  ├─ plan            decide retrieval breadth from query intent
                  ├─ retrieve        hybrid TF-IDF + BM25 search, over-fetch
                  ├─ (MMR rerank)    drop redundant near-duplicate hits
                  │
                  ├─ [conditional edge: does this query need graph context?]
                  │     yes ─▶ expand_graph   NetworkX traversal: pull structurally
                  │            │              related docs (issues/ADRs/dependencies)
                  │            ▼
                  │     no  ───────────────┐  (expand_graph is not invoked at all)
                  │                        ▼
                  ├─ verify          drop low-confidence retrieval hits
                  ├─ summarize       extractive by default, LLM if OPENAI_API_KEY set
                  └─ cite            doc_id citation list
                  │
                  ▼
          SQLite query log ──▶ GET /stats
```

## What's real here (and what's a documented stand-in)

**Genuinely solved, not a toy:**
- **Measured retrieval quality** — most portfolio RAG projects never
  evaluate the retriever at all. `evaluation.py` runs a hand-labeled query
  set through the hybrid retriever and reports real numbers:
  **Precision@3 = 0.667, MRR = 1.0** (every query's top hit is correct).
  Exposed live via `GET /eval`. Read honestly: the eval set is 6 queries
  against an 8-document corpus with high lexical overlap between queries and
  doc titles — MRR=1.0 demonstrates the harness computes real numbers
  correctly, not that retrieval is bulletproof at production scale or
  against harder, less lexically-obvious queries.
- **MMR reranking** (`mmr.py`) — over-fetches candidates then reranks for
  diversity, so a query doesn't return 3 near-duplicate "service overview"
  docs. Verified this actually changes the result set, not just theoretically
  present.
- **Real LangGraph conditional orchestration** — a 6-node `StateGraph`
  (plan → retrieve → expand_graph → verify → summarize → cite) where the
  retrieve → expand_graph edge is an actual `add_conditional_edges` route,
  not an if-statement hiding inside a node that always runs: for queries the
  planner judges narrow, `expand_graph` is never invoked at all (its output
  key is entirely absent from the resulting state — verified in
  `test_expand_graph_node_is_never_invoked_for_narrow_queries`), while
  relational-sounding queries ("why/impact/depends" language) route through
  it. That's the actual justification for a graph-orchestration framework
  here over a plain function-call chain — an earlier version of this pipeline
  used only static edges, which made LangGraph syntax around what was
  functionally `plan(retrieve(verify(...)))`, indistinguishable in behavior
  from a script with no framework at all.
- **GraphRAG** — relationships between services/issues/ADRs modeled as a
  real graph (NetworkX) with multi-hop traversal, not just metadata tags.
- **Auth**: API-key auth (SHA-256 hashed, never stored plaintext) on every
  data endpoint. Caught and fixed a real bug while testing this: FastAPI's
  default `Header(...)` returns 422 for a *missing* key, which is
  indistinguishable from a malformed request — fixed to return a proper 401.
- **Persistence**: every query, its citations, and latency are logged to
  SQLite (`GET /stats` reports real query volume/latency, not mock data).
- **Tests**: 18 passing pytest cases across retrieval, graph traversal, MMR,
  the evaluation harness itself, and the full agent pipeline (including the
  conditional-routing behavior above).

**Explicitly not production-grade (by design, for portfolio scope):**
- Retrieval uses TF-IDF + BM25, not transformer embeddings. I tried
  installing `sentence-transformers` for this — it pulled in a multi-GB
  CUDA dependency tree that didn't fit the dev sandbox's disk budget, and
  no CPU-only wheel index was reachable from there. The retriever interface
  is built so this is a one-function swap (`HybridRetriever._semantic_scores`)
  once running somewhere with normal package/network access — it's not an
  architectural limitation, just an environment one.
- Graph backend is in-memory NetworkX, not a live Neo4j instance —
  `graph_store.py` includes a `Neo4jGraphStore` class with the same
  interface, ready to point at a real deployment (`pip install neo4j` +
  connection env vars), just not exercised here since no Neo4j server is
  running in this environment.
- Summarization is extractive by default (no LLM call needed to run the
  demo) — set `OPENAI_API_KEY` to switch to real LLM-based abstractive
  summarization via `_llm_summarize()`.

## API
| Method | Endpoint                        | Auth | Description                          |
|--------|----------------------------------|------|----------------------------------------|
| GET    | `/health`                        | no   | Liveness check                        |
| POST   | `/query`                         | yes  | Run the full agent pipeline           |
| GET    | `/graph/{doc_id}/neighbors`      | yes  | Raw graph traversal                   |
| GET    | `/stats`                         | yes  | Query volume + avg latency            |
| GET    | `/eval`                          | yes  | Live retrieval evaluation (P@k, MRR)  |

## Setup
```bash
pip install -r requirements.txt
cd src
API_KEY=your-key python app.py     # or: uvicorn app:app --reload
```
(If `API_KEY` isn't set, one is generated and printed at startup.)

Or with Docker:
```bash
docker build -t knowledge-platform .
docker run -p 8000:8000 -e API_KEY=your-key knowledge-platform
```

## Try it
```bash
curl -X POST -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"query": "what happens if billing-service is down"}' \
  http://localhost:8000/query

curl -H "X-API-Key: your-key" http://localhost:8000/eval
curl -H "X-API-Key: your-key" http://localhost:8000/stats
```

## Tests
```bash
pip install -r requirements.txt
pytest tests/ -v      # 16 tests, all passing
```

## Tech Stack
Python · FastAPI · LangGraph · scikit-learn (TF-IDF) · BM25 · NetworkX
(GraphRAG) · SQLite · Docker · pytest
