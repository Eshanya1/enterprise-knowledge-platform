"""
FastAPI service exposing the multi-agent knowledge retrieval pipeline.

Endpoints:
  GET  /health
  POST /query                      - runs the full agent pipeline (requires X-API-Key)
  GET  /graph/{doc_id}/neighbors    - raw graph traversal for a given doc (requires X-API-Key)
  GET  /stats                       - query volume + avg latency (requires X-API-Key)
  GET  /eval                        - runs the retrieval evaluation harness (requires X-API-Key)
"""
import time
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from agent_pipeline import answer_query
from graph_store import GraphStore
from auth import verify_api_key
from evaluation import evaluate
import query_log

app = FastAPI(title="Enterprise Knowledge Intelligence Platform", version="0.2.0")
GRAPH = GraphStore()
query_log.init_db()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest, api_key: str = Depends(verify_api_key)):
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")

    t0 = time.time()
    result = answer_query(req.query)
    latency_ms = round((time.time() - t0) * 1000, 1)

    query_log.log_query(req.query, result["citations"], latency_ms)

    return {
        "query": req.query,
        "summary": result["summary"],
        "citations": result["citations"],
        "retrieved": result["retrieved"],
        "graph_expanded": result["expanded"],
        "latency_ms": latency_ms,
    }


@app.get("/graph/{doc_id}/neighbors")
def neighbors(doc_id: str, max_hops: int = 1, api_key: str = Depends(verify_api_key)):
    hits = GRAPH.neighbors(doc_id, max_hops=max_hops)
    if not hits and doc_id not in GRAPH.graph:
        raise HTTPException(404, "unknown doc_id")
    return {"doc_id": doc_id, "neighbors": hits}


@app.get("/stats")
def stats(api_key: str = Depends(verify_api_key)):
    return {**query_log.stats(), "recent_queries": query_log.recent_queries(limit=5)}


@app.get("/eval")
def eval_retriever(api_key: str = Depends(verify_api_key)):
    return evaluate(verbose=False)
