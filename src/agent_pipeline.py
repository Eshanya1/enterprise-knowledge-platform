"""
Multi-agent workflow implemented as a LangGraph StateGraph:

  plan -> retrieve -> [expand_graph?] -> verify -> summarize -> cite

The plan -> expand_graph edge is a real conditional edge (add_conditional_edges),
not a static chain with an if-statement hiding inside a node: for queries the
planner judges narrow (e.g. "billing service payments"), expand_graph is never
invoked at all -- the graph is genuinely shaped differently per query, which is
the actual reason to use a graph-orchestration framework here rather than a
plain `plan(retrieve(verify(...)))` call chain. (An earlier version always ran
expand_graph and had it silently no-op internally -- same end state, but
LangGraph syntax around what was functionally a linear script.)

Each node is a plain Python function operating on a shared state dict --
no LLM API key required to run this end-to-end (summarization uses
extractive sentence selection by default). Set OPENAI_API_KEY to switch
`summarize` to a real LLM call via LangChain's ChatOpenAI (see the
`_llm_summarize` stub) for higher-quality abstractive summaries.
"""
import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from retrieval import HybridRetriever
from graph_store import GraphStore
from mmr import mmr_rerank

RETRIEVER = HybridRetriever()
GRAPH = GraphStore()


class AgentState(TypedDict):
    query: str
    plan: Dict[str, Any]
    retrieved: List[Dict]
    expanded: List[Dict]
    verified: List[Dict]
    summary: str
    citations: List[str]


def plan_node(state: AgentState) -> AgentState:
    """Very lightweight query planner: decide retrieval breadth from query length/keywords."""
    query = state["query"]
    broaden = any(w in query.lower() for w in ["why", "impact", "affect", "related", "depends"])
    state["plan"] = {"top_k": 4 if broaden else 3, "expand_graph": broaden}
    return state


def retrieve_node(state: AgentState) -> AgentState:
    top_k = state["plan"]["top_k"]
    # Over-fetch, then MMR-rerank down to top_k to reduce redundant near-duplicate hits
    candidates = RETRIEVER.search(state["query"], top_k=max(top_k * 2, 6))
    hits = mmr_rerank(state["query"], candidates, top_k=top_k, lambda_param=0.7)
    state["retrieved"] = [
        {"doc_id": h.doc_id, "title": h.title, "text": h.text, "score": h.score, "source": "retrieval"}
        for h in hits
    ]
    return state


def expand_graph_node(state: AgentState) -> AgentState:
    # No internal skip-check here anymore -- routing to this node at all is
    # now the graph's job (see route_after_retrieve / add_conditional_edges
    # below), not a no-op branch hiding inside the node.
    seen = {d["doc_id"] for d in state["retrieved"]}
    expanded = []
    for d in state["retrieved"][:2]:  # expand from the top hits only
        for nbr in GRAPH.neighbors(d["doc_id"], max_hops=1):
            if nbr["doc_id"] not in seen:
                seen.add(nbr["doc_id"])
                expanded.append({
                    "doc_id": nbr["doc_id"],
                    "title": nbr["title"],
                    "relation_path": nbr["relation_path"],
                    "source": "graph_expansion",
                })
    state["expanded"] = expanded
    return state


def verify_node(state: AgentState) -> AgentState:
    """Drop low-confidence retrieval hits; graph-expanded docs are kept
    (they're structurally relevant even if lexically dissimilar). expand_graph
    may have been skipped entirely by the router above, so .get() defensively
    rather than assuming the key exists."""
    verified = [d for d in state["retrieved"] if d["score"] > 0.05] + state.get("expanded", [])
    state["verified"] = verified
    return state


def _extractive_summary(state: AgentState) -> str:
    lines = [f"Query: {state['query']}", ""]
    for d in state["verified"]:
        if d.get("source") == "graph_expansion":
            lines.append(f"- [{d['doc_id']}] {d['title']} (connected via {d['relation_path']})")
        else:
            snippet = d["text"][:160].rsplit(" ", 1)[0] + "..."
            lines.append(f"- [{d['doc_id']}] {d['title']}: {snippet}")
    return "\n".join(lines)


def _llm_summarize(state: AgentState) -> str:  # pragma: no cover - optional path
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context = "\n\n".join(d.get("text", d.get("title", "")) for d in state["verified"])
    prompt = (
        f"Answer the engineering question using only the context below. "
        f"Cite doc ids in brackets.\n\nQuestion: {state['query']}\n\nContext:\n{context}"
    )
    return llm.invoke(prompt).content


def summarize_node(state: AgentState) -> AgentState:
    if os.environ.get("OPENAI_API_KEY"):
        state["summary"] = _llm_summarize(state)
    else:
        state["summary"] = _extractive_summary(state)
    return state


def cite_node(state: AgentState) -> AgentState:
    state["citations"] = [d["doc_id"] for d in state["verified"]]
    return state


def route_after_retrieve(state: AgentState) -> str:
    """Real conditional routing, not an if-statement inside a node: only
    queries the planner judged relational/broad (state["plan"]["expand_graph"])
    actually visit expand_graph. Everything else skips straight to verify."""
    return "expand_graph" if state["plan"]["expand_graph"] else "verify"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("expand_graph", expand_graph_node)
    g.add_node("verify", verify_node)
    g.add_node("summarize", summarize_node)
    g.add_node("cite", cite_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"expand_graph": "expand_graph", "verify": "verify"})
    g.add_edge("expand_graph", "verify")
    g.add_edge("verify", "summarize")
    g.add_edge("summarize", "cite")
    g.add_edge("cite", END)
    return g.compile()


APP = build_graph()


def answer_query(query: str) -> AgentState:
    return APP.invoke({"query": query})


if __name__ == "__main__":
    result = answer_query("why does token refresh fail intermittently and what depends on auth-service?")
    print(result["summary"])
    print("\nCitations:", result["citations"])
