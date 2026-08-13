"""
GraphRAG layer: models relationships between repositories, services,
issues, and engineering decisions so retrieval can be expanded via graph
traversal ("what else is connected to this doc?") rather than text
similarity alone.

Default backend is in-memory NetworkX so the project runs with zero
infrastructure. `Neo4jGraphStore` below shows the swap-in for a real Neo4j
deployment (same interface, not exercised in this environment since it
requires a running Neo4j instance -- install `neo4j` and point
NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD env vars at a real database to use it).
"""
from typing import List, Dict
import networkx as nx

from sample_corpus import DOCUMENTS, RELATIONSHIPS


class GraphStore:
    """In-memory graph backend (NetworkX). No external services required."""

    def __init__(self, documents: List[dict] = None, relationships: List[tuple] = None):
        self.documents = documents or DOCUMENTS
        self.relationships = relationships or RELATIONSHIPS
        self.graph = nx.MultiDiGraph()
        for d in self.documents:
            self.graph.add_node(d["id"], **d)
        for src, rel, dst in self.relationships:
            self.graph.add_edge(src, dst, relation=rel)

    def neighbors(self, doc_id: str, max_hops: int = 1) -> List[Dict]:
        """Return docs connected to doc_id within max_hops, with the
        relation path used to reach them."""
        if doc_id not in self.graph:
            return []
        visited = {doc_id: []}
        frontier = [doc_id]
        for _ in range(max_hops):
            next_frontier = []
            for node in frontier:
                for _, nbr, data in self.graph.out_edges(node, data=True):
                    if nbr not in visited:
                        visited[nbr] = visited[node] + [data["relation"]]
                        next_frontier.append(nbr)
                for pred, _, data in self.graph.in_edges(node, data=True):
                    if pred not in visited:
                        visited[pred] = visited[node] + [f"<-{data['relation']}"]
                        next_frontier.append(pred)
            frontier = next_frontier

        results = []
        for node_id, path in visited.items():
            if node_id == doc_id:
                continue
            node_data = self.graph.nodes[node_id]
            results.append({
                "doc_id": node_id,
                "title": node_data.get("title"),
                "relation_path": " -> ".join(path),
            })
        return results


class Neo4jGraphStore:
    """
    Production swap-in for GraphStore, backed by a real Neo4j instance.
    Same `neighbors()` interface. Requires `pip install neo4j` and a
    running Neo4j server -- not exercised in this sandboxed environment.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # imported lazily; optional dependency
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def load(self, documents: List[dict], relationships: List[tuple]):
        with self.driver.session() as session:
            for d in documents:
                session.run(
                    "MERGE (n:Doc {id: $id}) SET n += $props",
                    id=d["id"], props={k: v for k, v in d.items() if k != "id"},
                )
            for src, rel, dst in relationships:
                session.run(
                    "MATCH (a:Doc {id: $src}), (b:Doc {id: $dst}) "
                    f"MERGE (a)-[:{rel}]->(b)",
                    src=src, dst=dst,
                )

    def neighbors(self, doc_id: str, max_hops: int = 1) -> List[Dict]:
        query = (
            f"MATCH (start:Doc {{id: $doc_id}})-[r*1..{max_hops}]-(n:Doc) "
            "RETURN DISTINCT n.id AS doc_id, n.title AS title"
        )
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query, doc_id=doc_id)]


if __name__ == "__main__":
    store = GraphStore()
    for hit in store.neighbors("doc-1", max_hops=1):
        print(hit)
