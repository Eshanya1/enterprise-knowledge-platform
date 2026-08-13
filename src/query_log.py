"""
SQLite-backed query log: persists every query, its citations, and latency
so the platform has an audit trail and the /stats endpoint has real data
to report -- rather than the pipeline being entirely stateless/ephemeral.
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "query_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    citations TEXT,
    latency_ms REAL,
    ts REAL DEFAULT (strftime('%s','now'))
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(SCHEMA)


def log_query(query: str, citations: list, latency_ms: float):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO queries (query, citations, latency_ms) VALUES (?, ?, ?)",
            (query, json.dumps(citations), latency_ms),
        )


def recent_queries(limit: int = 20) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT query, citations, latency_ms, ts FROM queries ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"query": q, "citations": json.loads(c), "latency_ms": lat, "ts": ts}
            for q, c, lat, ts in rows
        ]


def stats() -> dict:
    with _conn() as conn:
        row = conn.execute("SELECT COUNT(*), AVG(latency_ms) FROM queries").fetchone()
        count, avg_latency = row
        return {"total_queries": count or 0, "avg_latency_ms": round(avg_latency, 1) if avg_latency else None}
