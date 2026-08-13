"""
Small sample corpus standing in for 'distributed engineering knowledge'
(docs, repos, issues, technical decisions). Swap for real ingestion from
Confluence/GitHub/Jira connectors in production -- the retrieval and graph
layers only depend on this schema, not the source.
"""

DOCUMENTS = [
    {"id": "doc-1", "type": "doc", "title": "Auth Service Overview",
     "repo": "auth-service", "team": "platform",
     "text": "The auth-service handles login, token issuance, and session "
             "management for all internal APIs. It exposes REST endpoints "
             "for /login, /refresh, and /logout, and depends on the "
             "user-service for credential validation."},
    {"id": "doc-2", "type": "doc", "title": "User Service Overview",
     "repo": "user-service", "team": "platform",
     "text": "The user-service owns the canonical user profile store. It is "
             "consumed by auth-service for credential checks and by "
             "billing-service for account status lookups."},
    {"id": "doc-3", "type": "issue", "title": "Token refresh race condition",
     "repo": "auth-service", "team": "platform",
     "text": "Under concurrent refresh requests, two valid refresh tokens "
             "can be issued for the same session, causing intermittent "
             "401s in downstream services. Root cause traced to a missing "
             "row lock in the token table."},
    {"id": "doc-4", "type": "doc", "title": "Billing Service Overview",
     "repo": "billing-service", "team": "payments",
     "text": "billing-service calculates invoices and processes payments. "
             "It calls user-service to confirm account status before "
             "charging, and publishes payment-completed events consumed "
             "by the notification-service."},
    {"id": "doc-5", "type": "decision", "title": "ADR: Move sessions to Redis",
     "repo": "auth-service", "team": "platform",
     "text": "Decision record: session storage is moving from Postgres to "
             "Redis to reduce login latency. auth-service will read/write "
             "sessions through a new SessionStore interface, decoupling it "
             "from the underlying store."},
    {"id": "doc-6", "type": "issue", "title": "Notification service retry storm",
     "repo": "notification-service", "team": "payments",
     "text": "notification-service retries failed billing-service webhook "
             "deliveries without backoff, causing a retry storm during "
             "billing-service outages. Needs exponential backoff and a "
             "circuit breaker."},
    {"id": "doc-7", "type": "doc", "title": "Notification Service Overview",
     "repo": "notification-service", "team": "payments",
     "text": "notification-service sends email/SMS/push notifications "
             "triggered by events from billing-service and user-service."},
    {"id": "doc-8", "type": "decision", "title": "ADR: Circuit breaker library",
     "repo": "notification-service", "team": "payments",
     "text": "Decision record: adopt a circuit breaker library for all "
             "outbound webhook calls in notification-service, starting "
             "with billing-service webhook consumption, to prevent retry "
             "storms during downstream outages."},
]

# Explicit relationships for the knowledge graph layer (GraphRAG).
# (source_id, relation, target_id)
RELATIONSHIPS = [
    ("doc-1", "DEPENDS_ON", "doc-2"),
    ("doc-3", "AFFECTS", "doc-1"),
    ("doc-5", "MODIFIES", "doc-1"),
    ("doc-4", "DEPENDS_ON", "doc-2"),
    ("doc-4", "PUBLISHES_TO", "doc-7"),
    ("doc-6", "AFFECTS", "doc-7"),
    ("doc-6", "RELATES_TO", "doc-4"),
    ("doc-8", "RESOLVES", "doc-6"),
]
