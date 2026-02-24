-- ============================================================
-- V1__create_core_tables.sql
-- Iron Vault Registry — Full v1 Schema (reconciled)
-- ============================================================

-- ============================================================
-- 1. AUTH + SECURITY
-- ============================================================

CREATE TABLE clients (
    client_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    api_key_hash   TEXT NOT NULL,
    is_admin       INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    created_at     TEXT NOT NULL,
    last_used_at   TEXT
);

CREATE TABLE agents (
    agent_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    namespace         TEXT NOT NULL UNIQUE,
    role              TEXT NOT NULL DEFAULT 'agent',
    can_write_public  INTEGER NOT NULL DEFAULT 0 CHECK (can_write_public IN (0, 1)),
    registered_by     TEXT REFERENCES clients(client_id),
    registered_at     TEXT NOT NULL,
    last_seen_at      TEXT
);

CREATE TABLE agent_tokens (
    token_id       TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL REFERENCES agents(agent_id),
    client_id      TEXT NOT NULL REFERENCES clients(client_id),
    token_hash     TEXT NOT NULL,
    capabilities   TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(capabilities)),
    expires_at     TEXT NOT NULL,
    revoked        INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_agent_tokens_agent ON agent_tokens(agent_id);
CREATE INDEX idx_agent_tokens_expires ON agent_tokens(expires_at);
CREATE INDEX idx_agent_tokens_hash ON agent_tokens(token_hash);

CREATE TABLE sessions (
    session_id     TEXT PRIMARY KEY,
    client_id      TEXT NOT NULL REFERENCES clients(client_id),
    agent_id       TEXT NOT NULL REFERENCES agents(agent_id),
    token_id       TEXT NOT NULL REFERENCES agent_tokens(token_id),
    capabilities   TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(capabilities)),
    started_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    last_active_at TEXT
);
CREATE INDEX idx_sessions_agent ON sessions(agent_id);
CREATE INDEX idx_sessions_client ON sessions(client_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);

-- ============================================================
-- 2. LOOKUP / REGISTRATION
-- ============================================================

CREATE TABLE domains (
    domain_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT
);

INSERT INTO domains (name, description) VALUES
    ('systems',     'Infrastructure, DevOps, platform, networking'),
    ('security',    'Security, auth, cryptography'),
    ('finance',     'Quantitative finance, trading, risk'),
    ('ai_ml',       'Machine learning, AI, models'),
    ('data',        'Data engineering, pipelines, storage'),
    ('programming', 'Languages, frameworks, tooling'),
    ('math',        'Mathematics, statistics, optimization'),
    ('writing',     'Copywriting, documentation, commercial content'),
    ('product',     'Product design, UX, features');

CREATE TABLE tags (
    tag_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE
);

-- ============================================================
-- 3. CORE NOTE TABLES
-- ============================================================

-- Identity + head pointer
CREATE TABLE notes (
    note_id              TEXT PRIMARY KEY,
    namespace            TEXT NOT NULL DEFAULT 'ark',
    head_version_id      TEXT NOT NULL,
    canonical_version_id TEXT,
    author_agent_id      TEXT REFERENCES agents(agent_id),
    created_at           TEXT NOT NULL
);
CREATE INDEX idx_notes_namespace ON notes(namespace);

-- Append-only version history
CREATE TABLE note_versions (
    version_id       TEXT PRIMARY KEY,
    note_id          TEXT NOT NULL REFERENCES notes(note_id),
    prev_version_id  TEXT,
    author_agent_id  TEXT NOT NULL REFERENCES agents(agent_id),
    session_id       TEXT REFERENCES sessions(session_id),
    content_hash     TEXT NOT NULL,
    fm_hash          TEXT NOT NULL,
    md_hash          TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX idx_versions_note ON note_versions(note_id);
CREATE INDEX idx_versions_content ON note_versions(content_hash);
CREATE INDEX idx_versions_prev ON note_versions(prev_version_id);
CREATE INDEX idx_versions_created ON note_versions(created_at);

-- Materialized head view (hot path for browse + pack + search)
CREATE TABLE current_notes (
    note_id              TEXT PRIMARY KEY,
    namespace            TEXT NOT NULL DEFAULT 'ark',
    head_version_id      TEXT NOT NULL,
    canonical_version_id TEXT,
    author_agent_id      TEXT NOT NULL REFERENCES agents(agent_id),
    access               TEXT NOT NULL DEFAULT 'private',
    title                TEXT,
    domain               TEXT,
    intent               TEXT,
    kind                 TEXT,
    scope                TEXT,
    trust                TEXT,
    status               TEXT DEFAULT 'active',
    visibility           TEXT DEFAULT 'private',
    freshness            TEXT,
    compression_level    INTEGER DEFAULT 0 CHECK (compression_level >= 0),
    taint_level          TEXT DEFAULT 'agent',
    object_type          TEXT,
    object_name          TEXT,
    activation_score     REAL DEFAULT 0.0 CHECK (activation_score >= 0.0),
    updated_at           TEXT
);

-- Browse tree indexes (domain → intent → kind hierarchy)
CREATE INDEX idx_cn_domain         ON current_notes(domain);
CREATE INDEX idx_cn_domain_intent  ON current_notes(domain, intent);
CREATE INDEX idx_cn_browse         ON current_notes(domain, intent, kind, status);
CREATE INDEX idx_cn_namespace        ON current_notes(namespace);
CREATE INDEX idx_cn_namespace_browse ON current_notes(namespace, domain, intent, kind, status);
CREATE INDEX idx_cn_status         ON current_notes(status);
CREATE INDEX idx_cn_updated        ON current_notes(updated_at);
CREATE INDEX idx_cn_activation     ON current_notes(activation_score DESC);
CREATE INDEX idx_cn_object         ON current_notes(object_type, object_name);

-- Full-text search (head-only, 5 columns)
-- BM25 weights: title=5, body=1, spine=2, aliases=3, keywords=10
CREATE VIRTUAL TABLE note_text USING fts5(
    note_id UNINDEXED,
    title,
    body,
    spine,
    aliases,
    keywords,
    tokenize='unicode61'
);

-- Note graph (inline links extracted from body)
-- Syntax in body: note:<note_id> or note:<note_id>|<relation>
CREATE TABLE note_edges (
    src_note_id  TEXT NOT NULL,
    dst_note_id  TEXT NOT NULL,
    edge_type    TEXT NOT NULL DEFAULT 'link',
    relation     TEXT NOT NULL DEFAULT 'unspecified',
    version_id   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (src_note_id, dst_note_id, edge_type, relation)
);
CREATE INDEX idx_edges_src ON note_edges(src_note_id);
CREATE INDEX idx_edges_dst ON note_edges(dst_note_id);

-- ============================================================
-- 4. OWNERSHIP + COLLABORATION
-- ============================================================

CREATE TABLE note_tags (
    note_id  TEXT NOT NULL REFERENCES notes(note_id),
    tag_id   INTEGER NOT NULL REFERENCES tags(tag_id),
    PRIMARY KEY (note_id, tag_id)
);
CREATE INDEX idx_note_tags_tag ON note_tags(tag_id);

CREATE TABLE note_collaborators (
    note_id    TEXT NOT NULL REFERENCES notes(note_id),
    agent_id   TEXT NOT NULL REFERENCES agents(agent_id),
    granted_at TEXT NOT NULL,
    PRIMARY KEY (note_id, agent_id)
);

-- ============================================================
-- 5. AUDIT + META
-- ============================================================

CREATE TABLE audit_log (
    event_id     TEXT PRIMARY KEY,
    session_id   TEXT REFERENCES sessions(session_id),
    client_id    TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    action       TEXT NOT NULL,
    target_type  TEXT,
    target_id    TEXT,
    detail       TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_audit_agent  ON audit_log(agent_id);
CREATE INDEX idx_audit_target ON audit_log(target_type, target_id);
CREATE INDEX idx_audit_time   ON audit_log(created_at);
CREATE INDEX idx_audit_client_time ON audit_log(client_id, created_at);

