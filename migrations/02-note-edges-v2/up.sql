DROP TABLE IF EXISTS note_edges;

CREATE TABLE note_edges (
    src_note_id  TEXT NOT NULL,
    dst_note_id  TEXT NOT NULL,
    edge_type    TEXT NOT NULL DEFAULT 'references',
    weight       REAL NOT NULL DEFAULT 1.0,
    source_type  TEXT NOT NULL DEFAULT 'body',
    context      TEXT,
    version_id   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (src_note_id, dst_note_id, edge_type)
);
CREATE INDEX idx_edges_src ON note_edges(src_note_id);
CREATE INDEX idx_edges_dst ON note_edges(dst_note_id);

ALTER TABLE current_notes ADD COLUMN links_out_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE current_notes ADD COLUMN links_in_count INTEGER NOT NULL DEFAULT 0;
