//! Pure-Rust BM25 baseline using SQLite FTS5 with no ranking layers on top.
//! This is the floor every other adapter is benchmarked against.

use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;
use std::time::Instant;

use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

pub struct Fts5Adapter {
    conn: Option<Connection>,
}

impl Fts5Adapter {
    pub fn new() -> Self {
        Self { conn: None }
    }
}

impl Default for Fts5Adapter {
    fn default() -> Self { Self::new() }
}

impl Adapter for Fts5Adapter {
    fn name(&self) -> &str { "fts5" }

    fn version(&self) -> Result<String> {
        Ok(format!("sqlite-fts5 {}", rusqlite::version()))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        let db_path = workdir.join("fts5.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(doc_id UNINDEXED, body);"
        )?;
        self.conn = Some(conn);
        Ok(())
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        let t0 = Instant::now();
        let conn = self.conn.as_ref().ok_or_else(|| anyhow::anyhow!("setup not called"))?;
        conn.execute(
            "INSERT INTO docs (doc_id, body) VALUES (?1, ?2)",
            rusqlite::params![doc.id, doc.body],
        )?;
        Ok(WriteMetrics {
            latency_ms: t0.elapsed().as_millis() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        })
    }

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        let t0 = Instant::now();
        let conn = self.conn.as_ref().ok_or_else(|| anyhow::anyhow!("setup not called"))?;
        let sanitized = sanitize_fts5_query(query);
        let mut stmt = conn.prepare(
            "SELECT doc_id, bm25(docs) AS score, snippet(docs, 1, '[', ']', '...', 16) AS snip
             FROM docs WHERE docs MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(rusqlite::params![sanitized, k as i64], |row| {
                let doc_id: String = row.get(0)?;
                let raw: f64 = row.get(1)?;
                let snippet: String = row.get(2).unwrap_or_default();
                // FTS5 bm25() returns smaller (more-negative) is better. Negate so higher = better
                // and the downstream metrics see a "score" with conventional semantics.
                Ok(SearchResult {
                    doc_id,
                    score: (-raw) as f32,
                    snippet: if snippet.is_empty() { None } else { Some(snippet) },
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok((rows, SearchMetrics {
            latency_ms: t0.elapsed().as_millis() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        }))
    }

    fn teardown(&mut self) -> Result<()> {
        self.conn = None;
        Ok(())
    }
}

/// Strip FTS5 operator characters from a free-text query so that hyphens, colons,
/// and quotes don't get interpreted as column-prefix or phrase syntax. Replaces
/// any non-alphanumeric, non-space character with a space, then collapses runs
/// of whitespace.
fn sanitize_fts5_query(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut prev_space = true;
    for c in query.chars() {
        let ch = if c.is_alphanumeric() { c } else { ' ' };
        if ch == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn fts5_smoke_write_then_search() {
        let dir = tempdir().unwrap();
        let mut a = Fts5Adapter::new();
        a.setup(dir.path()).unwrap();

        let d1 = Document { id: "a".into(), body: "the quick brown fox".into(), metadata: json!({}) };
        let d2 = Document { id: "b".into(), body: "lazy dogs sleep".into(), metadata: json!({}) };
        a.write(&d1).unwrap();
        a.write(&d2).unwrap();

        let (results, _) = a.search("brown fox", 5).unwrap();
        assert!(!results.is_empty(), "expected at least one hit, got 0");
        assert_eq!(results[0].doc_id, "a", "expected 'a' to rank first, got {}", results[0].doc_id);
        a.teardown().unwrap();
    }

    #[test]
    fn sanitize_strips_special_chars() {
        assert_eq!(sanitize_fts5_query("pawn structure asymmetric c-file"), "pawn structure asymmetric c file");
        assert_eq!(sanitize_fts5_query("foo: bar"), "foo bar");
        assert_eq!(sanitize_fts5_query("\"hello\" world"), "hello world");
        assert_eq!(sanitize_fts5_query("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn fts5_handles_hyphenated_query() {
        let dir = tempdir().unwrap();
        let mut a = Fts5Adapter::new();
        a.setup(dir.path()).unwrap();
        a.write(&Document { id: "a".into(), body: "the c-file is open".into(), metadata: json!({}) }).unwrap();
        // Without sanitization, FTS5 would interpret `c-file` as a column prefix and error
        let (results, _) = a.search("c-file", 5).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn fts5_returns_no_hits_for_unrelated_query() {
        let dir = tempdir().unwrap();
        let mut a = Fts5Adapter::new();
        a.setup(dir.path()).unwrap();
        a.write(&Document { id: "x".into(), body: "completely unrelated content".into(), metadata: json!({}) }).unwrap();
        let (results, _) = a.search("zzzz never appears", 5).unwrap();
        assert!(results.is_empty(), "expected 0 hits, got {}", results.len());
    }
}
