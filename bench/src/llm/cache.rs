//! SQLite-backed cache for LLM responses.
//!
//! Cache key: sha256(prompt + "\n" + model_id + "\n" + prompt_version).
//! Same prompt + same model + same prompt version = cache hit.
//!
//! Persisted to disk at the path provided to `LlmCache::open`. The bench
//! defaults to `bench/cache/llm.db` and commits this file so re-runs are
//! free across contributors.
//!
//! Schema is created idempotently on first open.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use std::path::Path;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS responses (
  cache_key TEXT PRIMARY KEY,
  backend_name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  call_kind TEXT NOT NULL,
  request TEXT NOT NULL,
  response TEXT NOT NULL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_usd_micros INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_backend ON responses(backend_name, model_id);
";

pub struct LlmCache {
    conn: Connection,
}

/// Entry stored in the cache. `request` is the full prompt; `response` is the
/// full text returned by the LLM.
#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub backend_name: String,
    pub model_id: String,
    pub prompt_version: String,
    pub call_kind: String,
    pub request: String,
    pub response: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}

impl LlmCache {
    /// Open (or create) a cache at the given path. Use `":memory:"` for tests.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open llm cache at {:?}", path))?;
        conn.execute_batch(SCHEMA_SQL)
            .context("failed to initialize llm cache schema")?;
        Ok(Self { conn })
    }

    /// Compute the cache key from prompt + model_id + prompt_version.
    pub fn key(prompt: &str, model_id: &str, prompt_version: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(b"\n");
        hasher.update(model_id.as_bytes());
        hasher.update(b"\n");
        hasher.update(prompt_version.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT backend_name, model_id, prompt_version, call_kind, request, response,
                    tokens_in, tokens_out, cost_usd_micros
             FROM responses WHERE cache_key = ?1"
        )?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(CachedEntry {
                backend_name: row.get(0)?,
                model_id: row.get(1)?,
                prompt_version: row.get(2)?,
                call_kind: row.get(3)?,
                request: row.get(4)?,
                response: row.get(5)?,
                tokens_in: row.get::<_, i64>(6)? as u64,
                tokens_out: row.get::<_, i64>(7)? as u64,
                cost_usd_micros: row.get::<_, i64>(8)? as u64,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn put(&self, key: &str, entry: &CachedEntry) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR REPLACE INTO responses
             (cache_key, backend_name, model_id, prompt_version, call_kind,
              request, response, tokens_in, tokens_out, cost_usd_micros, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                key,
                entry.backend_name,
                entry.model_id,
                entry.prompt_version,
                entry.call_kind,
                entry.request,
                entry.response,
                entry.tokens_in as i64,
                entry.tokens_out as i64,
                entry.cost_usd_micros as i64,
                now,
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mem() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn key_is_deterministic() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "claude", "1");
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_with_prompt() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello!", "claude", "1");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_with_model() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "gpt", "1");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_with_prompt_version() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "claude", "2");
        assert_ne!(k1, k2);
    }

    #[test]
    fn put_then_get_roundtrips() {
        let cache = mem();
        let key = "abc";
        let entry = CachedEntry {
            backend_name: "echo".into(),
            model_id: "echo-v1".into(),
            prompt_version: "1".into(),
            call_kind: "judge".into(),
            request: "Q".into(),
            response: "A".into(),
            tokens_in: 1,
            tokens_out: 2,
            cost_usd_micros: 100,
        };
        cache.put(key, &entry).unwrap();
        let got = cache.get(key).unwrap().unwrap();
        assert_eq!(got.response, "A");
        assert_eq!(got.tokens_in, 1);
    }

    #[test]
    fn get_returns_none_for_unknown_key() {
        let cache = mem();
        let got = cache.get("nonexistent").unwrap();
        assert!(got.is_none());
    }
}
