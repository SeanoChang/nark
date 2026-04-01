use anyhow::{bail, Result};
use chrono::Utc;
use regex::Regex;
use std::path::Path;

use crate::cli::util::truncate_at_word;
use crate::config;
use crate::db;
use crate::embed;
use crate::registry::{embeddings, resolve, search::{self, CosineContext, SearchFilters, SearchMode}};
use crate::vault::fs::Vault;

/// Parse a relative temporal shorthand (e.g. "1d", "7d", "24h", "1w", "1mo")
/// into an ISO 8601 timestamp string.
pub fn parse_temporal(input: &str) -> Result<String> {
    let re = Regex::new(r"^(\d+)(h|d|w|mo)$").unwrap();
    let caps = re.captures(input).ok_or_else(|| {
        anyhow::anyhow!("invalid temporal format '{}'. Use: 1d, 7d, 24h, 1w, 1mo", input)
    })?;
    let n: i64 = caps[1].parse()?;
    let unit = &caps[2];
    let duration = match unit {
        "h" => chrono::Duration::hours(n),
        "d" => chrono::Duration::days(n),
        "w" => chrono::Duration::weeks(n),
        "mo" => chrono::Duration::days(n * 30),
        _ => unreachable!(),
    };
    let ts = Utc::now() - duration;
    Ok(ts.to_rfc3339())
}

pub fn run(
    vault_dir: &Path,
    query: &str,
    domain: Option<&str>,
    kind: Option<&str>,
    intent: Option<&str>,
    tags: &[String],
    limit: usize,
    bm25_only: bool,
    semantic: bool,
    since: Option<&str>,
    before: Option<&str>,
) -> Result<()> {
    if bm25_only && semantic {
        bail!("--bm25 and --semantic are mutually exclusive");
    }

    let conn = db::open_registry(vault_dir)?;
    let cfg = config::load(vault_dir)?;

    let since_ts = since.map(parse_temporal).transpose()?;
    let before_ts = before.map(parse_temporal).transpose()?;

    let filters = SearchFilters {
        domain,
        kind,
        intent,
        tags,
        since: since_ts.as_deref(),
        before: before_ts.as_deref(),
        limit,
    };

    let mode = if bm25_only {
        SearchMode::Bm25Only
    } else if semantic {
        SearchMode::Semantic
    } else {
        SearchMode::Normal
    };

    // Build cosine context if embeddings are available and there's a query.
    // Skip for BM25-only mode (doesn't use cosine).
    let cosine_ctx = if mode != SearchMode::Bm25Only && !query.is_empty() && embeddings::has_embeddings(&conn) {
        build_cosine_context(vault_dir, &cfg, &conn, query)
    } else {
        None
    };

    let mut hits = search::search(&conn, query, &filters, &cfg.search, cosine_ctx.as_ref(), mode)?;

    let vault = Vault::new(vault_dir.to_path_buf());
    fill_missing_snippets(&conn, &vault, query, &mut hits);

    let results: Vec<serde_json::Value> = hits.iter().map(|h| {
        serde_json::json!({
            "id": h.note_id,
            "title": h.title,
            "domain": h.domain,
            "kind": h.kind,
            "snippet": h.snippet,
            "rank": h.rank,
            "links_in": h.links_in,
            "links_out": h.links_out,
        })
    }).collect();

    let out = serde_json::json!({
        "query": query,
        "domain": domain,
        "mode": match mode {
            SearchMode::Bm25Only => "bm25",
            SearchMode::Semantic => "semantic",
            SearchMode::Normal => "normal",
        },
        "hits": results.len(),
        "results": results,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn build_cosine_context(
    vault_dir: &Path,
    cfg: &config::Config,
    conn: &rusqlite::Connection,
    query: &str,
) -> Option<CosineContext> {
    let mut provider = embed::init_provider(vault_dir, &cfg.embedding)?;
    let query_embedding = provider.embed_query(query).ok()?;
    let all = embeddings::get_all_embeddings(conn).ok()?;

    // Guard: skip cosine if stored embeddings have different dimensions than the query
    // (e.g. provider changed from local/768 to openai/1536 without re-embedding)
    if let Some((_, first_vec)) = all.first() {
        if first_vec.len() != query_embedding.len() {
            eprintln!(
                "Warning: embedding dimension mismatch (query={}, stored={}). Run `nark embed build` to re-embed.",
                query_embedding.len(), first_vec.len()
            );
            return None;
        }
    }

    let note_embeddings = all.into_iter().collect();
    Some(CosineContext { query_embedding, note_embeddings })
}

/// For search hits with empty snippets, try FTS5 snippet with the original query,
/// then fall back to first 150 chars of note body from vault.
fn fill_missing_snippets(
    conn: &rusqlite::Connection,
    vault: &Vault,
    query: &str,
    hits: &mut [search::SearchHit],
) {
    for hit in hits.iter_mut() {
        if !hit.snippet.is_empty() {
            continue;
        }

        // Try FTS5 snippet if we have a query
        if !query.is_empty() {
            if let Ok(snippet) = try_fts_snippet(conn, query, &hit.note_id) {
                hit.snippet = snippet;
                continue;
            }
        }

        // Fallback: first 150 chars of body from vault
        match read_body_preview(conn, vault, &hit.note_id) {
            Ok(body) => hit.snippet = body,
            Err(e) => eprintln!("Warning: failed to read body preview for {}: {}", hit.note_id, e),
        }
    }
}

fn try_fts_snippet(conn: &rusqlite::Connection, query: &str, note_id: &str) -> anyhow::Result<String> {
    let mut stmt = conn.prepare(
        "SELECT snippet(note_text, 2, '[', ']', '...', 32)
         FROM note_text
         WHERE note_text MATCH ?1 AND note_id = ?2"
    )?;
    let snippet: String = stmt.query_row(rusqlite::params![query, note_id], |row| row.get(0))?;
    if snippet.is_empty() {
        anyhow::bail!("empty snippet");
    }
    Ok(snippet)
}

fn read_body_preview(conn: &rusqlite::Connection, vault: &Vault, note_id: &str) -> anyhow::Result<String> {
    let refs = resolve::get_ref(conn, note_id)?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
    Ok(truncate_at_word(&body, 150).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_temporal_valid_hours() {
        let result = parse_temporal("24h");
        assert!(result.is_ok());
        let ts = result.unwrap();
        assert!(ts.contains("T")); // RFC 3339 format
        assert!(ts.ends_with("+00:00") || ts.ends_with("Z"));
    }

    #[test]
    fn test_parse_temporal_valid_days() {
        assert!(parse_temporal("1d").is_ok());
        assert!(parse_temporal("7d").is_ok());
        assert!(parse_temporal("30d").is_ok());
    }

    #[test]
    fn test_parse_temporal_valid_weeks() {
        assert!(parse_temporal("1w").is_ok());
        assert!(parse_temporal("4w").is_ok());
    }

    #[test]
    fn test_parse_temporal_valid_months() {
        assert!(parse_temporal("1mo").is_ok());
        assert!(parse_temporal("3mo").is_ok());
    }

    #[test]
    fn test_parse_temporal_invalid_formats() {
        assert!(parse_temporal("1y").is_err());
        assert!(parse_temporal("abc").is_err());
        assert!(parse_temporal("").is_err());
        assert!(parse_temporal("1D").is_err()); // case sensitive
        assert!(parse_temporal("d").is_err());  // missing number
        assert!(parse_temporal("last tuesday").is_err());
        assert!(parse_temporal("2026-01-01").is_err());
    }

    #[test]
    fn test_parse_temporal_error_message() {
        let err = parse_temporal("1y").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid temporal format '1y'"));
        assert!(msg.contains("1d, 7d, 24h, 1w, 1mo"));
    }

    #[test]
    fn test_parse_temporal_produces_past_timestamp() {
        let before = Utc::now();
        let ts_str = parse_temporal("1d").unwrap();
        let ts = chrono::DateTime::parse_from_rfc3339(&ts_str).unwrap();
        // The parsed timestamp should be roughly 24 hours before now
        let diff = before.signed_duration_since(ts);
        assert!(diff.num_hours() >= 23 && diff.num_hours() <= 25);
    }
}
