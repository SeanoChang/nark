use anyhow::{Result, bail};
use chrono::Utc;
use regex::Regex;
use std::path::Path;

use crate::cli::util::truncate_at_word;
use crate::config;
use crate::db;
use crate::embed;
use crate::registry::{
    embeddings, resolve,
    search::{self, CosineContext, SearchFilters, SearchMode},
};
use crate::serve;
use crate::vault::fs::Vault;

/// Build the `nark/search` JSON-RPC params object, mapping the CLI args onto the
/// exact field names the Phase-3 server's `search_params` parser expects (see
/// `serve::rpc`): `query`, `domain`, `kind`, `intent` (strings), `tag` (array),
/// `limit` (number), `bm25`/`semantic` (bools), `since`/`before` (strings).
/// Absent/`None`/`false`-default fields are omitted, so the server defaults them
/// the same way clap does on the direct path (the parser treats absent as the
/// default). `query` and `limit` are always sent (they are positional/has a
/// default on the CLI), matching what the direct path uses.
#[allow(clippy::too_many_arguments)]
fn search_params_json(
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
) -> serde_json::Value {
    let mut params = serde_json::Map::new();
    params.insert("query".into(), serde_json::json!(query));
    if let Some(d) = domain {
        params.insert("domain".into(), serde_json::json!(d));
    }
    if let Some(k) = kind {
        params.insert("kind".into(), serde_json::json!(k));
    }
    if let Some(i) = intent {
        params.insert("intent".into(), serde_json::json!(i));
    }
    if !tags.is_empty() {
        params.insert("tag".into(), serde_json::json!(tags));
    }
    params.insert("limit".into(), serde_json::json!(limit));
    if bm25_only {
        params.insert("bm25".into(), serde_json::json!(true));
    }
    if semantic {
        params.insert("semantic".into(), serde_json::json!(true));
    }
    if let Some(s) = since {
        params.insert("since".into(), serde_json::json!(s));
    }
    if let Some(b) = before {
        params.insert("before".into(), serde_json::json!(b));
    }
    serde_json::Value::Object(params)
}

/// Parse a relative temporal shorthand (e.g. "1d", "7d", "24h", "1w", "1mo")
/// into an ISO 8601 timestamp string.
pub fn parse_temporal(input: &str) -> Result<String> {
    let re = Regex::new(r"^(\d+)(h|d|w|mo)$").unwrap();
    let caps = re.captures(input).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid temporal format '{}'. Use: 1d, 7d, 24h, 1w, 1mo",
            input
        )
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

// The argument list mirrors the `nark search` CLI flags 1:1 (clap dispatch in
// `main.rs`); collapsing them into a struct would change that public call site,
// which is out of scope for the dual-mode slice — so the lint is allowed here.
#[allow(clippy::too_many_arguments)]
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

    // Dual-mode: ask a live `nark serve` first (one round-trip). The socket is
    // an optimization — `try_request` returns `None` on ANY failure (absent or
    // stale socket, connect timeout, `unauthorized`, error response, malformed
    // JSON, any I/O error), and we then fall through to the always-correct
    // direct-open path below, unchanged (including every `--bm25`/`--semantic`/
    // filter behavior). The server runs the SAME `registry::search` pipeline, so
    // a socket HIT yields the identical result shape.
    let socket = serve::client::default_socket(vault_dir);
    let params = search_params_json(
        query, domain, kind, intent, tags, limit, bm25_only, semantic, since, before,
    );
    if let Some(result) = serve::client::try_request(&socket, "nark/search", params) {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
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
    let cosine_ctx =
        if mode != SearchMode::Bm25Only && !query.is_empty() && embeddings::has_embeddings(&conn) {
            build_cosine_context(vault_dir, &cfg, &conn, query)
        } else {
            None
        };

    let mut hits = search::search(
        &conn,
        query,
        &filters,
        &cfg.search,
        cosine_ctx.as_ref(),
        mode,
    )?;

    let vault = Vault::new(vault_dir.to_path_buf());
    fill_missing_snippets(&conn, &vault, query, &mut hits);

    let results: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
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
        })
        .collect();

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
    if let Some((_, first_vec)) = all.first()
        && first_vec.len() != query_embedding.len()
    {
        eprintln!(
            "Warning: embedding dimension mismatch (query={}, stored={}). Run `nark embed build` to re-embed.",
            query_embedding.len(),
            first_vec.len()
        );
        return None;
    }

    let note_embeddings = all.into_iter().collect();
    Some(CosineContext {
        query_embedding,
        note_embeddings,
    })
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
        if !query.is_empty()
            && let Ok(snippet) = try_fts_snippet(conn, query, &hit.note_id)
        {
            hit.snippet = snippet;
            continue;
        }

        // Fallback: first 150 chars of body from vault
        match read_body_preview(conn, vault, &hit.note_id) {
            Ok(body) => hit.snippet = body,
            Err(e) => eprintln!(
                "Warning: failed to read body preview for {}: {}",
                hit.note_id, e
            ),
        }
    }
}

fn try_fts_snippet(
    conn: &rusqlite::Connection,
    query: &str,
    note_id: &str,
) -> anyhow::Result<String> {
    let mut stmt = conn.prepare(
        "SELECT snippet(note_text, 2, '[', ']', '...', 32)
         FROM note_text
         WHERE note_text MATCH ?1 AND note_id = ?2",
    )?;
    let snippet: String = stmt.query_row(rusqlite::params![query, note_id], |row| row.get(0))?;
    if snippet.is_empty() {
        anyhow::bail!("empty snippet");
    }
    Ok(snippet)
}

fn read_body_preview(
    conn: &rusqlite::Connection,
    vault: &Vault,
    note_id: &str,
) -> anyhow::Result<String> {
    let refs = resolve::get_ref(conn, note_id)?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
    Ok(truncate_at_word(&body, 150).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::client::default_socket;
    use crate::serve::client::test_support::{
        TestServer, current_uid_agent_map, seed_vault, temp_vault_dir,
    };

    /// The direct-open path's search JSON for a query over a seeded vault —
    /// exactly the `out` object `run` builds before printing, in `Normal` mode
    /// with no filters. Used to assert socket-hit == direct-open parity at the
    /// value level (both pretty-print through `serde_json::to_string_pretty`, so
    /// equal values mean byte-identical output).
    fn direct_search_value(vault_dir: &Path, query: &str) -> serde_json::Value {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let cfg = config::load(vault_dir).expect("load config");
        let filters = SearchFilters {
            domain: None,
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit: 10,
        };
        let mut hits = search::search(
            &conn,
            query,
            &filters,
            &cfg.search,
            None,
            SearchMode::Normal,
        )
        .expect("registry search");
        let vault = Vault::new(vault_dir.to_path_buf());
        fill_missing_snippets(&conn, &vault, query, &mut hits);
        let results: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
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
            })
            .collect();
        serde_json::json!({
            "query": query,
            "domain": serde_json::Value::Null,
            "mode": "normal",
            "hits": results.len(),
            "results": results,
        })
    }

    /// (a) With a live serve + a mapped uid, the dual-mode search path returns the
    /// SERVER's result, byte-identical (value-equal) to the direct-open path over
    /// the same seeded vault for a representative query that matches the note.
    #[cfg(target_os = "macos")]
    #[test]
    fn search_socket_hit_matches_direct_path() {
        let server = TestServer::start(current_uid_agent_map());

        // "client" matches the seeded note's title/body.
        let socket = default_socket(server.dir());
        let params = search_params_json(
            "client",
            None,
            None,
            None,
            &[],
            10,
            false,
            false,
            None,
            None,
        );
        let from_socket = serve::client::try_request(&socket, "nark/search", params)
            .expect("authenticated nark/search should return Some(result)");

        let from_direct = direct_search_value(server.dir(), "client");
        assert_eq!(
            from_socket, from_direct,
            "socket-hit search must match the direct-open path byte-for-byte"
        );
        assert!(
            from_socket["hits"].as_u64().unwrap() >= 1,
            "the 'client' query should hit the seeded note"
        );

        run(
            server.dir(),
            "client",
            None,
            None,
            None,
            &[],
            10,
            false,
            false,
            None,
            None,
        )
        .expect("dual-mode search over live serve");
    }

    /// (b) With NO serve (socket absent), the direct path is taken — `try_request`
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    /// The `--bm25` flag (the direct path's mode toggle) still works.
    #[test]
    fn search_no_serve_takes_direct_path_with_bm25_flag() {
        let dir = temp_vault_dir();
        let _id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        let params =
            search_params_json("client", None, None, None, &[], 10, true, false, None, None);
        assert!(
            serve::client::try_request(&socket, "nark/search", params).is_none(),
            "with no serve, try_request must return None so the direct path is taken"
        );

        // --bm25 still works on the direct path (mutually exclusive with --semantic
        // upheld; the direct path selects Bm25Only mode and returns hits).
        run(
            &dir,
            "client",
            None,
            None,
            None,
            &[],
            10,
            /*bm25*/ true,
            false,
            None,
            None,
        )
        .expect("direct-open search with --bm25 and no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
        assert!(parse_temporal("d").is_err()); // missing number
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
