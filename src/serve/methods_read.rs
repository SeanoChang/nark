//! READ method implementations for the `nark serve` daemon (Phase 3, slice 3.3).
//!
//! These functions build the **same** `serde_json::Value` that the matching CLI
//! handlers print, by calling the **same** `registry::*` functions — see
//! `cli/{peek,read,stats,search,orient}.rs`. They are the daemon-side mirror of
//! those commands: same shape, same fields, no behavioural change to the CLI.
//!
//! Key differences from the CLI handlers, both forced by the read-only serve
//! design (slice 3.1's [`ReadPool`](super::readpool::ReadPool)):
//!
//! * registry access goes through `ReadPool::with_conn` (a borrowed read-only
//!   [`rusqlite::Connection`]) instead of a fresh writable `db::open_registry`;
//! * [`read`] does **not** call `access::bump_access` — that is a write, and the
//!   pool's connection is read-only. The serve read path is side-effect-free; if
//!   access tracking over the socket is wanted it belongs to a later write-path
//!   slice, not here.
//!
//! Each function returns the bare result `Value`; the router ([`super::rpc`])
//! wraps it in an [`RPCResponse`](crate::wire::RPCResponse) and maps any `Err`
//! (unknown / ambiguous id, missing object) to a clean error response rather
//! than letting it panic.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;
use serde_json::{Value, json};

use super::readpool::ReadPool;
use crate::cli::search::parse_temporal;
use crate::cli::util::truncate_at_word;
use crate::config;
use crate::embed;
use crate::registry::search::{CosineContext, SearchFilters, SearchMode};
use crate::registry::{embeddings, resolve, search, stats, tags};
use crate::vault::fs::Vault;

/// Parsed parameters for [`search`], mapped from the JSON-RPC `params` object.
///
/// Mirrors the flags of `nark search` (`cli/search.rs`): `query` plus the
/// pre-filters and the mode toggles. Optional fields default the same way the
/// CLI's clap layer does (empty query, no filters, `limit` falls back to the
/// CLI default of 10, neither mode flag set).
pub struct SearchParams {
    pub query: String,
    pub domain: Option<String>,
    pub kind: Option<String>,
    pub intent: Option<String>,
    pub tags: Vec<String>,
    pub limit: usize,
    pub bm25: bool,
    pub semantic: bool,
    pub since: Option<String>,
    pub before: Option<String>,
}

/// Parsed parameters for [`orient`], mapped from the JSON-RPC `params` object.
///
/// Mirrors the flags of `nark orient` (`cli/orient.rs`): an optional `query`
/// (alias `topic`), the domain/kind/tag pre-filters, `limit`, and the
/// `since`/`before` temporal bounds. `orient` has no mode toggles and no
/// `intent` filter (it pins `intent: None`), so neither is parsed.
pub struct OrientParams {
    pub query: Option<String>,
    pub domain: Option<String>,
    pub kind: Option<String>,
    pub tags: Vec<String>,
    pub limit: usize,
    pub since: Option<String>,
    pub before: Option<String>,
}

/// `nark/peek`: resolve `id` to its head metadata, mirroring `cli::peek`.
///
/// Returns the same object `cli/peek.rs` prints: `id`, `title`, `domain`,
/// `intent`, `kind`, `status`, `tags`, `updated_at`, `links_in`, `links_out`.
/// An unknown or ambiguous id surfaces as the `Err` from [`resolve::get_meta`],
/// which the router turns into an error response.
pub fn peek(pool: &ReadPool, id: &str) -> Result<Value> {
    pool.with_conn(|conn| {
        let meta = resolve::get_meta(conn, id)?;
        Ok(json!({
            "id": meta.note_id,
            "title": meta.title,
            "domain": meta.domain,
            "intent": meta.intent,
            "kind": meta.kind,
            "status": meta.status,
            "tags": meta.tags,
            "updated_at": meta.updated_at,
            "links_in": meta.links_in,
            "links_out": meta.links_out,
        }))
    })
}

/// `nark/read`: resolve `id`, then read the head version's frontmatter and body
/// from the CAS, mirroring `cli::read`.
///
/// Returns the same object `cli/read.rs` prints: `id`, `title`, `frontmatter`
/// (the YAML frontmatter parsed to JSON), and `body` (the markdown body). Unlike
/// the CLI handler this does **not** bump access — the serve read path is
/// read-only and side-effect-free (see the module docs). A missing note or a
/// missing CAS object surfaces as an `Err` for the router to map.
pub fn read(pool: &ReadPool, vault_dir: &Path, id: &str) -> Result<Value> {
    let vault = Vault::new(vault_dir.to_path_buf());
    pool.with_conn(|conn| {
        let meta = resolve::get_meta(conn, id)?;
        let refs = resolve::get_ref(conn, &meta.note_id)?;

        let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

        let fm: Value = serde_yaml::from_str(&fm_raw)?;

        Ok(json!({
            "id": meta.note_id,
            "title": meta.title,
            "frontmatter": fm,
            "body": body,
        }))
    })
}

/// `nark/stats`: vault statistics overview, mirroring `cli::stats`.
///
/// Returns the same object `cli/stats.rs` prints: `total_notes`,
/// `total_versions`, `by_domain`/`by_kind` facet lists, the `recent` list, and
/// the nested `access` block (`total_reads`, `most_accessed`, `never_read`).
pub fn stats(pool: &ReadPool) -> Result<Value> {
    pool.with_conn(|conn| {
        let s = stats::overview(conn)?;

        let most_accessed = s
            .access
            .most_accessed
            .as_ref()
            .map(|m| json!({ "title": m.title, "count": m.count }));

        Ok(json!({
            "total_notes": s.total_notes,
            "total_versions": s.total_versions,
            "by_domain": s.by_domain.iter().map(|f| {
                json!({ "domain": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "by_kind": s.by_kind.iter().map(|f| {
                json!({ "kind": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "recent": s.recent.iter().map(|n| {
                json!({
                    "id": n.note_id,
                    "title": n.title,
                    "domain": n.domain,
                    "intent": n.intent,
                    "kind": n.kind,
                    "updated_at": n.updated_at,
                })
            }).collect::<Vec<_>>(),
            "access": {
                "total_reads": s.access.total_reads,
                "most_accessed": most_accessed,
                "never_read": s.access.never_read,
            },
        }))
    })
}

/// `nark/search`: ranked search over the vault, mirroring `cli::search`.
///
/// Maps `params` (see [`SearchParams`]) onto the same [`search::search`] call
/// the CLI makes: same pre-filters, same `SearchMode` selection, same cosine
/// dual-recall when embeddings *and* a query are present, and the same missing-
/// snippet backfill. Returns the same object `cli/search.rs` prints: `query`,
/// `domain`, `mode`, `hits` (count), and a `results` array of
/// `{id,title,domain,kind,snippet,rank,links_in,links_out}`.
///
/// Like the CLI, this degrades gracefully when the embedding provider is
/// unavailable: [`build_cosine_context`] returns `None` and the search runs on
/// BM25 + engagement instead of failing. The `bm25`/`semantic` flags are
/// mutually exclusive, matching the CLI's `--bm25`/`--semantic` guard.
pub fn search(pool: &ReadPool, vault_dir: &Path, params: &SearchParams) -> Result<Value> {
    if params.bm25 && params.semantic {
        anyhow::bail!("bm25 and semantic are mutually exclusive");
    }

    let cfg = config::load(vault_dir)?;
    let since_ts = params.since.as_deref().map(parse_temporal).transpose()?;
    let before_ts = params.before.as_deref().map(parse_temporal).transpose()?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let mode = if params.bm25 {
        SearchMode::Bm25Only
    } else if params.semantic {
        SearchMode::Semantic
    } else {
        SearchMode::Normal
    };

    pool.with_conn(|conn| {
        let filters = SearchFilters {
            domain: params.domain.as_deref(),
            kind: params.kind.as_deref(),
            intent: params.intent.as_deref(),
            tags: &params.tags,
            since: since_ts.as_deref(),
            before: before_ts.as_deref(),
            limit: params.limit,
        };

        // Build cosine context if embeddings are available and there's a query.
        // Skip for BM25-only mode (doesn't use cosine), mirroring the CLI.
        let cosine_ctx = if mode != SearchMode::Bm25Only
            && !params.query.is_empty()
            && embeddings::has_embeddings(conn)
        {
            build_cosine_context(vault_dir, &cfg, conn, &params.query)
        } else {
            None
        };

        let mut hits = search::search(
            conn,
            &params.query,
            &filters,
            &cfg.search,
            cosine_ctx.as_ref(),
            mode,
        )?;

        fill_missing_snippets(conn, &vault, &params.query, &mut hits);

        let results: Vec<Value> = hits
            .iter()
            .map(|h| {
                json!({
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

        Ok(json!({
            "query": params.query,
            "domain": params.domain,
            "mode": match mode {
                SearchMode::Bm25Only => "bm25",
                SearchMode::Semantic => "semantic",
                SearchMode::Normal => "normal",
            },
            "hits": results.len(),
            "results": results,
        }))
    })
}

/// `nark/orient`: a markdown vault briefing, mirroring `cli::orient`.
///
/// Maps `params` (see [`OrientParams`]) onto the same [`search::search`] call
/// the CLI makes (`intent` pinned to `None`, `SearchMode::Normal`, no cosine
/// context) and assembles the identical markdown briefing: a `# Vault Briefing`
/// header, a `## Key Notes` section with per-note previews, an `## Active Tags`
/// list, and a `## Recent Activity` count scoped to the same filters.
///
/// Unlike the CLI handler this does **not** call `access::bump_access` on each
/// surfaced note — the serve read path is read-only and side-effect-free (the
/// pool's connection is read-only; see the module docs). The returned `Value` is
/// the briefing markdown as a JSON string.
pub fn orient(pool: &ReadPool, vault_dir: &Path, params: &OrientParams) -> Result<Value> {
    let cfg = config::load(vault_dir)?;
    let since_ts = params.since.as_deref().map(parse_temporal).transpose()?;
    let before_ts = params.before.as_deref().map(parse_temporal).transpose()?;
    let vault = Vault::new(vault_dir.to_path_buf());

    pool.with_conn(|conn| {
        let filters = SearchFilters {
            domain: params.domain.as_deref(),
            kind: params.kind.as_deref(),
            intent: None,
            tags: &params.tags,
            since: since_ts.as_deref(),
            before: before_ts.as_deref(),
            limit: params.limit,
        };

        let q = params.query.as_deref().unwrap_or("");
        let hits = search::search(conn, q, &filters, &cfg.search, None, SearchMode::Normal)?;

        let mut md = String::new();
        let display_query = if q.is_empty() { "vault" } else { q };
        md.push_str(&format!("# Vault Briefing: {}\n\n", display_query));
        md.push_str(&format!("## Key Notes ({} most relevant)\n\n", hits.len()));

        let mut all_tags = BTreeSet::new();

        for hit in &hits {
            let refs = resolve::get_ref(conn, &hit.note_id)?;
            let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
            let preview = truncate_at_word(&body, 300).trim();

            let updated_at: String = conn.query_row(
                "SELECT COALESCE(updated_at, '') FROM current_notes WHERE note_id = ?1",
                [&hit.note_id],
                |row| row.get(0),
            )?;
            let date = updated_at.split('T').next().unwrap_or(&updated_at);

            md.push_str(&format!("### {}\n", hit.title));
            md.push_str(&format!("- Domain: {} | Kind: {}\n", hit.domain, hit.kind));
            md.push_str(&format!("- Updated: {}\n", date));
            md.push_str(&format!("> {}\n\n", preview.replace('\n', "\n> ")));

            if let Ok(note_tags) = tags::get_tags(conn, &hit.note_id) {
                for t in note_tags {
                    all_tags.insert(t);
                }
            }
            // Note: the serve read path does NOT bump access (read-only pool);
            // the CLI's `access::bump_access` is intentionally omitted here.
        }

        if !all_tags.is_empty() {
            md.push_str("## Active Tags\n");
            let tag_list: Vec<&str> = all_tags.iter().map(|s| s.as_str()).collect();
            md.push_str(&tag_list.join(", "));
            md.push_str("\n\n");
        }

        // Recent activity — scoped to the same domain/kind/tag filters.
        let seven_days_ago = parse_temporal("7d")?;
        let mut sql = String::from(
            "SELECT COUNT(*) FROM current_notes cn WHERE cn.updated_at >= ?1 AND cn.status != 'retracted'",
        );
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(seven_days_ago)];
        let mut pi = 2usize;
        if let Some(d) = params.domain.as_deref() {
            sql.push_str(&format!(" AND cn.domain = ?{}", pi));
            sql_params.push(Box::new(d.to_string()));
            pi += 1;
        }
        if let Some(k) = params.kind.as_deref() {
            sql.push_str(&format!(" AND cn.kind = ?{}", pi));
            sql_params.push(Box::new(k.to_string()));
            pi += 1;
        }
        for t in &params.tags {
            sql.push_str(&format!(
                " AND EXISTS (SELECT 1 FROM note_tags nt JOIN tags tg ON nt.tag_id = tg.tag_id WHERE nt.note_id = cn.note_id AND tg.name = ?{})",
                pi
            ));
            sql_params.push(Box::new(t.clone()));
            pi += 1;
        }
        let _ = pi;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        let recent_count: i64 = conn.query_row(&sql, param_refs.as_slice(), |row| row.get(0))?;
        let scope = if params.domain.is_some() || params.kind.is_some() || !params.tags.is_empty() {
            " (matching filters)"
        } else {
            ""
        };
        md.push_str("## Recent Activity\n");
        md.push_str(&format!(
            "{} notes updated in last 7 days{}\n",
            recent_count, scope
        ));

        Ok(Value::String(md))
    })
}

/// Build a cosine context (query embedding + per-note embeddings) for [`search`],
/// mirroring `cli::search::build_cosine_context`.
///
/// Returns `None` — degrading the search to BM25 + engagement instead of failing
/// — when the embedding provider is unavailable, when the query cannot be
/// embedded, when stored embeddings cannot be loaded, or when the stored
/// embedding dimension does not match the query's (a provider change without a
/// re-embed). This is the exact graceful-degradation contract the CLI has.
fn build_cosine_context(
    vault_dir: &Path,
    cfg: &config::Config,
    conn: &rusqlite::Connection,
    query: &str,
) -> Option<CosineContext> {
    let mut provider = embed::init_provider(vault_dir, &cfg.embedding)?;
    let query_embedding = provider.embed_query(query).ok()?;
    let all = embeddings::get_all_embeddings(conn).ok()?;

    if let Some((_, first_vec)) = all.first() {
        if first_vec.len() != query_embedding.len() {
            eprintln!(
                "Warning: embedding dimension mismatch (query={}, stored={}). Run `nark embed build` to re-embed.",
                query_embedding.len(),
                first_vec.len()
            );
            return None;
        }
    }

    let note_embeddings = all.into_iter().collect();
    Some(CosineContext {
        query_embedding,
        note_embeddings,
    })
}

/// Backfill empty snippets on search hits, mirroring
/// `cli::search::fill_missing_snippets`: try an FTS5 snippet for the query, then
/// fall back to the first 150 chars of the note body from the vault. A failed
/// body read is warned to stderr (not fatal), matching the CLI.
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

        if !query.is_empty() {
            if let Ok(snippet) = try_fts_snippet(conn, query, &hit.note_id) {
                hit.snippet = snippet;
                continue;
            }
        }

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
    use crate::registry::write::commit_version;

    /// A note document with a recognizable title/body and `namespace = ark`
    /// (set by `commit_version`), so it shows up in `current_notes` and stats.
    const NOTE: &str = "---\n\
title: Test Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - alpha\n\
  - beta\n\
---\n\
This is the body of the test note.\n";

    /// Build a temp vault: the writer (`db::open_registry`) creates/migrates/
    /// seeds the registry and enables WAL, then a single note is ingested into
    /// the CAS and committed. The writer connection is dropped so the read pool
    /// opens the same db read-only. Returns `(vault_dir, note_id)`.
    fn seeded_vault_with_note() -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!(
            "nark-methods-read-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");

        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit note version");
        let note_id = result.note_id.clone();
        drop(conn);

        (dir, note_id)
    }

    fn open_pool(dir: &Path) -> ReadPool {
        ReadPool::open_with_size(dir, 2).expect("open read pool")
    }

    #[test]
    fn peek_returns_same_key_fields_as_cli() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = peek(&pool, &note_id).expect("peek should succeed");
        assert_eq!(v["id"], note_id);
        assert_eq!(v["title"], "Test Note");
        assert_eq!(v["domain"], "engineering");
        assert_eq!(v["intent"], "reference");
        assert_eq!(v["kind"], "note");
        assert_eq!(v["status"], "active");
        assert_eq!(v["tags"], json!(["alpha", "beta"]));
        assert!(v["updated_at"].is_string());
        assert_eq!(v["links_in"], 0);
        assert_eq!(v["links_out"], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_returns_body_and_frontmatter() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = read(&pool, &dir, &note_id).expect("read should succeed");
        assert_eq!(v["id"], note_id);
        assert_eq!(v["title"], "Test Note");
        assert_eq!(
            v["body"], "This is the body of the test note.",
            "read should return the markdown body from the CAS"
        );
        // Frontmatter round-trips from YAML into a JSON object.
        assert_eq!(v["frontmatter"]["title"], "Test Note");
        assert_eq!(v["frontmatter"]["domain"], "engineering");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_returns_counts() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = stats(&pool).expect("stats should succeed");
        assert_eq!(v["total_notes"], 1, "the one ingested note is counted");
        assert_eq!(v["total_versions"], 1, "one version was committed");
        // The recent list surfaces the note we just wrote.
        assert_eq!(v["recent"][0]["id"], note_id);
        assert_eq!(v["recent"][0]["title"], "Test Note");
        // Never-read since the serve read path does not bump access.
        assert_eq!(v["access"]["never_read"], 1);
        assert_eq!(v["access"]["total_reads"], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peek_unknown_id_is_err_not_panic() {
        let (dir, _note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        // A well-formed but non-existent id prefix: resolve_id returns an error,
        // which propagates as Err (the router maps it to an error response).
        let result = peek(&pool, "ffffffff");
        assert!(
            result.is_err(),
            "unknown id should be an error, not a panic"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Slice 3.5 parity: the serve `peek` method's JSON must be byte-identical to
    /// the value built directly from `registry::resolve::get_meta` over a writer
    /// (`db::open_registry`) connection — the exact JSON `cli/peek.rs` prints.
    /// This proves the serve READ method has not drifted from the registry path
    /// the CLI uses for the same input (the Phase-3 non-breaking guarantee).
    #[test]
    fn peek_json_matches_direct_registry_call() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        // Serve path: through the read-only pool.
        let served = peek(&pool, &note_id).expect("serve peek should succeed");

        // Registry-direct path: the exact json! block `cli/peek.rs` builds from
        // `resolve::get_meta` over a writer connection.
        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let meta = resolve::get_meta(&conn, &note_id).expect("get_meta");
        let direct = json!({
            "id": meta.note_id,
            "title": meta.title,
            "domain": meta.domain,
            "intent": meta.intent,
            "kind": meta.kind,
            "status": meta.status,
            "tags": meta.tags,
            "updated_at": meta.updated_at,
            "links_in": meta.links_in,
            "links_out": meta.links_out,
        });
        drop(conn);

        assert_eq!(
            served, direct,
            "serve peek JSON must equal the direct registry::resolve::get_meta JSON \
             (CLI parity, byte-identical)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Slice 3.5 parity, stats edition: the serve `stats` method's JSON must be
    /// byte-identical to the value built directly from `registry::stats::overview`
    /// — the exact JSON `cli/stats.rs` prints. Mirrors `peek_json_matches_direct_
    /// registry_call` for the no-params overview method.
    #[test]
    fn stats_json_matches_direct_registry_call() {
        let (dir, _note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let served = stats(&pool).expect("serve stats should succeed");

        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let s = stats::overview(&conn).expect("overview");
        let most_accessed = s
            .access
            .most_accessed
            .as_ref()
            .map(|m| json!({ "title": m.title, "count": m.count }));
        let direct = json!({
            "total_notes": s.total_notes,
            "total_versions": s.total_versions,
            "by_domain": s.by_domain.iter().map(|f| {
                json!({ "domain": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "by_kind": s.by_kind.iter().map(|f| {
                json!({ "kind": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "recent": s.recent.iter().map(|n| {
                json!({
                    "id": n.note_id,
                    "title": n.title,
                    "domain": n.domain,
                    "intent": n.intent,
                    "kind": n.kind,
                    "updated_at": n.updated_at,
                })
            }).collect::<Vec<_>>(),
            "access": {
                "total_reads": s.access.total_reads,
                "most_accessed": most_accessed,
                "never_read": s.access.never_read,
            },
        });
        drop(conn);

        assert_eq!(
            served, direct,
            "serve stats JSON must equal the direct registry::stats::overview JSON \
             (CLI parity, byte-identical)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Three notes with distinct, queryable bodies so search/orient have
    /// something to rank. Each carries `namespace = ark` (via `commit_version`).
    fn note_doc(title: &str, body: &str, tag: &str) -> String {
        format!(
            "---\n\
title: {title}\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - {tag}\n\
---\n\
{body}\n"
        )
    }

    /// Seed a temp vault with three notes (writer creates/migrates/seeds), drop
    /// the writer, and return the read-only-openable vault dir.
    fn seeded_vault_three() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-methods-read-search-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");

        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        for (title, body, tag) in [
            ("Rust Ownership", "Borrowing and lifetimes in Rust.", "rust"),
            (
                "Tokio Tasks",
                "Async tasks scheduled on the tokio runtime.",
                "async",
            ),
            (
                "SQLite WAL",
                "Write-ahead logging for concurrent readers.",
                "db",
            ),
        ] {
            let result = vault
                .ingest(&note_doc(title, body, tag), None)
                .expect("ingest");
            commit_version(&conn, &result).expect("commit version");
        }
        drop(conn);
        dir
    }

    /// Default search params for a given query: no filters, no mode flags, the
    /// CLI default limit of 10.
    fn search_params(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            domain: None,
            kind: None,
            intent: None,
            tags: Vec::new(),
            limit: 10,
            bm25: false,
            semantic: false,
            since: None,
            before: None,
        }
    }

    #[test]
    fn search_returns_ranked_hits_matching_direct_registry_call() {
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let params = search_params("tokio");

        let v = search(&pool, &dir, &params).expect("search should succeed");

        assert_eq!(v["query"], "tokio");
        assert_eq!(v["mode"], "normal");
        assert!(v["hits"].as_u64().unwrap() >= 1, "tokio query should hit");
        assert_eq!(
            v["results"][0]["title"], "Tokio Tasks",
            "the tokio note should rank first"
        );

        // Same args through a direct registry::search call (no embeddings seeded,
        // so cosine_ctx is None either way) must yield the same ranked ids.
        let cfg = config::load(&dir).expect("load config");
        let direct_ids: Vec<String> = pool
            .with_conn(|conn| {
                let filters = SearchFilters {
                    domain: None,
                    kind: None,
                    intent: None,
                    tags: &[],
                    since: None,
                    before: None,
                    limit: 10,
                };
                let hits = search::search(
                    conn,
                    "tokio",
                    &filters,
                    &cfg.search,
                    None,
                    SearchMode::Normal,
                )?;
                Ok(hits.into_iter().map(|h| h.note_id).collect())
            })
            .expect("direct registry search");

        let method_ids: Vec<String> = v["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            method_ids, direct_ids,
            "serve search must mirror the direct registry::search ranking for the same args"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_bm25_mode_sets_mode_field() {
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let mut params = search_params("sqlite");
        params.bm25 = true;

        let v = search(&pool, &dir, &params).expect("bm25 search should succeed");
        assert_eq!(v["mode"], "bm25");
        assert_eq!(v["results"][0]["title"], "SQLite WAL");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_bm25_and_semantic_is_err() {
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let mut params = search_params("rust");
        params.bm25 = true;
        params.semantic = true;

        assert!(
            search(&pool, &dir, &params).is_err(),
            "bm25 + semantic are mutually exclusive"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn orient_returns_markdown_briefing() {
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let params = OrientParams {
            query: Some("rust".to_string()),
            domain: None,
            kind: None,
            tags: Vec::new(),
            limit: 5,
            since: None,
            before: None,
        };

        let v = orient(&pool, &dir, &params).expect("orient should succeed");
        let md = v.as_str().expect("orient returns a markdown string");

        assert!(
            md.starts_with("# Vault Briefing: rust"),
            "briefing header should echo the query, got: {md}"
        );
        assert!(
            md.contains("## Key Notes"),
            "should have a Key Notes section"
        );
        assert!(
            md.contains("### Rust Ownership"),
            "the rust note should be a key note, got: {md}"
        );
        assert!(
            md.contains("## Recent Activity"),
            "should have a Recent Activity section"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn orient_no_query_no_filter_is_err_like_cli() {
        // `nark orient` with neither a query nor a filter errors: the underlying
        // `search::search` bails (since/before are pre-filters and cannot stand
        // alone). The serve path faithfully mirrors that — it does not invent a
        // whole-vault scan the CLI does not do.
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let params = OrientParams {
            query: None,
            domain: None,
            kind: None,
            tags: Vec::new(),
            limit: 5,
            since: None,
            before: None,
        };

        assert!(
            orient(&pool, &dir, &params).is_err(),
            "orient with no query and no filter mirrors the CLI's error"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn orient_no_query_with_filter_briefs_the_whole_vault() {
        // A filter (here `domain`) makes the query optional, so the briefing
        // falls back to the 'vault' label — same as the CLI.
        let dir = seeded_vault_three();
        let pool = open_pool(&dir);
        let params = OrientParams {
            query: None,
            domain: Some("engineering".to_string()),
            kind: None,
            tags: Vec::new(),
            limit: 5,
            since: None,
            before: None,
        };

        let v = orient(&pool, &dir, &params).expect("orient with a filter should succeed");
        let md = v.as_str().unwrap();
        assert!(
            md.starts_with("# Vault Briefing: vault"),
            "no-query briefing falls back to the 'vault' label, got: {md}"
        );
        assert!(
            md.contains("(matching filters)"),
            "recent-activity scope should note the active filter, got: {md}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
