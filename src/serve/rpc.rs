//! JSON-RPC method router for the `nark serve` daemon.
//!
//! Phase 3 replaces the Phase-2 line protocol with one JSON-RPC request per
//! connection: the client writes a single [`crate::wire::RPCRequest`] line, the
//! connection handler parses it and calls [`dispatch`], and writes back the one
//! [`crate::wire::RPCResponse`] this returns.
//!
//! Slice 3.2 landed the framing and the router skeleton (`ping` + method-not-
//! found). Slice 3.3 adds the first READ methods, which need vault access, so
//! [`dispatch`] now takes a [`Ctx`] carrying the read-only connection pool and
//! the vault directory:
//!
//! * `ping` -> `{"pong": true}` (a success result, no vault access),
//! * `nark/peek` -> head metadata for `params.id` (see [`methods_read::peek`]),
//! * `nark/read` -> frontmatter + body for `params.id` ([`methods_read::read`]),
//! * `nark/stats` -> vault statistics ([`methods_read::stats`]),
//! * `nark/search` -> ranked hits for the search `params` ([`methods_read::search`]),
//! * `nark/orient` -> a markdown vault briefing ([`methods_read::orient`]),
//! * any other method -> JSON-RPC `-32601 method not found`.
//!
//! The read methods call the same `registry::*` functions the CLI handlers do
//! and build the same JSON shape. A missing/ambiguous id, a malformed params
//! object, or a missing CAS object produces a clean JSON-RPC error response
//! (`-32602 invalid params`) echoing the request id — never a panic.

use deadpool::managed::Pool;
use serde_json::{Value, json};

use super::dpool::{self, RoManager};
use super::methods_read;
use super::methods_read::{OrientParams, SearchParams};
use super::readpool::ReadPool;
use crate::wire::{RPCRequest, RPCResponse};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// CLI default `--limit` for `nark search` (`cli::mod`), used when
/// `params.limit` is omitted so the socket path defaults like the command line.
const SEARCH_DEFAULT_LIMIT: usize = 10;
/// CLI default `--limit` for `nark orient` (`cli::mod`); orient surfaces fewer
/// notes than search by default.
const ORIENT_DEFAULT_LIMIT: usize = 5;

/// JSON-RPC error code: the requested method is not implemented.
const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC error code: the method's params were missing/invalid, or the
/// requested note could not be resolved/read (a 404-style application error).
const INVALID_PARAMS: i64 = -32602;

/// Per-daemon context the router hands to the read methods plus the vault root
/// (needed by `nark/read` to resolve CAS object paths).
///
/// During Phase 3.5 the read path is split across two pools, both over
/// `<vault_dir>/registry.db` and both strictly read-only:
///
/// * `dpool` — the [`deadpool`]-managed pool ([`super::dpool`]). The cheap
///   methods (`peek` / `read` / `stats`) check a connection out of it and run
///   their blocking SQLite on a managed thread via `conn.interact(...)`; `get()`
///   backpressures when every connection is busy (slice 3.5.2). It is [`Clone`]
///   (internally `Arc`-based), so tests can hold a second handle to drive
///   contention.
/// * `ro_pool` — the hand-rolled [`ReadPool`] from Phase 3, kept only for the
///   embedding-bearing methods (`search` / `orient`), which still run their
///   blocking work (including ONNX inference) on a `spawn_blocking` thread this
///   slice. Slice 3.5.4 migrates them onto `dpool` and removes this field.
pub struct Ctx {
    dpool: Pool<RoManager>,
    ro_pool: Arc<ReadPool>,
    vault_dir: PathBuf,
}

impl Ctx {
    /// Build a context, opening both read-only pools against `vault_dir`.
    ///
    /// The registry must already exist (the writer owns creation/migration); both
    /// pools open it read-only. Async because the [`deadpool`] pool is built on
    /// the tokio runtime. Used by the serve daemon path.
    pub async fn open(vault_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            dpool: dpool::open_ro_pool(vault_dir, dpool::DEFAULT_POOL_SIZE).await?,
            ro_pool: Arc::new(ReadPool::open(vault_dir)?),
            vault_dir: vault_dir.to_path_buf(),
        })
    }

    /// Build a context from already-open pools and vault dir. Lets tests inject
    /// sized pools without re-opening. The [`deadpool`] pool is [`Clone`], so a
    /// test can keep a second handle (to occupy every connection) while the
    /// daemon dispatches against it.
    #[cfg(test)]
    pub fn new(dpool: Pool<RoManager>, ro_pool: ReadPool, vault_dir: PathBuf) -> Self {
        Self {
            dpool,
            ro_pool: Arc::new(ro_pool),
            vault_dir,
        }
    }
}

/// Dispatch a parsed [`RPCRequest`] to its method and return the response.
///
/// `ctx` provides the read-only vault access the read methods need; `ping`
/// ignores it. Unknown methods produce `-32601 method not found`; a read method
/// whose params are missing/invalid or whose note cannot be resolved/read
/// produces `-32602 invalid params`. Every path echoes the request id, so a
/// client always gets exactly one response per request and never a panic.
///
/// `dispatch` is `async`: the cheap methods (`peek` / `read` / `stats`) check a
/// connection out of the [`deadpool`] pool and run their blocking SQLite on a
/// managed thread via `conn.interact(...).await`, so they never park a tokio
/// worker (the listener awaits this directly — no `spawn_blocking` wrapper). The
/// embedding-bearing methods (`search` / `orient`) still run on the hand-rolled
/// [`ReadPool`], whose checkout blocks on a `Condvar`; until slice 3.5.4 moves
/// them onto the deadpool path, this routes them through
/// [`tokio::task::spawn_blocking`] so that blocking checkout (and the ONNX
/// inference under it) cannot park a worker either.
pub async fn dispatch(ctx: &Ctx, req: &RPCRequest) -> RPCResponse {
    match req.method.as_str() {
        "ping" => RPCResponse::result(req.id.clone(), json!({"pong": true})),
        "nark/peek" => match id_param(req) {
            Ok(id) => result_or_invalid(req, methods_read::peek(&ctx.dpool, &id).await),
            Err(resp) => resp,
        },
        "nark/read" => match id_param(req) {
            Ok(id) => result_or_invalid(
                req,
                methods_read::read(&ctx.dpool, &ctx.vault_dir, &id).await,
            ),
            Err(resp) => resp,
        },
        "nark/stats" => result_or_invalid(req, methods_read::stats(&ctx.dpool).await),
        "nark/search" => match search_params(req) {
            Ok(params) => result_or_invalid(req, blocking_search(ctx, params).await),
            Err(resp) => resp,
        },
        "nark/orient" => match orient_params(req) {
            Ok(params) => result_or_invalid(req, blocking_orient(ctx, params).await),
            Err(resp) => resp,
        },
        _ => RPCResponse::error(req.id.clone(), METHOD_NOT_FOUND, "method not found", None),
    }
}

/// Run `search` on a [`tokio::task::spawn_blocking`] thread. `search` still uses
/// the hand-rolled [`ReadPool`] (blocking `Condvar` checkout) and may run ONNX
/// inference, so it must not execute on an async worker. A panic in the blocking
/// task surfaces as an `Err` (joined below), which the router maps to a clean
/// `-32602` rather than tearing down the connection. Removed in slice 3.5.4 when
/// `search` moves onto the deadpool `interact` path.
async fn blocking_search(ctx: &Ctx, params: SearchParams) -> anyhow::Result<Value> {
    let pool = Arc::clone(&ctx.ro_pool);
    let vault_dir = ctx.vault_dir.clone();
    tokio::task::spawn_blocking(move || methods_read::search(&pool, &vault_dir, &params))
        .await
        .map_err(|e| anyhow::anyhow!("search task join: {e}"))?
}

/// Run `orient` on a [`tokio::task::spawn_blocking`] thread, for the same reason
/// as [`blocking_search`]. Removed in slice 3.5.4.
async fn blocking_orient(ctx: &Ctx, params: OrientParams) -> anyhow::Result<Value> {
    let pool = Arc::clone(&ctx.ro_pool);
    let vault_dir = ctx.vault_dir.clone();
    tokio::task::spawn_blocking(move || methods_read::orient(&pool, &vault_dir, &params))
        .await
        .map_err(|e| anyhow::anyhow!("orient task join: {e}"))?
}

/// Parse the `nark/search` `params` object into [`SearchParams`].
///
/// All fields are optional and default exactly as the CLI does: empty `query`,
/// no pre-filters, `limit` -> [`SEARCH_DEFAULT_LIMIT`], `bm25`/`semantic` off.
/// A `params` value that is present but not a JSON object, or a field of the
/// wrong type, yields a `-32602 invalid params` error response.
fn search_params(req: &RPCRequest) -> Result<SearchParams, RPCResponse> {
    let obj = params_object(req)?;
    Ok(SearchParams {
        query: opt_string(req, obj, "query")?.unwrap_or_default(),
        domain: opt_string(req, obj, "domain")?,
        kind: opt_string(req, obj, "kind")?,
        intent: opt_string(req, obj, "intent")?,
        tags: opt_string_array(req, obj, "tag")?,
        limit: opt_limit(req, obj, SEARCH_DEFAULT_LIMIT)?,
        bm25: opt_bool(req, obj, "bm25")?,
        semantic: opt_bool(req, obj, "semantic")?,
        since: opt_string(req, obj, "since")?,
        before: opt_string(req, obj, "before")?,
    })
}

/// Parse the `nark/orient` `params` object into [`OrientParams`].
///
/// `orient` accepts `query` or its alias `topic` (query wins if both are given),
/// the domain/kind/tag filters, `limit` -> [`ORIENT_DEFAULT_LIMIT`], and the
/// `since`/`before` bounds. All optional; type mismatches yield `-32602`.
fn orient_params(req: &RPCRequest) -> Result<OrientParams, RPCResponse> {
    let obj = params_object(req)?;
    let query = match opt_string(req, obj, "query")? {
        Some(q) => Some(q),
        None => opt_string(req, obj, "topic")?,
    };
    Ok(OrientParams {
        query,
        domain: opt_string(req, obj, "domain")?,
        kind: opt_string(req, obj, "kind")?,
        tags: opt_string_array(req, obj, "tag")?,
        limit: opt_limit(req, obj, ORIENT_DEFAULT_LIMIT)?,
        since: opt_string(req, obj, "since")?,
        before: opt_string(req, obj, "before")?,
    })
}

/// Borrow `req.params` as a JSON object. A missing `params` is treated as an
/// empty object (all fields default); a non-object `params` is an error.
fn params_object(req: &RPCRequest) -> Result<&serde_json::Map<String, Value>, RPCResponse> {
    static EMPTY: std::sync::OnceLock<serde_json::Map<String, Value>> = std::sync::OnceLock::new();
    match req.params.as_ref() {
        None => Ok(EMPTY.get_or_init(serde_json::Map::new)),
        Some(Value::Object(map)) => Ok(map),
        Some(_) => Err(invalid(req, "invalid params: expected an object")),
    }
}

/// Read an optional string field; `null`/absent -> `None`, non-string -> error.
fn opt_string(
    req: &RPCRequest,
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<String>, RPCResponse> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(invalid(
            req,
            &format!("invalid params: '{key}' must be a string"),
        )),
    }
}

/// Read an optional string-array field (e.g. `tag`); absent/`null` -> empty vec.
/// A non-array, or an array element that is not a string, is an error.
fn opt_string_array(
    req: &RPCRequest,
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<String>, RPCResponse> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(s) => out.push(s.clone()),
                    _ => {
                        return Err(invalid(
                            req,
                            &format!("invalid params: '{key}' must be an array of strings"),
                        ));
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(invalid(
            req,
            &format!("invalid params: '{key}' must be an array of strings"),
        )),
    }
}

/// Read an optional boolean flag; absent/`null` -> `false`, non-bool -> error.
fn opt_bool(
    req: &RPCRequest,
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<bool, RPCResponse> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(invalid(
            req,
            &format!("invalid params: '{key}' must be a boolean"),
        )),
    }
}

/// Read an optional `limit`; absent/`null` -> `default`. Must be a non-negative
/// integer that fits `usize`; anything else is an error.
fn opt_limit(
    req: &RPCRequest,
    obj: &serde_json::Map<String, Value>,
    default: usize,
) -> Result<usize, RPCResponse> {
    match obj.get("limit") {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => {
            n.as_u64()
                .and_then(|u| usize::try_from(u).ok())
                .ok_or_else(|| {
                    invalid(
                        req,
                        "invalid params: 'limit' must be a non-negative integer",
                    )
                })
        }
        Some(_) => Err(invalid(
            req,
            "invalid params: 'limit' must be a non-negative integer",
        )),
    }
}

/// Build a `-32602 invalid params` error response echoing the request id.
fn invalid(req: &RPCRequest, message: &str) -> RPCResponse {
    RPCResponse::error(req.id.clone(), INVALID_PARAMS, message, None)
}

/// Extract the required string `params.id` for `nark/peek` / `nark/read`.
///
/// On a missing `params` or a missing/non-string `id`, returns a ready-made
/// `-32602 invalid params` error response (echoing the request id) instead of an
/// `Ok`, so the caller can short-circuit.
fn id_param(req: &RPCRequest) -> Result<String, RPCResponse> {
    match req.params.as_ref().and_then(|p| p.get("id")) {
        Some(Value::String(id)) => Ok(id.clone()),
        _ => Err(RPCResponse::error(
            req.id.clone(),
            INVALID_PARAMS,
            "invalid params: expected {\"id\": string}",
            None,
        )),
    }
}

/// Wrap a method's `Result<Value>` into a response: `Ok` becomes a success
/// result echoing the request id; `Err` (unknown/ambiguous id, missing object)
/// becomes a `-32602 invalid params` error carrying the error text as `data`.
fn result_or_invalid(req: &RPCRequest, out: anyhow::Result<Value>) -> RPCResponse {
    match out {
        Ok(value) => RPCResponse::result(req.id.clone(), value),
        Err(e) => RPCResponse::error(
            req.id.clone(),
            INVALID_PARAMS,
            "invalid params",
            Some(json!({ "detail": format!("{e:#}") })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::write::commit_version;
    use crate::vault::fs::Vault;
    use crate::wire::RPCResponse;

    const NOTE: &str = "---\n\
title: Router Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - gamma\n\
---\n\
Router body text.\n";

    /// Build a [`Ctx`] over `dir`'s seeded registry: the deadpool pool the cheap
    /// methods check out from, and the hand-rolled [`ReadPool`] search/orient
    /// still use this slice. Async because the deadpool pool is built on the
    /// runtime.
    async fn ctx_for(dir: &std::path::Path) -> Ctx {
        let dpool = dpool::open_ro_pool(dir, 2)
            .await
            .expect("open deadpool pool");
        let ro_pool = ReadPool::open_with_size(dir, 2).expect("open read pool");
        Ctx::new(dpool, ro_pool, dir.to_path_buf())
    }

    /// Seed a temp vault with one note (writer creates/migrates/seeds), drop the
    /// writer, and return a `Ctx` (read-only pools) plus the note id.
    async fn seeded_ctx() -> (Ctx, String, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-read-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit version");
        let note_id = result.note_id.clone();
        drop(conn);

        let ctx = ctx_for(&dir).await;
        (ctx, note_id, dir)
    }

    fn request(id: &str, method: &str, params: Option<Value>) -> RPCRequest {
        RPCRequest {
            id: id.to_string(),
            method: method.to_string(),
            params,
        }
    }

    fn expect_result(resp: RPCResponse) -> Value {
        match resp {
            RPCResponse::Result(r) => r.result,
            RPCResponse::Error(e) => panic!("expected result, got error {e:?}"),
        }
    }

    fn expect_error(resp: RPCResponse) -> crate::wire::RPCError {
        match resp {
            RPCResponse::Error(e) => e.error,
            RPCResponse::Result(r) => panic!("expected error, got result {r:?}"),
        }
    }

    #[tokio::test]
    async fn ping_returns_pong_with_matching_id() {
        let (ctx, _id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("42", "ping", None)).await;
        match resp {
            RPCResponse::Result(r) => {
                assert_eq!(r.id, "42");
                assert_eq!(r.result, json!({"pong": true}));
            }
            RPCResponse::Error(e) => panic!("ping should succeed, got error {e:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn unknown_method_is_method_not_found() {
        let (ctx, _id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("7", "no-such-method", None)).await;
        let err = expect_error(resp);
        assert_eq!(err.code, METHOD_NOT_FOUND);
        assert_eq!(err.message, "method not found");
        assert!(err.data.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn peek_returns_meta_fields() {
        let (ctx, id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("1", "nark/peek", Some(json!({"id": id})))).await;
        let v = expect_result(resp);
        assert_eq!(v["id"], id);
        assert_eq!(v["title"], "Router Note");
        assert_eq!(v["domain"], "engineering");
        assert_eq!(v["tags"], json!(["gamma"]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_returns_body() {
        let (ctx, id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("2", "nark/read", Some(json!({"id": id})))).await;
        let v = expect_result(resp);
        assert_eq!(v["body"], "Router body text.");
        assert_eq!(v["frontmatter"]["title"], "Router Note");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn stats_returns_counts() {
        let (ctx, _id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("3", "nark/stats", None)).await;
        let v = expect_result(resp);
        assert_eq!(v["total_notes"], 1);
        assert_eq!(v["total_versions"], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bad_id_returns_invalid_params_not_panic() {
        let (ctx, _id, dir) = seeded_ctx().await;
        let resp = dispatch(
            &ctx,
            &request("9", "nark/peek", Some(json!({"id": "ffffffff"}))),
        )
        .await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn missing_params_returns_invalid_params() {
        let (ctx, _id, dir) = seeded_ctx().await;
        let resp = dispatch(&ctx, &request("10", "nark/peek", None)).await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn note_doc(title: &str, body: &str) -> String {
        format!(
            "---\n\
title: {title}\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - gamma\n\
---\n\
{body}\n"
        )
    }

    /// Seed a temp vault with a few notes and return a read-only `Ctx`.
    async fn seeded_ctx_multi() -> (Ctx, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-search-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        for (title, body) in [
            ("Rust Ownership", "Borrowing and lifetimes in Rust."),
            ("Tokio Tasks", "Async tasks on the tokio runtime."),
        ] {
            let result = vault.ingest(&note_doc(title, body), None).expect("ingest");
            commit_version(&conn, &result).expect("commit version");
        }
        drop(conn);
        let ctx = ctx_for(&dir).await;
        (ctx, dir)
    }

    #[tokio::test]
    async fn search_returns_ranked_hits() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request("11", "nark/search", Some(json!({"query": "tokio"}))),
        )
        .await;
        let v = expect_result(resp);
        assert_eq!(v["query"], "tokio");
        assert_eq!(v["mode"], "normal");
        assert!(v["hits"].as_u64().unwrap() >= 1);
        assert_eq!(v["results"][0]["title"], "Tokio Tasks");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn search_bm25_flag_selects_bm25_mode() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request(
                "12",
                "nark/search",
                Some(json!({"query": "rust", "bm25": true})),
            ),
        )
        .await;
        let v = expect_result(resp);
        assert_eq!(v["mode"], "bm25");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn search_with_no_query_and_no_filters_is_invalid_params() {
        // registry::search bails when there is neither a query nor a filter; the
        // router maps that Err to -32602 rather than panicking.
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(&ctx, &request("13", "nark/search", Some(json!({})))).await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn search_bad_param_type_is_invalid_params() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request("14", "nark/search", Some(json!({"query": 7}))),
        )
        .await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn orient_returns_markdown_briefing() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request("15", "nark/orient", Some(json!({"query": "rust"}))),
        )
        .await;
        let v = expect_result(resp);
        let md = v.as_str().expect("orient returns a markdown string");
        assert!(md.starts_with("# Vault Briefing: rust"));
        assert!(md.contains("## Key Notes"));
        assert!(md.contains("## Recent Activity"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn orient_accepts_topic_alias_and_omitted_params() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request("16", "nark/orient", Some(json!({"topic": "tokio"}))),
        )
        .await;
        let v = expect_result(resp);
        let md = v.as_str().unwrap();
        assert!(
            md.starts_with("# Vault Briefing: tokio"),
            "the `topic` alias should feed the query, got: {md}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn orient_no_query_no_filter_is_invalid_params() {
        // Mirrors the CLI: orient with neither a query nor a filter bails in
        // registry::search; the router maps that to -32602 (not a panic).
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(&ctx, &request("17", "nark/orient", None)).await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn orient_with_filter_and_omitted_query_briefs_vault() {
        let (ctx, dir) = seeded_ctx_multi().await;
        let resp = dispatch(
            &ctx,
            &request("18", "nark/orient", Some(json!({"domain": "engineering"}))),
        )
        .await;
        let v = expect_result(resp);
        let md = v.as_str().unwrap();
        assert!(
            md.starts_with("# Vault Briefing: vault"),
            "omitted query with a filter defaults to a whole-vault briefing, got: {md}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
