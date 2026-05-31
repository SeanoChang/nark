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
//! * `nark/write` -> ingest a note via the single serializing writer queue, the
//!   first WRITE method (see [`methods_write::write`]),
//! * `nark/link` -> create typed edges from sources to a target via the writer
//!   queue (see [`methods_write::link`]),
//! * `nark/delete` -> soft-retract / hard-delete / purge notes via the writer
//!   queue (see [`methods_write::delete`]),
//! * any other method -> JSON-RPC `-32601 method not found`.
//!
//! The read methods call the same `registry::*` functions the CLI handlers do
//! and build the same JSON shape. A missing/ambiguous id, a malformed params
//! object, or a missing CAS object produces a clean JSON-RPC error response
//! (`-32602 invalid params`) echoing the request id — never a panic.

use deadpool::managed::Pool;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use super::dpool::{self, RoManager};
use super::embed_permit;
use super::methods_read;
use super::methods_read::{OrientParams, SearchParams};
use super::methods_write;
use super::methods_write::{DeleteParams, LinkParams, WriteParams};
use super::writer::Writer;
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
/// As of Phase 3.5 the whole read path runs over a single [`deadpool`]-managed,
/// strictly read-only pool against `<vault_dir>/registry.db`, with the embedding
/// work split out under a permit. `Ctx` therefore holds exactly two things — the
/// pool and the permit semaphore — plus the vault dir:
///
/// * `dpool` — the [`deadpool`]-managed pool ([`super::dpool`]), the **only**
///   connection pool. **Every** read method (`peek` / `read` / `stats` /
///   `search` / `orient`) checks a connection out of it and runs its blocking
///   SQLite on a managed thread via `conn.interact(...)`; `get()` backpressures
///   when every connection is busy. It is [`Clone`] (internally `Arc`-based), so
///   tests can hold a second handle to drive contention. The Phase-3 hand-rolled
///   read pool is gone (retired in slice 3.5.5) — there is no second pool.
///
/// * `embed_sem` — the bounded embedding-worker semaphore: an `Arc<Semaphore>`
///   with [`embed_permit::DEFAULT_EMBED_PERMITS`] permits that caps how many ONNX
///   inferences run concurrently. The `search` method's inference step runs
///   through [`embed_permit::with_embed_permit`] using it — the embedding work
///   happens under a permit and **outside** any DB checkout (the 2B payoff), so a
///   burst of `search` load cannot hold a connection across inference and stall
///   the cheap reads (no head-of-line blocking).
pub struct Ctx {
    dpool: Pool<RoManager>,
    embed_sem: Arc<Semaphore>,
    vault_dir: PathBuf,
    /// The single serializing, off-reactor writer (Phase 6). `Ctx::open` (the
    /// daemon path) builds one; the read-path test injector [`Ctx::new`] leaves it
    /// `None` because reads never touch it. Held as `Option<Arc<Writer>>` so the
    /// writer can be shared and so read-only tests need not stand one up. The
    /// write methods (`nark/write`, slice 6.2 onward) route mutations through it
    /// via [`Ctx::writer`].
    writer: Option<Arc<Writer>>,
}

impl Ctx {
    /// Build a context, opening the read-only pool against `vault_dir` and the
    /// single read-write [`Writer`] (Phase 6).
    ///
    /// The registry must already exist (the writer owns creation/migration); the
    /// read pool opens it read-only and the [`Writer`] opens one read-write
    /// connection via the shared inner path. Async because the [`deadpool`] pool
    /// is built on the tokio runtime. Used by the serve daemon path.
    ///
    /// Serve already holds the advisory write lock for its lifetime (see
    /// [`super::run_until`]), so the [`Writer`] opens an *unlocked* connection and
    /// does NOT take a second lock.
    pub async fn open(vault_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            dpool: dpool::open_ro_pool(vault_dir, dpool::DEFAULT_POOL_SIZE).await?,
            embed_sem: embed_permit::default_embed_semaphore(),
            vault_dir: vault_dir.to_path_buf(),
            writer: Some(Arc::new(Writer::open(vault_dir)?)),
        })
    }

    /// Build a read-only context from an already-open pool and vault dir, with no
    /// writer. Lets the read-path tests inject a sized pool without re-opening (or
    /// standing up a writer they do not use). The [`deadpool`] pool is [`Clone`],
    /// so a test can keep a second handle (to occupy every connection) while the
    /// daemon dispatches against it.
    #[cfg(test)]
    pub fn new(dpool: Pool<RoManager>, vault_dir: PathBuf) -> Self {
        Self {
            dpool,
            embed_sem: embed_permit::default_embed_semaphore(),
            vault_dir,
            writer: None,
        }
    }

    /// Build a context from an already-open pool and vault dir **with** a
    /// [`Writer`], for the write-method tests (`nark/write`). Mirrors [`Ctx::new`]
    /// but stands up the single serializing writer so a test can drive a write
    /// without binding a socket or owning the runtime that [`Ctx::open`] needs.
    #[cfg(test)]
    pub fn with_writer(dpool: Pool<RoManager>, vault_dir: PathBuf, writer: Arc<Writer>) -> Self {
        Self {
            dpool,
            embed_sem: embed_permit::default_embed_semaphore(),
            vault_dir,
            writer: Some(writer),
        }
    }

    /// The single serializing [`Writer`] the write methods submit jobs to, or
    /// `None` for a read-only context (the read-path test injector). The daemon
    /// path ([`Ctx::open`]) always has one.
    pub fn writer(&self) -> Option<&Arc<Writer>> {
        self.writer.as_ref()
    }

    /// The vault root, needed by the write methods to load config and build the
    /// [`Vault`](crate::vault::fs::Vault) for ingest.
    pub fn vault_dir(&self) -> &Path {
        &self.vault_dir
    }

    /// Borrow the read-only pool, for the write-method tests that write through the
    /// writer and then read the note back via [`methods_read::read`] (which takes
    /// the pool directly). Test-only so the production pool stays encapsulated.
    #[cfg(test)]
    pub fn dpool_for_test(&self) -> &Pool<RoManager> {
        &self.dpool
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
/// `dispatch` is `async`: **every** read method (`peek` / `read` / `stats` /
/// `search` / `orient`) checks a connection out of the [`deadpool`] pool and runs
/// its blocking SQLite on a managed thread via `conn.interact(...).await`, so
/// none ever parks a tokio worker (the listener awaits this directly — no
/// `spawn_blocking` wrapper). For `search`, the ONNX query embedding runs under
/// an embedding permit and **outside** the DB checkout (slice 3.5.4), so a burst
/// of `search` load cannot hold a connection across inference and stall the cheap
/// reads (no head-of-line blocking).
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
            Ok(params) => result_or_invalid(
                req,
                methods_read::search(&ctx.dpool, &ctx.embed_sem, &ctx.vault_dir, &params).await,
            ),
            Err(resp) => resp,
        },
        "nark/orient" => match orient_params(req) {
            Ok(params) => result_or_invalid(
                req,
                methods_read::orient(&ctx.dpool, &ctx.vault_dir, &params).await,
            ),
            Err(resp) => resp,
        },
        "nark/write" => match write_params(req) {
            Ok(params) => result_or_invalid(req, methods_write::write(ctx, params).await),
            Err(resp) => resp,
        },
        "nark/link" => match link_params(req) {
            Ok(params) => result_or_invalid(req, methods_write::link(ctx, params).await),
            Err(resp) => resp,
        },
        "nark/delete" => match delete_params(req) {
            Ok(params) => result_or_invalid(req, methods_write::delete(ctx, params).await),
            Err(resp) => resp,
        },
        _ => RPCResponse::error(req.id.clone(), METHOD_NOT_FOUND, "method not found", None),
    }
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

/// Parse the `nark/write` `params` object into [`WriteParams`].
///
/// `note` is **required** — the full note markdown document (frontmatter + body)
/// `vault.ingest` consumes, exactly what `nark write` reads from a file/stdin. A
/// missing or non-string `note` is `-32602 invalid params`. `auto_link` is an
/// optional boolean (default `false`), mirroring `nark write --auto-link`.
/// `idempotency_key` is an optional string (Phase 6, slice 6.3): when present the
/// single serializing writer applies the write at most once for that key and
/// returns the cached result on a retry; absent -> the write always applies. A
/// non-string `idempotency_key` is `-32602`. A missing `params`, or a `params`
/// that is not an object, is an error (unlike the read methods, `write` has a
/// required field, so an empty object is rejected too via the missing `note`).
fn write_params(req: &RPCRequest) -> Result<WriteParams, RPCResponse> {
    let obj = params_object(req)?;
    let note = match opt_string(req, obj, "note")? {
        Some(note) => note,
        None => {
            return Err(invalid(
                req,
                "invalid params: 'note' (the note markdown) is required",
            ));
        }
    };
    Ok(WriteParams {
        note,
        auto_link: opt_bool(req, obj, "auto_link")?,
        idempotency_key: opt_string(req, obj, "idempotency_key")?,
    })
}

/// Parse the `nark/link` `params` object into [`LinkParams`].
///
/// Mirrors `nark link <sources...> --target <id> [--rel <rel>]` (`cli/link.rs`):
/// `sources` is a **required**, non-empty array of note id strings; `target` is a
/// **required** note id string; `rel` is optional and defaults to `"references"`
/// (the CLI's `--rel` default). `idempotency_key` is an optional string (slice
/// 6.3). A missing/empty `sources`, a missing `target`, or a wrong-typed field is
/// `-32602 invalid params`.
fn link_params(req: &RPCRequest) -> Result<LinkParams, RPCResponse> {
    let obj = params_object(req)?;
    let sources = opt_string_array(req, obj, "sources")?;
    if sources.is_empty() {
        return Err(invalid(
            req,
            "invalid params: 'sources' (a non-empty array of note ids) is required",
        ));
    }
    let target = match opt_string(req, obj, "target")? {
        Some(target) => target,
        None => {
            return Err(invalid(
                req,
                "invalid params: 'target' (the target note id) is required",
            ));
        }
    };
    Ok(LinkParams {
        sources,
        target,
        rel: opt_string(req, obj, "rel")?.unwrap_or_else(|| "references".to_string()),
        idempotency_key: opt_string(req, obj, "idempotency_key")?,
    })
}

/// Parse the `nark/delete` `params` object into [`DeleteParams`].
///
/// Mirrors `nark delete <ids...> [-f] [-rf]` (`cli/delete.rs`): `ids` is an array
/// of note id strings (absent -> empty, exactly the clap positional default — an
/// empty delete is a no-op deleting zero notes), `force` (`-f`) and `recursive`
/// (`-r`) are optional booleans defaulting to `false`. The CLI's clap layer makes
/// `recursive` require `force`; this enforces the same guard so `recursive` without
/// `force` is `-32602 invalid params` rather than a silently-ignored flag.
/// `idempotency_key` is an optional string (slice 6.3). A wrong-typed field is
/// `-32602`.
fn delete_params(req: &RPCRequest) -> Result<DeleteParams, RPCResponse> {
    let obj = params_object(req)?;
    let force = opt_bool(req, obj, "force")?;
    let recursive = opt_bool(req, obj, "recursive")?;
    if recursive && !force {
        return Err(invalid(
            req,
            "invalid params: 'recursive' requires 'force' (the -rf purge mode)",
        ));
    }
    Ok(DeleteParams {
        ids: opt_string_array(req, obj, "ids")?,
        force,
        recursive,
        idempotency_key: opt_string(req, obj, "idempotency_key")?,
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

    /// Build a [`Ctx`] over `dir`'s seeded registry: the single deadpool pool
    /// every read method (cheap reads plus `search`/`orient`) checks out from as
    /// of slice 3.5.4. Async because the deadpool pool is built on the runtime.
    async fn ctx_for(dir: &std::path::Path) -> Ctx {
        let dpool = dpool::open_ro_pool(dir, 2)
            .await
            .expect("open deadpool pool");
        Ctx::new(dpool, dir.to_path_buf())
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

    /// Build a [`Ctx`] over `dir` with a real writer (the daemon's write path)
    /// plus the read-only pool — what the router needs to route `nark/write`.
    async fn ctx_with_writer(dir: &std::path::Path) -> Ctx {
        let pool = dpool::open_ro_pool(dir, 2).await.expect("open read pool");
        let writer = std::sync::Arc::new(Writer::open(dir).expect("open writer"));
        Ctx::with_writer(pool, dir.to_path_buf(), writer)
    }

    /// The router routes `nark/write` to the write method: the parsed `note`
    /// markdown is ingested and the success result carries the note id + title.
    #[tokio::test]
    async fn write_routes_through_dispatch_and_creates_note() {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-write-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        drop(crate::db::open_registry(&dir).expect("seed registry"));
        let ctx = ctx_with_writer(&dir).await;

        let resp = dispatch(
            &ctx,
            &request("w1", "nark/write", Some(json!({"note": NOTE}))),
        )
        .await;
        let v = expect_result(resp);
        assert!(v["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert_eq!(v["title"], "Router Note");

        // The committed note is then readable through the same daemon.
        let id = v["id"].as_str().unwrap();
        let read = dispatch(&ctx, &request("r1", "nark/read", Some(json!({"id": id})))).await;
        let rv = expect_result(read);
        assert_eq!(rv["body"], "Router body text.");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/write` requests carrying the SAME `idempotency_key` route through
    /// the writer's dedup: exactly one note/version is created and the second
    /// response is the identical cached result (same id). A retried write over the
    /// wire is idempotent.
    #[tokio::test]
    async fn write_same_idempotency_key_dedups_through_dispatch() {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-write-idem-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        drop(crate::db::open_registry(&dir).expect("seed registry"));
        let ctx = ctx_with_writer(&dir).await;

        let params = json!({ "note": NOTE, "idempotency_key": "wire-key" });
        let first =
            expect_result(dispatch(&ctx, &request("i1", "nark/write", Some(params.clone()))).await);
        let second =
            expect_result(dispatch(&ctx, &request("i2", "nark/write", Some(params))).await);

        assert_eq!(
            first, second,
            "the retried write returns the identical cached result"
        );

        // Exactly one note + one version exist despite two write requests.
        let stats = expect_result(dispatch(&ctx, &request("s1", "nark/stats", None)).await);
        assert_eq!(stats["total_notes"], 1, "same key => one note");
        assert_eq!(stats["total_versions"], 1, "same key => one version");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/write` with a missing `note` is `-32602 invalid params` (the required
    /// field guard), not a panic and not a write.
    #[tokio::test]
    async fn write_missing_note_is_invalid_params() {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-write-missing-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        drop(crate::db::open_registry(&dir).expect("seed registry"));
        let ctx = ctx_with_writer(&dir).await;

        let resp = dispatch(&ctx, &request("w2", "nark/write", Some(json!({})))).await;
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a fresh write-capable vault dir for the link/delete router tests.
    fn fresh_write_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-rpc-{tag}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        drop(crate::db::open_registry(&dir).expect("seed registry"));
        dir
    }

    /// Write a note through the daemon and return its committed id.
    async fn write_note(ctx: &Ctx, title: &str, body: &str) -> String {
        let note = format!(
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
        );
        let resp = dispatch(
            ctx,
            &request("seed", "nark/write", Some(json!({ "note": note }))),
        )
        .await;
        expect_result(resp)["id"]
            .as_str()
            .expect("write returns id")
            .to_string()
    }

    /// The router routes `nark/link`: a typed edge is created between two seeded
    /// notes (verified via a follow-up `nark/peek` link-count read over the same
    /// daemon), and the response mirrors the CLI's `{target, rel, linked, ...}`.
    #[tokio::test]
    async fn link_routes_through_dispatch_and_creates_edge() {
        let dir = fresh_write_dir("link");
        let ctx = ctx_with_writer(&dir).await;

        let src = write_note(&ctx, "Src", "Source body.").await;
        let dst = write_note(&ctx, "Dst", "Target body.").await;

        let resp = dispatch(
            &ctx,
            &request(
                "l1",
                "nark/link",
                Some(json!({ "sources": [src], "target": dst, "rel": "depends-on" })),
            ),
        )
        .await;
        let v = expect_result(resp);
        assert_eq!(v["target"], dst);
        assert_eq!(v["rel"], "depends-on");
        assert_eq!(v["linked"], 1);

        // The typed edge materialized (verified directly): src -> dst, depends-on.
        let outgoing = {
            let conn = crate::db::open_registry(&dir).expect("open registry");
            let (out, _in) = crate::registry::edges::get_edges(&conn, &src).expect("get_edges");
            drop(conn);
            out
        };
        assert!(
            outgoing
                .iter()
                .any(|e| e.note_id == dst && e.edge_type == "depends-on"),
            "the depends-on edge from src to dst must exist"
        );

        // And it is visible through a read: the target gained incoming link(s).
        let peek = expect_result(
            dispatch(
                &ctx,
                &request("p1", "nark/peek", Some(json!({ "id": dst }))),
            )
            .await,
        );
        assert!(
            peek["links_in"].as_i64().unwrap() >= 1,
            "target has incoming link(s)"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The router routes `nark/delete`: a soft delete (default) retracts the note —
    /// a follow-up `nark/peek` shows `status: retracted` — and the response mirrors
    /// the CLI's `{deleted, mode, notes}`.
    #[tokio::test]
    async fn delete_routes_through_dispatch_and_retracts() {
        let dir = fresh_write_dir("delete");
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Doomed", "Body.").await;

        let resp = dispatch(
            &ctx,
            &request("d1", "nark/delete", Some(json!({ "ids": [id] }))),
        )
        .await;
        let v = expect_result(resp);
        assert_eq!(v["deleted"], 1);
        assert_eq!(v["mode"], "retract");

        let peek = expect_result(
            dispatch(&ctx, &request("p2", "nark/peek", Some(json!({ "id": id })))).await,
        );
        assert_eq!(peek["status"], "retracted", "soft delete retracts the note");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/delete` requests with the SAME `idempotency_key` dedup through the
    /// writer: the second returns the identical cached result without re-applying
    /// (which would otherwise error on the already-deleted note).
    #[tokio::test]
    async fn delete_same_idempotency_key_dedups_through_dispatch() {
        let dir = fresh_write_dir("delete-idem");
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Once", "Body.").await;
        let params = json!({ "ids": [id], "force": true, "idempotency_key": "wire-del" });

        let first = expect_result(
            dispatch(&ctx, &request("d1", "nark/delete", Some(params.clone()))).await,
        );
        let second =
            expect_result(dispatch(&ctx, &request("d2", "nark/delete", Some(params))).await);

        assert_eq!(
            first, second,
            "a retried delete with the same key returns the identical cached result"
        );
        assert_eq!(first["mode"], "hard_delete");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/link` with missing `sources` / `target` is `-32602 invalid params`
    /// (the required-field guards), not a panic.
    #[tokio::test]
    async fn link_missing_required_params_is_invalid_params() {
        let dir = fresh_write_dir("link-bad");
        let ctx = ctx_with_writer(&dir).await;

        // Missing target.
        let err = expect_error(
            dispatch(
                &ctx,
                &request("l1", "nark/link", Some(json!({ "sources": ["abc"] }))),
            )
            .await,
        );
        assert_eq!(err.code, INVALID_PARAMS);

        // Missing / empty sources.
        let err = expect_error(
            dispatch(
                &ctx,
                &request("l2", "nark/link", Some(json!({ "target": "abc" }))),
            )
            .await,
        );
        assert_eq!(err.code, INVALID_PARAMS);

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/delete` with `recursive` but no `force` is `-32602` (mirrors clap's
    /// `requires = "force"` guard) — `recursive` is never silently ignored.
    #[tokio::test]
    async fn delete_recursive_without_force_is_invalid_params() {
        let dir = fresh_write_dir("delete-bad");
        let ctx = ctx_with_writer(&dir).await;

        let err = expect_error(
            dispatch(
                &ctx,
                &request(
                    "d1",
                    "nark/delete",
                    Some(json!({ "ids": ["abc"], "recursive": true })),
                ),
            )
            .await,
        );
        assert_eq!(err.code, INVALID_PARAMS);

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
