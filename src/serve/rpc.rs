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
//! * any other method -> JSON-RPC `-32601 method not found`.
//!
//! The read methods call the same `registry::*` functions the CLI handlers do
//! and build the same JSON shape. A missing/ambiguous id, a malformed params
//! object, or a missing CAS object produces a clean JSON-RPC error response
//! (`-32602 invalid params`) echoing the request id — never a panic.

use serde_json::{Value, json};

use super::methods_read;
use super::readpool::ReadPool;
use crate::wire::{RPCRequest, RPCResponse};
use std::path::{Path, PathBuf};

/// JSON-RPC error code: the requested method is not implemented.
const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC error code: the method's params were missing/invalid, or the
/// requested note could not be resolved/read (a 404-style application error).
const INVALID_PARAMS: i64 = -32602;

/// Per-daemon context the router hands to the read methods: a read-only
/// connection pool over `<vault_dir>/registry.db` plus the vault root (needed by
/// `nark/read` to resolve CAS object paths).
pub struct Ctx {
    pool: ReadPool,
    vault_dir: PathBuf,
}

impl Ctx {
    /// Build a context, opening the read-only pool against `vault_dir`.
    ///
    /// The registry must already exist (the writer owns creation/migration); the
    /// pool opens it read-only. Used by the serve daemon path.
    pub fn open(vault_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            pool: ReadPool::open(vault_dir)?,
            vault_dir: vault_dir.to_path_buf(),
        })
    }

    /// Build a context from an already-open pool and vault dir. Lets tests inject
    /// a sized pool without re-opening.
    #[cfg(test)]
    pub fn new(pool: ReadPool, vault_dir: PathBuf) -> Self {
        Self { pool, vault_dir }
    }
}

/// Dispatch a parsed [`RPCRequest`] to its method and return the response.
///
/// `ctx` provides the read-only vault access the read methods need; `ping`
/// ignores it. Unknown methods produce `-32601 method not found`; a read method
/// whose params are missing/invalid or whose note cannot be resolved/read
/// produces `-32602 invalid params`. Every path echoes the request id, so a
/// client always gets exactly one response per request and never a panic.
pub fn dispatch(ctx: &Ctx, req: &RPCRequest) -> RPCResponse {
    match req.method.as_str() {
        "ping" => RPCResponse::result(req.id.clone(), json!({"pong": true})),
        "nark/peek" => match id_param(req) {
            Ok(id) => result_or_invalid(req, methods_read::peek(&ctx.pool, &id)),
            Err(resp) => resp,
        },
        "nark/read" => match id_param(req) {
            Ok(id) => result_or_invalid(req, methods_read::read(&ctx.pool, &ctx.vault_dir, &id)),
            Err(resp) => resp,
        },
        "nark/stats" => result_or_invalid(req, methods_read::stats(&ctx.pool)),
        _ => RPCResponse::error(req.id.clone(), METHOD_NOT_FOUND, "method not found", None),
    }
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

    /// Seed a temp vault with one note (writer creates/migrates/seeds), drop the
    /// writer, and return a `Ctx` (read-only pool) plus the note id.
    fn seeded_ctx() -> (Ctx, String, std::path::PathBuf) {
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

        let pool = ReadPool::open_with_size(&dir, 2).expect("open read pool");
        (Ctx::new(pool, dir.clone()), note_id, dir)
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

    #[test]
    fn ping_returns_pong_with_matching_id() {
        let (ctx, _id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("42", "ping", None));
        match resp {
            RPCResponse::Result(r) => {
                assert_eq!(r.id, "42");
                assert_eq!(r.result, json!({"pong": true}));
            }
            RPCResponse::Error(e) => panic!("ping should succeed, got error {e:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let (ctx, _id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("7", "no-such-method", None));
        let err = expect_error(resp);
        assert_eq!(err.code, METHOD_NOT_FOUND);
        assert_eq!(err.message, "method not found");
        assert!(err.data.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peek_returns_meta_fields() {
        let (ctx, id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("1", "nark/peek", Some(json!({"id": id}))));
        let v = expect_result(resp);
        assert_eq!(v["id"], id);
        assert_eq!(v["title"], "Router Note");
        assert_eq!(v["domain"], "engineering");
        assert_eq!(v["tags"], json!(["gamma"]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_returns_body() {
        let (ctx, id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("2", "nark/read", Some(json!({"id": id}))));
        let v = expect_result(resp);
        assert_eq!(v["body"], "Router body text.");
        assert_eq!(v["frontmatter"]["title"], "Router Note");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_returns_counts() {
        let (ctx, _id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("3", "nark/stats", None));
        let v = expect_result(resp);
        assert_eq!(v["total_notes"], 1);
        assert_eq!(v["total_versions"], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_id_returns_invalid_params_not_panic() {
        let (ctx, _id, dir) = seeded_ctx();
        let resp = dispatch(
            &ctx,
            &request("9", "nark/peek", Some(json!({"id": "ffffffff"}))),
        );
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_params_returns_invalid_params() {
        let (ctx, _id, dir) = seeded_ctx();
        let resp = dispatch(&ctx, &request("10", "nark/peek", None));
        let err = expect_error(resp);
        assert_eq!(err.code, INVALID_PARAMS);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
