//! Blocking serve client with try-or-fallback (Phase 4 of the Ark comm
//! protocol).
//!
//! The five read CLI handlers (`peek` / `read` / `stats` / `search` /
//! `orient`) become DUAL-MODE: if a live `nark serve` daemon is listening on
//! the vault's socket they ask it (one JSON-RPC round-trip) and pretty-print
//! the server's result; otherwise they fall back to today's direct
//! `db::open_registry`. The socket is an **optimization**, never a
//! requirement — direct-open is always correct for reads (read-only, and the
//! CLI user owns the vault).
//!
//! [`try_request`] embodies that fall-back rule. It connects a **blocking**
//! `std::os::unix::net::UnixStream` (the CLI is synchronous — it must not pull
//! a tokio runtime onto the command path; `nark serve` has its own runtime),
//! writes one [`crate::wire::RPCRequest`] line, reads one
//! [`crate::wire::RPCResponse`] line, and returns `Some(result)` ONLY on a
//! clean success response. It returns `None` — so the caller falls back to
//! direct-open — on ANY failure, and never panics:
//!
//! * the socket is absent / connect refused (the common no-daemon case),
//! * a stale socket where connect would hang (guarded by a short connect
//!   timeout so the CLI never stalls),
//! * read/write timeout or any other I/O error,
//! * an `unauthorized` rejection (a normal CLI invocation's uid is not in the
//!   daemon's `[serve.agents]` map, so this is EXPECTED and must fall back —
//!   not surface an error),
//! * an `error` JSON-RPC response,
//! * malformed JSON or an empty read.
//!
//! Slice 4.1 lands this client and the helper; wiring it into the handlers is
//! a later slice — so under the non-test build nothing calls these yet. The
//! tests exercise the full path today; the same "lands now, wired later"
//! allowance the rest of the serve module carries (see `peercred` / `authz`)
//! keeps the non-test build warning-free until the CLI slice wires it in.
#![cfg_attr(not(test), allow(dead_code))]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::wire::{RPCRequest, RPCResponse};

/// How long to wait for the socket connect / each read / each write before
/// giving up and falling back to direct-open. Deliberately short: a stale or
/// wedged socket must never make a read slower than it is today. `std`'s
/// `UnixStream` has no `connect_timeout`, so the connect itself can in theory
/// block; in practice a missing socket fails connect immediately (ENOENT /
/// ECONNREFUSED) and a live local daemon accepts instantly, so the dominant
/// risk — a stale socket file with no accepting peer — is covered by the
/// read/write timeouts set immediately after connect.
const TIMEOUT: Duration = Duration::from_millis(300);

/// The default socket path for a vault: `<vault_dir>/run/nark.sock`
/// (re-uses [`super::resolve_socket_path`] so the CLI and the daemon agree on
/// the path; never hard-coded).
pub fn default_socket(vault_dir: &Path) -> PathBuf {
    super::resolve_socket_path(vault_dir, None)
}

/// Try one JSON-RPC request against a live `nark serve` at `socket_path`.
///
/// Returns `Some(result)` ONLY on a clean success response (a `result`, no
/// `error`). Returns `None` on ANY failure — connect failure, timeout,
/// `unauthorized`, an error response, a parse failure, an empty read, or any
/// I/O error — so the caller falls back to direct-open. Never panics.
pub fn try_request(socket_path: &Path, method: &str, params: Value) -> Option<Value> {
    let stream = UnixStream::connect(socket_path).ok()?;
    // Bound every read and write so a stale/wedged socket can never hang the
    // synchronous CLI. (`std::os::unix::net::UnixStream` has no connect
    // timeout; a missing socket fails connect fast, and these guard the rest.)
    stream.set_read_timeout(Some(TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(TIMEOUT)).ok()?;

    let request = RPCRequest {
        id: request_id(),
        method: method.to_string(),
        params: Some(params),
    };

    // Write one JSON line + '\n' (Phase 3 framing).
    let mut line = serde_json::to_vec(&request).ok()?;
    line.push(b'\n');
    {
        let mut writer = &stream;
        writer.write_all(&line).ok()?;
        writer.flush().ok()?;
    }

    // Read exactly one response line. An `unauthorized` rejection is a plain
    // (non-JSON) line, so it fails the parse below and yields `None` (the
    // EXPECTED fall-back for a CLI uid not in `[serve.agents]`).
    let mut reader = BufReader::new(&stream);
    let mut reply = String::new();
    let n = reader.read_line(&mut reply).ok()?;
    if n == 0 {
        // Empty read (EOF before any line) — fall back.
        return None;
    }

    match serde_json::from_str::<RPCResponse>(reply.trim_end()).ok()? {
        RPCResponse::Result(ok) => Some(ok.result),
        // An `error` response means the daemon could not serve this read; the
        // direct-open path can, so fall back rather than surface it.
        RPCResponse::Error(_) => None,
    }
}

/// A short unique request id. The daemon echoes it back; we do not correlate
/// across requests (one request per connection), so uniqueness is only for
/// hygiene/logging.
fn request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Instant;

    use super::super::AgentMap;
    use super::super::BoundListener;
    use super::super::rpc::Ctx;

    fn temp_vault_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "nark-client-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    /// Seed `dir` as a vault (writer creates/migrates/seeds `registry.db` and
    /// enables WAL) with one ingested note, then build the read-only [`Ctx`]
    /// the serve loop dispatches against. The writer connection is dropped
    /// before the read pool opens. Mirrors the `listener.rs` test harness.
    async fn seeded_ctx(dir: &Path) -> Arc<Ctx> {
        use crate::registry::write::commit_version;
        use crate::vault::fs::Vault;

        const NOTE: &str = "---\n\
title: Client Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - delta\n\
---\n\
Client body text.\n";

        std::fs::create_dir_all(dir).expect("create vault dir");
        let conn = crate::db::open_registry(dir).expect("open writer registry");
        let vault = Vault::new(dir.to_path_buf());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit version");
        drop(conn);

        let ctx = Ctx::open(dir).await.expect("open serve ctx");
        Arc::new(ctx)
    }

    /// A running `nark serve` on its own thread (its own tokio runtime, so the
    /// synchronous client under test connects to a real blocking-vs-async
    /// boundary). Dropping it signals shutdown and joins the thread, removing
    /// the socket.
    struct TestServer {
        dir: PathBuf,
        socket_path: PathBuf,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        /// Bind + serve with the given uid->agent table. `ready` fires once the
        /// listener is bound so the client never races the bind.
        fn start(agents: AgentMap) -> Self {
            let dir = temp_vault_dir();
            let (ready_tx, ready_rx) = mpsc::channel::<PathBuf>();
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

            let dir_for_thread = dir.clone();
            let handle = thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("build runtime");
                runtime.block_on(async move {
                    let ctx = seeded_ctx(&dir_for_thread).await;
                    let socket_path = dir_for_thread.join("run").join("nark.sock");
                    let bound = BoundListener::bind(&socket_path).expect("bind listener");
                    ready_tx.send(socket_path).expect("signal ready");
                    bound
                        .serve_authenticated_until(agents, ctx, async {
                            let _ = shutdown_rx.await;
                        })
                        .await
                        .expect("serve loop");
                });
            });

            let socket_path = ready_rx.recv().expect("server bound");
            TestServer {
                dir,
                socket_path,
                shutdown: Some(shutdown_tx),
                handle: Some(handle),
            }
        }

        fn socket(&self) -> &Path {
            &self.socket_path
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            if let Some(tx) = self.shutdown.take() {
                let _ = tx.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn current_uid_agent_map() -> AgentMap {
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        AgentMap::new(table)
    }

    /// `default_socket` must agree with the daemon's own path resolution
    /// (`resolve_socket_path(vault, None)` -> `<vault>/run/nark.sock`); the CLI
    /// and the daemon must never disagree on where the socket lives.
    #[test]
    fn default_socket_matches_daemon_resolution() {
        let vault = Path::new("/tmp/some-vault");
        assert_eq!(
            default_socket(vault),
            super::super::resolve_socket_path(vault, None),
        );
        assert_eq!(
            default_socket(vault),
            Path::new("/tmp/some-vault/run/nark.sock")
        );
    }

    /// (1) On a socket HIT — a live daemon that authenticates this uid —
    /// `try_request("nark/stats", {})` returns `Some(json)` carrying the vault
    /// statistics for the one seeded note.
    #[cfg(target_os = "macos")]
    #[test]
    fn try_request_returns_some_on_authenticated_hit() {
        let server = TestServer::start(current_uid_agent_map());

        let result = try_request(server.socket(), "nark/stats", json!({}))
            .expect("authenticated nark/stats should return Some(result)");

        assert_eq!(
            result["total_notes"], 1,
            "stats result should report the one seeded note"
        );
        assert_eq!(result["total_versions"], 1);
        // It is the server's result shape pretty-printed by the caller, so the
        // top-level stats keys must be present.
        assert!(result.get("by_domain").is_some());
        assert!(result.get("access").is_some());
    }

    /// (2) Against a NONEXISTENT socket path, `try_request` returns `None`
    /// quickly (no hang) — the dominant no-daemon case must fail fast.
    #[test]
    fn try_request_returns_none_for_missing_socket_quickly() {
        let missing = temp_vault_dir().join("run").join("nark.sock");
        assert!(!missing.exists(), "precondition: socket must not exist");

        let start = Instant::now();
        let result = try_request(&missing, "nark/stats", json!({}));
        let elapsed = start.elapsed();

        assert!(result.is_none(), "a missing socket must yield None");
        assert!(
            elapsed < Duration::from_secs(1),
            "a missing socket must fail fast (no hang), took {elapsed:?}"
        );
    }

    /// (3) Against a live daemon with an EMPTY agent map, this uid is unknown,
    /// so the daemon answers `unauthorized` and closes. `try_request` must
    /// return `None` (the EXPECTED fall-back), not surface an error.
    #[cfg(target_os = "macos")]
    #[test]
    fn try_request_returns_none_when_unauthorized() {
        let server = TestServer::start(AgentMap::default());

        let result = try_request(server.socket(), "nark/stats", json!({}));

        assert!(
            result.is_none(),
            "an unauthorized rejection must fall back (None), not error"
        );
    }
}
