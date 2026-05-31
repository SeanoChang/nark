//! Unix domain socket listener for the `nark serve` daemon.
//!
//! Slice 2.2: bind a `UnixListener`, accept connections, and answer a
//! line-oriented `ping` with `pong`. The socket file is created under a
//! `0700` parent directory and removed on shutdown (ctrl-c or listener drop).
//!
//! Slice 2.5 assembles the authenticated path: each connection's peer uid is
//! extracted ([`super::peercred::peer_uid`]), resolved to an agent via the
//! injected [`AgentMap`], and only known agents are served. Unknown / forged
//! uids get an `unauthorized` line and the connection is closed (fail closed).
//!
//! Slice 3.2 replaces the line protocol with one JSON-RPC request per
//! connection: after auth, the handler reads one [`crate::wire::RPCRequest`]
//! line, dispatches it via [`super::rpc::dispatch`], and writes one
//! [`crate::wire::RPCResponse`] line. `ping` is now a JSON-RPC method
//! (`{"pong": true}`); unknown methods return `-32601`, malformed JSON returns
//! `-32700`. The unknown-uid rejection still precedes any RPC parsing.
//!
//! Before binding, a pre-bind lstat guard ([`guard_preexisting_socket_path`])
//! refuses to serve if a symlink or a foreign-owned file already sits at the
//! socket path; only a self-owned stale socket is then unlinked and rebound.
//! Full liveness probing lands in a later slice.
//!
//! Each per-connection request read is bounded by [`READ_TIMEOUT`] so an idle
//! peer that never sends a line cannot park its task forever.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use super::AgentMap;
use super::authz::guard_preexisting_socket_path;
use super::peercred::peer_uid;
use super::rpc::{self, Ctx};
use crate::wire::{RPCRequest, RPCResponse};

/// JSON-RPC error code: the request line was not valid JSON.
const PARSE_ERROR: i64 = -32700;
/// JSON-RPC error code: the request was not a valid request (here: too large).
const INVALID_REQUEST: i64 = -32600;

/// How long to wait for an authenticated peer to send its request line before
/// the connection is logged and closed. A peer that connects and never sends a
/// newline would otherwise park the spawned task forever.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of bytes accepted for a single request line. An authenticated
/// peer that streams bytes without a newline would otherwise grow the read
/// buffer unbounded until [`READ_TIMEOUT`] fires (a large allocation). The read
/// is capped at this many bytes (Spec §16 max message size); a line that hits
/// the cap without a terminating newline is rejected with a `-32600` error and
/// the connection is closed cleanly.
const MAX_REQUEST_BYTES: u64 = 1 << 20; // 1 MiB

/// Resolve the socket path for the daemon.
///
/// When `socket` is provided it is used verbatim; otherwise the path defaults
/// to `<vault_dir>/run/nark.sock`. The default deliberately nests the socket in
/// a dedicated `run/` subdir: [`ensure_socket_dir`] chmods the socket's *parent*
/// to `0700`, and that parent must never be the vault root (`~/.ark`), which
/// holds `objects/`, `registry.db`, and `config.toml` shared by the whole CLI.
pub fn resolve_socket_path(vault_dir: &Path, socket: Option<String>) -> PathBuf {
    match socket {
        Some(s) => PathBuf::from(s),
        None => vault_dir.join("run").join("nark.sock"),
    }
}

/// Ensure the parent directory of `socket_path` exists at mode `0700`.
///
/// This chmods the socket's *immediate parent* (the dedicated `run/` dir for the
/// default path). It must only ever be pointed at a socket-only directory — the
/// default path keeps the socket out of the vault root so this never alters the
/// vault root's mode. With a `--socket` override the operator owns that choice.
fn ensure_socket_dir(socket_path: &Path) -> Result<()> {
    let parent = socket_path
        .parent()
        .context("socket path has no parent directory")?;
    if !parent.exists() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating socket directory {}", parent.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o700);
        std::fs::set_permissions(parent, perms)
            .with_context(|| format!("setting 0700 on {}", parent.display()))?;
    }
    Ok(())
}

/// Remove a pre-existing socket file (stale-cleanup).
fn unlink_existing(socket_path: &Path) -> Result<()> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => {
            Err(e).with_context(|| format!("removing stale socket {}", socket_path.display()))
        }
    }
}

/// A bound listener whose socket file is removed when dropped.
///
/// Holding the `UnixListener` and the bound path together lets us guarantee the
/// socket file is cleaned up on both graceful shutdown and unwinding.
pub struct BoundListener {
    listener: UnixListener,
    path: PathBuf,
}

impl BoundListener {
    /// Bind a fresh listener at `socket_path`.
    ///
    /// Steps, in order:
    /// 1. create the `0700` parent directory ([`ensure_socket_dir`]) — primary
    ///    containment;
    /// 2. **pre-bind** lstat guard ([`guard_preexisting_socket_path`]): refuse if
    ///    a symlink or a foreign-owned file already sits at the path. This must
    ///    run before unlink+bind, because once we bind our own socket the path is
    ///    owned by us and the check can no longer fire;
    /// 3. unlink any (now-verified self-owned) stale socket;
    /// 4. bind the `UnixListener`.
    pub fn bind(socket_path: &Path) -> Result<Self> {
        ensure_socket_dir(socket_path)?;
        guard_preexisting_socket_path(socket_path)?;
        unlink_existing(socket_path)?;
        let listener = UnixListener::bind(socket_path)
            .with_context(|| format!("binding unix socket {}", socket_path.display()))?;
        Ok(Self {
            listener,
            path: socket_path.to_path_buf(),
        })
    }

    /// The path the listener is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Accept-loop until `shutdown` resolves. Each connection is handled by
    /// reading a single JSON-RPC request line and writing its response.
    ///
    /// This is the unauthenticated baseline from slice 2.2; the daemon path now
    /// uses [`Self::serve_authenticated_until`]. It is retained (and exercised
    /// by tests) as the documented pre-auth round-trip primitive.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn serve_until<F>(&self, ctx: Arc<Ctx>, shutdown: F) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                accepted = self.listener.accept() => {
                    let (stream, _addr) = accepted.context("accepting connection")?;
                    let ctx = Arc::clone(&ctx);
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, &ctx).await {
                            eprintln!("nark serve: connection error: {e:#}");
                        }
                    });
                }
                () = &mut shutdown => {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Authenticated accept-loop until `shutdown` resolves.
    ///
    /// For each connection the peer's uid is extracted and resolved against
    /// `agents`. Known uids are served one JSON-RPC request/response round-trip;
    /// unknown / forged uids receive an `unauthorized` line and the connection is
    /// closed (fail closed) before any RPC parsing. This is the Phase 3 serve
    /// path built on the Phase 2 authenticated accept loop.
    ///
    /// The per-connection request read is bounded by [`READ_TIMEOUT`]; a peer
    /// that connects and never sends a line is closed rather than parking its
    /// task forever.
    pub async fn serve_authenticated_until<F>(
        &self,
        agents: AgentMap,
        ctx: Arc<Ctx>,
        shutdown: F,
    ) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        self.serve_authenticated_until_with_limits(
            agents,
            ctx,
            READ_TIMEOUT,
            MAX_REQUEST_BYTES,
            shutdown,
        )
        .await
    }

    /// As [`Self::serve_authenticated_until`], but with an explicit per-connection
    /// read timeout. Lets tests drive the timeout path with a short duration
    /// without waiting the production [`READ_TIMEOUT`]. The request-size cap stays
    /// at the production [`MAX_REQUEST_BYTES`].
    #[cfg(test)]
    pub async fn serve_authenticated_until_with_timeout<F>(
        &self,
        agents: AgentMap,
        ctx: Arc<Ctx>,
        read_timeout: Duration,
        shutdown: F,
    ) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        self.serve_authenticated_until_with_limits(
            agents,
            ctx,
            read_timeout,
            MAX_REQUEST_BYTES,
            shutdown,
        )
        .await
    }

    /// As [`Self::serve_authenticated_until`], but with an explicit per-connection
    /// read timeout *and* request-size cap. Lets tests drive both the timeout path
    /// (short duration) and the over-size path (small cap) without the production
    /// [`READ_TIMEOUT`] / [`MAX_REQUEST_BYTES`].
    ///
    /// Each connection's JSON-RPC dispatch is awaited directly (see
    /// [`handle_authenticated_connection`]): every read method runs its blocking
    /// SQLite via the deadpool pool's `interact` (own thread) and `get().await`
    /// backpressures, so a saturated pool can never park a tokio worker and starve
    /// the accept loop / shutdown future (Spec §10). As of slice 3.5.4 `search` /
    /// `orient` run on the same deadpool pool, and `search`'s ONNX inference runs
    /// under an embedding permit **outside** the DB checkout, so no `spawn_blocking`
    /// wrapper remains.
    pub async fn serve_authenticated_until_with_limits<F>(
        &self,
        agents: AgentMap,
        ctx: Arc<Ctx>,
        read_timeout: Duration,
        max_request_bytes: u64,
        shutdown: F,
    ) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        let agents = Arc::new(agents);
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                accepted = self.listener.accept() => {
                    let (stream, _addr) = accepted.context("accepting connection")?;
                    let agents = Arc::clone(&agents);
                    let ctx = Arc::clone(&ctx);
                    tokio::spawn(async move {
                        if let Err(e) = handle_authenticated_connection(
                            stream,
                            &agents,
                            ctx,
                            read_timeout,
                            max_request_bytes,
                        )
                        .await
                        {
                            eprintln!("nark serve: connection error: {e:#}");
                        }
                    });
                }
                () = &mut shutdown => {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for BoundListener {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Handle a single connection: read one JSON-RPC request line and write its
/// response.
///
/// Unauthenticated baseline from slice 2.2, superseded on the daemon path by
/// [`handle_authenticated_connection`]; retained as a tested primitive. Phase 3
/// moves it to the same JSON-RPC framing as the authenticated path so there is a
/// single protocol on the wire (no dual line/JSON-RPC handling). Malformed JSON
/// yields a `-32700 parse error` with an empty id.
#[cfg_attr(not(test), allow(dead_code))]
async fn handle_connection(stream: UnixStream, ctx: &Ctx) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .context("reading request line")?;
    if n == 0 {
        return Ok(());
    }
    let response = match serde_json::from_str::<RPCRequest>(line.trim_end()) {
        Ok(req) => rpc::dispatch(ctx, &req).await,
        Err(e) => {
            eprintln!("nark serve: unauthenticated peer sent malformed JSON-RPC: {e}");
            RPCResponse::error("", PARSE_ERROR, "parse error", None)
        }
    };
    write_response(&mut write_half, &response).await
}

/// Handle a single connection with peer authentication, then one JSON-RPC
/// request/response round-trip.
///
/// Order, fail-closed:
/// 1. Extract the peer uid and resolve it to an agent via `agents`. Unknown
///    uids get an `unauthorized` line and the connection is closed *before any
///    RPC parsing* (the unknown-uid rejection precedes everything else).
/// 2. Read exactly one line, bounded by `read_timeout` *and* `max_request_bytes`:
///    a peer that authenticates but never sends a request is logged and closed
///    instead of parking the spawned task forever, and a peer that streams bytes
///    without a newline is cut off at the cap (not allowed to grow the buffer
///    unbounded) and answered with a `-32600 request too large` error.
/// 3. Parse the line as a [`RPCRequest`]. Malformed JSON yields a JSON-RPC
///    `-32700 parse error` with an empty id (there is no id to echo).
/// 4. Dispatch via [`rpc::dispatch`] and write the single [`RPCResponse`] as one
///    JSON line + newline.
///
/// [`rpc::dispatch`] is now `async`: every read method (`peek` / `read` /
/// `stats` / `search` / `orient`) checks a connection out of the [`deadpool`]
/// pool and runs its blocking SQLite on a managed thread via
/// `conn.interact(...).await`, so this handler can `await` the dispatch directly
/// — there is no `spawn_blocking` wrapper around it (slice 3.5.2/3.5.4). The
/// reactor stays free even when the pool is saturated: `pool.get().await`
/// backpressures and `interact` owns its own blocking thread, so no tokio worker
/// is parked (Spec §10: reads must never block the reactor). `search`'s ONNX
/// inference runs under a bounded embedding permit and **outside** the DB
/// checkout (slice 3.5.4), so it never holds a connection across inference.
///
/// The accept loop is unaffected by the per-connection timeout because this runs
/// inside the spawned per-connection task.
async fn handle_authenticated_connection(
    stream: UnixStream,
    agents: &AgentMap,
    ctx: Arc<Ctx>,
    read_timeout: Duration,
    max_request_bytes: u64,
) -> Result<()> {
    let uid = peer_uid(&stream).context("extracting peer uid")?;
    let agent = match agents.resolve(uid) {
        Some(agent) => agent.to_string(),
        None => {
            eprintln!("nark serve: rejecting unauthorized peer (uid {uid})");
            let (_read_half, mut write_half) = stream.into_split();
            write_half
                .write_all(b"unauthorized\n")
                .await
                .context("writing rejection")?;
            write_half.flush().await.context("flushing rejection")?;
            // Dropping `write_half` (and the moved read half) closes the
            // connection so the client sees EOF after the rejection.
            return Ok(());
        }
    };

    eprintln!("nark serve: authenticated {agent} (uid {uid})");

    let (read_half, mut write_half) = stream.into_split();
    // Cap the read at `max_request_bytes`: `take` makes the underlying reader
    // return EOF once the cap is reached, so `read_line` cannot grow `line`
    // beyond the cap regardless of whether the peer ever sends a newline.
    let mut reader = BufReader::new(read_half).take(max_request_bytes);
    let mut line = String::new();
    let n = match tokio::time::timeout(read_timeout, reader.read_line(&mut line)).await {
        Ok(read) => read.context("reading request line")?,
        Err(_elapsed) => {
            eprintln!(
                "nark serve: {agent} (uid {uid}) sent no request within {}s; closing",
                read_timeout.as_secs()
            );
            // Dropping the halves closes the connection.
            return Ok(());
        }
    };
    if n == 0 {
        return Ok(());
    }

    // If the read filled the cap without a terminating newline, the request is
    // over-size (or unterminated). Answer with `-32600 request too large` and
    // close cleanly rather than parse a truncated line or keep reading.
    if n as u64 >= max_request_bytes && !line.ends_with('\n') {
        eprintln!(
            "nark serve: {agent} (uid {uid}) sent an over-size request (>= {max_request_bytes} bytes); closing"
        );
        let response = RPCResponse::error("", INVALID_REQUEST, "request too large", None);
        return write_response(&mut write_half, &response).await;
    }

    // Parse the one request line as JSON-RPC; malformed JSON is reported with a
    // `-32700 parse error` and an empty id (we have no id to echo).
    let response = match serde_json::from_str::<RPCRequest>(line.trim_end()) {
        Ok(req) => {
            // dispatch is async: cheap methods run their blocking SQLite via
            // deadpool `interact` (own thread), and `get().await` backpressures —
            // so awaiting here never parks the reactor (Spec §10). No
            // `spawn_blocking` wrapper needed (search/orient handle their own
            // blocking internally until slice 3.5.4).
            rpc::dispatch(&ctx, &req).await
        }
        Err(e) => {
            eprintln!("nark serve: {agent} (uid {uid}) sent malformed JSON-RPC: {e}");
            RPCResponse::error("", PARSE_ERROR, "parse error", None)
        }
    };

    write_response(&mut write_half, &response).await
}

/// Serialize a [`RPCResponse`] to one JSON line + newline and flush it.
async fn write_response<W>(write_half: &mut W, response: &RPCResponse) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let mut bytes = serde_json::to_vec(response).context("serializing RPC response")?;
    bytes.push(b'\n');
    write_half
        .write_all(&bytes)
        .await
        .context("writing RPC response")?;
    write_half.flush().await.context("flushing RPC response")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::oneshot;

    fn temp_socket_dir() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "nark-serve-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        );
        base.join(unique)
    }

    #[test]
    fn resolve_defaults_to_vault_run_nark_sock() {
        let vault = Path::new("/tmp/some-vault");
        let resolved = resolve_socket_path(vault, None);
        assert_eq!(resolved, Path::new("/tmp/some-vault/run/nark.sock"));
    }

    #[test]
    fn resolve_uses_override() {
        let vault = Path::new("/tmp/some-vault");
        let resolved = resolve_socket_path(vault, Some("/run/custom.sock".to_string()));
        assert_eq!(resolved, Path::new("/run/custom.sock"));
    }

    /// Seed `dir` as a vault (writer creates/migrates/seeds `registry.db` and
    /// enables WAL) with one ingested note, then build the read-only [`Ctx`] the
    /// serve loops dispatch against. Returns `(ctx, note_id)`. The writer
    /// connection is dropped before the read pool opens.
    async fn seeded_ctx(dir: &Path) -> (Arc<Ctx>, String) {
        use crate::registry::write::commit_version;
        use crate::vault::fs::Vault;

        const NOTE: &str = "---\n\
title: Socket Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - delta\n\
---\n\
Socket body text.\n";

        std::fs::create_dir_all(dir).expect("create vault dir");
        let conn = crate::db::open_registry(dir).expect("open writer registry");
        let vault = Vault::new(dir.to_path_buf());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit version");
        let note_id = result.note_id.clone();
        drop(conn);

        let ctx = Ctx::open(dir).await.expect("open serve ctx");
        (Arc::new(ctx), note_id)
    }

    /// Read one newline-terminated line from `stream` and parse it as JSON.
    /// Shared by the JSON-RPC round-trip tests.
    async fn read_json_line(stream: UnixStream) -> serde_json::Value {
        let mut reply = String::new();
        let mut reader = BufReader::new(stream);
        reader
            .read_line(&mut reply)
            .await
            .expect("read response line");
        serde_json::from_str(reply.trim_end()).expect("response is valid JSON")
    }

    /// Phase 3 exit: an authenticated JSON-RPC `ping` request round-trips to a
    /// `{"pong": true}` result whose id echoes the request id, when the
    /// connecting uid is a known agent.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_ping_known_uid_gets_pong() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");
        let (ctx, _note_id) = seeded_ctx(&dir).await;

        // Register the *test process's own* uid as a known agent so the
        // connecting client (this process) authenticates as "tester".
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream
            .write_all(b"{\"id\":\"abc\",\"method\":\"ping\"}\n")
            .await
            .expect("write ping request");
        stream.flush().await.expect("flush ping request");

        let v = read_json_line(stream).await;
        assert_eq!(v["id"], "abc", "response id should echo request id");
        assert_eq!(v["result"]["pong"], true, "ping should yield pong=true");
        assert!(v.get("error").is_none(), "ping should not be an error");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Phase 3 slice 3.3 exit: the three READ methods (`nark/peek`, `nark/read`,
    /// `nark/stats`) round-trip over the socket against a seeded note, and a bad
    /// id returns a clean error response (not a panic / dropped connection).
    /// One request per connection, so each method opens a fresh connection.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_read_methods_round_trip_over_socket() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");
        let (ctx, note_id) = seeded_ctx(&dir).await;

        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        // Helper: one request line -> one parsed JSON response over a fresh conn.
        async fn round_trip(socket_path: &Path, line: &str) -> serde_json::Value {
            let mut stream = UnixStream::connect(socket_path)
                .await
                .expect("connect to socket");
            stream
                .write_all(line.as_bytes())
                .await
                .expect("write request");
            stream.flush().await.expect("flush request");
            read_json_line(stream).await
        }

        // nark/peek
        let peek = round_trip(
            &socket_path,
            &format!(
                "{{\"id\":\"p\",\"method\":\"nark/peek\",\"params\":{{\"id\":\"{note_id}\"}}}}\n"
            ),
        )
        .await;
        assert_eq!(peek["id"], "p");
        assert_eq!(peek["result"]["id"], note_id);
        assert_eq!(peek["result"]["title"], "Socket Note");

        // nark/read
        let read = round_trip(
            &socket_path,
            &format!(
                "{{\"id\":\"r\",\"method\":\"nark/read\",\"params\":{{\"id\":\"{note_id}\"}}}}\n"
            ),
        )
        .await;
        assert_eq!(read["result"]["body"], "Socket body text.");
        assert_eq!(read["result"]["frontmatter"]["title"], "Socket Note");

        // nark/stats
        let stats = round_trip(&socket_path, "{\"id\":\"s\",\"method\":\"nark/stats\"}\n").await;
        assert_eq!(stats["result"]["total_notes"], 1);
        assert_eq!(stats["result"]["total_versions"], 1);

        // Bad id -> clean error response, connection stays well-behaved.
        let bad = round_trip(
            &socket_path,
            "{\"id\":\"b\",\"method\":\"nark/peek\",\"params\":{\"id\":\"ffffffff\"}}\n",
        )
        .await;
        assert_eq!(bad["id"], "b");
        assert_eq!(
            bad["error"]["code"], -32602,
            "bad id is invalid-params, not a panic"
        );
        assert!(bad.get("result").is_none());

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Slice 3.5 end-to-end: an authenticated client (its own uid mapped to an
    /// agent) issues `nark/peek` and then `nark/search` over two one-shot
    /// connections and gets correct, distinct responses. This exercises the full
    /// assembled path — peer auth -> JSON-RPC framing -> router -> deadpool pool
    /// -> `registry::*` — for more than one method on a single running daemon.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_client_peek_then_search_end_to_end() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");
        let (ctx, note_id) = seeded_ctx(&dir).await;

        // Map the test process's own uid to a known agent so we authenticate.
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        async fn round_trip(socket_path: &Path, line: &str) -> serde_json::Value {
            let mut stream = UnixStream::connect(socket_path)
                .await
                .expect("connect to socket");
            stream
                .write_all(line.as_bytes())
                .await
                .expect("write request");
            stream.flush().await.expect("flush request");
            read_json_line(stream).await
        }

        // 1) nark/peek over its own one-shot connection.
        let peek = round_trip(
            &socket_path,
            &format!(
                "{{\"id\":\"pk\",\"method\":\"nark/peek\",\"params\":{{\"id\":\"{note_id}\"}}}}\n"
            ),
        )
        .await;
        assert_eq!(peek["id"], "pk");
        assert_eq!(peek["result"]["id"], note_id);
        assert_eq!(peek["result"]["title"], "Socket Note");
        assert!(peek.get("error").is_none());

        // 2) nark/search over a fresh one-shot connection. The seeded note has a
        // distinctive body word ("Socket") that the query should match.
        let search = round_trip(
            &socket_path,
            "{\"id\":\"se\",\"method\":\"nark/search\",\"params\":{\"query\":\"socket\"}}\n",
        )
        .await;
        assert_eq!(search["id"], "se");
        assert_eq!(search["result"]["query"], "socket");
        assert_eq!(search["result"]["mode"], "normal");
        assert!(
            search["result"]["hits"].as_u64().unwrap() >= 1,
            "the socket query should hit the seeded note"
        );
        assert_eq!(
            search["result"]["results"][0]["id"], note_id,
            "the seeded note should be the top search hit"
        );
        assert!(search.get("error").is_none());

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Slice 3.5: an unknown-uid connection is rejected *before any method runs*.
    /// The client sends a well-formed `nark/peek` JSON-RPC request, but the
    /// server (empty `AgentMap`) must answer with the plain `unauthorized` line
    /// and close — never a JSON-RPC response — proving auth gates dispatch.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn unknown_uid_rejected_before_any_method_runs() {
        use super::super::AgentMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Empty map -> the connecting uid is unknown -> rejected at the app layer.
        let agents = AgentMap::default();
        let (ctx, note_id) = seeded_ctx(&dir).await;

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // A perfectly valid peek request: if auth did not gate dispatch, this
        // would return a JSON-RPC result. The server must reject before running
        // the method.
        let request = format!(
            "{{\"id\":\"x\",\"method\":\"nark/peek\",\"params\":{{\"id\":\"{note_id}\"}}}}\n"
        );
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write peek request");
        stream.flush().await.expect("flush peek request");

        // The very first line back must be the rejection, not a JSON-RPC reply.
        let mut reply = String::new();
        let mut reader = BufReader::new(stream);
        reader
            .read_line(&mut reply)
            .await
            .expect("read rejection line");
        assert_eq!(
            reply, "unauthorized\n",
            "unknown uid must be rejected before nark/peek runs"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(reply.trim_end()).is_err(),
            "rejection must not be a JSON-RPC response (no method ran)"
        );

        // And the connection is then closed (EOF), with no peek result trailing.
        let mut rest = String::new();
        let n = reader
            .read_line(&mut rest)
            .await
            .expect("read after reject");
        assert_eq!(
            n, 0,
            "server must close after rejecting; no method response follows"
        );

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An authenticated unknown method returns a JSON-RPC `-32601` error echoing
    /// the request id.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_unknown_method_returns_32601() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        let (ctx, _note_id) = seeded_ctx(&dir).await;
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream
            .write_all(b"{\"id\":\"7\",\"method\":\"nope\"}\n")
            .await
            .expect("write request");
        stream.flush().await.expect("flush request");

        let v = read_json_line(stream).await;
        assert_eq!(v["id"], "7", "error response should echo request id");
        assert_eq!(v["error"]["code"], -32601, "unknown method is -32601");
        assert_eq!(v["error"]["message"], "method not found");
        assert!(v.get("result").is_none(), "error has no result");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Malformed JSON on an authenticated connection returns a JSON-RPC
    /// `-32700 parse error` with an empty id.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_malformed_json_returns_32700() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        let (ctx, _note_id) = seeded_ctx(&dir).await;
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream
            .write_all(b"this is not json\n")
            .await
            .expect("write garbage");
        stream.flush().await.expect("flush garbage");

        let v = read_json_line(stream).await;
        assert_eq!(v["id"], "", "parse error carries an empty id");
        assert_eq!(v["error"]["code"], -32700, "malformed JSON is -32700");
        assert_eq!(v["error"]["message"], "parse error");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn unknown_uid_is_rejected_with_unauthorized() {
        use super::super::AgentMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Empty map: the connecting uid is unknown -> rejected at app layer.
        let agents = AgentMap::default();
        // Ctx is required by the loop signature but never reached: rejection
        // precedes any RPC dispatch.
        let (ctx, _note_id) = seeded_ctx(&dir).await;

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // Send a ping; the server must reject before answering.
        stream.write_all(b"ping\n").await.expect("write ping");
        stream.flush().await.expect("flush ping");

        // Read whatever the server sends back: it must be the rejection line,
        // and there must be no `pong`.
        let mut reply = String::new();
        let mut reader = BufReader::new(stream);
        reader
            .read_line(&mut reply)
            .await
            .expect("read rejection line");
        assert_eq!(
            reply, "unauthorized\n",
            "unknown uid should be rejected at the application layer"
        );

        // Connection should be closed right after the rejection (EOF).
        let mut rest = String::new();
        let n = reader
            .read_line(&mut rest)
            .await
            .expect("read after reject");
        assert_eq!(n, 0, "server should close the connection after rejecting");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ping_pong_over_uds_and_socket_lifecycle() {
        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");

        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Socket directory must be mode 0700.
        let dir_mode = std::fs::metadata(&dir)
            .expect("stat socket dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700, "socket dir should be 0700");

        // Socket file exists once bound.
        assert!(socket_path.exists(), "socket file should exist after bind");

        let (ctx, _note_id) = seeded_ctx(&dir).await;
        let (tx, rx) = oneshot::channel::<()>();
        let serve_path = socket_path.clone();
        let server = tokio::spawn(async move {
            bound
                .serve_until(ctx, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
            // Listener dropped here -> socket removed.
            drop(serve_path);
        });

        // Connect and exercise a JSON-RPC ping -> pong round-trip.
        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream
            .write_all(b"{\"id\":\"life\",\"method\":\"ping\"}\n")
            .await
            .expect("write ping request");
        stream.flush().await.expect("flush ping request");

        let v = read_json_line(stream).await;
        assert_eq!(v["id"], "life", "response id should echo request id");
        assert_eq!(v["result"]["pong"], true, "ping should yield pong=true");

        // Trigger shutdown and wait for the server task to finish.
        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        // Socket file must be gone after shutdown (listener drop).
        assert!(
            !socket_path.exists(),
            "socket file should be removed after shutdown"
        );

        // Cleanup temp dir.
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- FIX 1: pre-bind ownership guard ----

    #[test]
    fn bind_refuses_when_socket_path_is_a_symlink() {
        // An attacker plants a symlink at the socket path. `bind` must refuse
        // before unlinking/binding through it.
        let dir = temp_socket_dir();
        std::fs::create_dir_all(&dir).expect("create socket dir");
        let socket_path = dir.join("nark.sock");
        let target = dir.join("elsewhere.sock");
        std::fs::write(&target, b"").expect("create symlink target");
        std::os::unix::fs::symlink(&target, &socket_path).expect("plant symlink");

        // `BoundListener` is not `Debug`, so match rather than `expect_err`.
        let err = match BoundListener::bind(&socket_path) {
            Ok(_) => panic!("bind must refuse a symlinked socket path"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("symlink"),
            "rejection should name the symlink-swap case, got: {err:#}"
        );
        // The symlink must be left in place (not unlinked) since we refused.
        assert!(
            std::fs::symlink_metadata(&socket_path)
                .expect("symlink should still exist")
                .file_type()
                .is_symlink(),
            "guard must refuse before unlinking the planted symlink"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bind_succeeds_for_normal_cold_start() {
        // No pre-existing path: a normal start binds successfully.
        // (`UnixListener::bind` needs a tokio reactor, hence the async test.)
        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("cold-start bind should succeed");
        assert!(socket_path.exists(), "socket file should exist after bind");
        assert_eq!(bound.path(), socket_path.as_path());
        drop(bound); // removes the socket file
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- FIX 2: dedicated socket dir, vault root untouched ----

    #[tokio::test]
    async fn ensure_socket_dir_chmods_run_dir_not_vault_root() {
        // Simulate the default layout: vault_dir/run/nark.sock. The 0700 chmod
        // must land on run/, and the vault root's mode must be left alone.
        let vault = temp_socket_dir();
        std::fs::create_dir_all(&vault).expect("create vault dir");
        // Give the vault root a recognizable, non-0700 mode.
        std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o755))
            .expect("set vault mode");
        let vault_mode_before = std::fs::metadata(&vault)
            .expect("stat vault")
            .permissions()
            .mode()
            & 0o777;

        let socket_path = resolve_socket_path(&vault, None);
        assert!(
            socket_path.ends_with("run/nark.sock"),
            "default socket should live under run/, got {}",
            socket_path.display()
        );

        let bound = BoundListener::bind(&socket_path).expect("bind under run/");

        // run/ must be 0700.
        let run_dir = socket_path.parent().expect("run dir");
        let run_mode = std::fs::metadata(run_dir)
            .expect("stat run dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(run_mode, 0o700, "run/ should be chmodded to 0700");

        // Vault root mode must be unchanged.
        let vault_mode_after = std::fs::metadata(&vault)
            .expect("stat vault after")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            vault_mode_after, vault_mode_before,
            "vault root mode must not be modified by serve"
        );
        assert_ne!(
            vault_mode_after, 0o700,
            "vault root should retain its original (non-0700) mode"
        );

        drop(bound);
        let _ = std::fs::remove_dir_all(&vault);
    }

    // ---- FIX 3: per-connection read timeout ----

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_connection_with_no_request_is_closed_after_timeout() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Authenticate the test process as a known agent.
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);
        let (ctx, _note_id) = seeded_ctx(&dir).await;

        // Short timeout so the test does not hang.
        let read_timeout = Duration::from_millis(150);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until_with_timeout(agents, ctx, read_timeout, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // Deliberately send nothing (no newline). The server must time out the
        // read and close the connection, so our next read sees EOF.
        let mut buf = Vec::new();
        let read = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut buf))
            .await
            .expect("server should close the idle connection within the test budget")
            .expect("read to EOF");
        assert_eq!(read, 0, "idle connection should be closed at EOF, no bytes");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- FIX 1: a saturated read pool must not park a tokio worker ----

    /// Seed a vault on disk (writer ingests one note, drops) and return its dir.
    /// Used by the saturation test, which then opens its own sized pool.
    fn seeded_vault_dir() -> PathBuf {
        use crate::registry::write::commit_version;
        use crate::vault::fs::Vault;

        const NOTE: &str = "---\n\
title: Saturation Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - delta\n\
---\n\
Saturation body text.\n";

        let dir = temp_socket_dir();
        std::fs::create_dir_all(&dir).expect("create vault dir");
        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit version");
        drop(conn);
        dir
    }

    /// FIX 1 (slice 3.5.2 edition) — when every deadpool connection is checked
    /// out, a concurrent `ping` request still round-trips promptly on a 1-worker
    /// runtime, proving the migrated cheap-read path keeps the reactor free
    /// **without** the old `spawn_blocking` wrapper.
    ///
    /// The cheap methods now check a connection out of the deadpool pool and run
    /// their blocking SQLite via `conn.interact(...).await`; `pool.get().await`
    /// backpressures (async wait) when the pool is saturated rather than blocking.
    /// So when `nark/stats` is fired against a fully-checked-out pool, its dispatch
    /// `await`s the checkout and yields the single worker — a concurrent `ping`
    /// (which needs no connection) is still driven to completion. Under the old
    /// synchronous dispatch a blocking checkout would have parked the only worker
    /// and the ping would time out; here it does not, because nothing blocks the
    /// reactor. Once the held connections are dropped the blocked stats completes.
    #[cfg(target_os = "macos")]
    #[test]
    fn saturated_pool_does_not_block_concurrent_request() {
        use super::super::AgentMap;
        use super::super::dpool::open_ro_pool;
        use std::collections::HashMap;

        const N: usize = 2;

        // A 1-worker multi-thread runtime: if the dispatch blocked the worker on a
        // saturated checkout, the concurrent ping could never be driven and the
        // timeout below would trip.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("build runtime");

        runtime.block_on(async {
            let dir = seeded_vault_dir();
            let socket_path = dir.join("nark.sock");
            let bound = BoundListener::bind(&socket_path).expect("bind listener");

            // One deadpool of N conns, shared between the daemon `Ctx` and this
            // test (the pool is `Clone`/`Arc`-backed, so the daemon and the test
            // checkout from the same pool). As of slice 3.5.4 this single pool
            // backs every read method; the cheap `nark/stats` below uses it.
            let dpool = open_ro_pool(&dir, N).await.expect("open deadpool pool");
            let ctx = Arc::new(Ctx::new(dpool.clone(), dir.clone()));

            let me = nix::unistd::getuid().as_raw();
            let mut table = HashMap::new();
            table.insert(me, "tester".to_string());
            let agents = AgentMap::new(table);

            let (tx, rx) = oneshot::channel::<()>();
            let server = tokio::spawn(async move {
                bound
                    .serve_authenticated_until(agents, ctx, async {
                        let _ = rx.await;
                    })
                    .await
                    .expect("serve loop");
            });

            // Occupy ALL N deadpool connections by checking them out and holding
            // the `Object`s alive (they return to the pool only on drop). With
            // every connection held, the daemon's `nark/stats` dispatch must wait
            // on `pool.get().await` — an async backpressure wait, not a worker park.
            let mut holders = Vec::with_capacity(N);
            for _ in 0..N {
                holders.push(dpool.get().await.expect("checkout connection"));
            }

            // Fire a pool-NEEDING request (`nark/stats`) against the saturated
            // pool. Its dispatch `await`s the checkout (cannot complete: every
            // connection is held) and yields the worker.
            let blocked = tokio::spawn({
                let socket_path = socket_path.clone();
                async move {
                    let mut stream = UnixStream::connect(&socket_path)
                        .await
                        .expect("connect to socket");
                    stream
                        .write_all(b"{\"id\":\"blk\",\"method\":\"nark/stats\"}\n")
                        .await
                        .expect("write stats");
                    stream.flush().await.expect("flush stats");
                    read_json_line(stream).await
                }
            });

            // Give the blocked stats request time to reach the checkout wait. Then
            // a concurrent `ping` must STILL round-trip promptly. If awaiting the
            // saturated checkout parked the single worker, this ping would time out.
            tokio::time::sleep(Duration::from_millis(200)).await;
            let ping = async {
                let mut stream = UnixStream::connect(&socket_path)
                    .await
                    .expect("connect to socket");
                stream
                    .write_all(b"{\"id\":\"sat\",\"method\":\"ping\"}\n")
                    .await
                    .expect("write ping");
                stream.flush().await.expect("flush ping");
                read_json_line(stream).await
            };
            let v = tokio::time::timeout(Duration::from_secs(5), ping)
                .await
                .expect("ping must make progress while the pool is saturated");
            assert_eq!(v["id"], "sat");
            assert_eq!(v["result"]["pong"], true);

            // Release the held connections back to the pool; the previously-blocked
            // stats request now wins a checkout and completes.
            drop(holders);
            let stats = tokio::time::timeout(Duration::from_secs(5), blocked)
                .await
                .expect("blocked stats must complete once the pool frees up")
                .expect("stats task join");
            assert_eq!(stats["id"], "blk");
            assert_eq!(stats["result"]["total_notes"], 1);
            tx.send(()).expect("send shutdown");
            server.await.expect("server task join");
            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    // ---- FIX 2: capped request-line read ----

    /// FIX 2 — an authenticated connection that streams bytes without a newline
    /// past the (test-injected small) cap is answered with a `-32600 request too
    /// large` error and the connection is closed cleanly — it does not hang until
    /// the read timeout nor grow the buffer unbounded.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn oversize_request_gets_invalid_request_and_clean_close() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);
        let (ctx, _note_id) = seeded_ctx(&dir).await;

        // Small injected cap so the test sends a modest over-size payload. Generous
        // read timeout so a hang (not the cap) would be the failure, not a timeout.
        let max_request_bytes: u64 = 64;
        let read_timeout = Duration::from_secs(30);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until_with_limits(
                    agents,
                    ctx,
                    read_timeout,
                    max_request_bytes,
                    async {
                        let _ = rx.await;
                    },
                )
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // Send more than the cap with NO newline: the server must cut us off at the
        // cap and answer with an error rather than waiting for a newline.
        let payload = vec![b'x'; (max_request_bytes as usize) * 4];
        stream.write_all(&payload).await.expect("write oversize");
        stream.flush().await.expect("flush oversize");

        // The reply must be the `-32600 request too large` error, arriving well
        // within the read timeout (the cap, not the timeout, ends the read).
        let mut reader = BufReader::new(stream);
        let mut reply = String::new();
        let read = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut reply))
            .await
            .expect("oversize request must be answered promptly, not after the read timeout")
            .expect("read error line");
        assert!(read > 0, "server should send an error line, not just EOF");
        let v: serde_json::Value =
            serde_json::from_str(reply.trim_end()).expect("error reply is JSON");
        assert_eq!(v["id"], "", "request-too-large carries an empty id");
        assert_eq!(v["error"]["code"], -32600, "over-size request is -32600");
        assert_eq!(v["error"]["message"], "request too large");

        // After the error the connection is closed (EOF), no second response.
        let mut rest = String::new();
        let n = reader.read_line(&mut rest).await.expect("read after error");
        assert_eq!(n, 0, "server must close after the over-size error");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// FIX 2 — a normal-size request still round-trips under the same explicit
    /// limits path (a small cap that comfortably fits the request), proving the
    /// cap does not break the happy path.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn normal_request_still_works_under_explicit_cap() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);
        let (ctx, _note_id) = seeded_ctx(&dir).await;

        // 256-byte cap comfortably fits the ping request line below.
        let max_request_bytes: u64 = 256;
        let read_timeout = Duration::from_secs(30);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until_with_limits(
                    agents,
                    ctx,
                    read_timeout,
                    max_request_bytes,
                    async {
                        let _ = rx.await;
                    },
                )
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream
            .write_all(b"{\"id\":\"ok\",\"method\":\"ping\"}\n")
            .await
            .expect("write ping");
        stream.flush().await.expect("flush ping");

        let v = read_json_line(stream).await;
        assert_eq!(v["id"], "ok");
        assert_eq!(v["result"]["pong"], true);
        assert!(v.get("error").is_none());

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
