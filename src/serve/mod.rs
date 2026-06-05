//! `nark serve` daemon — the Ark comm protocol's read path.
//!
//! Listens on a Unix domain socket and serves the vault to local agents.
//! Slice 2.2 lands the listener loop (bind / accept / ping->pong / clean
//! shutdown); slice 2.3 extracts peer credentials; slice 2.4 adds peer
//! authorization (uid -> agent map + socket-ownership guard); slice 2.5
//! assembles the authenticated ping path end-to-end (peer uid -> agent ->
//! ping/pong, with unknown uids rejected with `unauthorized`).
//!
//! Phase 3 added the JSON-RPC read methods; Phase 3.5 reshaped the read path
//! into a single pooling story with an embedding split:
//!
//! * the `dpool` module is the one connection pool — a `deadpool`-managed,
//!   strictly **read-only** SQLite pool against `<vault>/registry.db`. Every
//!   read method (`peek` / `read` / `stats` / `search` / `orient`) checks a
//!   connection out of it and runs its blocking SQLite on a managed thread via
//!   `conn.interact(...).await`; `get()` backpressures when every connection is
//!   busy. There is no second pool — slice 3.5.5 retired the Phase-3 hand-rolled
//!   `ReadPool` (the `Mutex<Vec<Connection>>` + `Condvar` consumed via
//!   `spawn_blocking`).
//! * the `embed_permit` module is the embedding split (the 2B payoff): the ONNX
//!   query embedding `search` needs runs under a bounded `tokio::sync::Semaphore`
//!   permit and **outside** any DB checkout, so a burst of `search` load can
//!   never hold a connection across inference and stall the cheap reads (no
//!   head-of-line blocking).

mod authz;
// Phase 4: the blocking serve client the dual-mode CLI handlers will use to ask
// a live `nark serve` (try-or-fallback). Public so `cli::*` can call
// `serve::client::try_request`. Slice 4.1 lands the client; wiring is later.
pub mod client;
mod dpool;
mod embed_permit;
mod listener;
mod methods_read;
// Phase 6: the WRITE method implementations (`nark/write`, and later
// `nark/link` / `nark/delete`) that submit their mutations to the single
// serializing writer queue. Slice 6.2 lands `nark/write`.
mod methods_write;
mod peercred;
mod rpc;
// Phase 6: the single serializing, off-reactor writer queue the write methods
// (`nark/write` / `nark/link` / `nark/delete`) run their mutations on. Slice 6.1
// landed the queue primitive; slice 6.2 wires `nark/write` onto it.
mod writer;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

pub use authz::AgentMap;
// `guard_preexisting_socket_path` is the pre-bind lstat guard; it is invoked by
// `BoundListener::bind` before unlink+bind, so the daemon path does not call it
// directly. Re-exported as part of the serve module's surface (and used by tests).
#[allow(unused_imports)]
pub use authz::guard_preexisting_socket_path;
pub use listener::{BoundListener, resolve_socket_path};

use crate::config;
use crate::db::wlock::WriteLock;

/// Run the serve daemon until ctrl-c.
///
/// `vault_dir` is the vault root (default `~/.ark`); `socket` is an optional
/// override for the Unix socket path. Delegates to [`run_until`] with the ctrl-c
/// shutdown signal — see [`run_until`] for the write-lock + bind + serve-loop
/// behavior.
pub fn run(vault_dir: &Path, socket: Option<String>) -> Result<()> {
    run_until(vault_dir, socket, async {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("nark serve: shutting down");
    })
}

/// Run the serve daemon until `shutdown` resolves.
///
/// The serve daemon is the registry's single live writer-in-waiting (writes land
/// in Phase 6): to enforce **no dual writers**, it acquires the advisory write
/// lock ([`WriteLock`]) at startup, *before* the serve loop, and holds it for the
/// daemon's entire lifetime. While serve runs, a direct write CLI's
/// `db::open_registry_guarded` finds the lock held and refuses with the plain
/// "registry is write-locked by another process" error; reads (the read-only
/// deadpool and `db::open_registry`) never touch the lock and are unaffected.
///
/// Order:
/// 1. Acquire the write lock. If another writer already holds it,
///    [`WriteLock::try_acquire`] returns `Ok(None)` and startup fails with a
///    clear "another nark is already writing" error — serve never binds.
/// 2. Build the runtime, open the read-only [`rpc::Ctx`], bind the socket, and
///    serve until `shutdown` resolves.
///
/// The lock guard is bound in this synchronous frame and dropped only when `run`
/// returns (graceful shutdown, bind failure, or unwind), so it outlives the whole
/// `block_on`. Phase 6 will move actual writes behind this same lock; for now
/// serve only *owns* it so direct CLI writes refuse while it runs.
fn run_until<F>(vault_dir: &Path, socket: Option<String>, shutdown: F) -> Result<()>
where
    F: std::future::Future<Output = ()>,
{
    // Acquire the write lock before anything else. Held in `_write_lock` for the
    // lifetime of this call (RAII): released on return/unwind, and the OS also
    // releases the flock if the process crashes (no stale lock to reap).
    let _write_lock = match WriteLock::try_acquire(vault_dir)? {
        Some(lock) => lock,
        None => {
            return Err(anyhow!(
                "registry is write-locked by another process: another nark is already writing (serve or a write command); refusing to start a second writer"
            ));
        }
    };

    let socket_path = resolve_socket_path(vault_dir, socket);
    let cfg = config::load(vault_dir)?;
    let agents = AgentMap::from_config(&cfg.serve.agents);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        // Read-only registry pools + vault dir the READ methods dispatch against.
        // The writer (the CLI's `db::open_registry`) owns creation/migration/
        // seeding and has WAL enabled; this only reads. `Ctx::open` is async
        // because the deadpool pool is built on the tokio runtime.
        let ctx = Arc::new(rpc::Ctx::open(vault_dir).await?);
        // `bind` runs the pre-bind lstat guard (symlink-swap / socket-planting)
        // before it unlinks any stale socket and binds — see `BoundListener::bind`.
        let bound = BoundListener::bind(&socket_path)?;
        eprintln!("nark serve: listening on {}", bound.path().display());
        bound.serve_authenticated_until(agents, ctx, shutdown).await
    })
    // `_write_lock` drops here, releasing the advisory write lock.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use std::time::{Duration, Instant};

    #[test]
    fn resolve_socket_path_default() {
        // The default socket lives in a dedicated `run/` subdir so the daemon
        // never touches the vault root's mode (see `ensure_socket_dir`).
        let path = resolve_socket_path(Path::new("/tmp/vault"), None);
        assert_eq!(path, Path::new("/tmp/vault/run/nark.sock"));
    }

    /// Create a fresh, unique temp vault directory with a migrated+seeded
    /// `registry.db` (the writer owns creation; serve only reads + owns the
    /// lock). The lock test below has `serve` bind `<dir>/run/nark.sock`, so the
    /// dir must stay short enough to fit `sun_path` (104 bytes on macOS) under
    /// the long real `$TMPDIR` — hence the shared short-path helper rather than
    /// the usual long `nark-<module>-test-...` name. See
    /// `client::test_support::short_socket_dir`.
    fn seeded_vault() -> std::path::PathBuf {
        let dir = crate::serve::client::test_support::short_socket_dir();
        std::fs::create_dir_all(&dir).expect("create temp vault");
        // Create + migrate + seed the registry via the plain (unlocked) writer
        // open, then drop it so no lock is held going into the test.
        let conn = db::open_registry(&dir).expect("seed registry");
        drop(conn);
        dir
    }

    /// End-to-end no-dual-writer: a running `nark serve` owns the advisory write
    /// lock for its whole lifetime, so a direct write CLI (`open_registry_guarded`)
    /// is refused while serve runs, reads still succeed, and once serve shuts down
    /// the guarded writer open succeeds again.
    ///
    /// `run_until` builds its own blocking runtime, so serve runs on a dedicated
    /// std thread and a `oneshot` drives its shutdown from the test thread.
    #[test]
    fn serve_holds_write_lock_so_guarded_writes_refuse_then_succeed_after_shutdown() {
        let dir = seeded_vault();

        // Drive serve's shutdown from the test thread. The receiver is awaited
        // inside serve's runtime; the sender fires from here. `send` is sync and
        // safe across threads.
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Serve binds the socket only AFTER it has acquired the write lock, so the
        // socket file appearing is a reliable "serve owns the lock and is up"
        // signal. The test must NOT poll `open_registry_guarded` to detect this:
        // that would itself take the lock and race serve's own startup
        // acquisition (whoever wins, the loser is refused — and serve does not
        // retry). Waiting on the socket lets serve win the lock uncontended.
        let socket_path = dir.join("run").join("nark.sock");
        let serve_socket = socket_path.to_string_lossy().into_owned();
        let serve_dir = dir.clone();
        let serve = std::thread::spawn(move || {
            run_until(&serve_dir, Some(serve_socket), async {
                let _ = shutdown_rx.await;
            })
        });

        // Wait for serve to bind (=> it holds the lock). Bounded so a regression
        // fails the test instead of hanging.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !socket_path.exists() {
            assert!(
                Instant::now() < deadline,
                "serve never bound the socket (so never acquired the write lock) within the budget"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        // Serve now holds the write lock: a direct write CLI's guarded open is
        // refused with the plain honest error and opens no writer connection.
        let err = db::open_registry_guarded(&dir)
            .expect_err("a guarded write must be refused while serve holds the lock");
        assert_eq!(
            err.to_string(),
            "registry is write-locked by another process",
            "the conflict must be the plain honest error (no serve-specific text)"
        );
        assert!(
            !err.to_string().contains("serve"),
            "the safety-net refusal must not mention serve"
        );

        // While serve holds the write lock, a READ (plain open_registry / the
        // RO path) must STILL succeed — reads are never gated by the write lock.
        {
            let read_conn =
                db::open_registry(&dir).expect("a read must succeed while serve holds the lock");
            let agent_count: i64 = read_conn
                .query_row(
                    "SELECT COUNT(*) FROM agents WHERE agent_id = ?1",
                    rusqlite::params![db::DEFAULT_AGENT_ID],
                    |row| row.get(0),
                )
                .expect("read query through plain open");
            assert_eq!(agent_count, 1, "read sees the seeded default agent");
            drop(read_conn);
        }

        // Shut serve down and wait for the thread to finish: the lock guard drops
        // when `run_until` returns, releasing the advisory write lock.
        shutdown_tx.send(()).expect("send shutdown");
        serve
            .join()
            .expect("serve thread join")
            .expect("serve run_until returns Ok on clean shutdown");

        // After serve has fully shut down, a guarded writer open succeeds again
        // (the lock was released on serve's return).
        let after = db::open_registry_guarded(&dir)
            .expect("guarded write succeeds again once serve has released the lock");
        drop(after);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Serve startup itself refuses when another writer already holds the lock:
    /// `run_until` must fail fast (before binding) with a clear error and never
    /// take a second copy of the lock.
    #[test]
    fn serve_startup_refuses_when_write_lock_already_held() {
        let dir = seeded_vault();

        // Simulate a direct write CLI (or another serve) already holding the lock.
        let held = db::open_registry_guarded(&dir).expect("acquire the write lock first");

        // Serve must refuse to start: try_acquire returns Ok(None), run_until
        // turns that into a clear startup error — and never binds the socket.
        let err = run_until(&dir, None, async {})
            .expect_err("serve must refuse to start while the write lock is held");
        let msg = err.to_string();
        assert!(
            msg.contains("write-locked"),
            "startup refusal should name the write-lock conflict, got: {msg}"
        );
        assert!(
            msg.contains("writing"),
            "startup refusal should explain another writer is active, got: {msg}"
        );

        // The conflict did not bind the socket (startup bailed before the loop).
        assert!(
            !dir.join("run").join("nark.sock").exists(),
            "serve must not bind when it cannot acquire the write lock"
        );

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
