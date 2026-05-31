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
mod dpool;
mod embed_permit;
mod listener;
mod methods_read;
mod peercred;
mod rpc;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

pub use authz::AgentMap;
// `guard_preexisting_socket_path` is the pre-bind lstat guard; it is invoked by
// `BoundListener::bind` before unlink+bind, so the daemon path does not call it
// directly. Re-exported as part of the serve module's surface (and used by tests).
#[allow(unused_imports)]
pub use authz::guard_preexisting_socket_path;
pub use listener::{BoundListener, resolve_socket_path};

use crate::config;

/// Run the serve daemon.
///
/// `vault_dir` is the vault root (default `~/.ark`); `socket` is an optional
/// override for the Unix socket path. Builds its own tokio runtime so the
/// caller (`main`) stays synchronous, binds the socket, and serves until
/// ctrl-c.
pub fn run(vault_dir: &Path, socket: Option<String>) -> Result<()> {
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
        bound
            .serve_authenticated_until(agents, ctx, async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("nark serve: shutting down");
            })
            .await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_socket_path_default() {
        // The default socket lives in a dedicated `run/` subdir so the daemon
        // never touches the vault root's mode (see `ensure_socket_dir`).
        let path = resolve_socket_path(Path::new("/tmp/vault"), None);
        assert_eq!(path, Path::new("/tmp/vault/run/nark.sock"));
    }
}
