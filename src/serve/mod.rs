//! `nark serve` daemon — Phase 2 of the Ark comm protocol.
//!
//! Listens on a Unix domain socket and serves the vault to local agents.
//! Slice 2.2 lands the listener loop (bind / accept / ping->pong / clean
//! shutdown); slice 2.3 extracts peer credentials; slice 2.4 adds peer
//! authorization (uid -> agent map + socket-ownership guard); slice 2.5
//! assembles the authenticated ping path end-to-end (peer uid -> agent ->
//! ping/pong, with unknown uids rejected with `unauthorized`).

mod authz;
mod listener;
mod peercred;

use std::path::Path;

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
        // `bind` runs the pre-bind lstat guard (symlink-swap / socket-planting)
        // before it unlinks any stale socket and binds — see `BoundListener::bind`.
        let bound = BoundListener::bind(&socket_path)?;
        eprintln!("nark serve: listening on {}", bound.path().display());
        bound
            .serve_authenticated_until(agents, async {
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
