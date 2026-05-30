//! `nark serve` daemon — Phase 2 of the Ark comm protocol.
//!
//! Listens on a Unix domain socket and serves the vault to local agents.
//! Slice 2.2 lands the listener loop (bind / accept / ping->pong / clean
//! shutdown); peer-credential authentication arrives in a later slice.

mod listener;

use std::path::Path;

use anyhow::Result;

pub use listener::{BoundListener, resolve_socket_path};

/// Run the serve daemon.
///
/// `vault_dir` is the vault root (default `~/.ark`); `socket` is an optional
/// override for the Unix socket path. Builds its own tokio runtime so the
/// caller (`main`) stays synchronous, binds the socket, and serves until
/// ctrl-c.
pub fn run(vault_dir: &Path, socket: Option<String>) -> Result<()> {
    let socket_path = resolve_socket_path(vault_dir, socket);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let bound = BoundListener::bind(&socket_path)?;
        eprintln!("nark serve: listening on {}", bound.path().display());
        bound
            .serve_until(async {
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
        let path = resolve_socket_path(Path::new("/tmp/vault"), None);
        assert_eq!(path, Path::new("/tmp/vault/nark.sock"));
    }
}
