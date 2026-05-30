//! `nark serve` daemon — Phase 2 of the Ark comm protocol.
//!
//! Listens on a Unix domain socket and serves the vault to local agents.
//! This module currently holds the command skeleton; the listener loop and
//! peer-credential authentication land in later slices.

use std::path::Path;

/// Run the serve daemon.
///
/// `vault_dir` is the vault root (default `~/.ark`); `socket` is an optional
/// override for the Unix socket path. Currently a stub that logs and returns.
pub fn run(_vault_dir: &Path, _socket: Option<String>) -> anyhow::Result<()> {
    eprintln!("nark serve: starting (stub)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_stub_returns_ok() {
        let tmp = std::env::temp_dir();
        let result = run(&tmp, None);
        assert!(result.is_ok());
    }
}
