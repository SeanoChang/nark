//! Peer authorization for the `nark serve` daemon.
//!
//! Two complementary guards:
//!
//! 1. **uid -> agent resolution** ([`AgentMap`]). Peer credentials extracted
//!    from a connection (see [`super::peercred`]) yield a numeric uid; this map
//!    turns a known uid into a logical agent name. Unknown uids resolve to
//!    `None` and are rejected by the caller (fail closed).
//!
//! 2. **pre-bind socket-path guard** ([`guard_preexisting_socket_path`]). The
//!    `0700` socket directory is the primary containment — only the owning user
//!    can traverse into it. As defence in depth, *before* unlinking and binding,
//!    the daemon lstats the socket path: it refuses to serve if a SYMLINK exists
//!    there (symlink-swap), or if a real file/socket exists there owned by some
//!    other uid (socket-planting). This must run before bind, because after we
//!    bind our own socket the owner is always our own uid and the check is moot.

use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use anyhow::{Context, Result};

/// Map of peer uid -> logical agent name.
///
/// Built from the optional `[serve.agents]` config table. In dev (no config /
/// no table) the map is empty and every uid resolves to `None`.
///
/// `resolve`/`new`/`len`/`is_empty` are exercised by tests today; the daemon
/// accept path wires `resolve` in a later slice, so they carry the same
/// "tests exercise it today" allowance as the peercred primitives.
#[derive(Debug, Clone, Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct AgentMap(HashMap<u32, String>);

#[cfg_attr(not(test), allow(dead_code))]
impl AgentMap {
    /// Construct directly from a uid -> name map (primarily for tests and
    /// programmatic use).
    pub fn new(map: HashMap<u32, String>) -> Self {
        Self(map)
    }

    /// Build from the `[serve.agents]` config table, whose keys are uid strings
    /// (TOML table keys are always strings). Entries whose key does not parse
    /// as a `u32` uid are skipped rather than failing the whole daemon, so a
    /// single typo in the config does not lock everyone out.
    pub fn from_config(agents: &HashMap<String, String>) -> Self {
        let map = agents
            .iter()
            .filter_map(|(uid_str, name)| {
                uid_str.parse::<u32>().ok().map(|uid| (uid, name.clone()))
            })
            .collect();
        Self(map)
    }

    /// Resolve a peer uid to its agent name. `None` means "unknown uid" and the
    /// caller MUST reject the connection (fail closed).
    pub fn resolve(&self, uid: u32) -> Option<&str> {
        self.0.get(&uid).map(String::as_str)
    }

    /// Number of configured agents.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the map has no configured agents (the dev default).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The serving process's own (real) uid.
fn process_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

/// Pre-bind guard against a planted socket path.
///
/// Call this *before* unlinking and binding the socket. It `lstat`s `path`
/// (`std::fs::symlink_metadata`, which does NOT follow symlinks) and:
///
/// * `Ok(())` if nothing exists at `path` — the normal cold-start case.
/// * `Err` if `path` is a SYMLINK — an attacker may have pointed it at a socket
///   they control (symlink-swap), so we refuse rather than unlink+bind through
///   it.
/// * `Err` if `path` is a real file/socket owned by some *other* uid — another
///   user planted it at the expected path (socket-planting); refuse.
/// * `Ok(())` if `path` exists, is not a symlink, and is owned by our own uid —
///   this is our own stale socket from a previous run, which the caller then
///   unlinks and rebinds.
///
/// This is defence in depth: the `0700` socket directory is the primary
/// containment (only the owning user can traverse into it).
pub fn guard_preexisting_socket_path(path: &Path) -> Result<()> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        // Nothing there yet — the normal cold-start path.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(e).with_context(|| format!("lstat socket path {}", path.display()));
        }
    };

    if meta.file_type().is_symlink() {
        anyhow::bail!(
            "refusing to serve: socket path {} is a symlink \
             (possible symlink-swap attack); refusing to bind through it",
            path.display()
        );
    }

    let owner = meta.uid();
    let me = process_uid();
    if owner != me {
        anyhow::bail!(
            "refusing to serve: existing path {} is owned by uid {owner}, \
             not the serving uid {me} (possible socket-planting attack)",
            path.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_uid_returns_agent() {
        let mut map = HashMap::new();
        map.insert(501, "ark-agent".to_string());
        let agents = AgentMap::new(map);

        assert_eq!(agents.resolve(501), Some("ark-agent"));
    }

    #[test]
    fn resolve_unknown_uid_returns_none() {
        let mut map = HashMap::new();
        map.insert(501, "ark-agent".to_string());
        let agents = AgentMap::new(map);

        assert_eq!(agents.resolve(999), None);
    }

    #[test]
    fn from_config_parses_uid_string_keys() {
        let mut table = HashMap::new();
        table.insert("501".to_string(), "ark-agent".to_string());
        table.insert("not-a-uid".to_string(), "ignored".to_string());
        let agents = AgentMap::from_config(&table);

        // Valid uid key parsed; bogus key skipped (not a lockout).
        assert_eq!(agents.resolve(501), Some("ark-agent"));
        assert_eq!(agents.len(), 1);
    }

    #[test]
    fn empty_map_is_dev_default() {
        let agents = AgentMap::default();
        assert!(agents.is_empty());
        assert_eq!(agents.resolve(0), None);
    }

    fn authz_temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-authz-{tag}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn pre_bind_guard_allows_absent_path() {
        // Cold start: nothing exists at the socket path yet.
        let dir = authz_temp_dir("absent");
        let socket = dir.join("nark.sock");
        guard_preexisting_socket_path(&socket)
            .expect("absent socket path is the normal cold-start case");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pre_bind_guard_allows_self_owned_stale_socket() {
        // A regular file we created stands in for our own stale socket: same
        // uid, not a symlink -> the caller is cleared to unlink and rebind.
        let dir = authz_temp_dir("stale");
        let socket = dir.join("nark.sock");
        std::fs::write(&socket, b"").expect("create stale file");

        guard_preexisting_socket_path(&socket)
            .expect("self-owned stale socket should be allowed (caller unlinks it)");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pre_bind_guard_refuses_symlink() {
        // A symlink planted at the socket path is refused regardless of where it
        // points (symlink-swap). lstat sees the link itself, not its target.
        let dir = authz_temp_dir("symlink");
        let socket = dir.join("nark.sock");
        let target = dir.join("elsewhere.sock");
        std::fs::write(&target, b"").expect("create link target");
        std::os::unix::fs::symlink(&target, &socket).expect("create symlink at socket path");

        let err = guard_preexisting_socket_path(&socket)
            .expect_err("a symlink at the socket path must be refused");
        assert!(
            err.to_string().contains("symlink"),
            "error should name the symlink-swap rejection, got: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pre_bind_guard_uses_lstat_not_target_owner() {
        // The guard must NOT follow the symlink to inspect its target's owner —
        // it rejects on the symlink itself. (A self-owned target would falsely
        // pass if we followed it.) This pins the lstat (vs metadata) behaviour.
        let dir = authz_temp_dir("lstat");
        let socket = dir.join("nark.sock");
        let target = dir.join("self-owned.sock");
        std::fs::write(&target, b"").expect("create self-owned target");
        std::os::unix::fs::symlink(&target, &socket).expect("create symlink");

        // Sanity: the target is owned by us, so a target-following guard would
        // wrongly allow this; lstat-based guard must still refuse.
        assert_eq!(
            std::fs::metadata(&target).expect("stat target").uid(),
            nix::unistd::getuid().as_raw()
        );
        guard_preexisting_socket_path(&socket)
            .expect_err("lstat-based guard must refuse the symlink even with a self-owned target");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
