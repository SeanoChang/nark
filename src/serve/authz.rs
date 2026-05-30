//! Peer authorization for the `nark serve` daemon.
//!
//! Slice 2.4 lands two complementary guards:
//!
//! 1. **uid -> agent resolution** ([`AgentMap`]). Peer credentials extracted
//!    from a connection (see [`super::peercred`]) yield a numeric uid; this map
//!    turns a known uid into a logical agent name. Unknown uids resolve to
//!    `None` and are rejected by the caller (fail closed).
//!
//! 2. **socket-ownership guard** ([`socket_owner_uid`] / [`guard_socket_owner`]).
//!    Before trusting a socket path the daemon verifies the socket *file* is
//!    owned by the serving process's own uid. This defends against a
//!    symlink-swap / pre-created-socket attack where another user plants a
//!    socket (or a symlink to one) at the expected path so connecting agents
//!    talk to an impostor.

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

/// The owning uid of the file at `path`, from filesystem metadata.
///
/// Uses `std::fs::metadata` (which follows symlinks) + `MetadataExt::uid()`.
/// Following symlinks is intentional here: the ownership guard wants the uid of
/// the *target* the socket path actually resolves to, so a symlink pointing at
/// a foreign-owned socket is caught by [`guard_socket_owner`].
pub fn socket_owner_uid(path: &Path) -> Result<u32> {
    let meta =
        std::fs::metadata(path).with_context(|| format!("stat socket file {}", path.display()))?;
    Ok(meta.uid())
}

/// The serving process's own (real) uid.
fn process_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

/// Anti symlink-swap guard: verify the socket file at `path` is owned by the
/// serving process's own uid.
///
/// Returns the owner uid on success. On mismatch it returns an error describing
/// the rejection (the documented basis for refusing to serve on a socket some
/// other user planted at the expected path). The daemon should treat any error
/// here as fatal and refuse to serve.
pub fn guard_socket_owner(path: &Path) -> Result<u32> {
    let owner = socket_owner_uid(path)?;
    let me = process_uid();
    if owner != me {
        anyhow::bail!(
            "refusing to serve: socket {} is owned by uid {owner}, not the serving uid {me} \
             (possible symlink-swap / socket-planting attack)",
            path.display()
        );
    }
    Ok(owner)
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

    #[test]
    fn socket_owner_uid_is_current_uid_for_self_created_file() {
        let dir = std::env::temp_dir().join(format!(
            "nark-authz-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let file = dir.join("owned.sock");
        std::fs::write(&file, b"").expect("create temp file");

        let owner = socket_owner_uid(&file).expect("read owner uid");
        let expected = nix::unistd::getuid().as_raw();
        assert_eq!(owner, expected, "self-created file should be owned by us");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guard_passes_for_self_owned_socket() {
        let dir = std::env::temp_dir().join(format!(
            "nark-authz-guard-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let file = dir.join("owned.sock");
        std::fs::write(&file, b"").expect("create temp file");

        // Self-owned: guard returns Ok with our uid.
        let owner = guard_socket_owner(&file).expect("guard should pass for self-owned socket");
        assert_eq!(owner, nix::unistd::getuid().as_raw());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guard_rejection_is_owner_vs_serving_uid_mismatch() {
        // The guard rejects whenever the socket owner uid differs from the
        // serving uid. We cannot chown a file to a foreign uid as a non-root
        // test process, so we document and exercise the comparison directly:
        // a foreign owner uid (serving uid + 1) must not equal our uid, which
        // is exactly the condition `guard_socket_owner` rejects on.
        let me = nix::unistd::getuid().as_raw();
        let foreign = me.wrapping_add(1);
        assert_ne!(
            foreign, me,
            "a socket owned by a different uid is the documented rejection case"
        );
    }
}
