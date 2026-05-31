use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::stats;
use crate::serve;

pub fn run(vault_dir: &Path) -> Result<()> {
    // Dual-mode: ask a live `nark serve` first (one round-trip). The socket is
    // an optimization — `try_request` returns `None` on ANY failure (absent or
    // stale socket, connect timeout, `unauthorized`, error response, malformed
    // JSON, any I/O error), and we then fall through to the always-correct
    // direct-open path below, unchanged.
    let socket = serve::client::default_socket(vault_dir);
    if let Some(result) = serve::client::try_request(&socket, "nark/stats", serde_json::json!({})) {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let conn = db::open_registry(vault_dir)?;
    let s = stats::overview(&conn)?;

    let most_accessed = s
        .access
        .most_accessed
        .as_ref()
        .map(|m| serde_json::json!({ "title": m.title, "count": m.count }));

    let out = serde_json::json!({
        "total_notes": s.total_notes,
        "total_versions": s.total_versions,
        "by_domain": s.by_domain.iter().map(|f| {
            serde_json::json!({ "domain": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "by_kind": s.by_kind.iter().map(|f| {
            serde_json::json!({ "kind": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "recent": s.recent.iter().map(|n| {
            serde_json::json!({
                "id": n.note_id,
                "title": n.title,
                "domain": n.domain,
                "intent": n.intent,
                "kind": n.kind,
                "updated_at": n.updated_at,
            })
        }).collect::<Vec<_>>(),
        "access": {
            "total_reads": s.access.total_reads,
            "most_accessed": most_accessed,
            "never_read": s.access.never_read,
        },
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::client::default_socket;
    use crate::serve::client::test_support::{
        TestServer, current_uid_agent_map, seed_vault, temp_vault_dir,
    };

    /// The direct-open path's JSON for `stats` over a seeded vault — exactly the
    /// `out` object `run` builds before printing. Used to assert socket-hit ==
    /// direct-open parity at the value level.
    fn direct_stats_value(vault_dir: &Path) -> serde_json::Value {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let s = stats::overview(&conn).expect("stats overview");

        let most_accessed = s
            .access
            .most_accessed
            .as_ref()
            .map(|m| serde_json::json!({ "title": m.title, "count": m.count }));

        serde_json::json!({
            "total_notes": s.total_notes,
            "total_versions": s.total_versions,
            "by_domain": s.by_domain.iter().map(|f| {
                serde_json::json!({ "domain": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "by_kind": s.by_kind.iter().map(|f| {
                serde_json::json!({ "kind": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "recent": s.recent.iter().map(|n| {
                serde_json::json!({
                    "id": n.note_id,
                    "title": n.title,
                    "domain": n.domain,
                    "intent": n.intent,
                    "kind": n.kind,
                    "updated_at": n.updated_at,
                })
            }).collect::<Vec<_>>(),
            "access": {
                "total_reads": s.access.total_reads,
                "most_accessed": most_accessed,
                "never_read": s.access.never_read,
            },
        })
    }

    /// (a) With a live serve + a mapped uid, the dual-mode command path returns
    /// the SERVER's result, byte-identical (value-equal) to the direct-open path.
    #[cfg(target_os = "macos")]
    #[test]
    fn stats_socket_hit_matches_direct_path() {
        let server = TestServer::start(current_uid_agent_map());

        let socket = default_socket(server.dir());
        let from_socket = serve::client::try_request(&socket, "nark/stats", serde_json::json!({}))
            .expect("authenticated nark/stats should return Some(result)");

        let from_direct = direct_stats_value(server.dir());
        assert_eq!(
            from_socket, from_direct,
            "socket-hit stats must match the direct-open path byte-for-byte"
        );

        run(server.dir()).expect("dual-mode stats over live serve");
    }

    /// (b) With NO serve (socket absent), the direct path is taken — `try_request`
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    #[test]
    fn stats_no_serve_takes_direct_path() {
        let dir = temp_vault_dir();
        let _id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        assert!(
            serve::client::try_request(&socket, "nark/stats", serde_json::json!({})).is_none(),
            "with no serve, try_request must return None so the direct path is taken"
        );

        run(&dir).expect("direct-open stats with no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
