use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::resolve;
use crate::serve;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    // Dual-mode: ask a live `nark serve` first (one round-trip). The socket is
    // an optimization — `try_request` returns `None` on ANY failure (absent or
    // stale socket, connect timeout, `unauthorized`, error response, malformed
    // JSON, any I/O error), and we then fall through to the always-correct
    // direct-open path below, unchanged.
    let socket = serve::client::default_socket(vault_dir);
    if let Some(result) =
        serve::client::try_request(&socket, "nark/peek", serde_json::json!({ "id": id }))
    {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let conn = db::open_registry(vault_dir)?;
    let meta = resolve::get_meta(&conn, id)?;

    let out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "domain": meta.domain,
        "intent": meta.intent,
        "kind": meta.kind,
        "status": meta.status,
        "tags": meta.tags,
        "updated_at": meta.updated_at,
        "links_in": meta.links_in,
        "links_out": meta.links_out,
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

    /// The direct-open path's JSON for `peek` over a seeded vault — exactly the
    /// `out` object `run` builds before printing. Used to assert socket-hit ==
    /// direct-open parity at the value level (both pretty-print through
    /// `serde_json::to_string_pretty`, so equal values mean byte-identical output).
    fn direct_peek_value(vault_dir: &Path, id: &str) -> serde_json::Value {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let meta = resolve::get_meta(&conn, id).expect("resolve meta");
        serde_json::json!({
            "id": meta.note_id,
            "title": meta.title,
            "domain": meta.domain,
            "intent": meta.intent,
            "kind": meta.kind,
            "status": meta.status,
            "tags": meta.tags,
            "updated_at": meta.updated_at,
            "links_in": meta.links_in,
            "links_out": meta.links_out,
        })
    }

    /// (a) With a live serve + a mapped uid, the dual-mode command path returns
    /// the SERVER's result, and that result is byte-identical (value-equal) to
    /// the direct-open path over the same seeded vault.
    #[cfg(target_os = "macos")]
    #[test]
    fn peek_socket_hit_matches_direct_path() {
        let server = TestServer::start(current_uid_agent_map());

        // Socket HIT: the handler resolves the server's socket from its vault dir.
        let socket = default_socket(server.dir());
        let from_socket = serve::client::try_request(
            &socket,
            "nark/peek",
            serde_json::json!({ "id": server.note_id() }),
        )
        .expect("authenticated nark/peek should return Some(result)");

        let from_direct = direct_peek_value(server.dir(), server.note_id());
        assert_eq!(
            from_socket, from_direct,
            "socket-hit peek must match the direct-open path byte-for-byte"
        );

        // The wired handler hits the socket and succeeds end-to-end.
        run(server.dir(), server.note_id()).expect("dual-mode peek over live serve");
    }

    /// (b) With NO serve (socket absent), the direct path is taken — `try_request`
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    #[test]
    fn peek_no_serve_takes_direct_path() {
        let dir = temp_vault_dir();
        let id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        assert!(
            serve::client::try_request(&socket, "nark/peek", serde_json::json!({ "id": id }))
                .is_none(),
            "with no serve, try_request must return None so the direct path is taken"
        );

        run(&dir, &id).expect("direct-open peek with no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
