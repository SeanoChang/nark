use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::resolve;
use crate::serve;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    // Dual-mode: ask a live `nark serve` first (one round-trip via the shared
    // `try_vault_request` seam, which resolves the vault's socket itself). The
    // socket is an optimization — the seam returns `None` on ANY failure (absent
    // or stale socket, connect timeout, `unauthorized`, error response, malformed
    // JSON, any I/O error), and we then fall through to the always-correct
    // direct-open path below, unchanged.
    if let Some(result) =
        serve::client::try_vault_request(vault_dir, "nark/peek", serde_json::json!({ "id": id }))
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

    /// (b) With NO serve (socket absent), the direct path is taken — the seam
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    #[test]
    fn peek_no_serve_takes_direct_path() {
        let dir = temp_vault_dir();
        let id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        assert!(
            serve::client::try_vault_request(&dir, "nark/peek", serde_json::json!({ "id": id }))
                .is_none(),
            "with no serve, the seam must return None so the direct path is taken"
        );

        run(&dir, &id).expect("direct-open peek with no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// PARITY SWEEP (peek): the BYTES `run` would print on a socket HIT must be
    /// byte-identical to the bytes it prints on the direct path over the SAME
    /// seeded vault. Both go through `serde_json::to_string_pretty(..)` + a
    /// trailing newline (`println!`), so rendering each path's value the exact
    /// way `run` does and asserting equality proves socket-present vs
    /// socket-absent output is identical to the byte.
    #[cfg(target_os = "macos")]
    #[test]
    fn peek_socket_vs_direct_output_is_byte_identical() {
        let server = TestServer::start(current_uid_agent_map());

        let socket_value = serve::client::try_vault_request(
            server.dir(),
            "nark/peek",
            serde_json::json!({ "id": server.note_id() }),
        )
        .expect("socket hit");
        let direct_value = direct_peek_value(server.dir(), server.note_id());

        let socket_bytes = format!(
            "{}\n",
            serde_json::to_string_pretty(&socket_value).expect("render socket")
        );
        let direct_bytes = format!(
            "{}\n",
            serde_json::to_string_pretty(&direct_value).expect("render direct")
        );
        assert_eq!(
            socket_bytes, direct_bytes,
            "peek output must be byte-identical socket-present vs socket-absent"
        );
    }

    /// FALLBACK HARDENING (peek): a STALE socket — a bound listener that never
    /// accepts — must NOT make the read fail. The seam times out and returns
    /// `None`, and `run` produces the correct direct-path result with no hang.
    #[test]
    fn peek_stale_socket_falls_back_to_direct() {
        use std::os::unix::net::UnixListener;

        let dir = temp_vault_dir();
        let id = seed_vault(&dir);
        let socket = default_socket(&dir);
        std::fs::create_dir_all(socket.parent().unwrap()).expect("create run dir");
        let _listener = UnixListener::bind(&socket).expect("bind stale listener");

        let start = std::time::Instant::now();
        run(&dir, &id).expect("peek must fall back to direct over a stale socket");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "a stale socket must not hang the peek read"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
