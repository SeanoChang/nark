use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::{access, resolve};
use crate::serve;
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    // Dual-mode: ask a live `nark serve` first (one round-trip via the shared
    // `try_vault_request` seam). On a socket hit we pretty-print the server's
    // result and return; the serve read path is read-only and side-effect-free,
    // so unlike the direct path it does not bump access. The seam returns `None`
    // on ANY failure (absent/stale socket, connect timeout, `unauthorized`,
    // error response, malformed JSON, any I/O error), and we then fall through
    // to the always-correct direct-open path below, unchanged (including its
    // access bump).
    if let Some(result) =
        serve::client::try_vault_request(vault_dir, "nark/read", serde_json::json!({ "id": id }))
    {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let meta = resolve::get_meta(&conn, id)?;
    let refs = resolve::get_ref(&conn, &meta.note_id)?;

    let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

    let fm: serde_json::Value = serde_yaml::from_str(&fm_raw)?;

    let out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "frontmatter": fm,
        "body": body,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);

    // Bump access after successful read
    access::bump_access(&conn, &meta.note_id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::client::default_socket;
    use crate::serve::client::test_support::{
        TestServer, current_uid_agent_map, seed_vault, temp_vault_dir,
    };

    /// The direct-open path's JSON for `read` over a seeded vault — exactly the
    /// `out` object `run` builds before printing (and before its access bump).
    /// Used to assert socket-hit == direct-open parity at the value level.
    fn direct_read_value(vault_dir: &Path, id: &str) -> serde_json::Value {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let vault = Vault::new(vault_dir.to_path_buf());
        let meta = resolve::get_meta(&conn, id).expect("resolve meta");
        let refs = resolve::get_ref(&conn, &meta.note_id).expect("resolve ref");
        let fm_raw = vault
            .read_object("objects/fm", &refs.fm_hash, "yaml")
            .expect("read fm");
        let body = vault
            .read_object("objects/md", &refs.md_hash, "md")
            .expect("read body");
        let fm: serde_json::Value = serde_yaml::from_str(&fm_raw).expect("parse fm");
        serde_json::json!({
            "id": meta.note_id,
            "title": meta.title,
            "frontmatter": fm,
            "body": body,
        })
    }

    /// (a) With a live serve + a mapped uid, the dual-mode command path returns
    /// the SERVER's result, byte-identical (value-equal) to the direct-open path.
    #[cfg(target_os = "macos")]
    #[test]
    fn read_socket_hit_matches_direct_path() {
        let server = TestServer::start(current_uid_agent_map());

        let socket = default_socket(server.dir());
        let from_socket = serve::client::try_request(
            &socket,
            "nark/read",
            serde_json::json!({ "id": server.note_id() }),
        )
        .expect("authenticated nark/read should return Some(result)");

        let from_direct = direct_read_value(server.dir(), server.note_id());
        assert_eq!(
            from_socket, from_direct,
            "socket-hit read must match the direct-open path byte-for-byte"
        );

        run(server.dir(), server.note_id()).expect("dual-mode read over live serve");

        // The serve read path is side-effect-free: a socket HIT must NOT bump
        // access. This is also the decisive proof the handler took the socket
        // branch and not the fallback (the direct path bumps access on every
        // read, which would make `total_reads` 1).
        let conn = db::open_registry(server.dir()).expect("open registry");
        let s = crate::registry::stats::overview(&conn).expect("stats overview");
        assert_eq!(
            s.access.total_reads, 0,
            "a socket-hit read must not bump access (proves the socket path, not fallback)"
        );
    }

    /// (b) With NO serve (socket absent), the direct path is taken — the seam
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    #[test]
    fn read_no_serve_takes_direct_path() {
        let dir = temp_vault_dir();
        let id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        assert!(
            serve::client::try_vault_request(&dir, "nark/read", serde_json::json!({ "id": id }))
                .is_none(),
            "with no serve, the seam must return None so the direct path is taken"
        );

        run(&dir, &id).expect("direct-open read with no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// PARITY SWEEP (read): the BYTES `run` would print on a socket HIT must be
    /// byte-identical to the direct path's printed bytes over the same seeded
    /// vault. Both render via `serde_json::to_string_pretty(..)` + a trailing
    /// newline.
    #[cfg(target_os = "macos")]
    #[test]
    fn read_socket_vs_direct_output_is_byte_identical() {
        let server = TestServer::start(current_uid_agent_map());

        let socket_value = serve::client::try_vault_request(
            server.dir(),
            "nark/read",
            serde_json::json!({ "id": server.note_id() }),
        )
        .expect("socket hit");
        let direct_value = direct_read_value(server.dir(), server.note_id());

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
            "read output must be byte-identical socket-present vs socket-absent"
        );
    }

    /// FALLBACK HARDENING (read): a STALE socket — bound but never accepting —
    /// must NOT make the read fail. The seam times out, `run` falls back to the
    /// full direct path, and the decisive proof it took the DIRECT branch (not
    /// silently the socket) is that the direct path bumps access: `total_reads`
    /// becomes 1.
    #[test]
    fn read_stale_socket_falls_back_to_direct_and_bumps_access() {
        use std::os::unix::net::UnixListener;

        let dir = temp_vault_dir();
        let id = seed_vault(&dir);
        let socket = default_socket(&dir);
        std::fs::create_dir_all(socket.parent().unwrap()).expect("create run dir");
        let _listener = UnixListener::bind(&socket).expect("bind stale listener");

        let start = std::time::Instant::now();
        run(&dir, &id).expect("read must fall back to direct over a stale socket");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "a stale socket must not hang the read"
        );

        let conn = db::open_registry(&dir).expect("open registry");
        let s = crate::registry::stats::overview(&conn).expect("stats overview");
        assert_eq!(
            s.access.total_reads, 1,
            "fall-back must run the full direct path, which bumps access"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
