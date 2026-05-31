//! READ method implementations for the `nark serve` daemon (Phase 3, slice 3.3).
//!
//! These functions build the **same** `serde_json::Value` that the matching CLI
//! handlers print, by calling the **same** `registry::*` functions — see
//! `cli/{peek,read,stats}.rs`. They are the daemon-side mirror of those
//! commands: same shape, same fields, no behavioural change to the CLI.
//!
//! Key differences from the CLI handlers, both forced by the read-only serve
//! design (slice 3.1's [`ReadPool`](super::readpool::ReadPool)):
//!
//! * registry access goes through `ReadPool::with_conn` (a borrowed read-only
//!   [`rusqlite::Connection`]) instead of a fresh writable `db::open_registry`;
//! * [`read`] does **not** call `access::bump_access` — that is a write, and the
//!   pool's connection is read-only. The serve read path is side-effect-free; if
//!   access tracking over the socket is wanted it belongs to a later write-path
//!   slice, not here.
//!
//! Each function returns the bare result `Value`; the router ([`super::rpc`])
//! wraps it in an [`RPCResponse`](crate::wire::RPCResponse) and maps any `Err`
//! (unknown / ambiguous id, missing object) to a clean error response rather
//! than letting it panic.

use std::path::Path;

use anyhow::Result;
use serde_json::{Value, json};

use super::readpool::ReadPool;
use crate::registry::{resolve, stats};
use crate::vault::fs::Vault;

/// `nark/peek`: resolve `id` to its head metadata, mirroring `cli::peek`.
///
/// Returns the same object `cli/peek.rs` prints: `id`, `title`, `domain`,
/// `intent`, `kind`, `status`, `tags`, `updated_at`, `links_in`, `links_out`.
/// An unknown or ambiguous id surfaces as the `Err` from [`resolve::get_meta`],
/// which the router turns into an error response.
pub fn peek(pool: &ReadPool, id: &str) -> Result<Value> {
    pool.with_conn(|conn| {
        let meta = resolve::get_meta(conn, id)?;
        Ok(json!({
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
        }))
    })
}

/// `nark/read`: resolve `id`, then read the head version's frontmatter and body
/// from the CAS, mirroring `cli::read`.
///
/// Returns the same object `cli/read.rs` prints: `id`, `title`, `frontmatter`
/// (the YAML frontmatter parsed to JSON), and `body` (the markdown body). Unlike
/// the CLI handler this does **not** bump access — the serve read path is
/// read-only and side-effect-free (see the module docs). A missing note or a
/// missing CAS object surfaces as an `Err` for the router to map.
pub fn read(pool: &ReadPool, vault_dir: &Path, id: &str) -> Result<Value> {
    let vault = Vault::new(vault_dir.to_path_buf());
    pool.with_conn(|conn| {
        let meta = resolve::get_meta(conn, id)?;
        let refs = resolve::get_ref(conn, &meta.note_id)?;

        let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

        let fm: Value = serde_yaml::from_str(&fm_raw)?;

        Ok(json!({
            "id": meta.note_id,
            "title": meta.title,
            "frontmatter": fm,
            "body": body,
        }))
    })
}

/// `nark/stats`: vault statistics overview, mirroring `cli::stats`.
///
/// Returns the same object `cli/stats.rs` prints: `total_notes`,
/// `total_versions`, `by_domain`/`by_kind` facet lists, the `recent` list, and
/// the nested `access` block (`total_reads`, `most_accessed`, `never_read`).
pub fn stats(pool: &ReadPool) -> Result<Value> {
    pool.with_conn(|conn| {
        let s = stats::overview(conn)?;

        let most_accessed = s
            .access
            .most_accessed
            .as_ref()
            .map(|m| json!({ "title": m.title, "count": m.count }));

        Ok(json!({
            "total_notes": s.total_notes,
            "total_versions": s.total_versions,
            "by_domain": s.by_domain.iter().map(|f| {
                json!({ "domain": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "by_kind": s.by_kind.iter().map(|f| {
                json!({ "kind": f.label, "count": f.count })
            }).collect::<Vec<_>>(),
            "recent": s.recent.iter().map(|n| {
                json!({
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
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::write::commit_version;

    /// A note document with a recognizable title/body and `namespace = ark`
    /// (set by `commit_version`), so it shows up in `current_notes` and stats.
    const NOTE: &str = "---\n\
title: Test Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - alpha\n\
  - beta\n\
---\n\
This is the body of the test note.\n";

    /// Build a temp vault: the writer (`db::open_registry`) creates/migrates/
    /// seeds the registry and enables WAL, then a single note is ingested into
    /// the CAS and committed. The writer connection is dropped so the read pool
    /// opens the same db read-only. Returns `(vault_dir, note_id)`.
    fn seeded_vault_with_note() -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!(
            "nark-methods-read-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");

        let conn = crate::db::open_registry(&dir).expect("open writer registry");
        let vault = Vault::new(dir.clone());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit note version");
        let note_id = result.note_id.clone();
        drop(conn);

        (dir, note_id)
    }

    fn open_pool(dir: &Path) -> ReadPool {
        ReadPool::open_with_size(dir, 2).expect("open read pool")
    }

    #[test]
    fn peek_returns_same_key_fields_as_cli() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = peek(&pool, &note_id).expect("peek should succeed");
        assert_eq!(v["id"], note_id);
        assert_eq!(v["title"], "Test Note");
        assert_eq!(v["domain"], "engineering");
        assert_eq!(v["intent"], "reference");
        assert_eq!(v["kind"], "note");
        assert_eq!(v["status"], "active");
        assert_eq!(v["tags"], json!(["alpha", "beta"]));
        assert!(v["updated_at"].is_string());
        assert_eq!(v["links_in"], 0);
        assert_eq!(v["links_out"], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_returns_body_and_frontmatter() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = read(&pool, &dir, &note_id).expect("read should succeed");
        assert_eq!(v["id"], note_id);
        assert_eq!(v["title"], "Test Note");
        assert_eq!(
            v["body"], "This is the body of the test note.",
            "read should return the markdown body from the CAS"
        );
        // Frontmatter round-trips from YAML into a JSON object.
        assert_eq!(v["frontmatter"]["title"], "Test Note");
        assert_eq!(v["frontmatter"]["domain"], "engineering");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_returns_counts() {
        let (dir, note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        let v = stats(&pool).expect("stats should succeed");
        assert_eq!(v["total_notes"], 1, "the one ingested note is counted");
        assert_eq!(v["total_versions"], 1, "one version was committed");
        // The recent list surfaces the note we just wrote.
        assert_eq!(v["recent"][0]["id"], note_id);
        assert_eq!(v["recent"][0]["title"], "Test Note");
        // Never-read since the serve read path does not bump access.
        assert_eq!(v["access"]["never_read"], 1);
        assert_eq!(v["access"]["total_reads"], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peek_unknown_id_is_err_not_panic() {
        let (dir, _note_id) = seeded_vault_with_note();
        let pool = open_pool(&dir);

        // A well-formed but non-existent id prefix: resolve_id returns an error,
        // which propagates as Err (the router maps it to an error response).
        let result = peek(&pool, "ffffffff");
        assert!(
            result.is_err(),
            "unknown id should be an error, not a panic"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
