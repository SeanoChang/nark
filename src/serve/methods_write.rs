//! WRITE method implementations for the `nark serve` daemon (Phase 6, slice 6.2).
//!
//! Serve is the registry's single authoritative writer: it holds the advisory
//! write lock for its whole lifetime (see [`super::run_until`]) and owns one
//! read-write connection behind the single serializing [`Writer`]
//! ([`super::writer`]). The write methods here submit **one** job to that writer
//! per logical mutation, so every write the daemon performs is applied serially,
//! off the tokio reactor, on the writer's owning thread.
//!
//! [`write`] is the daemon-side mirror of `nark write` / `nark jot`
//! (`cli/{write,jot}.rs`): it ingests one note markdown document into the CAS,
//! commits a new version, and — when an embedding provider is available — upserts
//! the note embedding and computes similarity suggestions (auto-linking when
//! asked). It builds the **same** `serde_json::Value` the CLI prints for a single
//! note: `{ "id", "title" }`, plus the optional `similar` / `auto_linked` keys
//! the embedding path appends. There is no behavioural change to the CLI.
//!
//! Differences from the CLI handlers, forced by the serve design:
//!
//! * the registry mutation runs inside a [`Writer::submit`] job on the writer's
//!   single read-write connection, **not** a fresh `db::open_registry_guarded`.
//!   Serve already holds the advisory write lock, so the writer opened an
//!   *unlocked* connection (a second lock would self-deadlock the daemon);
//! * Phase 6 writes as the single lock-holding authority — there is **no**
//!   per-agent author check here (peer-agent write authz is Phase 7). The
//!   authenticated peer is gated at the listener (unknown uids are rejected
//!   before any method runs); a known peer may write.
//!
//! The whole job — ingest, commit, embed, link — runs in one closure on the
//! writer thread so the connection is touched by exactly that thread (single-
//! writer discipline) and the reactor is never blocked: [`Writer::submit`] awaits
//! a oneshot for the result. The `params` shape is the full note markdown
//! document (frontmatter + body), exactly what `vault.ingest` consumes — the
//! cleanest fit for an agent, and identical to what `nark write` reads from a
//! file or stdin.

use anyhow::Result;
use serde_json::{Value, json};

use super::rpc::Ctx;
use crate::config;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, similarity, write::commit_version};
use crate::vault::fs::Vault;

/// Parsed parameters for [`write`], mapped from the JSON-RPC `params` object.
///
/// `note` is the full note markdown document — frontmatter delimited by `---`
/// then the body — exactly what `vault.ingest` consumes and what `nark write`
/// reads from a file/stdin. `auto_link` mirrors `nark write --auto-link`: when
/// set, similarity suggestions above the auto-link threshold become edges.
pub struct WriteParams {
    /// The full note markdown document (frontmatter + body) to ingest.
    pub note: String,
    /// Whether to create auto-link edges from similarity suggestions.
    pub auto_link: bool,
    /// Optional caller-supplied idempotency key (Phase 6, slice 6.3). When set,
    /// the single serializing writer applies this write at most once for the key
    /// and returns the cached result on a repeat — so a client that retries
    /// (e.g. after a dropped connection) never commits a second version. Absent
    /// -> never deduped (the write always applies). The key scope is **global**
    /// across the daemon's write methods; see [`super::writer`] for the bound.
    pub idempotency_key: Option<String>,
}

/// `nark/write`: ingest one note markdown document, mirroring `cli::write` /
/// `cli::jot` for a single note.
///
/// Submits ONE job to the single serializing [`Writer`]: on the writer's owning
/// thread, against its one read-write connection, the job
///
/// 1. ingests `params.note` into the CAS ([`Vault::ingest`]),
/// 2. commits a new version ([`commit_version`]),
/// 3. if an embedding provider is available, embeds the note, upserts the
///    embedding ([`embeddings::upsert_embedding`]) and computes similarity
///    suggestions ([`similarity::compute_suggestions`], auto-linking when
///    `params.auto_link`),
///
/// and returns the **same** JSON object the CLI prints for one note:
/// `{ "id": note_id, "title": title }`, plus `similar` / `auto_linked` when the
/// embedding path produced suggestions (see [`similarity::append_to_json`]).
///
/// A malformed note (no frontmatter, bad YAML) surfaces as the `Err` from
/// `ingest`, which the router maps to a clean error response. Embedding failures
/// degrade gracefully exactly as the CLI does — the note is still written, just
/// without the `similar` block.
///
/// The reactor is never blocked: the blocking SQLite + CAS I/O run on the
/// writer thread and this `await`s the job's oneshot reply. When the writer queue
/// is full, [`Writer::submit_idempotent`] returns a clean backpressure `Err` (no
/// hang).
///
/// When `params.idempotency_key` is set, the write is submitted through
/// [`Writer::submit_idempotent`]: the single writer applies it at most once for
/// the key and returns the cached `{ "id", "title", ... }` result on a retry, so
/// a re-sent write commits no second version. An absent key always applies (no
/// dedup). The dedup check/cache is atomic on the one writer thread — see
/// [`super::writer`].
pub async fn write(ctx: &Ctx, params: WriteParams) -> Result<Value> {
    let writer = ctx
        .writer()
        .ok_or_else(|| anyhow::anyhow!("serve writer is unavailable"))?;

    // Config + vault dir are read off-thread; the closure owns clones so it is
    // `'static + Send` for the writer thread. `init_provider` runs inside the
    // job (on the writer thread) so all blocking work — ONNX init/inference and
    // SQLite — stays off the reactor and on the single writer thread.
    let vault_dir = ctx.vault_dir().to_path_buf();
    let WriteParams {
        note,
        auto_link,
        idempotency_key,
    } = params;

    writer
        .submit_idempotent(idempotency_key, move |conn| {
            let cfg = config::load(&vault_dir)?;

            let vault = Vault::new(vault_dir.clone());
            let result = vault.ingest(&note, None)?;
            commit_version(conn, &result)?;

            // Embedding step — graceful degradation matches the CLI: if no
            // provider is available, or the embed fails, the note is still
            // written; we just skip the `similar` block.
            let last_embedding =
                if let Some(ref mut prov) = embed::init_provider(&vault_dir, &cfg.embedding) {
                    let fm = &result.frontmatter;
                    let input = build_embed_input(
                        &fm.title,
                        &fm.domain,
                        &fm.kind,
                        &fm.intent,
                        &fm.tags,
                        &fm.aliases,
                        &result.body,
                    );
                    match prov.embed_document(&input) {
                        Ok(embedding) => {
                            let _ = embeddings::upsert_embedding(
                                conn,
                                &result.note_id,
                                &embedding,
                                prov.model_name(),
                            );
                            Some(embedding)
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

            let mut output = json!({
                "id": result.note_id,
                "title": result.frontmatter.title,
            });

            if let Some(ref embedding) = last_embedding
                && embeddings::has_embeddings(conn)
            {
                let all = embeddings::get_all_embeddings(conn).unwrap_or_default();
                if let Some(sim_result) = similarity::compute_suggestions(
                    conn,
                    &result.note_id,
                    embedding,
                    &all,
                    cfg.embedding.similarity_threshold as f32,
                    cfg.embedding.auto_link_threshold as f32,
                    cfg.embedding.max_suggestions,
                    auto_link,
                ) {
                    similarity::append_to_json(&sim_result, &mut output);
                }
            }

            Ok(output)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::super::dpool::{self, RoManager};
    use super::super::methods_read;
    use super::super::rpc::Ctx;
    use super::super::writer::Writer;
    use super::*;
    use std::sync::Arc;

    const NOTE: &str = "---\n\
title: Written Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - gamma\n\
---\n\
Written body text.\n";

    /// A fresh, unique temp vault whose `registry.db` is created/migrated/seeded
    /// by the writer open, then dropped so the read pool can open it. Matches the
    /// repo's temp-dir + pid + uuid convention (no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-methods-write-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        let conn = crate::db::open_registry(&dir).expect("seed registry");
        drop(conn);
        dir
    }

    /// Build a `Ctx` over `dir` with a real writer (the daemon's write path) and a
    /// read-only pool (the daemon's read path), without binding a socket.
    async fn ctx_with_writer(dir: &std::path::Path) -> Ctx {
        let pool = dpool::open_ro_pool(dir, 2).await.expect("open read pool");
        let writer = Arc::new(Writer::open(dir).expect("open writer"));
        Ctx::with_writer(pool, dir.to_path_buf(), writer)
    }

    /// `nark/write` ingests a note and returns the single-note JSON object the CLI
    /// prints (`{ "id", "title" }`); a follow-up `nark/read` over the same daemon
    /// returns that note's body and frontmatter.
    #[tokio::test]
    async fn write_creates_note_then_read_returns_it() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let out = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("write should succeed");

        let note_id = out["id"]
            .as_str()
            .expect("write returns a note id")
            .to_string();
        assert!(!note_id.is_empty(), "note id must be non-empty");
        assert_eq!(out["title"], "Written Note", "write echoes the note title");

        // The note is now durably committed: a follow-up serve read returns it.
        // (Phase 3 read path over the read-only pool, against the same db.)
        let read = methods_read::read(ctx_pool(&ctx), &dir, &note_id)
            .await
            .expect("read of the just-written note should succeed");
        assert_eq!(read["id"], note_id);
        assert_eq!(read["title"], "Written Note");
        assert_eq!(read["body"], "Written body text.");
        assert_eq!(read["frontmatter"]["domain"], "engineering");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Parity: the JSON `nark/write` returns for a given note must equal the JSON a
    /// direct CLI-style write produces for the same input — `vault.ingest` ->
    /// `commit_version` over a writer connection, then the exact `{ "id", "title" }`
    /// object `cli/jot.rs` builds. (No embeddings present in either path, so the
    /// optional `similar`/`auto_linked` keys are absent in both.)
    #[tokio::test]
    async fn write_json_matches_direct_cli_write_shape() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let served = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("serve write should succeed");

        // Direct registry-path write of the SAME note into a SEPARATE vault: the
        // ingest is content-addressed and deterministic, so the title is identical
        // and the JSON shape (keys + title) must match. The note_id is a fresh
        // uuid per write, so parity is asserted on shape + title (not the id).
        let dir2 = fresh_vault();
        let conn = crate::db::open_registry(&dir2).expect("open writer registry");
        let vault = Vault::new(dir2.clone());
        let result = vault.ingest(NOTE, None).expect("direct ingest");
        commit_version(&conn, &result).expect("direct commit");
        let direct = json!({
            "id": result.note_id,
            "title": result.frontmatter.title,
        });
        drop(conn);

        // Same set of top-level keys, same title; neither path attached `similar`.
        let served_obj = served.as_object().expect("write returns an object");
        let direct_obj = direct.as_object().unwrap();
        let served_keys: std::collections::BTreeSet<&String> = served_obj.keys().collect();
        let direct_keys: std::collections::BTreeSet<&String> = direct_obj.keys().collect();
        assert_eq!(
            served_keys, direct_keys,
            "serve write JSON must have the same shape as a direct CLI write"
        );
        assert_eq!(
            served["title"], direct["title"],
            "serve write title must match the direct write for the same note"
        );
        assert!(
            served.get("similar").is_none(),
            "no embeddings -> no similar block, matching the CLI's degraded path"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    /// A malformed note (no frontmatter) is a clean `Err` (the router maps it to
    /// `-32602`), not a panic — and nothing is committed.
    #[tokio::test]
    async fn write_malformed_note_is_err() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let result = write(
            &ctx,
            WriteParams {
                note: "no frontmatter here".to_string(),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await;
        assert!(result.is_err(), "a note without frontmatter must error");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Borrow the read pool out of a `Ctx` for the follow-up read. The write tests
    /// need both the writer (to write) and the read pool (to read back) on one
    /// `Ctx`; `methods_read::read` takes the pool directly, so expose it here via a
    /// tiny test shim rather than widening the public `Ctx` surface.
    fn ctx_pool(ctx: &Ctx) -> &deadpool::managed::Pool<RoManager> {
        ctx.dpool_for_test()
    }

    /// Total notes and total versions in the vault — the dedup invariant for the
    /// idempotency tests (a retried write must not bump either count).
    async fn counts(ctx: &Ctx) -> (i64, i64) {
        let stats = methods_read::stats(ctx_pool(ctx))
            .await
            .expect("stats should succeed");
        (
            stats["total_notes"].as_i64().expect("total_notes"),
            stats["total_versions"].as_i64().expect("total_versions"),
        )
    }

    /// Two `nark/write` with the SAME `idempotency_key` create exactly ONE
    /// note/version, and the second call returns the IDENTICAL cached result (same
    /// id) without re-applying — a retried write is idempotent.
    #[tokio::test]
    async fn same_idempotency_key_writes_once_returns_cached() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let first = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: Some("retry-key-1".to_string()),
            },
        )
        .await
        .expect("first write ok");
        let (notes_after_first, versions_after_first) = counts(&ctx).await;
        assert_eq!(notes_after_first, 1, "first write creates one note");
        assert_eq!(versions_after_first, 1, "first write creates one version");

        // The retry: same key, same note. Must NOT create a second note/version,
        // and must return the identical cached result.
        let second = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: Some("retry-key-1".to_string()),
            },
        )
        .await
        .expect("retry write ok (served from cache)");

        let (notes_after_retry, versions_after_retry) = counts(&ctx).await;
        assert_eq!(
            notes_after_retry, 1,
            "a retried write with the same key must not create a second note"
        );
        assert_eq!(
            versions_after_retry, 1,
            "a retried write with the same key must not create a second version"
        );
        assert_eq!(
            first, second,
            "the retry must return the identical cached result (same id + title)"
        );
        assert_eq!(
            first["id"], second["id"],
            "the cached result carries the same note id"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/write` with DIFFERENT idempotency keys create TWO notes/versions
    /// (no cross-key dedup), each with its own id.
    #[tokio::test]
    async fn different_idempotency_keys_write_twice() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let a = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: Some("key-a".to_string()),
            },
        )
        .await
        .expect("write a ok");
        let b = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: Some("key-b".to_string()),
            },
        )
        .await
        .expect("write b ok");

        let (notes, versions) = counts(&ctx).await;
        assert_eq!(notes, 2, "two distinct keys create two notes");
        assert_eq!(versions, 2, "two distinct keys create two versions");
        assert_ne!(
            a["id"], b["id"],
            "two distinct keys produce two distinct notes"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/write` with NO idempotency key create TWO notes/versions — an
    /// absent key is never deduped.
    #[tokio::test]
    async fn absent_idempotency_key_writes_twice() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let a = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("keyless write a ok");
        let b = write(
            &ctx,
            WriteParams {
                note: NOTE.to_string(),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("keyless write b ok");

        let (notes, versions) = counts(&ctx).await;
        assert_eq!(notes, 2, "two keyless writes create two notes");
        assert_eq!(versions, 2, "two keyless writes create two versions");
        assert_ne!(
            a["id"], b["id"],
            "keyless writes never dedup: two distinct notes"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
