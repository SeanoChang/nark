//! WRITE method implementations for the `nark serve` daemon (Phase 6).
//!
//! Serve is the registry's single authoritative writer: it holds the advisory
//! write lock for its whole lifetime (see [`super::run_until`]) and owns one
//! read-write connection behind the single serializing [`Writer`]
//! ([`super::writer`]). The write methods here submit **one** job to that writer
//! per logical mutation, so every write the daemon performs is applied serially,
//! off the tokio reactor, on the writer's owning thread.
//!
//! The methods are the daemon-side mirror of the write CLI commands, each
//! building the **same** `serde_json::Value` its command prints:
//!
//! * [`write`] mirrors `nark write` / `nark jot` (slice 6.2),
//! * [`link`] mirrors `nark link` (slice 6.4),
//! * [`delete`] mirrors `nark delete` (slice 6.4).
//!
//! All three honor an optional `idempotency_key` (slice 6.3) through
//! [`Writer::submit_idempotent`]: a retried request with the same key returns the
//! cached result without re-applying.
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
use crate::registry::delete::{self, DeletedNote};
use crate::registry::{embeddings, resolve, similarity, write::commit_version};
use crate::types::markdown::{Frontmatter, FrontmatterLink};
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

/// Parsed parameters for [`link`], mapped from the JSON-RPC `params` object.
///
/// Mirrors `nark link <sources...> <target> --rel <rel>` (`cli/link.rs`):
/// `sources` and `target` are note ids (or unambiguous prefixes), `rel` is the
/// edge type. `idempotency_key` is the optional dedup key (slice 6.3).
pub struct LinkParams {
    /// The source note ids/prefixes that gain a `rel` edge to `target`.
    pub sources: Vec<String>,
    /// The destination note id/prefix every source links to.
    pub target: String,
    /// The edge type (`references`, `depends-on`, ...), exactly as `--rel`.
    pub rel: String,
    /// Optional caller-supplied idempotency key (slice 6.3). See [`WriteParams`].
    pub idempotency_key: Option<String>,
}

/// Parsed parameters for [`delete`], mapped from the JSON-RPC `params` object.
///
/// Mirrors `nark delete <ids...> [-f] [-rf]` (`cli/delete.rs`): `ids` are the
/// notes to delete, and the two booleans select the mode exactly as the CLI's
/// `--force` (`-f`) / `--recursive` (`-r`) flags do — soft-retract by default,
/// `force` -> hard delete, `force` + `recursive` -> purge (hard delete plus CAS
/// object removal). `idempotency_key` is the optional dedup key (slice 6.3).
pub struct DeleteParams {
    /// The note ids/prefixes to delete.
    pub ids: Vec<String>,
    /// `--force`: hard-delete (remove registry rows) instead of soft-retract.
    pub force: bool,
    /// `--recursive`: with `force`, also purge the CAS objects (the `-rf` mode).
    pub recursive: bool,
    /// Optional caller-supplied idempotency key (slice 6.3). See [`WriteParams`].
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

/// `nark/link`: create typed edges from each source note to a target, mirroring
/// `cli::link`.
///
/// Submits ONE job to the single serializing [`Writer`]: on the writer's owning
/// thread, against its one read-write connection, the job — exactly as
/// `cli/link.rs` does — resolves `target` and each `source`, rejects self-links,
/// and for each source that does not already carry the link adds a
/// [`FrontmatterLink`] (`target`, `rel`) to its frontmatter plus a `[[target]]`
/// wikilink in the matching body section, then re-ingests the source note
/// ([`Vault::ingest`] with the source's id) and commits a new version
/// ([`commit_version`], which materializes the edge via `sync_edges`). A source
/// that already has both the frontmatter link and the body wikilink is a `no-op`.
///
/// Returns the **same** JSON object the CLI prints:
/// `{ "target", "target_title", "rel", "linked": <count>, "results": [ { "source",
/// "source_title", "status": "linked" | "no-op" } ] }`.
///
/// A missing target/source surfaces as the job's `Err` (the router maps it to a
/// clean error response). The whole op runs in one writer job so every mutated
/// source is committed serially on the single writer connection, off the reactor.
/// `idempotency_key` is honored via [`Writer::submit_idempotent`] (slice 6.3): a
/// retried link with the same key returns the cached result without re-applying.
pub async fn link(ctx: &Ctx, params: LinkParams) -> Result<Value> {
    let writer = ctx
        .writer()
        .ok_or_else(|| anyhow::anyhow!("serve writer is unavailable"))?;

    let vault_dir = ctx.vault_dir().to_path_buf();
    let LinkParams {
        sources,
        target,
        rel,
        idempotency_key,
    } = params;

    writer
        .submit_idempotent(idempotency_key, move |conn| {
            let vault = Vault::new(vault_dir.clone());

            // Validate target exists and resolve prefix.
            let target_meta = resolve::get_meta(conn, &target)
                .map_err(|_| anyhow::anyhow!("target note not found: {}", target))?;
            let target_id = &target_meta.note_id;

            let mut results: Vec<Value> = Vec::new();

            for source in &sources {
                // Validate source exists and resolve prefix.
                let source_meta = resolve::get_meta(conn, source)
                    .map_err(|_| anyhow::anyhow!("source note not found: {}", source))?;
                let source_id = &source_meta.note_id;

                // Reject self-links.
                if source_id == target_id {
                    anyhow::bail!("cannot link a note to itself: {}", source_id);
                }

                // Read current source note content.
                let refs = resolve::get_ref(conn, source_id)?;
                let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
                let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

                let mut fm: Frontmatter = serde_yaml::from_str(&fm_raw)?;

                // Idempotency — both frontmatter and body already have the link.
                let has_fm_link = fm
                    .links
                    .iter()
                    .any(|l| l.target == *target_id && l.rel == rel);
                let wikilink_bare = format!("[[{}]]", target_id);
                let has_body_link = body.contains(&wikilink_bare);

                if has_fm_link && has_body_link {
                    results.push(json!({
                        "source": source_id,
                        "source_title": source_meta.title,
                        "status": "no-op",
                    }));
                    continue;
                }

                // Mutate frontmatter.
                if !has_fm_link {
                    fm.links.push(FrontmatterLink {
                        target: target_id.to_string(),
                        rel: rel.clone(),
                    });
                }

                // Mutate body — insert into the correct rel section.
                let (new_body, _) = if has_body_link {
                    (body, false)
                } else {
                    insert_body_link(&body, target_id, &rel)
                };

                // Reassemble and re-ingest.
                let full_note = format!("---\n{}---\n{}", serde_yaml::to_string(&fm)?, new_body);
                let result = vault.ingest(&full_note, Some(source_id))?;
                commit_version(conn, &result)?;

                results.push(json!({
                    "source": source_id,
                    "source_title": source_meta.title,
                    "status": "linked",
                }));
            }

            Ok(json!({
                "target": target_id,
                "target_title": target_meta.title,
                "rel": rel,
                "linked": results.iter().filter(|r| r["status"] == "linked").count(),
                "results": results,
            }))
        })
        .await
}

/// `nark/delete`: delete one or more notes, mirroring `cli::delete`.
///
/// Submits ONE job to the single serializing [`Writer`]: on the writer's owning
/// thread, against its one read-write connection, the job validates the ids
/// ([`delete::validate_ids`]) and applies the mode the booleans select, exactly
/// as `cli/delete.rs`:
///
/// * default — [`delete::soft_delete`] (status becomes `retracted`), `mode`
///   `"retract"`;
/// * `force` — [`delete::hard_delete`] (registry rows removed), `mode`
///   `"hard_delete"`;
/// * `force` + `recursive` — hard delete plus CAS object removal (the `-rf`
///   purge), `mode` `"purge"`.
///
/// Returns the **same** JSON object the CLI prints:
/// `{ "deleted": <count>, "mode", "notes": [ { "id", "title" } ] }`.
///
/// An unknown id surfaces as the job's `Err` (the router maps it to a clean
/// error response) before any deletion runs. The whole op runs in one writer job,
/// serially on the single writer connection, off the reactor. `idempotency_key`
/// is honored via [`Writer::submit_idempotent`] (slice 6.3): a retried delete with
/// the same key returns the cached result without re-applying.
pub async fn delete(ctx: &Ctx, params: DeleteParams) -> Result<Value> {
    let writer = ctx
        .writer()
        .ok_or_else(|| anyhow::anyhow!("serve writer is unavailable"))?;

    let vault_dir = ctx.vault_dir().to_path_buf();
    let DeleteParams {
        ids,
        force,
        recursive,
        idempotency_key,
    } = params;

    writer
        .submit_idempotent(idempotency_key, move |conn| {
            let notes = delete::validate_ids(conn, &ids)?;

            let mode = if force && recursive {
                purge(conn, &notes, &vault_dir)?;
                "purge"
            } else if force {
                delete::hard_delete(conn, &notes)?;
                "hard_delete"
            } else {
                delete::soft_delete(conn, &notes)?;
                "retract"
            };

            Ok(json!({
                "deleted": notes.len(),
                "mode": mode,
                "notes": notes.iter().map(|n| json!({
                    "id": n.note_id,
                    "title": n.title,
                })).collect::<Vec<_>>(),
            }))
        })
        .await
}

/// Hard delete the notes, then remove their CAS objects — the `-rf` purge,
/// mirroring `cli::delete::purge`. Runs on the writer connection inside the
/// [`delete`] job, so registry rows and CAS objects are removed under the single
/// writer.
fn purge(
    conn: &rusqlite::Connection,
    notes: &[DeletedNote],
    vault_dir: &std::path::Path,
) -> Result<()> {
    delete::hard_delete(conn, notes)?;

    let vault = Vault::new(vault_dir.to_path_buf());
    for note in notes {
        vault.remove_object("objects/fm", &note.fm_hash, "yaml")?;
        vault.remove_object("objects/md", &note.md_hash, "md")?;
    }

    Ok(())
}

/// Convert a rel type like `depends-on` to a section heading like `## Depends On`,
/// mirroring `cli::link::rel_to_heading` (the CLI helper is private, so the serve
/// link path ports it verbatim — same output for the same rel).
fn rel_to_heading(rel: &str) -> String {
    let words: Vec<String> = rel
        .split('-')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    format!("{}{}", upper, c.as_str())
                }
            }
        })
        .collect();
    format!("## {}", words.join(" "))
}

/// Insert a wikilink into the correct rel section of the body, mirroring
/// `cli::link::insert_body_link` (the CLI helper is private, so the serve link
/// path ports it verbatim).
///
/// Cases:
/// 1. Section exists -> append `- [[target]]` after the last `- [[...]]` line in
///    that section.
/// 2. Section doesn't exist -> append the section + link at the end of the body.
/// 3. Link already present in body anywhere -> return body unchanged.
fn insert_body_link(body: &str, target: &str, rel: &str) -> (String, bool) {
    let wikilink_entry = format!("- [[{}]]", target);
    let wikilink_bare = format!("[[{}]]", target);

    // Already present anywhere in body — skip.
    if body.contains(&wikilink_bare) {
        return (body.to_string(), false);
    }

    let heading = rel_to_heading(rel);
    let lines: Vec<&str> = body.lines().collect();

    // Find the section for this rel type.
    if let Some(section_idx) = lines.iter().position(|l| l.trim() == heading) {
        // Find the last `- [[...]]` line within this section (before next ## or end).
        let mut insert_after = section_idx;
        for (i, line) in lines.iter().enumerate().skip(section_idx + 1) {
            let trimmed = line.trim();
            if trimmed.starts_with("## ") {
                break;
            }
            if trimmed.starts_with("- [[") {
                insert_after = i;
            }
        }

        let mut result: Vec<&str> = Vec::with_capacity(lines.len() + 1);
        result.extend_from_slice(&lines[..=insert_after]);
        let mut out = result.join("\n");
        out.push('\n');
        out.push_str(&wikilink_entry);
        if insert_after + 1 < lines.len() {
            out.push('\n');
            out.push_str(&lines[insert_after + 1..].join("\n"));
        }
        return (out, true);
    }

    // Section doesn't exist — append at end.
    let trimmed_body = body.trim_end();
    let new_body = format!("{}\n\n{}\n{}", trimmed_body, heading, wikilink_entry);
    (new_body, true)
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

    /// A note document with a given title/body, `namespace = ark` (via
    /// `commit_version`), for the link/delete tests.
    fn note_doc(title: &str, body: &str) -> String {
        format!(
            "---\n\
title: {title}\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - gamma\n\
---\n\
{body}\n"
        )
    }

    /// Write one note through the serve writer and return its committed id.
    async fn write_note(ctx: &Ctx, title: &str, body: &str) -> String {
        write(
            ctx,
            WriteParams {
                note: note_doc(title, body),
                auto_link: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("seed write ok")["id"]
            .as_str()
            .expect("write returns an id")
            .to_string()
    }

    /// `nark/link` creates a typed edge from the source to the target: the
    /// response mirrors the CLI's JSON, the source's frontmatter + body gain the
    /// link (visible through a serve read), and the edge materializes — the source
    /// gains an outgoing link and the target an incoming one (visible via `peek`).
    #[tokio::test]
    async fn link_creates_edge_between_two_notes() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let src = write_note(&ctx, "Source Note", "Source body text.").await;
        let dst = write_note(&ctx, "Target Note", "Target body text.").await;

        let out = link(
            &ctx,
            LinkParams {
                sources: vec![src.clone()],
                target: dst.clone(),
                rel: "depends-on".to_string(),
                idempotency_key: None,
            },
        )
        .await
        .expect("link should succeed");

        // Same JSON shape the CLI prints.
        assert_eq!(out["target"], dst);
        assert_eq!(out["target_title"], "Target Note");
        assert_eq!(out["rel"], "depends-on");
        assert_eq!(out["linked"], 1, "one source was linked");
        assert_eq!(out["results"][0]["source"], src);
        assert_eq!(out["results"][0]["status"], "linked");

        // The source note now carries the frontmatter link + body wikilink.
        let read = methods_read::read(ctx_pool(&ctx), &dir, &src)
            .await
            .expect("read source");
        assert_eq!(
            read["frontmatter"]["links"][0]["target"], dst,
            "the frontmatter link targets dst"
        );
        assert_eq!(read["frontmatter"]["links"][0]["rel"], "depends-on");
        assert!(
            read["body"]
                .as_str()
                .unwrap()
                .contains(&format!("[[{dst}]]")),
            "the body gains a wikilink to dst, got: {}",
            read["body"]
        );

        // The edge materialized. Re-ingesting the linked source created BOTH a
        // `depends-on` frontmatter edge AND a `references` body edge (the auto-
        // extracted wikilink) — distinct rows (PK is src,dst,edge_type) — exactly
        // as `nark link` does. Assert the typed edge exists and the counts reflect
        // the real edges, not an invented single edge.
        let edges = {
            let conn = crate::db::open_registry(&dir).expect("open registry");
            let (outgoing, _incoming) =
                crate::registry::edges::get_edges(&conn, &src).expect("get_edges");
            drop(conn);
            outgoing
        };
        assert!(
            edges
                .iter()
                .any(|e| e.note_id == dst && e.edge_type == "depends-on"),
            "the typed depends-on edge from src to dst must exist, got: {:?}",
            edges
                .iter()
                .map(|e| (&e.note_id, &e.edge_type))
                .collect::<Vec<_>>()
        );

        // src gained outgoing link(s), dst incoming link(s) (>=1, since the body
        // wikilink adds a second edge alongside the typed frontmatter one).
        let src_meta = methods_read::peek(ctx_pool(&ctx), &src)
            .await
            .expect("peek source");
        assert!(
            src_meta["links_out"].as_i64().unwrap() >= 1,
            "source has outgoing link(s)"
        );
        let dst_meta = methods_read::peek(ctx_pool(&ctx), &dst)
            .await
            .expect("peek target");
        assert!(
            dst_meta["links_in"].as_i64().unwrap() >= 1,
            "target has incoming link(s)"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Re-running `nark/link` with the same args (no idempotency key) is a `no-op`:
    /// the CLI's own idempotency (both frontmatter + body link already present)
    /// reports `no-op` and creates no second version on the source.
    #[tokio::test]
    async fn link_repeat_same_args_is_noop() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let src = write_note(&ctx, "Src", "Body.").await;
        let dst = write_note(&ctx, "Dst", "Body.").await;

        let params = || LinkParams {
            sources: vec![src.clone()],
            target: dst.clone(),
            rel: "references".to_string(),
            idempotency_key: None,
        };

        let first = link(&ctx, params()).await.expect("first link ok");
        assert_eq!(first["results"][0]["status"], "linked");
        let (_n1, versions_after_link) = counts(&ctx).await;

        let second = link(&ctx, params()).await.expect("second link ok");
        assert_eq!(
            second["results"][0]["status"], "no-op",
            "re-linking the same edge is a no-op (CLI idempotency)"
        );
        let (_n2, versions_after_repeat) = counts(&ctx).await;
        assert_eq!(
            versions_after_link, versions_after_repeat,
            "a no-op link must not commit a new version on the source"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/link` with the SAME `idempotency_key` apply once: the second
    /// returns the identical cached result without re-applying (no extra version),
    /// even though it would otherwise be a fresh `linked`.
    #[tokio::test]
    async fn link_same_idempotency_key_dedups() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let src = write_note(&ctx, "Src", "Body.").await;
        let dst = write_note(&ctx, "Dst", "Body.").await;

        let params = || LinkParams {
            sources: vec![src.clone()],
            target: dst.clone(),
            rel: "references".to_string(),
            idempotency_key: Some("link-key".to_string()),
        };

        let first = link(&ctx, params()).await.expect("first keyed link ok");
        let (_n, versions_after_first) = counts(&ctx).await;

        let second = link(&ctx, params())
            .await
            .expect("second keyed link ok (served from cache)");
        let (_n2, versions_after_second) = counts(&ctx).await;

        assert_eq!(
            first, second,
            "a retried link with the same key returns the identical cached result"
        );
        assert_eq!(
            first["results"][0]["status"], "linked",
            "the cached result is the original 'linked' (not a re-applied no-op)"
        );
        assert_eq!(
            versions_after_first, versions_after_second,
            "the deduped retry must not commit a second version"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/delete` (default) soft-retracts the note: the status becomes
    /// `retracted` (visible via `peek`), the JSON mode is `retract`, and the row
    /// survives (one note still in the registry).
    #[tokio::test]
    async fn delete_soft_retracts_status() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Doomed", "Body.").await;

        let out = delete(
            &ctx,
            DeleteParams {
                ids: vec![id.clone()],
                force: false,
                recursive: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("soft delete ok");

        assert_eq!(out["deleted"], 1);
        assert_eq!(out["mode"], "retract");
        assert_eq!(out["notes"][0]["id"], id);
        assert_eq!(out["notes"][0]["title"], "Doomed");

        // The note survives (soft) and its status is now retracted.
        let meta = methods_read::peek(ctx_pool(&ctx), &id)
            .await
            .expect("peek after soft delete");
        assert_eq!(
            meta["status"], "retracted",
            "soft delete sets status = retracted"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/delete -f` hard-deletes: the registry rows are removed (the note can
    /// no longer be read), mode is `hard_delete`.
    #[tokio::test]
    async fn delete_force_hard_deletes_rows() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Gone", "Body.").await;

        let out = delete(
            &ctx,
            DeleteParams {
                ids: vec![id.clone()],
                force: true,
                recursive: false,
                idempotency_key: None,
            },
        )
        .await
        .expect("hard delete ok");
        assert_eq!(out["mode"], "hard_delete");
        assert_eq!(out["deleted"], 1);

        // The registry rows are gone: the note no longer resolves.
        assert!(
            methods_read::read(ctx_pool(&ctx), &dir, &id).await.is_err(),
            "a hard-deleted note must no longer be readable"
        );
        let (notes, _versions) = counts(&ctx).await;
        assert_eq!(notes, 0, "hard delete removed the only note");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/delete -rf` purges: registry rows AND the CAS objects are removed.
    /// The note no longer reads and the body object is gone from the vault.
    #[tokio::test]
    async fn delete_recursive_purges_cas_objects() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Purged", "Purged body.").await;

        // Capture the CAS object hashes before purge so we can assert removal.
        let (fm_hash, md_hash) = {
            let conn = crate::db::open_registry(&dir).expect("open registry");
            let r = resolve::get_ref(&conn, &id).expect("get_ref before purge");
            drop(conn);
            (r.fm_hash, r.md_hash)
        };
        let vault = Vault::new(dir.clone());
        assert!(
            vault.read_object("objects/md", &md_hash, "md").is_ok(),
            "body object exists before purge"
        );

        let out = delete(
            &ctx,
            DeleteParams {
                ids: vec![id.clone()],
                force: true,
                recursive: true,
                idempotency_key: None,
            },
        )
        .await
        .expect("purge ok");
        assert_eq!(out["mode"], "purge");
        assert_eq!(out["deleted"], 1);

        // Rows gone AND CAS objects removed.
        assert!(
            methods_read::read(ctx_pool(&ctx), &dir, &id).await.is_err(),
            "a purged note must no longer be readable"
        );
        assert!(
            vault.read_object("objects/md", &md_hash, "md").is_err(),
            "purge removed the body CAS object"
        );
        assert!(
            vault.read_object("objects/fm", &fm_hash, "yaml").is_err(),
            "purge removed the frontmatter CAS object"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two `nark/delete` with the SAME `idempotency_key` apply once: the second
    /// returns the identical cached result without re-applying. A retried delete
    /// of an already-deleted note would otherwise error ("not found"); dedup
    /// returns the original success instead.
    #[tokio::test]
    async fn delete_same_idempotency_key_dedups() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let id = write_note(&ctx, "Once", "Body.").await;

        let params = || DeleteParams {
            ids: vec![id.clone()],
            force: true,
            recursive: false,
            idempotency_key: Some("del-key".to_string()),
        };

        let first = delete(&ctx, params()).await.expect("first keyed delete ok");
        assert_eq!(first["mode"], "hard_delete");

        // Without dedup this retry would error (the note is already gone). With the
        // same key it returns the cached success and never re-runs the closure.
        let second = delete(&ctx, params())
            .await
            .expect("retried keyed delete is served from cache, not re-applied");
        assert_eq!(
            first, second,
            "a retried delete with the same key returns the identical cached result"
        );

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/link` to a non-existent target is a clean `Err` (the router maps it to
    /// an error response), not a panic.
    #[tokio::test]
    async fn link_missing_target_is_err() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;
        let src = write_note(&ctx, "Src", "Body.").await;

        let result = link(
            &ctx,
            LinkParams {
                sources: vec![src],
                target: "ffffffff".to_string(),
                rel: "references".to_string(),
                idempotency_key: None,
            },
        )
        .await;
        assert!(result.is_err(), "linking to a missing target must error");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `nark/delete` of an unknown id is a clean `Err`, not a panic.
    #[tokio::test]
    async fn delete_unknown_id_is_err() {
        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        let result = delete(
            &ctx,
            DeleteParams {
                ids: vec!["ffffffff".to_string()],
                force: false,
                recursive: false,
                idempotency_key: None,
            },
        )
        .await;
        assert!(result.is_err(), "deleting an unknown id must error");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Phase 6 slice 6.5: a READ over the read-only deadpool runs CONCURRENTLY
    /// with an in-flight WRITE. The single writer thread is parked inside an open
    /// `BEGIN IMMEDIATE` transaction (so it genuinely holds the WAL write lock and
    /// occupies the one writer thread), gated by a channel the test controls; while
    /// that write is in flight a `nark/read` through the read-only pool must still
    /// complete PROMPTLY (WAL readers do not block on the writer). The read is
    /// asserted to finish *before* the gate is opened, proving the overlap is real,
    /// not an artifact of the write having already finished.
    #[tokio::test]
    async fn read_runs_concurrently_with_in_flight_write() {
        use std::time::Duration;

        let dir = fresh_vault();
        let ctx = ctx_with_writer(&dir).await;

        // Seed a note to read back.
        let id = write_note(&ctx, "Concurrent", "Concurrent body.").await;

        // A barrier the gated write job blocks on while INSIDE an open write
        // transaction, so it holds the WAL write lock and occupies the single
        // writer thread until the test opens the gate.
        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let gate_rx = std::sync::Mutex::new(gate_rx);
        // Signals that the gated job has actually started (so the write is truly
        // in flight before we issue the concurrent read).
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();

        let writer = Arc::clone(ctx.writer().expect("writer present"));
        let inflight = tokio::spawn(async move {
            writer
                .submit(move |conn| {
                    // BEGIN IMMEDIATE takes the WAL write lock now; the single
                    // writer thread is parked here until the gate opens.
                    conn.execute_batch("BEGIN IMMEDIATE")?;
                    let _ = started_tx.send(());
                    let _ = gate_rx.lock().expect("gate lock").recv();
                    conn.execute_batch("ROLLBACK")?;
                    Ok::<_, anyhow::Error>(())
                })
                .await
        });

        // Wait until the write is genuinely in flight (transaction open, thread
        // parked on the gate).
        started_rx.await.expect("gated write started");

        // While the write is in flight, a read through the read-only pool must
        // complete promptly (WAL readers are not blocked by the writer).
        let read = tokio::time::timeout(
            Duration::from_secs(5),
            methods_read::read(ctx_pool(&ctx), &dir, &id),
        )
        .await
        .expect("a WAL read must not block on an in-flight write")
        .expect("read succeeds");
        assert_eq!(
            read["body"], "Concurrent body.",
            "the concurrent read returns the note while the write holds the lock"
        );

        // The read completed BEFORE we open the gate, proving the overlap was real
        // (the writer thread is still parked in its transaction).
        assert!(
            !inflight.is_finished(),
            "the gated write must still be in flight when the read completes"
        );

        // Release the writer; the in-flight job rolls back and completes cleanly.
        gate_tx.send(()).expect("open the gate");
        tokio::time::timeout(Duration::from_secs(10), inflight)
            .await
            .expect("the in-flight write completes once the gate opens")
            .expect("in-flight task join")
            .expect("in-flight submit ok");

        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
