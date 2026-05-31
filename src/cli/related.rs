use anyhow::{Result, bail};
use std::ops::Deref;
use std::path::Path;

use crate::config;
use crate::db;
use crate::registry::{embeddings, resolve, similarity};

/// Registry connection for `related`, opened according to whether this
/// invocation writes.
///
/// `related <id>` is a pure read by default and must NOT be gated by the write
/// lock; with `--link` it mutates (auto-link edges) and so must hold the
/// advisory write lock for the whole op. Both variants `Deref` to the
/// underlying `&Connection`, so all the logic below is identical regardless of
/// which open path was taken. When `--link` is set, the held [`db::WriteHandle`]
/// keeps the write lock alive until this value drops at end of function.
enum RegistryConn {
    /// `link == false`: pure read, opened unguarded (never lock-gated).
    Read(rusqlite::Connection),
    /// `link == true`: write path, opened through the lock-checked guard.
    Write(db::WriteHandle),
}

impl Deref for RegistryConn {
    type Target = rusqlite::Connection;

    fn deref(&self) -> &rusqlite::Connection {
        match self {
            RegistryConn::Read(conn) => conn,
            RegistryConn::Write(handle) => handle,
        }
    }
}

pub fn run(vault_dir: &Path, id: &str, limit: usize, link: bool) -> Result<()> {
    // `related` is a READ by default and a WRITE only with `--link` (it creates
    // auto-link edges). Reads must never be lock-gated, so open unguarded; the
    // write path holds the advisory write lock for the whole op via the guarded
    // open. A conflicting guarded open is refused with the plain write-locked
    // error, and the held handle derefs to the `Connection` so the logic below
    // is unchanged. The lock is released when `conn` drops at end of function.
    let conn = if link {
        RegistryConn::Write(db::open_registry_guarded(vault_dir)?)
    } else {
        RegistryConn::Read(db::open_registry(vault_dir)?)
    };
    let cfg = config::load(vault_dir)?;

    let meta = resolve::get_meta(&conn, id)?;

    // Get this note's embedding
    let note_embedding = match embeddings::get_embedding(&conn, &meta.note_id)? {
        Some(emb) => emb,
        None => bail!(
            "no embedding for note {}. Run `nark embed build`.",
            meta.note_id
        ),
    };

    // Load all embeddings
    let all = embeddings::get_all_embeddings(&conn)?;
    if all.is_empty() {
        bail!("no embeddings found. Run `nark embed init` then `nark embed build`.");
    }

    // Dimension check
    if let Some((_, first_vec)) = all.first() {
        if first_vec.len() != note_embedding.len() {
            bail!(
                "embedding dimension mismatch (note={}, stored={}). Run `nark embed build` to re-embed.",
                note_embedding.len(),
                first_vec.len()
            );
        }
    }

    let similar = similarity::find_similar_notes(
        &conn,
        &meta.note_id,
        &all,
        &note_embedding,
        cfg.embedding.similarity_threshold as f32,
        limit,
    );

    let linked = if link && !similar.is_empty() {
        similarity::create_auto_edges(
            &conn,
            &meta.note_id,
            &similar,
            cfg.embedding.auto_link_threshold as f32,
        )
        .unwrap_or(0)
    } else {
        0
    };

    let sim_json: Vec<serde_json::Value> = similar
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.note_id,
                "title": s.title,
                "similarity": (s.similarity * 1000.0).round() / 1000.0,
            })
        })
        .collect();

    let mut out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "similar": sim_json,
    });

    if link {
        out["auto_linked"] = serde_json::json!(linked);
    }

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    /// Fresh, unique temp vault dir (matches the repo's temp-dir + pid + uuid
    /// convention; no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-related-lock-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    /// Seed two notes plus a near-identical embedding pair into the file-backed
    /// registry so that `related --link` on the first note WOULD create one
    /// auto-edge to the second if it were allowed to run. Returns the two
    /// (full-UUID) note IDs. Uses a plain (unguarded) open for setup only.
    ///
    /// Real UUIDs are required because `related` resolves the id through
    /// `resolve::get_meta` → `resolve_id`, which only accepts hex+hyphen ids.
    fn seed_two_similar_notes(vault_dir: &Path) -> (String, String) {
        let conn = db::open_registry(vault_dir).expect("open registry for seeding");
        let note_a = uuid::Uuid::new_v4().to_string();
        let note_b = uuid::Uuid::new_v4().to_string();
        insert_note(&conn, &note_a, "Note A");
        insert_note(&conn, &note_b, "Note B");
        // Cosine-similarity ~1.0 between these two (both default thresholds pass).
        upsert_test_embedding(&conn, &note_a, &make_embedding(1.0, 16));
        upsert_test_embedding(&conn, &note_b, &make_embedding(1.0, 16));
        (note_a, note_b)
    }

    fn insert_note(conn: &Connection, note_id: &str, title: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let version_id = format!("v-{}", note_id);
        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, 'ch', 'fh', 'mh', ?4)",
            rusqlite::params![version_id, note_id, db::DEFAULT_AGENT_ID, now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, 'test', 'reference', 'active', ?5)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, title, now],
        )
        .unwrap();
    }

    fn upsert_test_embedding(conn: &Connection, note_id: &str, embedding: &[f32]) {
        embeddings::upsert_embedding(conn, note_id, embedding, "test-model")
            .expect("seed embedding");
    }

    fn make_embedding(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| seed + i as f32).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn edge_count(vault_dir: &Path) -> i64 {
        let conn = db::open_registry(vault_dir).expect("open registry for edge count");
        conn.query_row("SELECT COUNT(*) FROM note_edges", [], |r| r.get(0))
            .expect("count edges")
    }

    /// `related --link` is a WRITE path: while the registry write lock is held by
    /// another handle it must be refused with the plain write-locked error
    /// (through the guarded open, before any work) and must NOT create any edge.
    /// Releasing the lock lets the same call succeed and actually create the
    /// auto-edge, proving the guarded path's success behavior is unchanged.
    #[test]
    fn related_link_refuses_and_creates_no_edge_when_write_locked() {
        let dir = fresh_vault();
        let (note_a, _note_b) = seed_two_similar_notes(&dir);
        assert_eq!(
            edge_count(&dir),
            0,
            "no edges before the locked link attempt"
        );

        // Hold the write lock for the duration of the link attempt.
        let held = db::open_registry_guarded(&dir).expect("pre-hold the write lock");

        let err = run(&dir, &note_a, 10, true)
            .expect_err("related --link must be refused while the write lock is held");
        let msg = err.to_string();
        assert_eq!(
            msg, "registry is write-locked by another process",
            "conflict must surface the plain honest error"
        );
        assert!(
            !msg.contains("serve"),
            "safety-net error must not mention serve"
        );
        assert_eq!(
            edge_count(&dir),
            0,
            "a refused related --link must not have created any edge"
        );

        // After the lock is released the same write path runs to completion and
        // creates the auto-edge.
        drop(held);
        run(&dir, &note_a, 10, true).expect("related --link succeeds after lock released");
        assert_eq!(
            edge_count(&dir),
            1,
            "related --link must create the auto-edge once the lock is free"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `related <id>` (no `--link`) is a pure READ and must NEVER be gated by the
    /// write lock: while the lock is held it opens through the unguarded path and
    /// runs to completion (producing its normal output) without surfacing the
    /// write-locked error.
    #[test]
    fn related_read_is_not_gated_by_write_lock() {
        let dir = fresh_vault();
        let (note_a, _note_b) = seed_two_similar_notes(&dir);

        let held = db::open_registry_guarded(&dir).expect("hold the write lock");

        // link == false: the read path must succeed even though the write lock is
        // held — it does NOT acquire the lock, so it must not see the conflict.
        run(&dir, &note_a, 10, false)
            .expect("related (read, link=false) must succeed while write-locked");

        // And a pure read never creates edges.
        assert_eq!(
            edge_count(&dir),
            0,
            "the read path must not have written any edge"
        );

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
