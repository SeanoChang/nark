use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;
use std::path::Path;

use crate::db::wlock::WriteLock;

/// Bump access tracking for a note.
/// Called after content is returned in read and about commands.
pub fn bump_access(conn: &Connection, note_id: &str) -> Result<()> {
    conn.execute(
        "UPDATE current_notes \
         SET access_count = access_count + 1, \
             last_accessed = ?1 \
         WHERE note_id = ?2",
        rusqlite::params![Utc::now().to_rfc3339(), note_id],
    )?;
    Ok(())
}

/// Best-effort, NON-BLOCKING, lock-respecting access bump for read commands.
///
/// The access bump is a registry write ([`bump_access`] runs an `UPDATE
/// current_notes ...`), but it is issued from the read CLIs (`read`, `about`,
/// `orient`), which open the *unguarded* connection so reads are NEVER blocked.
/// To avoid an unguarded dual-write that bypasses the no-dual-writer guard, this
/// helper gates the bump on the advisory write lock:
///
/// - Try to acquire `<vault>/registry.write.lock` WITHOUT blocking
///   ([`WriteLock::try_acquire`]). This returns immediately, so the read is
///   never gated by the lock.
/// - `Some(lock)` — we hold the lock: bump every `note_id` under it, then drop
///   the lock (RAII at end of scope).
/// - `None` — a writer (or `nark serve`) holds the lock: SKIP the bump entirely.
///   Access tracking is non-critical (best-effort), so dropping it on contention
///   is correct; the read's printed output is unaffected because the bump is a
///   pure side effect.
/// - `Err` (a genuine I/O error acquiring the lock) — propagated, matching the
///   prior unguarded `bump_access?` which propagated its own errors.
///
/// `conn` must be the same registry connection the reads ran on. The bump only
/// happens while the lock is held, so it is no longer an unguarded write.
pub fn try_bump_access(vault_dir: &Path, conn: &Connection, note_ids: &[&str]) -> Result<()> {
    if note_ids.is_empty() {
        return Ok(());
    }
    // Non-blocking: returns immediately whether or not the lock is free, so the
    // calling read is never blocked. On contention (`None`) we skip silently.
    if let Some(_lock) = WriteLock::try_acquire(vault_dir)? {
        for note_id in note_ids {
            bump_access(conn, note_id)?;
        }
        // `_lock` drops here, releasing the advisory write lock.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-access-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    /// Insert one note row into `current_notes` and return its id, so the bump
    /// has a target row to increment. `head_version_id` and `author_agent_id`
    /// are NOT NULL; the latter references `agents`, so we point it at the
    /// `noah` default seeded by `seed_defaults` on open.
    fn seed_note(conn: &Connection) -> String {
        let note_id = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO current_notes \
             (note_id, namespace, head_version_id, author_agent_id, title, status, access_count) \
             VALUES (?1, 'ark', 'v1', ?2, 'T', 'active', 0)",
            rusqlite::params![note_id, db::DEFAULT_AGENT_ID],
        )
        .expect("insert note");
        note_id
    }

    fn access_count(conn: &Connection, note_id: &str) -> i64 {
        conn.query_row(
            "SELECT access_count FROM current_notes WHERE note_id = ?1",
            rusqlite::params![note_id],
            |r| r.get(0),
        )
        .expect("read access_count")
    }

    /// With NO lock held, `try_bump_access` acquires the lock, bumps, releases.
    #[test]
    fn try_bump_increments_when_unlocked() {
        let dir = fresh_vault();
        let conn = db::open_registry(&dir).expect("open");
        let id = seed_note(&conn);

        assert_eq!(access_count(&conn, &id), 0);
        try_bump_access(&dir, &conn, &[&id]).expect("bump when unlocked");
        assert_eq!(
            access_count(&conn, &id),
            1,
            "with no lock held the bump must run"
        );

        // The helper must have released the lock: a fresh acquire still works.
        let relock = WriteLock::try_acquire(&dir).expect("io ok");
        assert!(
            relock.is_some(),
            "try_bump_access must release the lock after bumping"
        );

        drop(relock);
        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With the lock HELD by a separate handle, `try_bump_access` SKIPS the bump
    /// (best-effort) instead of blocking or erroring: access_count is unchanged.
    #[test]
    fn try_bump_skips_when_locked() {
        let dir = fresh_vault();
        let conn = db::open_registry(&dir).expect("open");
        let id = seed_note(&conn);

        // A separate holder owns the lock for the duration of the bump attempt.
        let held = WriteLock::try_acquire(&dir)
            .expect("io ok")
            .expect("acquire lock");

        let start = std::time::Instant::now();
        try_bump_access(&dir, &conn, &[&id]).expect("bump must not error on contention");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "try_bump_access must not block when the lock is held"
        );
        assert_eq!(
            access_count(&conn, &id),
            0,
            "with the lock held the bump must be SKIPPED, not blocked or applied"
        );

        drop(held);
        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Multi-note: all targets bump under a single lock acquisition when unlocked.
    #[test]
    fn try_bump_bumps_all_notes_when_unlocked() {
        let dir = fresh_vault();
        let conn = db::open_registry(&dir).expect("open");
        let a = seed_note(&conn);
        let b = seed_note(&conn);

        try_bump_access(&dir, &conn, &[&a, &b]).expect("bump both");
        assert_eq!(access_count(&conn, &a), 1);
        assert_eq!(access_count(&conn, &b), 1);

        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
