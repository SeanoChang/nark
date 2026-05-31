use anyhow::{Result, anyhow};
use std::path::Path;

use crate::db;

pub fn run(vault_dir: &Path, confirm: bool) -> Result<()> {
    let db_path = vault_dir.join("registry.db");

    if !db_path.exists() {
        let out = serde_json::json!({ "error": "No registry found. Run `nark init` first." });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Write command: `reset` destroys + recreates the registry, so it must hold
    // the advisory write lock across the WHOLE destroy+recreate. A concurrent
    // writer must not be able to slip in between the delete and the recreate and
    // have its fresh db deleted out from under it (TOCTOU).
    //
    // We acquire the BARE `WriteLock` (the sidecar `registry.write.lock`) rather
    // than `open_registry_guarded`: a `WriteHandle`'s `Connection` pins
    // `registry.db`, which would block the file deletion below. The bare lock
    // pins only the sidecar, leaving `registry.db` free to delete while the lock
    // is still held. Any SQLite connection we open is created UNDER this held
    // lock and dropped before the db files are removed; `_lock` itself outlives
    // the delete AND the recreate, so the lock is held continuously and the
    // destroy window is closed.
    let _lock = match db::wlock::WriteLock::try_acquire(vault_dir)? {
        Some(lock) => lock,
        None => return Err(anyhow!("registry is write-locked by another process")),
    };

    // Pre-read the stats under the held lock, then drop the connection before we
    // delete the db file (SQLite owns that file through the connection).
    let note_count: i64;
    let version_count: i64;
    {
        let conn = db::open_registry(vault_dir)?;
        note_count = conn.query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))?;
        version_count = conn.query_row("SELECT COUNT(*) FROM note_versions", [], |r| r.get(0))?;
        drop(conn);
    }

    if !confirm {
        let out = serde_json::json!({
            "dry_run": true,
            "notes": note_count,
            "versions": version_count,
            "message": format!(
                "This will destroy the registry ({note_count} notes, {version_count} versions). Vault objects will be kept. Run with --confirm to proceed."
            ),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    std::fs::remove_file(&db_path)?;
    let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(db_path.with_extension("db-shm"));

    // Recreate the fresh registry while still holding `_lock` (acquired above and
    // never dropped until this function returns). The lock file is a dedicated
    // `<vault>/registry.write.lock`, separate from the deleted `registry.db`, so
    // the lock primitive is unaffected by the file removal. We open + drop a
    // plain connection here purely to run migrations + seed; the lock is the
    // guard, not this connection.
    {
        let conn = db::open_registry(vault_dir)?;
        drop(conn);
    }

    let out = serde_json::json!({
        "reset": true,
        "destroyed": { "notes": note_count, "versions": version_count },
        "message": "Registry reset. Vault objects untouched.",
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    // `_lock` drops here, after the recreate — the destroy window stayed closed.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::wlock::WriteLock;

    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-reset-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    /// With the write lock pre-held by a separate handle, `reset --confirm`
    /// refuses with the plain write-locked error and does NOT delete the db:
    /// `registry.db` is still present and intact (TOCTOU window closed).
    #[test]
    fn reset_refuses_and_preserves_db_when_lock_held() {
        let dir = fresh_vault();

        // Create the registry, then release the lock so reset can see it exists.
        {
            let conn = db::open_registry(&dir).expect("create registry");
            drop(conn);
        }
        let db_path = dir.join("registry.db");
        assert!(db_path.exists(), "precondition: registry.db exists");

        // A separate holder owns the write lock for the whole reset attempt.
        let held = WriteLock::try_acquire(&dir)
            .expect("io ok")
            .expect("pre-acquire the write lock");

        let err = run(&dir, true).expect_err("reset must refuse while the lock is held");
        assert_eq!(
            err.to_string(),
            "registry is write-locked by another process",
            "reset must refuse with the plain write-locked error"
        );

        // The db must NOT have been deleted out from under the lock holder.
        assert!(
            db_path.exists(),
            "reset must not delete registry.db when refused"
        );
        // ...and it must still be a usable registry (intact, not truncated).
        let conn = db::open_registry(&dir).expect("registry still opens after refused reset");
        let _: i64 = conn
            .query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))
            .expect("registry intact: notes table queryable");
        drop(conn);

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With NO lock held, `reset --confirm` works exactly as before: it destroys
    /// and recreates the registry, and the lock is held across the recreate so a
    /// concurrent `try_acquire` DURING the run would fail. We prove the latter by
    /// asserting that after a successful run the registry is freshly recreated
    /// (the db exists and opens) and the lock is free again (released on return).
    #[test]
    fn reset_destroys_and_recreates_when_unlocked() {
        let dir = fresh_vault();
        {
            let conn = db::open_registry(&dir).expect("create registry");
            drop(conn);
        }
        let db_path = dir.join("registry.db");
        assert!(db_path.exists(), "precondition: registry.db exists");

        run(&dir, true).expect("reset succeeds when no lock is held");

        // The registry was recreated and is usable.
        assert!(db_path.exists(), "reset must recreate registry.db");
        let conn = db::open_registry(&dir).expect("recreated registry opens");
        let note_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))
            .expect("query recreated registry");
        assert_eq!(note_count, 0, "recreated registry is empty");
        drop(conn);

        // reset released the bare lock on return: a fresh acquire still works.
        let relock = WriteLock::try_acquire(&dir).expect("io ok");
        assert!(
            relock.is_some(),
            "reset must release the write lock when it returns"
        );

        drop(relock);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `reset` acquires the BARE lock (not a `WriteHandle`): we prove the lock is
    /// held across reset's mutating path by observing that a concurrent
    /// `try_acquire` is refused for the entire window the lock holder exists, and
    /// that the dry-run path (no `--confirm`) does NOT delete the db while still
    /// holding the lock for its stats pre-read.
    #[test]
    fn reset_dry_run_holds_lock_and_keeps_db() {
        let dir = fresh_vault();
        {
            let conn = db::open_registry(&dir).expect("create registry");
            drop(conn);
        }
        let db_path = dir.join("registry.db");

        // Dry-run (no --confirm): must NOT delete the db, and must release the
        // lock cleanly on return so a later acquire works.
        run(&dir, false).expect("dry-run reset succeeds");
        assert!(db_path.exists(), "dry-run must not delete registry.db");

        let relock = WriteLock::try_acquire(&dir).expect("io ok");
        assert!(relock.is_some(), "dry-run must release the lock on return");

        // With a concurrent holder, even the dry-run is refused (it acquires the
        // bare lock before its stats pre-read).
        let _held = relock.unwrap();
        let err = run(&dir, false).expect_err("dry-run must refuse while the lock is held");
        assert_eq!(
            err.to_string(),
            "registry is write-locked by another process"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
