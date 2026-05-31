//! Advisory write-lock primitive for the registry (Phase 5, slice 5.1).
//!
//! A [`WriteLock`] holds a non-blocking exclusive `flock(2)` on a *dedicated*
//! lock file at `<vault>/registry.write.lock`. We deliberately flock this
//! sidecar file and **never** `registry.db` itself — SQLite owns that handle,
//! and an advisory lock on it would race with SQLite's own file management.
//!
//! The lock is *advisory*: it only blocks other callers that also try to take
//! this same flock. Readers (`db::open_registry` and the read-only deadpool)
//! never touch this file, so reads are never gated by the write lock.
//!
//! Because `flock` is a per-process advisory lock, the OS releases it
//! automatically when the holding process exits or crashes (verified on macOS:
//! fs2 uses `flock(2)` with `LOCK_EX | LOCK_NB`). A crashed writer therefore
//! never wedges the registry — there is no stale lock file to reap. Dropping a
//! [`WriteLock`] releases the lock explicitly (`LOCK_UN`) and closes the file
//! (which would release it anyway).
//!
//! As of slice 5.3 this primitive is live: `db::open_registry_guarded` takes
//! the lock and every mutating CLI command opens through it. `db::open_registry`
//! (reads/back-compat) and the read-only deadpool never touch this lock.

use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result};
use fs2::FileExt;

/// File name of the dedicated advisory write-lock sidecar, created inside the
/// vault directory next to `registry.db`. We flock *this* file, never the db.
pub const WRITE_LOCK_FILENAME: &str = "registry.write.lock";

/// RAII guard holding an exclusive advisory `flock` on `<vault>/registry.write.lock`.
///
/// Construct via [`try_acquire`]. While a `WriteLock` is alive, any other
/// attempt to acquire the same flock (from another process, or another open
/// file handle to the same path) is refused. Dropping the guard releases the
/// lock.
#[derive(Debug)]
pub struct WriteLock {
    /// The locked file handle. The flock lives as long as this handle is open;
    /// `Drop` unlocks it explicitly and then closes it. Kept private so callers
    /// cannot accidentally tamper with the lock.
    file: File,
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        // Best-effort explicit release. Even if this errors, closing `file`
        // (which happens right after) releases the flock at the OS level, and
        // the OS also releases it on process exit. Nothing actionable to do on
        // failure, so we intentionally ignore the result.
        let _ = FileExt::unlock(&self.file);
    }
}

/// Try to acquire the registry write lock for `vault_dir`, without blocking.
///
/// Creates `<vault>/registry.write.lock` if it does not exist, then attempts a
/// non-blocking exclusive `flock` on it.
///
/// - `Ok(Some(lock))` — the lock was acquired; hold the returned guard for the
///   duration of the write, and drop it to release.
/// - `Ok(None)` — another holder currently owns the lock (the non-blocking
///   `flock` returned `EWOULDBLOCK`). The caller should treat this as a clean
///   "registry is write-locked by another process" refusal.
/// - `Err(_)` — a real I/O error (e.g. the lock file could not be created or
///   opened, or `flock` failed for a reason other than contention).
pub fn try_acquire(vault_dir: &Path) -> Result<Option<WriteLock>> {
    let lock_path = vault_dir.join(WRITE_LOCK_FILENAME);

    // Open (creating if absent) the dedicated lock file. We need write access
    // for an exclusive flock to be meaningful across platforms.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening write-lock file {}", lock_path.display()))?;

    match file.try_lock_exclusive() {
        Ok(()) => Ok(Some(WriteLock { file })),
        Err(err) => {
            // Distinguish lock contention (another holder) from a genuine I/O
            // error. fs2 reports contention as `lock_contended_error()`, which
            // is `EWOULDBLOCK` on Unix; compare the raw OS error code so a real
            // failure is never silently swallowed as a clean refusal.
            if err.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                Ok(None)
            } else {
                Err(err).with_context(|| format!("acquiring write lock on {}", lock_path.display()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a fresh, unique temp vault directory (matches the repo's
    /// `std::env::temp_dir()` + pid + uuid convention; no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-wlock-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    #[test]
    fn acquire_on_fresh_vault_succeeds_and_creates_lockfile() {
        let dir = fresh_vault();

        let lock = try_acquire(&dir).expect("io ok");
        assert!(lock.is_some(), "fresh vault should acquire the write lock");

        // The dedicated lockfile must exist at the documented path, and it must
        // NOT be registry.db.
        let lock_path = dir.join(WRITE_LOCK_FILENAME);
        assert!(
            lock_path.exists(),
            "lockfile should exist at {}",
            lock_path.display()
        );
        assert_ne!(
            lock_path.file_name().unwrap(),
            std::ffi::OsStr::new("registry.db"),
            "must flock a dedicated lockfile, never registry.db"
        );
        assert!(
            !dir.join("registry.db").exists(),
            "acquiring the lock must not create or touch registry.db"
        );

        drop(lock);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// While a `WriteLock` is held, a second acquisition through a SEPARATE OS
    /// file handle to the same path is refused with `Ok(None)`.
    ///
    /// macOS semantics: fs2 uses `flock(2)` with `LOCK_EX | LOCK_NB`. flock
    /// associates the lock with the open file description, so a distinct
    /// `OpenOptions::open` of the same path is a *different* description and the
    /// non-blocking exclusive lock fails with `EWOULDBLOCK` — surfaced by
    /// `try_acquire` as `Ok(None)`. (fs2's own unix test suite asserts exactly
    /// this cross-handle contention behavior.) We use a second handle in-process
    /// rather than spawning a child, which is what fs2's semantics support
    /// portably.
    #[test]
    fn second_acquisition_while_held_is_refused() {
        let dir = fresh_vault();

        let held = try_acquire(&dir)
            .expect("io ok")
            .expect("first acquire succeeds");

        // Second acquisition via try_acquire opens its own File handle to the
        // same lockfile, so this exercises real cross-handle flock contention.
        let refused = try_acquire(&dir).expect("io ok (contention is Ok(None), not Err)");
        assert!(
            refused.is_none(),
            "a second acquisition while the lock is held must be refused"
        );

        // Sanity: the held lock is still the only thing keeping the file locked.
        drop(held);

        // After release, acquisition succeeds again.
        let reacquired = try_acquire(&dir).expect("io ok");
        assert!(
            reacquired.is_some(),
            "after the first lock is dropped, a new acquire should succeed"
        );

        drop(reacquired);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_is_released_on_drop() {
        let dir = fresh_vault();

        {
            let lock = try_acquire(&dir).expect("io ok");
            assert!(lock.is_some());
        } // dropped here

        let after = try_acquire(&dir).expect("io ok");
        assert!(
            after.is_some(),
            "dropping the WriteLock should release the flock"
        );

        drop(after);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
