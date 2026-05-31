use anyhow::{Result, anyhow};
use include_dir::{Dir, include_dir};
use rusqlite::Connection;
use rusqlite_migration::Migrations;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::LazyLock;

use wlock::WriteLock;

/// Advisory registry write-lock primitive (Phase 5). Wired into
/// [`open_registry_guarded`] as of slice 5.2: a guarded writer-open takes this
/// lock before opening a RW connection. `pub` so callers can reason about the
/// lock; reads (`open_registry` and the read-only deadpool) never touch it.
///
/// `allow(dead_code)`: as of this slice (5.2) the lock is exercised by the lib
/// tests and by [`open_registry_guarded`], but no write-command call site in
/// the *binary* takes a guarded handle yet, so the bin target still sees these
/// symbols as unused. The slice that wires the write CLIs onto
/// `open_registry_guarded` MUST drop this allow.
#[allow(dead_code)]
pub mod wlock;

pub const DEFAULT_AGENT_ID: &str = "noah";
pub const DEFAULT_CLIENT_ID: &str = "cli_default";

static MIGRATIONS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations");

pub(crate) static MIGRATIONS: LazyLock<Migrations<'static>> =
    LazyLock::new(|| Migrations::from_directory(&MIGRATIONS_DIR).unwrap());

/// Open the registry read-write at `<vault>/registry.db`, applying the WAL +
/// foreign-keys pragmas, running migrations, and seeding defaults.
///
/// This is the SINGLE shared open path. Both the unlocked [`open_registry`]
/// (reads/back-compat) and the lock-guarded [`open_registry_guarded`] (writes)
/// delegate here so the two can never diverge.
fn open_registry_inner(vault_dir: &Path) -> Result<Connection> {
    let db_path = vault_dir.join("registry.db");
    let mut conn = Connection::open(db_path)?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA foreign_keys=ON;",
    )?;

    MIGRATIONS.to_latest(&mut conn)?;

    seed_defaults(&conn)?;

    Ok(conn)
}

pub fn open_registry(vault_dir: &Path) -> Result<Connection> {
    open_registry_inner(vault_dir)
}

/// A registry connection opened read-write while holding the advisory write
/// lock. The lock is released when the handle is dropped.
///
/// `WriteHandle` derefs to the underlying [`Connection`], so callers use it
/// exactly like the `Connection` returned by [`open_registry`]. The held
/// [`WriteLock`] is private (`_lock`): callers cannot release it early without
/// dropping the whole handle, which guarantees the lock outlives every write
/// issued through this connection.
///
/// `allow(dead_code)`: exercised by the lib tests this slice, but the binary
/// has no guarded write-command call site yet (next slice). The wiring slice
/// MUST drop this allow.
#[allow(dead_code)]
#[derive(Debug)]
pub struct WriteHandle {
    conn: Connection,
    /// Held for the lifetime of the handle; released on drop (RAII). Never read
    /// directly — its only job is to keep the flock alive.
    _lock: WriteLock,
}

impl WriteHandle {
    /// Borrow the underlying connection. Equivalent to deref; provided for
    /// call sites that prefer an explicit accessor over auto-deref.
    ///
    /// `allow(dead_code)`: exercised by the lib tests, but the binary has no
    /// guarded call site yet (next slice). The wiring slice MUST drop this allow.
    #[allow(dead_code)]
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

impl Deref for WriteHandle {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        &self.conn
    }
}

impl DerefMut for WriteHandle {
    fn deref_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }
}

/// Open the registry read-write for a writer, gated by the advisory write lock.
///
/// 1. Try to acquire `<vault>/registry.write.lock` without blocking. If another
///    process already holds it, return a plain refusal — *without opening any
///    writer connection*.
/// 2. Otherwise open the registry exactly as [`open_registry`] does (shared
///    inner path: WAL + foreign-keys + migrate + seed) and return a
///    [`WriteHandle`] that owns both the connection and the lock guard.
///
/// The conflict error is intentionally generic ("registry is write-locked by
/// another process") — see the Phase-5 scope decision: the write CLIs are a
/// defense-in-depth safety net, not a UX surface, so there is no serve-specific
/// message and no wait/retry.
///
/// `allow(dead_code)`: exercised by the lib tests this slice, but no write
/// command in the binary calls it yet (next slice). The wiring slice MUST drop
/// this allow.
#[allow(dead_code)]
pub fn open_registry_guarded(vault_dir: &Path) -> Result<WriteHandle> {
    let lock = match wlock::try_acquire(vault_dir)? {
        Some(lock) => lock,
        None => return Err(anyhow!("registry is write-locked by another process")),
    };

    let conn = open_registry_inner(vault_dir)?;

    Ok(WriteHandle { conn, _lock: lock })
}

pub(crate) fn seed_defaults(conn: &Connection) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT OR IGNORE INTO clients (client_id, name, api_key_hash, is_admin, created_at)
         VALUES (?1, ?2, '', 1, ?3)",
        rusqlite::params![DEFAULT_CLIENT_ID, DEFAULT_CLIENT_ID, now],
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO agents (agent_id, name, namespace, role, can_write_public, registered_by, registered_at)
         VALUES (?1, ?2, 'lib', 'admin', 1, ?3, ?4)",
        rusqlite::params![DEFAULT_AGENT_ID, DEFAULT_AGENT_ID, DEFAULT_CLIENT_ID, now],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    /// Create a fresh, unique temp vault directory (matches the repo's
    /// `std::env::temp_dir()` + pid + uuid convention; no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-db-guarded-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    #[test]
    fn test_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        MIGRATIONS.to_latest(&mut conn).unwrap();

        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap();
        let table_names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|res| res.unwrap())
            .collect();

        assert!(table_names.contains(&"notes".to_string()));
        assert!(table_names.contains(&"note_versions".to_string()));
    }

    /// With no lock held, `open_registry_guarded` opens a usable writer handle:
    /// it must hold the advisory write lock AND open the registry exactly like
    /// `open_registry` (migrated + seeded), so the conn can query.
    #[test]
    fn guarded_open_succeeds_and_conn_queries() {
        let dir = fresh_vault();

        let handle = open_registry_guarded(&dir).expect("guarded open succeeds on fresh vault");

        // The migrate+seed path ran: the default agent seeded by seed_defaults
        // must be present, queried through the handle's conn (Deref).
        let agent_count: i64 = handle
            .query_row(
                "SELECT COUNT(*) FROM agents WHERE agent_id = ?1",
                rusqlite::params![DEFAULT_AGENT_ID],
                |row| row.get(0),
            )
            .expect("query through guarded handle (Deref)");
        assert_eq!(
            agent_count, 1,
            "seed_defaults must have run via the guarded path"
        );

        // The explicit .conn() accessor returns the same usable connection.
        let client_count: i64 = handle
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM clients WHERE client_id = ?1",
                rusqlite::params![DEFAULT_CLIENT_ID],
                |row| row.get(0),
            )
            .expect("query through guarded handle (.conn())");
        assert_eq!(
            client_count, 1,
            "seed_defaults must have seeded the default client"
        );

        // The dedicated lock file exists; the registry db was actually opened.
        assert!(dir.join(wlock::WRITE_LOCK_FILENAME).exists());
        assert!(dir.join("registry.db").exists());

        drop(handle);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A second guarded open, while a first handle is alive, is refused with the
    /// plain write-locked error and NO writer connection is opened. Dropping the
    /// first handle releases the lock so a later guarded open succeeds.
    #[test]
    fn guarded_open_refuses_when_lock_held_then_succeeds_after_drop() {
        let dir = fresh_vault();

        let first = open_registry_guarded(&dir).expect("first guarded open succeeds");

        let err = open_registry_guarded(&dir)
            .expect_err("second guarded open must be refused while the lock is held");
        let msg = err.to_string();
        assert_eq!(
            msg, "registry is write-locked by another process",
            "conflict must be the plain honest error"
        );
        // Safety-net only: must NOT leak any serve-specific text.
        assert!(!msg.contains("serve"), "error must not mention serve");

        // First handle releases the lock on drop.
        drop(first);

        let second = open_registry_guarded(&dir)
            .expect("guarded open succeeds again after the first handle is dropped");
        drop(second);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `open_registry_guarded` and plain `open_registry` must produce an
    /// identical schema/seed (no logic divergence): the inner open path is
    /// shared, so a plain open of the same vault sees the seeded defaults too.
    #[test]
    fn guarded_open_matches_plain_open_schema_and_seed() {
        let dir = fresh_vault();

        // Open + close guarded so the lock is released.
        {
            let handle = open_registry_guarded(&dir).expect("guarded open");
            drop(handle);
        }

        // A plain open of the same db must see the same seeded defaults and the
        // same migrated tables — i.e. the guarded path did not diverge.
        let conn = open_registry(&dir).expect("plain open after guarded open");
        let agent_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM agents WHERE agent_id = ?1",
                rusqlite::params![DEFAULT_AGENT_ID],
                |row| row.get(0),
            )
            .expect("query default agent");
        assert_eq!(agent_count, 1);

        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap();
        let table_names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|res| res.unwrap())
            .collect();
        assert!(table_names.contains(&"notes".to_string()));
        assert!(table_names.contains(&"note_versions".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
