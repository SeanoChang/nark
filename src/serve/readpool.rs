//! Read-only WAL connection pool for the `nark serve` daemon.
//!
//! Slice 3.1 of the Ark comm protocol: serve's read methods (peek / read /
//! stats / search / orient, landed in later slices) need cheap, concurrent,
//! read-only access to the vault registry. They must **not** re-run migrations
//! or `seed_defaults` — the writer (`db::open_registry`, used by the CLI) owns
//! schema evolution and seeding, and WAL is already enabled by that writer. The
//! pool therefore opens each connection with
//! [`OpenFlags::SQLITE_OPEN_READ_ONLY`] only.
//!
//! A WAL database supports many concurrent readers, so the pool simply holds a
//! fixed set of read-only [`Connection`]s and hands one out per `with_conn`
//! call. Checkout blocks (via a [`Condvar`]) until a connection is free, so the
//! pool degrades to serialized access under contention rather than failing or
//! opening unbounded connections.
//!
//! Slice 3.1 lands the pool and proves it (read-back, RO-enforced, concurrent)
//! via tests; the serve connection handler does not consume it until a later
//! slice. Until then the pool's public surface reads as dead code in the binary
//! target (the lib/test target exercises all of it), so the module carries a
//! scoped `dead_code` allow rather than leaving the new code un-plumbed-but-warned.
#![allow(dead_code)]

use std::path::Path;
use std::sync::{Condvar, Mutex};

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};

/// Default number of read-only connections in the pool.
///
/// Four covers the expected per-daemon read concurrency (a handful of agents
/// issuing interleaved reads) without holding many file descriptors open. Reads
/// that exceed this simply queue on [`ReadPool::with_conn`].
///
/// `open` / `DEFAULT_POOL_SIZE` are the entry points the serve read methods call
/// in a later slice; until then only the explicit-size constructor is exercised
/// (by tests).
pub const DEFAULT_POOL_SIZE: usize = 4;

/// A fixed-size pool of read-only SQLite connections to `<vault>/registry.db`.
///
/// Each connection is opened with [`OpenFlags::SQLITE_OPEN_READ_ONLY`] and no
/// `SQLITE_OPEN_CREATE`, so the pool can never create or migrate the database —
/// the writer owns that. Any attempt to write through a pooled connection fails
/// at the SQLite layer, which is the intended guarantee.
pub struct ReadPool {
    /// Idle connections available for checkout. A connection is removed on
    /// checkout and pushed back on return (see [`PoolGuard`]).
    idle: Mutex<Vec<Connection>>,
    /// Signaled whenever a connection is returned, to wake a waiting checkout.
    available: Condvar,
}

impl ReadPool {
    /// Open a pool of [`DEFAULT_POOL_SIZE`] read-only connections against
    /// `<vault_dir>/registry.db`.
    ///
    /// The database (and its schema) must already exist — opening read-only
    /// without `SQLITE_OPEN_CREATE` fails if the file is missing. This is by
    /// design: the writer (`db::open_registry`) creates, migrates, and seeds the
    /// registry; the pool only reads it.
    pub fn open(vault_dir: &Path) -> Result<Self> {
        Self::open_with_size(vault_dir, DEFAULT_POOL_SIZE)
    }

    /// As [`Self::open`], but with an explicit connection count. Mainly for
    /// tests that want to drive contention or assert the pool size.
    pub fn open_with_size(vault_dir: &Path, size: usize) -> Result<Self> {
        assert!(size > 0, "read pool size must be at least 1");
        let db_path = vault_dir.join("registry.db");
        // Read-only, never create: the writer owns creation/migration/seeding.
        // WAL is already enabled by the writer, so readers inherit it.
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let mut conns = Vec::with_capacity(size);
        for _ in 0..size {
            let conn = Connection::open_with_flags(&db_path, flags)
                .with_context(|| format!("opening read-only registry at {}", db_path.display()))?;
            conns.push(conn);
        }
        Ok(Self {
            idle: Mutex::new(conns),
            available: Condvar::new(),
        })
    }

    /// Check out a connection, run `f` against it, and return the connection to
    /// the pool (even if `f` returns an error or panics, via [`PoolGuard`]).
    ///
    /// If every connection is currently in use, this blocks until one is
    /// returned. The borrow handed to `f` is read-only at the SQLite layer, so a
    /// write statement run through it returns `Err`.
    pub fn with_conn<R>(&self, f: impl FnOnce(&Connection) -> Result<R>) -> Result<R> {
        let guard = self.checkout();
        f(guard.conn())
    }

    /// Block until a connection is available and check it out, wrapping it in a
    /// [`PoolGuard`] that returns it on drop.
    fn checkout(&self) -> PoolGuard<'_> {
        let mut idle = self.idle.lock().expect("read pool mutex poisoned");
        loop {
            if let Some(conn) = idle.pop() {
                return PoolGuard {
                    pool: self,
                    conn: Some(conn),
                };
            }
            idle = self
                .available
                .wait(idle)
                .expect("read pool mutex poisoned while waiting");
        }
    }

    /// Return a connection to the idle set and wake one waiter.
    fn checkin(&self, conn: Connection) {
        let mut idle = self.idle.lock().expect("read pool mutex poisoned");
        idle.push(conn);
        drop(idle);
        self.available.notify_one();
    }
}

/// RAII guard that holds a checked-out connection and returns it to the pool on
/// drop, so a panic or early `?` in the closure never leaks a connection.
struct PoolGuard<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
}

impl PoolGuard<'_> {
    fn conn(&self) -> &Connection {
        self.conn
            .as_ref()
            .expect("pooled connection taken before drop")
    }
}

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.checkin(conn);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Build a temp vault whose `registry.db` is created, migrated, and seeded
    /// by the writer (`db::open_registry`), then drop the writer connection.
    /// Returns the temp dir; the seeded `agents` row (`agent_id = "noah"`) is
    /// the "seeded note" the read-only pool reads back.
    fn seeded_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-readpool-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        // Writer opens RW, migrates, and seeds; dropping it releases the file.
        let conn = crate::db::open_registry(&dir).expect("seed registry via writer");
        drop(conn);
        dir
    }

    #[test]
    fn pool_opens_n_readonly_conns_and_reads_seeded_row() {
        let dir = seeded_vault();

        let pool = ReadPool::open_with_size(&dir, 4).expect("open read pool");
        assert_eq!(
            pool.idle.lock().unwrap().len(),
            4,
            "pool should hold the requested number of connections"
        );

        // SELECT the seeded agent row back through a pooled connection.
        let name: String = pool
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT name FROM agents WHERE agent_id = ?1",
                    [crate::db::DEFAULT_AGENT_ID],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .expect("read seeded row");
        assert_eq!(name, crate::db::DEFAULT_AGENT_ID);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_through_pool_conn_errors() {
        let dir = seeded_vault();
        let pool = ReadPool::open_with_size(&dir, 2).expect("open read pool");

        let result: Result<()> = pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO agents (agent_id, name, namespace, role, registered_at)
                 VALUES ('intruder', 'intruder', 'intruder-ns', 'agent', '2026-05-31T00:00:00Z')",
                [],
            )
            .map(|_| ())
            .map_err(Into::into)
        });

        assert!(
            result.is_err(),
            "a write through a read-only pool connection must fail"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_concurrent_with_conn_calls_both_succeed() {
        let dir = seeded_vault();
        let pool = Arc::new(ReadPool::open_with_size(&dir, 4).expect("open read pool"));

        // Barrier so both threads hold a connection simultaneously, proving the
        // pool serves concurrent readers (WAL allows it) rather than serializing
        // to a single connection.
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    pool.with_conn(|conn| {
                        // Both threads block here until the other has also
                        // checked out a connection.
                        barrier.wait();
                        let count: i64 =
                            conn.query_row("SELECT COUNT(*) FROM agents", [], |row| row.get(0))?;
                        Ok(count)
                    })
                    .expect("concurrent read")
                })
            })
            .collect();

        for handle in handles {
            let count = handle.join().expect("thread join");
            assert!(count >= 1, "seeded agent row should be visible");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
