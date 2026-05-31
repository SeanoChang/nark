//! Deadpool-managed read-only connection pool for the `nark serve` daemon.
//!
//! Phase 3.5 (slice 3.5.1) of the Ark comm protocol. The Phase-3 read path uses
//! a hand-rolled [`super::readpool::ReadPool`] (`Mutex<Vec<Connection>>` +
//! `Condvar`) consumed synchronously via `spawn_blocking`. This module lands the
//! replacement: a [`deadpool`]-managed pool whose connections are still strictly
//! **read-only**, but whose checkout (`pool.get().await`) and blocking SQLite
//! work (`conn.interact(...).await`) are async-native — backpressure when every
//! connection is busy, and blocking queries run on a managed thread rather than
//! holding a worker.
//!
//! Why not `deadpool-sqlite`? Its built-in `Config`/`Manager` hardcode
//! `Connection::open()` (= `SQLITE_OPEN_READ_WRITE | SQLITE_OPEN_CREATE`) with no
//! `OpenFlags` hook (verified against deadpool-sqlite 0.13.0). That would let the
//! pool create or write the registry, violating serve's read-only guarantee — the
//! writer (`db::open_registry`, used by the CLI) owns creation/migration/seeding,
//! and WAL is already enabled there. So we pair [`deadpool`]'s generic managed
//! pool with [`deadpool_sync::SyncWrapper`] and the custom [`RoManager`] below,
//! which opens each connection with
//! [`OpenFlags::SQLITE_OPEN_READ_ONLY`] | [`OpenFlags::SQLITE_OPEN_NO_MUTEX`] and
//! no `SQLITE_OPEN_CREATE`. Any write through a pooled connection fails at the
//! SQLite layer (`SQLITE_READONLY`), which is the intended guarantee.
//!
//! Slice 3.5.1 lands the manager + pool **in parallel**: nothing here is wired
//! into the router yet (that is a later slice), so the public surface reads as
//! dead code in the binary target until then. The lib/test target exercises all
//! of it, so the module carries a scoped `dead_code` allow rather than leaving
//! the new code un-plumbed-but-warned.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use deadpool::Runtime;
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use deadpool_sync::SyncWrapper;
use rusqlite::{Connection, OpenFlags};

/// Default number of read-only connections in the pool.
///
/// Four covers the expected per-daemon read concurrency (a handful of agents
/// issuing interleaved reads) without holding many file descriptors open. Reads
/// that exceed this back-pressure on [`Pool::get`] (subject to the wait timeout).
pub const DEFAULT_POOL_SIZE: usize = 4;

/// How long [`Pool::get`] waits for a free connection before erroring rather
/// than blocking an async task indefinitely under sustained contention. A
/// `wait` timeout requires the pool to be built with a [`Runtime`] (we set
/// [`Runtime::Tokio1`]), else `build()` fails with `NoRuntimeSpecified`.
const GET_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// [`deadpool::managed::Manager`] that creates **read-only** SQLite connections
/// to `<vault>/registry.db`, wrapped in a [`SyncWrapper`] so blocking SQLite
/// runs on a managed thread (`conn.interact(...).await`).
///
/// `create()` opens with [`OpenFlags::SQLITE_OPEN_READ_ONLY`] |
/// [`OpenFlags::SQLITE_OPEN_NO_MUTEX`] and **no** `SQLITE_OPEN_CREATE`, so the
/// manager can never create or migrate the database — the writer owns that, and
/// opening a missing registry fails (by design). `recycle()` runs a cheap
/// `SELECT 1` to confirm the handle is still usable before reuse.
#[derive(Debug)]
pub struct RoManager {
    /// Absolute path to `<vault>/registry.db`.
    path: PathBuf,
}

impl RoManager {
    /// Build a manager that opens `<vault_dir>/registry.db` read-only.
    fn new(vault_dir: &Path) -> Self {
        Self {
            path: vault_dir.join("registry.db"),
        }
    }
}

impl Manager for RoManager {
    type Type = SyncWrapper<Connection>;
    type Error = rusqlite::Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        let path = self.path.clone();
        // Read-only, never create: the writer owns creation/migration/seeding.
        // WAL is already enabled by the writer, so readers inherit it. The
        // blocking `open_with_flags` runs on a managed thread via `SyncWrapper`.
        SyncWrapper::new(Runtime::Tokio1, move || {
            Connection::open_with_flags(
                &path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
        })
        .await
    }

    async fn recycle(&self, conn: &mut Self::Type, _: &Metrics) -> RecycleResult<Self::Error> {
        // Cheap liveness check before reuse. `interact` returns
        // `Result<rusqlite::Result<i64>, InteractError>`: an `InteractError`
        // (panicked/poisoned handle) maps to a `Message` recycle error so the
        // connection is dropped and recreated; a SQLite error is a `Backend`
        // recycle error (via the blanket `From<E>`).
        conn.interact(|c| c.query_row("SELECT 1", [], |row| row.get::<_, i64>(0)))
            .await
            .map_err(|e| RecycleError::message(format!("recycle interact failed: {e}")))??;
        Ok(())
    }
}

/// Open a [`deadpool`]-managed pool of `size` read-only connections against
/// `<vault_dir>/registry.db`.
///
/// The database (and its schema) must already exist — opening read-only without
/// `SQLITE_OPEN_CREATE` fails if the file is missing, so a vault with no
/// `registry.db` is rejected (lazily, on first [`Pool::get`], when the manager
/// runs `create`). The pool caps at `size` connections and backpressures
/// further `get()` calls until one frees (up to [`GET_WAIT_TIMEOUT`]).
pub async fn open_ro_pool(vault_dir: &Path, size: usize) -> Result<Pool<RoManager>> {
    assert!(size > 0, "read pool size must be at least 1");
    let manager = RoManager::new(vault_dir);
    Pool::builder(manager)
        .max_size(size)
        .wait_timeout(Some(GET_WAIT_TIMEOUT))
        // A `wait` timeout requires a runtime, else `build()` errors.
        .runtime(Runtime::Tokio1)
        .build()
        .with_context(|| {
            format!(
                "building read-only registry pool for {}",
                vault_dir.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::write::commit_version;
    use crate::vault::fs::Vault;

    const NOTE: &str = "---\n\
title: Pool Note\n\
author: tester\n\
domain: engineering\n\
intent: reference\n\
kind: note\n\
status: active\n\
tags:\n\
  - delta\n\
---\n\
Pool body text.\n";

    /// Build a temp vault whose `registry.db` is created, migrated, and seeded
    /// by the writer (`db::open_registry`), ingest one note so `current_notes`
    /// has a row, then drop the writer connection to release the file. Returns
    /// the vault dir; the seeded `current_notes` count is `1`.
    fn seeded_vault() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-dpool-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        let conn = crate::db::open_registry(&dir).expect("seed registry via writer");
        let vault = Vault::new(dir.clone());
        let result = vault.ingest(NOTE, None).expect("ingest note");
        commit_version(&conn, &result).expect("commit version");
        drop(conn);
        dir
    }

    #[tokio::test]
    async fn pool_reads_seeded_current_notes_count() {
        let dir = seeded_vault();

        let pool = open_ro_pool(&dir, 4).await.expect("open ro pool");
        let conn = pool.get().await.expect("checkout connection");
        let count: i64 = conn
            .interact(|c| c.query_row("SELECT count(*) FROM current_notes", [], |r| r.get(0)))
            .await
            .expect("interact join")
            .expect("query row");
        assert_eq!(count, 1, "the one ingested note should be visible");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_through_pool_conn_errors_readonly() {
        let dir = seeded_vault();
        let pool = open_ro_pool(&dir, 2).await.expect("open ro pool");
        let conn = pool.get().await.expect("checkout connection");

        // An INSERT through a read-only connection must fail at the SQLite layer
        // (SQLITE_READONLY) — this proves the pool's read-only guarantee.
        let inner: rusqlite::Result<usize> = conn
            .interact(|c| {
                c.execute(
                    "INSERT INTO agents (agent_id, name, namespace, role, registered_at)
                     VALUES ('intruder', 'intruder', 'intruder-ns', 'agent', '2026-05-31T00:00:00Z')",
                    [],
                )
            })
            .await
            .expect("interact join");

        let err = inner.expect_err("write through a read-only pool connection must fail");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("read") && msg.contains("only") || msg.contains("readonly"),
            "expected a read-only error from SQLite, got: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pool_against_missing_registry_errors_no_create() {
        // A vault dir with no registry.db: opening read-only without
        // SQLITE_OPEN_CREATE must fail rather than creating an empty db. The
        // failure surfaces on first `get()` (lazy connection creation).
        let dir = std::env::temp_dir().join(format!(
            "nark-dpool-noregistry-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create empty temp vault");
        assert!(
            !dir.join("registry.db").exists(),
            "precondition: no registry.db in the temp vault"
        );

        let pool = open_ro_pool(&dir, 2)
            .await
            .expect("build pool (lazy create)");
        let result = pool.get().await;
        assert!(
            result.is_err(),
            "opening a missing read-only registry must error (no CREATE)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
