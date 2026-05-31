//! Single serializing, off-reactor writer queue for the `nark serve` daemon
//! (Phase 6, slice 6.1).
//!
//! Serve is the registry's single authoritative writer: it already holds the
//! advisory write lock for its whole lifetime (see [`super::run_until`]), and
//! SQLite's WAL mode permits exactly one writer. The natural shape is therefore
//! **one** read-write connection, fed by **one** task that applies jobs
//! sequentially. This module is that primitive:
//!
//! * [`Writer::open`] opens a single read-write [`rusqlite::Connection`] against
//!   `<vault>/registry.db` through the SAME inner path the CLI writer uses
//!   ([`crate::db::open_registry`]: WAL + foreign-keys + migrate + seed). It does
//!   **not** take a second copy of the advisory lock — serve already owns it; a
//!   second `open_registry_guarded` here would self-deadlock.
//! * The connection is owned by a dedicated **std thread** (not a tokio worker)
//!   that runs a blocking `recv` loop and applies each job in turn. Running the
//!   owning loop on its own thread is the simplest way to keep blocking SQLite
//!   entirely off the tokio reactor — no `spawn_blocking` per job, and no chance
//!   of parking an async worker.
//! * Jobs arrive over a **bounded** [`tokio::sync::mpsc`] channel (capacity
//!   [`WRITER_QUEUE_CAPACITY`]). [`Writer::submit`] boxes the caller's closure,
//!   pairs it with a [`tokio::sync::oneshot`] reply, and `try_send`s it: when the
//!   queue is full the submit returns a clean backpressure [`Err`] immediately
//!   rather than growing unbounded or hanging.
//!
//! This slice lands the queue as a standalone primitive; wiring the write
//! methods (`nark/write` / `nark/link` / `nark/delete`) onto it is a later slice.

use std::path::Path;
use std::thread::JoinHandle;

use anyhow::{Result, anyhow};
use rusqlite::Connection;
use tokio::sync::{mpsc, oneshot};

/// Bound on the in-flight write-job queue.
///
/// Sixty-four lets a healthy burst of writes queue behind the single serializing
/// connection without unbounded memory growth; a submit that arrives with the
/// queue full is refused (clean backpressure) rather than buffered forever. The
/// queue only ever drains as fast as the one writer connection can apply jobs,
/// so an unbounded channel here would let a runaway producer grow memory without
/// limit — hence a hard cap.
pub const WRITER_QUEUE_CAPACITY: usize = 64;

/// A unit of work for the writer thread: a boxed closure that borrows the
/// owning connection and is responsible for delivering its own typed result back
/// over the oneshot it captured. Type-erased (`R` lives inside the closure) so
/// heterogeneous jobs share one channel.
type Job = Box<dyn FnOnce(&Connection) + Send>;

/// A single serializing, off-reactor writer over one read-write connection.
///
/// Cloning is intentionally *not* derived: the [`Writer`] owns the send side of
/// the job channel and the owning thread's [`JoinHandle`]. The serve daemon holds
/// exactly one, shared via the `Arc<Ctx>` the listener already clones.
pub struct Writer {
    /// Send side of the bounded job channel, in an `Option` so [`Writer::drop`]
    /// can `take()` (and thus drop) the sole sender *before* joining the owning
    /// thread — dropping the last sender is what ends the thread's `recv` loop,
    /// so joining first would deadlock. `None` only transiently, during drop.
    tx: Option<mpsc::Sender<Job>>,
    /// Join handle for the owning thread. Held so the thread is joined on drop;
    /// it exits cleanly once the sole sender is dropped and the channel drains.
    handle: Option<JoinHandle<()>>,
}

impl Writer {
    /// Open the writer: one read-write connection on a dedicated owning thread.
    ///
    /// The connection is opened with [`crate::db::open_registry`] (the shared
    /// inner path: WAL + foreign-keys + migrate + seed), the SAME path the CLI
    /// writer uses, so the schema can never diverge. Serve already holds the
    /// advisory write lock, so this deliberately uses the *unlocked* open — it
    /// must NOT take a second lock (that would self-deadlock the daemon).
    ///
    /// The opened connection is moved onto a freshly spawned std thread that owns
    /// it for the writer's lifetime and runs [`run_loop`]. Open failures (a
    /// missing/corrupt registry, a failed migration) surface here synchronously,
    /// before any job can be submitted.
    pub fn open(vault_dir: &Path) -> Result<Self> {
        // Open on the calling thread so an open error is reported synchronously,
        // then hand the connection to the owning thread. `Connection` is `Send`,
        // so the move across the thread boundary is sound; from this point only
        // the owning thread ever touches it (single-writer discipline).
        let conn = crate::db::open_registry(vault_dir)
            .map_err(|e| anyhow!("opening writer registry connection: {e:#}"))?;

        let (tx, rx) = mpsc::channel::<Job>(WRITER_QUEUE_CAPACITY);

        let handle = std::thread::Builder::new()
            .name("nark-writer".to_string())
            .spawn(move || run_loop(conn, rx))
            .map_err(|e| anyhow!("spawning writer thread: {e}"))?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    /// Submit a write job and await its result, applied serially on the writer
    /// connection.
    ///
    /// `job` borrows the owning [`Connection`] and returns `Result<R>`; it runs on
    /// the writer thread, in submit order, never concurrently with another job —
    /// so callers get single-writer serialization for free. The reactor is never
    /// blocked: the blocking SQLite runs on the owning thread and this `await`s a
    /// oneshot.
    ///
    /// Backpressure is explicit. The bounded channel is probed with `try_send`:
    /// if the queue is **full** the submit returns a clean `Err` immediately
    /// rather than waiting, so a flood of writes can never grow memory unbounded
    /// or park the caller indefinitely. A `closed` channel (the writer thread is
    /// gone) is likewise an `Err`. On success this awaits the job's oneshot reply
    /// and returns whatever the closure produced.
    ///
    /// `cfg_attr(not(test), allow(dead_code))`: slice 6.1 lands the queue as a
    /// primitive exercised by the lib tests; the write methods that call `submit`
    /// in the binary land in a later slice, so the bin target sees it as unused
    /// until then. Mirrors `db::WriteHandle::conn` / `listener::serve_until`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn submit<R, F>(&self, job: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let (reply_tx, reply_rx) = oneshot::channel::<Result<R>>();

        // Box a closure that runs `job` against the connection and ships the
        // typed result back over the oneshot. The result type `R` is erased into
        // the boxed `Job` here, so heterogeneous jobs share the one channel. If
        // the receiver is already gone (caller dropped the future), the send is a
        // no-op — the work still ran, but nobody is waiting.
        let boxed: Job = Box::new(move |conn: &Connection| {
            let out = job(conn);
            let _ = reply_tx.send(out);
        });

        // try_send => clean, bounded backpressure. Full or closed both map to a
        // descriptive Err; neither hangs. `tx` is only `None` transiently during
        // drop, when no caller can hold a `&self` to reach here.
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("writer is shut down"))?;
        tx.try_send(boxed).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                anyhow!("writer queue is full ({WRITER_QUEUE_CAPACITY} jobs in flight); try again")
            }
            mpsc::error::TrySendError::Closed(_) => anyhow!("writer is shut down"),
        })?;

        // Await the job's reply. A dropped reply sender (the owning thread died
        // mid-job, e.g. a panic) surfaces as a recv error rather than a hang.
        reply_rx
            .await
            .map_err(|_| anyhow!("writer dropped the job before replying"))?
    }
}

impl Drop for Writer {
    /// Shut the writer down deterministically: drop the sole sender so the owning
    /// thread's `recv` loop ends, then join the thread so the read-write
    /// connection is closed before the caller (e.g. a test's vault dir) is torn
    /// down. The join is best-effort — a panicked owning thread must not panic
    /// the dropper.
    fn drop(&mut self) {
        // Drop the sole sender FIRST. tokio's channel closes when its last sender
        // is dropped, which is what makes `blocking_recv` return `None` and ends
        // `run_loop`. Joining before this would deadlock (the loop would never
        // see the channel close). `take()` drops the `Sender` immediately.
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The owning thread's loop: pull jobs off the bounded channel and apply each to
/// the single connection, in order, until the channel closes.
///
/// Uses [`mpsc::Receiver::blocking_recv`] so this thread genuinely blocks waiting
/// for work — appropriate because it is a dedicated std thread, NOT a tokio
/// worker, so blocking here never starves the reactor. When the last [`Writer`]
/// (the sole sender) is dropped, `blocking_recv` returns `None` and the loop
/// ends, closing the connection on return.
fn run_loop(conn: Connection, mut rx: mpsc::Receiver<Job>) {
    while let Some(job) = rx.blocking_recv() {
        // Each job is fully responsible for sending its own result back over the
        // oneshot it captured; we just hand it the connection. Jobs run strictly
        // one at a time on this single thread => single-writer serialization.
        job(&conn);
    }
    // Channel closed: all senders dropped. Returning drops `conn`, closing the
    // read-write SQLite connection cleanly.
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// Create a fresh, unique temp vault whose `registry.db` is created, migrated
    /// and seeded by the writer open. Matches the repo's
    /// `std::env::temp_dir()` + pid + uuid convention (no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-serve-writer-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        // Pre-create + migrate + seed the registry so the writer opens an existing
        // db (open_registry would also create it, but this mirrors serve, where
        // the vault already exists). Drop immediately so no handle lingers.
        let conn = crate::db::open_registry(&dir).expect("seed registry");
        drop(conn);
        dir
    }

    /// A submitted SELECT job runs on the writer connection and returns Ok with
    /// the queried value.
    #[tokio::test]
    async fn select_job_returns_ok() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let agent_count: i64 = writer
            .submit(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM agents WHERE agent_id = ?1",
                    rusqlite::params![crate::db::DEFAULT_AGENT_ID],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .await
            .expect("select job completes");
        assert_eq!(
            agent_count, 1,
            "the seeded default agent must be visible through the writer connection"
        );

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A probe WRITE job succeeds: the writer connection is genuinely read-write
    /// (not the read-only pool). It inserts a row and reads it back.
    #[tokio::test]
    async fn probe_write_job_succeeds_conn_is_rw() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        // INSERT a fresh agent through the writer connection: this would fail with
        // SQLITE_READONLY on the read-only pool, so success proves RW.
        let inserted: usize = writer
            .submit(|conn| {
                conn.execute(
                    "INSERT INTO agents (agent_id, name, namespace, role, registered_at)
                     VALUES ('writer-probe', 'writer-probe', 'probe-ns', 'agent', '2026-05-31T00:00:00Z')",
                    [],
                )
                .map_err(Into::into)
            })
            .await
            .expect("probe write job completes");
        assert_eq!(inserted, 1, "the probe insert must affect exactly one row");

        // Read it back through the same writer to confirm it persisted.
        let count: i64 = writer
            .submit(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM agents WHERE agent_id = 'writer-probe'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .await
            .expect("read-back job completes");
        assert_eq!(count, 1, "the probe-written row must be visible afterwards");

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two concurrent submits both complete (serialized through the single
    /// connection, no deadlock) within a bounded budget.
    #[tokio::test]
    async fn two_concurrent_submits_both_complete() {
        let dir = fresh_vault();
        let writer = Arc::new(Writer::open(&dir).expect("open writer"));

        let w1 = Arc::clone(&writer);
        let a = tokio::spawn(async move {
            w1.submit(|conn| {
                conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                    .map_err(Into::into)
            })
            .await
        });
        let w2 = Arc::clone(&writer);
        let b = tokio::spawn(async move {
            w2.submit(|conn| {
                conn.query_row("SELECT 2", [], |row| row.get::<_, i64>(0))
                    .map_err(Into::into)
            })
            .await
        });

        let (ra, rb) = tokio::time::timeout(Duration::from_secs(5), async { (a.await, b.await) })
            .await
            .expect("both concurrent submits must complete within the budget (no deadlock)");

        assert_eq!(
            ra.expect("task a join").expect("submit a ok"),
            1,
            "first concurrent submit returns its value"
        );
        assert_eq!(
            rb.expect("task b join").expect("submit b ok"),
            2,
            "second concurrent submit returns its value"
        );

        let writer = Arc::try_unwrap(writer)
            .ok()
            .expect("sole owner at teardown");
        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Filling the bounded queue forces clean backpressure: once the queue is
    /// saturated by an in-flight slow job plus `WRITER_QUEUE_CAPACITY` queued
    /// jobs, a further submit returns an `Err` promptly (bounded time) rather than
    /// hanging. After the backlog drains, submits succeed again.
    #[tokio::test]
    async fn full_queue_backpressures_with_err_not_hang() {
        let dir = fresh_vault();
        let writer = Arc::new(Writer::open(&dir).expect("open writer"));

        // A barrier the in-flight job blocks on, so it occupies the writer thread
        // and lets jobs pile up in the bounded channel behind it.
        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let gate_rx = std::sync::Mutex::new(gate_rx);

        // Job 0: occupies the writer thread (blocks on the gate). We do NOT await
        // its reply yet — we just want it running so subsequent sends queue up.
        let w0 = Arc::clone(&writer);
        let inflight = tokio::spawn(async move {
            w0.submit(move |_conn| {
                // Block the single writer thread until the test opens the gate.
                let _ = gate_rx.lock().expect("gate lock").recv();
                Ok::<_, anyhow::Error>(())
            })
            .await
        });

        // Wait until the in-flight job is actually executing on the writer thread
        // (so the channel is empty and ready to be filled to capacity). We detect
        // this indirectly by giving it a brief moment; the gate guarantees it
        // cannot finish, so the writer thread is occupied.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Now fire submits that will queue behind the blocked job. Each of these
        // futures parks awaiting its oneshot; the SENDS land in the bounded
        // channel. Fill it to capacity.
        let mut queued = Vec::with_capacity(WRITER_QUEUE_CAPACITY);
        for _ in 0..WRITER_QUEUE_CAPACITY {
            let w = Arc::clone(&writer);
            queued.push(tokio::spawn(async move {
                w.submit(|conn| {
                    conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                        .map_err(Into::into)
                })
                .await
            }));
        }

        // Give the queued sends time to land in the channel (the consumer is
        // blocked on the gate, so they cannot drain).
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The queue is now full (capacity jobs buffered + one executing). A direct
        // submit must return an Err promptly — NOT hang.
        let backpressured = tokio::time::timeout(
            Duration::from_secs(2),
            writer.submit(|conn| {
                conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                    .map_err(Into::into)
            }),
        )
        .await
        .expect("a full-queue submit must return promptly, not hang");
        let err = backpressured.expect_err("a full-queue submit must be a backpressure Err");
        assert!(
            err.to_string().contains("full"),
            "backpressure error should name the full queue, got: {err}"
        );

        // Open the gate: the in-flight job finishes, the backlog drains, and the
        // queued + in-flight submits all complete.
        gate_tx.send(()).expect("open the gate");
        tokio::time::timeout(Duration::from_secs(10), inflight)
            .await
            .expect("in-flight job completes once the gate opens")
            .expect("in-flight task join")
            .expect("in-flight submit ok");
        for (i, q) in queued.into_iter().enumerate() {
            let v = tokio::time::timeout(Duration::from_secs(10), q)
                .await
                .unwrap_or_else(|_| panic!("queued submit {i} drains after the gate opens"))
                .expect("queued task join")
                .expect("queued submit ok");
            assert_eq!(v, 1);
        }

        // And a fresh submit succeeds again now that the queue has drained.
        let after: i64 = writer
            .submit(|conn| {
                conn.query_row("SELECT 7", [], |row| row.get::<_, i64>(0))
                    .map_err(Into::into)
            })
            .await
            .expect("submit succeeds again after the backlog drains");
        assert_eq!(after, 7);

        let writer = Arc::try_unwrap(writer)
            .ok()
            .expect("sole owner at teardown");
        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
