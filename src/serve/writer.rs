//! Single serializing, off-reactor writer queue for the `nark serve` daemon
//! (Phase 6, slices 6.1–6.3).
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
//! ## Idempotency-key dedup (slice 6.3)
//!
//! [`Writer::submit_idempotent`] adds at-most-once application keyed by an
//! optional caller-supplied `idempotency_key`. The dedup store lives **inside**
//! the owning thread's loop ([`run_loop`]) — it is owned by exactly the one
//! thread that serializes every write, so the check-then-apply-then-cache is
//! atomic *by construction*: no lock, no race, no second writer can interleave.
//! On a submit with a key the thread:
//!
//! 1. looks the key up; on a live (un-expired) hit it returns the **cached**
//!    `serde_json::Value` result WITHOUT re-running the closure (so a retried
//!    write creates no second version);
//! 2. otherwise runs the closure against the connection, and — only on `Ok` —
//!    caches the result under the key before replying. A failed job is **not**
//!    cached, so a transient error never poisons the key.
//!
//! An **absent** key is never deduped: the closure always applies. The cached
//! value type is `serde_json::Value` because every write method (`nark/write` /
//! `nark/link` / `nark/delete`) returns that shape.
//!
//! **Key scope is GLOBAL** across all write methods sharing this writer: a key
//! is matched purely on its string, regardless of method. Callers that need
//! per-method isolation must namespace the key themselves (e.g. `write:<uuid>`).
//! The store is **bounded** — at most [`DEDUP_CAPACITY`] entries with a
//! [`DEDUP_TTL`] time bound — and evicts the oldest insert when full, so it can
//! never grow without limit (an evicted or expired key simply re-applies).

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use rusqlite::Connection;
use serde_json::Value;
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

/// Maximum number of cached idempotency-key results held by the writer's dedup
/// store. The store is keyed by `idempotency_key` and bounded both by this entry
/// count and by [`DEDUP_TTL`]; when a fresh insert would exceed this many live
/// entries the oldest insert is evicted (FIFO). Small on purpose — dedup only
/// needs to cover a client's retry window, not the whole history of writes — so
/// the cache cannot grow without limit no matter how many distinct keys arrive.
pub const DEDUP_CAPACITY: usize = 256;

/// Time bound on a cached idempotency-key result. A hit older than this is
/// treated as a miss (and re-applied), so the dedup window is the retry window,
/// not forever. Twenty-four hours comfortably covers an agent's reconnect/retry
/// loop while keeping the store from holding stale results indefinitely.
const DEDUP_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// A unit of work for the writer thread: a boxed closure that borrows the
/// owning connection and is responsible for delivering its own typed result back
/// over the oneshot it captured. Type-erased (`R` lives inside the closure) so
/// heterogeneous jobs share one channel.
type Job = Box<dyn FnOnce(&Connection) + Send>;

/// The mutation an [`IdempotentJob`] applies: a boxed closure that borrows the
/// owning connection and produces the JSON result every write method returns.
/// Boxed (a `type` alias, not an inline `dyn`) so heterogeneous write closures
/// share one channel without a `clippy::type_complexity` thicket at the call.
type IdempotentRun = Box<dyn FnOnce(&Connection) -> Result<Value> + Send>;

/// A dedup-aware unit of work for the writer thread.
///
/// Unlike a plain [`Job`] (which erases its result type and ships its own reply),
/// an idempotent job hands the *thread* its optional key, its `Value`-producing
/// closure, and the reply channel, so the owning thread — the single
/// serialization point — performs the check / apply / cache atomically against
/// the dedup store it alone owns.
struct IdempotentJob {
    /// The caller's `idempotency_key`, or `None` to never dedup (always apply).
    key: Option<String>,
    /// The mutation to apply on the connection, producing the JSON result that
    /// write methods return. Boxed so heterogeneous closures share one channel.
    run: IdempotentRun,
    /// Where the (cached or freshly applied) result is delivered.
    reply: oneshot::Sender<Result<Value>>,
}

/// What travels over the writer's bounded channel: either a type-erased plain
/// [`Job`] (the [`Writer::submit`] path) or a dedup-aware [`IdempotentJob`] (the
/// [`Writer::submit_idempotent`] path). One channel keeps strict submit-order
/// serialization across both kinds.
enum Msg {
    /// A plain job that ships its own typed reply (no dedup).
    Plain(Job),
    /// A dedup-aware job the owning thread resolves against its dedup store.
    Idempotent(IdempotentJob),
}

/// The writer thread's idempotency-key dedup store: a bounded, time-bounded map
/// from `idempotency_key` to the cached JSON result of the write that key
/// committed.
///
/// Owned exclusively by [`run_loop`] (the single writer thread), so every
/// `get`/`insert` is serialized with the writes themselves — the dedup decision
/// is atomic w.r.t. the writer with no locking. Bounded two ways: at most
/// [`DEDUP_CAPACITY`] entries (oldest-insert eviction) and a [`DEDUP_TTL`] age
/// bound (a stale hit is a miss). An `order` queue records insertion order for
/// O(1) FIFO eviction.
struct DedupStore {
    entries: HashMap<String, (Instant, Value)>,
    order: VecDeque<String>,
}

impl DedupStore {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Return the cached result for `key` if present and still within
    /// [`DEDUP_TTL`]. An expired entry is removed and reported as a miss, so a
    /// stale key re-applies rather than serving an aged result.
    fn get(&mut self, key: &str) -> Option<Value> {
        match self.entries.get(key) {
            Some((stored_at, value)) if stored_at.elapsed() < DEDUP_TTL => Some(value.clone()),
            Some(_) => {
                // Expired: drop it so the key re-applies and the slot frees up.
                self.entries.remove(key);
                self.order.retain(|k| k != key);
                None
            }
            None => None,
        }
    }

    /// Cache `value` under `key`, evicting the oldest live entry first if the
    /// store is at capacity. Re-inserting an existing key refreshes its value and
    /// its position so it is treated as newly inserted.
    fn insert(&mut self, key: String, value: Value) {
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else {
            while self.entries.len() >= DEDUP_CAPACITY {
                // Pop the oldest inserted key; skip any already gone (e.g. expired
                // out from under us) so the loop always makes progress.
                match self.order.pop_front() {
                    Some(oldest) => {
                        if self.entries.remove(&oldest).is_some() {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, (Instant::now(), value));
    }
}

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
    tx: Option<mpsc::Sender<Msg>>,
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

        let (tx, rx) = mpsc::channel::<Msg>(WRITER_QUEUE_CAPACITY);

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

        self.enqueue(Msg::Plain(boxed))?;

        // Await the job's reply. A dropped reply sender (the owning thread died
        // mid-job, e.g. a panic) surfaces as a recv error rather than a hang.
        reply_rx
            .await
            .map_err(|_| anyhow!("writer dropped the job before replying"))?
    }

    /// Submit a write job with optional idempotency-key dedup, applied serially
    /// on the writer connection, and await its `serde_json::Value` result.
    ///
    /// When `key` is `Some`, the writer thread (the single serialization point)
    /// performs an atomic check / apply / cache against its own dedup store:
    ///
    /// * a live cached result for the key is returned WITHOUT re-running `run`
    ///   (so a retried write commits no second version);
    /// * otherwise `run` is applied to the connection and, only on `Ok`, the
    ///   result is cached under the key before replying. A failed job is not
    ///   cached.
    ///
    /// When `key` is `None`, `run` always applies (no dedup). The dedup store is
    /// bounded ([`DEDUP_CAPACITY`] entries, [`DEDUP_TTL`] age) and the key scope
    /// is **global** across write methods — see the module docs. Backpressure is
    /// identical to [`Writer::submit`]: a full queue is a clean `Err`, never a
    /// hang.
    ///
    /// `cfg_attr(not(test), allow(dead_code))`: the write methods that call this
    /// in the binary land alongside their `idempotency_key` params; until every
    /// caller is wired the bin target may see it as unused. Mirrors
    /// [`Writer::submit`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn submit_idempotent<F>(&self, key: Option<String>, run: F) -> Result<Value>
    where
        F: FnOnce(&Connection) -> Result<Value> + Send + 'static,
    {
        let (reply, reply_rx) = oneshot::channel::<Result<Value>>();

        self.enqueue(Msg::Idempotent(IdempotentJob {
            key,
            run: Box::new(run),
            reply,
        }))?;

        reply_rx
            .await
            .map_err(|_| anyhow!("writer dropped the job before replying"))?
    }

    /// Hand a [`Msg`] to the owning thread with clean, bounded backpressure.
    ///
    /// `try_send` so a **full** queue returns a descriptive `Err` immediately
    /// rather than buffering unbounded or hanging; a **closed** channel (the
    /// writer thread is gone) likewise errors. `tx` is only `None` transiently
    /// during drop, when no caller can hold a `&self` to reach here.
    fn enqueue(&self, msg: Msg) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("writer is shut down"))?;
        tx.try_send(msg).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                anyhow!("writer queue is full ({WRITER_QUEUE_CAPACITY} jobs in flight); try again")
            }
            mpsc::error::TrySendError::Closed(_) => anyhow!("writer is shut down"),
        })
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

/// The owning thread's loop: pull messages off the bounded channel and apply
/// each to the single connection, in order, until the channel closes.
///
/// Uses [`mpsc::Receiver::blocking_recv`] so this thread genuinely blocks waiting
/// for work — appropriate because it is a dedicated std thread, NOT a tokio
/// worker, so blocking here never starves the reactor. When the last [`Writer`]
/// (the sole sender) is dropped, `blocking_recv` returns `None` and the loop
/// ends, closing the connection on return.
///
/// The idempotency-key [`DedupStore`] is owned here, by this one thread, so the
/// check / apply / cache for an [`Msg::Idempotent`] job is atomic with the writes
/// themselves — no lock, no second writer, no race.
fn run_loop(conn: Connection, mut rx: mpsc::Receiver<Msg>) {
    let mut dedup = DedupStore::new();
    while let Some(msg) = rx.blocking_recv() {
        // Messages run strictly one at a time on this single thread =>
        // single-writer serialization (and serialized dedup-store access).
        match msg {
            // A plain job is fully responsible for sending its own result back
            // over the oneshot it captured; we just hand it the connection.
            Msg::Plain(job) => job(&conn),
            Msg::Idempotent(job) => apply_idempotent(&conn, &mut dedup, job),
        }
    }
    // Channel closed: all senders dropped. Returning drops `conn`, closing the
    // read-write SQLite connection cleanly.
}

/// Resolve one [`IdempotentJob`] against the writer's [`DedupStore`].
///
/// With a key: a live cached hit is returned WITHOUT running the closure; a miss
/// runs the closure and, only on `Ok`, caches the result before replying (a
/// failed job is never cached, so a transient error does not poison the key).
/// Without a key the closure always runs (no dedup). Either way the result is
/// shipped over the job's oneshot; a dropped receiver (caller gone) is a no-op.
fn apply_idempotent(conn: &Connection, dedup: &mut DedupStore, job: IdempotentJob) {
    let IdempotentJob { key, run, reply } = job;

    let result = match key {
        Some(key) => match dedup.get(&key) {
            // Cache hit: do NOT re-apply; return the stored result verbatim.
            Some(cached) => Ok(cached),
            // Miss: apply, then cache only a successful result under the key.
            None => {
                let out = run(conn);
                if let Ok(ref value) = out {
                    dedup.insert(key, value.clone());
                }
                out
            }
        },
        // No key: never dedup — always apply.
        None => run(conn),
    };

    let _ = reply.send(result);
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

    /// Two `submit_idempotent` calls with the SAME key apply the job exactly
    /// once: the second returns the cached result of the first WITHOUT
    /// re-running the closure. A shared counter proves the closure ran once.
    #[tokio::test]
    async fn same_key_applies_once_and_caches() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let r1 = runs.clone();
        let first: serde_json::Value = writer
            .submit_idempotent(Some("k1".to_string()), move |_conn| {
                let n = r1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "applied_on_run": n }))
            })
            .await
            .expect("first idempotent submit ok");

        let r2 = runs.clone();
        let second: serde_json::Value = writer
            .submit_idempotent(Some("k1".to_string()), move |_conn| {
                // This closure MUST NOT run on a cache hit.
                r2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "applied_on_run": 999 }))
            })
            .await
            .expect("second idempotent submit ok (served from cache)");

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the closure must run exactly once across two same-key submits"
        );
        assert_eq!(
            first, second,
            "the repeat key must return the identical cached result"
        );
        assert_eq!(first["applied_on_run"], 0);

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two DIFFERENT keys both apply: each runs the closure and returns its own
    /// result. No cross-key dedup.
    #[tokio::test]
    async fn different_keys_both_apply() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let ra = runs.clone();
        let a: serde_json::Value = writer
            .submit_idempotent(Some("ka".to_string()), move |_conn| {
                ra.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "k": "a" }))
            })
            .await
            .expect("submit a ok");
        let rb = runs.clone();
        let b: serde_json::Value = writer
            .submit_idempotent(Some("kb".to_string()), move |_conn| {
                rb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "k": "b" }))
            })
            .await
            .expect("submit b ok");

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "two distinct keys must each apply the closure"
        );
        assert_eq!(a["k"], "a");
        assert_eq!(b["k"], "b");

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An absent key is NEVER deduped: each submit applies the closure, even when
    /// the produced result is identical.
    #[tokio::test]
    async fn absent_key_never_dedups() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for _ in 0..2 {
            let r = runs.clone();
            let _: serde_json::Value = writer
                .submit_idempotent(None, move |_conn| {
                    r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(serde_json::json!({ "x": 1 }))
                })
                .await
                .expect("keyless submit ok");
        }

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a keyless submit must always apply (never dedup)"
        );

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A job that fails is NOT cached: a retry with the same key re-applies the
    /// closure (so a transient failure does not poison the key forever).
    #[tokio::test]
    async fn failed_job_is_not_cached() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let r1 = runs.clone();
        let first = writer
            .submit_idempotent(Some("kf".to_string()), move |_conn| {
                r1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err::<serde_json::Value, _>(anyhow!("boom"))
            })
            .await;
        assert!(first.is_err(), "the first submit must surface the error");

        let r2 = runs.clone();
        let second: serde_json::Value = writer
            .submit_idempotent(Some("kf".to_string()), move |_conn| {
                r2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "ok": true }))
            })
            .await
            .expect("retry after a failure applies");

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a failed job must not be cached: the same key re-applies on retry"
        );
        assert_eq!(second["ok"], true);

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The dedup store is bounded: inserting more than `DEDUP_CAPACITY` distinct
    /// keys evicts the oldest, so an evicted key re-applies on its next submit
    /// (the cache can never grow without limit).
    #[tokio::test]
    async fn dedup_store_is_bounded_evicts_oldest() {
        let dir = fresh_vault();
        let writer = Writer::open(&dir).expect("open writer");

        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // The very first key, which we will later evict by overflowing the cache.
        let r0 = runs.clone();
        let _: serde_json::Value = writer
            .submit_idempotent(Some("evict-me".to_string()), move |_conn| {
                r0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "first": true }))
            })
            .await
            .expect("seed key ok");

        // Fill the cache past capacity with distinct keys, evicting the oldest.
        for i in 0..DEDUP_CAPACITY {
            let r = runs.clone();
            let _: serde_json::Value = writer
                .submit_idempotent(Some(format!("fill-{i}")), move |_conn| {
                    r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(serde_json::json!({ "fill": i }))
                })
                .await
                .expect("fill key ok");
        }

        // "evict-me" should have fallen out of the bounded store, so re-submitting
        // it re-applies the closure (a second run) rather than serving a cache hit.
        let before = runs.load(std::sync::atomic::Ordering::SeqCst);
        let r1 = runs.clone();
        let _: serde_json::Value = writer
            .submit_idempotent(Some("evict-me".to_string()), move |_conn| {
                r1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({ "first": "again" }))
            })
            .await
            .expect("evicted key re-applies");
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            before + 1,
            "an evicted key must re-apply (the bounded store dropped it)"
        );

        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
