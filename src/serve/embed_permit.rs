//! Bounded embedding-worker permits for the `nark serve` daemon (Phase 3.5,
//! slice 3.5.3 — the 2B primitive).
//!
//! The embedding-bearing read methods (`search` / `orient`) run ONNX inference,
//! which is **CPU-blocking** and unbounded in fan-out: every concurrent
//! `search`/`orient` would otherwise spin up its own inference at once and
//! saturate the box. This module lands the bound: a [`tokio::sync::Semaphore`]
//! (held in the router's `Ctx` as an `Arc<Semaphore>`) caps how many embedding
//! computations run concurrently, and [`with_embed_permit`] is the RAII gate
//! every embedding computation must pass through.
//!
//! Two things are deliberately kept separate here:
//!
//! * the **permit** bounds embedding *concurrency* (this module), and
//! * the **deadpool connection** bounds registry *I/O* (`super::dpool`).
//!
//! Embedding compute is **not** a deadpool connection op — it does not touch the
//! registry — so [`with_embed_permit`] runs the closure on
//! [`tokio::task::spawn_blocking`] (a blocking-pool thread), never on a pooled
//! SQLite connection's `interact` thread. `search`'s `build_cosine_context`
//! (`super::methods_read`) acquires a permit for the ONNX inference step and a
//! separate deadpool connection for the query step, so a connection is never
//! held across inference.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Semaphore;

/// Default number of concurrent embedding computations permitted per daemon.
///
/// Two keeps a steady stream of `search`/`orient` requests making progress
/// without letting an unbounded fan-out of ONNX inferences saturate the CPU
/// (each inference is a blocking, compute-heavy job). Requests beyond this
/// back-pressure on [`with_embed_permit`]'s `acquire` until a permit frees.
pub const DEFAULT_EMBED_PERMITS: usize = 2;

/// Build the embedding-worker semaphore with [`DEFAULT_EMBED_PERMITS`] permits.
///
/// Returned as an `Arc<Semaphore>` so the router's `Ctx` can hold one and hand
/// `&Semaphore` borrows to [`with_embed_permit`] across many concurrent
/// connections.
pub fn default_embed_semaphore() -> Arc<Semaphore> {
    Arc::new(Semaphore::new(DEFAULT_EMBED_PERMITS))
}

/// Run CPU-blocking embedding work `f` under a single embedding-worker permit.
///
/// Acquires one permit from `sem` (awaiting — i.e. back-pressuring — when all
/// permits are in use), then runs `f` on [`tokio::task::spawn_blocking`] because
/// ONNX inference is compute-blocking and must not run on an async worker. The
/// permit is an [`tokio::sync::OwnedSemaphorePermit`] held for the lifetime of
/// the blocking task and dropped (RAII) the moment the task finishes — whether
/// `f` returns `Ok`, returns `Err`, **or panics** — so a later acquire always
/// succeeds once the slot frees. A panic in `f` surfaces as an `Err` (the join
/// error) rather than tearing anything down.
///
/// Note this takes `&Semaphore` but acquires an *owned* permit: it clones the
/// `Arc` the caller's `Ctx` holds via [`Semaphore::acquire_owned`], so the
/// permit can move into the `'static` blocking closure without borrowing `sem`
/// for the task's lifetime.
pub async fn with_embed_permit<F, R>(sem: &Arc<Semaphore>, f: F) -> Result<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    // Owned permit so it can move into the 'static blocking closure; dropped
    // (RAII) when the closure's task completes — on Ok, Err, or panic.
    let permit = Arc::clone(sem)
        .acquire_owned()
        .await
        .map_err(|e| anyhow::anyhow!("embedding semaphore closed: {e}"))?;

    tokio::task::spawn_blocking(move || {
        let _permit = permit; // held for the duration of the blocking work
        f()
    })
    .await
    .map_err(|e| anyhow::anyhow!("embedding worker task join: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Semaphore;

    /// Acquiring all permits blocks a further acquisition until one frees, while
    /// a path that needs no permit proceeds unimpeded.
    ///
    /// With a 2-permit semaphore, two `with_embed_permit` calls park inside their
    /// blocking closures (each on a synchronous channel `recv` that only returns
    /// when signalled), holding both permits. A third `with_embed_permit` must NOT
    /// resolve within a short timeout (no permit is free). A non-permit future
    /// resolves promptly in the meantime. Signalling the two holders frees the
    /// permits, and the third call then completes.
    #[tokio::test]
    async fn all_permits_held_blocks_third_acquire_but_not_a_non_permit_path() {
        let sem = Arc::new(Semaphore::new(2));

        // Two synchronous channels: each held closure blocks on `recv()` (a
        // genuine thread-blocking wait, suitable inside spawn_blocking) until the
        // test sends on the matching sender, returning the permit.
        let (tx1, rx1) = std::sync::mpsc::channel::<()>();
        let (tx2, rx2) = std::sync::mpsc::channel::<()>();

        // Hold the two permits: each closure blocks until its channel is signalled.
        let h1 = tokio::spawn({
            let sem = Arc::clone(&sem);
            async move {
                with_embed_permit(&sem, move || {
                    // Block the blocking thread until signalled (RAII permit held).
                    let _ = rx1.recv();
                })
                .await
                .expect("holder 1 completes")
            }
        });
        let h2 = tokio::spawn({
            let sem = Arc::clone(&sem);
            async move {
                with_embed_permit(&sem, move || {
                    let _ = rx2.recv();
                })
                .await
                .expect("holder 2 completes")
            }
        });

        // Wait until both permits are actually taken (the spawned tasks have
        // reached the blocking closure). available_permits drops to 0.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while sem.available_permits() > 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "both permits should be taken by the two holders"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // A third acquire must NOT resolve while both permits are held.
        let third = tokio::spawn({
            let sem = Arc::clone(&sem);
            async move { with_embed_permit(&sem, || 7u32).await.expect("third value") }
        });

        // Give the third task a window to (fail to) acquire, then assert it has
        // not finished — no permit is free for it.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            !third.is_finished(),
            "a third acquire must not resolve while both permits are held"
        );

        // Meanwhile a non-permit path resolves promptly.
        let non_permit = tokio::time::timeout(Duration::from_millis(100), async { 42u32 })
            .await
            .expect("a non-permit future resolves promptly while permits are held");
        assert_eq!(non_permit, 42);

        // Signal the two holders -> permits return -> third acquire completes.
        tx1.send(()).expect("signal holder 1");
        tx2.send(()).expect("signal holder 2");
        h1.await.expect("holder 1 join");
        h2.await.expect("holder 2 join");

        let got = tokio::time::timeout(Duration::from_secs(5), third)
            .await
            .expect("third acquire completes once permits free")
            .expect("third task join");
        assert_eq!(got, 7);
    }

    /// The permit is released even when the closure returns `Err` or panics, so a
    /// later acquire on a 1-permit semaphore succeeds.
    #[tokio::test]
    async fn permit_released_on_err_and_on_panic() {
        let sem = Arc::new(Semaphore::new(1));

        // Closure returns an Err-carrying value: with_embed_permit itself is Ok
        // (the work ran), and the permit drops when the task ends.
        let err_result: Result<core::result::Result<(), String>> =
            with_embed_permit(&sem, || Err("boom".to_string())).await;
        assert!(
            matches!(err_result, Ok(Err(_))),
            "the closure's Err is carried through; the call itself succeeds"
        );
        assert_eq!(
            sem.available_permits(),
            1,
            "permit must be released after an Err-returning closure"
        );

        // Closure panics: with_embed_permit returns Err (join error), and the
        // permit is still released (RAII on the blocking task unwinding).
        let panicked: Result<()> = with_embed_permit(&sem, || panic!("kaboom")).await;
        assert!(
            panicked.is_err(),
            "a panicking closure surfaces as a join Err, not a torn-down runtime"
        );
        // available_permits may take a beat to reflect the dropped permit after
        // the panicking task unwinds; poll briefly.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while sem.available_permits() < 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "permit must be released even after the closure panics"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // And a later acquire succeeds (permit is free again).
        let after = with_embed_permit(&sem, || 99u32)
            .await
            .expect("acquire after panic should succeed");
        assert_eq!(after, 99);
    }
}
