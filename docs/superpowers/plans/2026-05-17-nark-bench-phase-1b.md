# nark Bench — Phase 1b (Vector Baseline + Embeddings-Enabled nark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert nark into a binary+library hybrid, add a pure-Rust vector baseline adapter in `bench/`, wire the existing nark adapter to actually use embeddings, apply three deferred code-review fixes (microsecond latency / schema bump, query-count regression check, filename sanitization), and regenerate the locked-in baselines.

**Architecture:** nark grows a `lib.rs` that re-exports its existing modules; the bench crate gains a path dependency on `nark` and a new `model_cache` module that downloads the ONNX embedding model once into `~/.cache/nark-bench/models/` and hard-links it into per-adapter workdirs. A new `VectorAdapter` uses nark's `OnnxProvider` in-process for embedding and stores vectors in a `HashMap`. The existing `NarkAdapter` stages model files in `setup()` so `nark write` produces embeddings inline.

**Tech Stack:** Rust 2024; ort (ONNX Runtime, already a nark dep); tokenizers (already a nark dep); dirs crate; existing `bench/` workspace member; the `nomic-embed-text-v1.5` model (768-dim L2-normalized embeddings, ~270MB download).

**Spec reference:** `docs/superpowers/specs/2026-05-16-phase-1b-design.md`

**Branch:** `feat/bench-phase-1a` (continuation; Phase 1a already shipped on this branch — these commits build on top).

---

## File structure

**Files to create:**

| Path | Responsibility |
|---|---|
| `nark/src/lib.rs` | Library entry point; declares `pub mod` for each module the bench needs |
| `bench/src/model_cache.rs` | Locate/populate/stage the shared ONNX model cache |
| `bench/src/adapters/vector.rs` | In-process vector baseline using nark's `OnnxProvider` + brute-force cosine |

**Files to modify:**

| Path | Change |
|---|---|
| `nark/Cargo.toml` | Add `[lib]` target |
| `nark/src/embed/mod.rs:159` | Add `pub` to `onnx_dylib_name()` |
| `nark/src/embed/download.rs` | Add `pub fn install_into(target: &Path) -> Result<()>` |
| `bench/Cargo.toml` | Add `nark = { path = ".." }` and `dirs = "6"` |
| `bench/src/protocol.rs` | Rename `latency_ms` → `latency_us` in both metric structs |
| `bench/src/result.rs` | Rename `latency_p50_ms/p99_ms` → `latency_p50_us/p99_us`; bump `schema_version` to "2"; add filename sanitization in `write_to_disk` |
| `bench/src/tasks/ir.rs` | Switch all `as_millis()` → `as_micros()` |
| `bench/src/adapters/fts5.rs` | Switch `as_millis()` → `as_micros()` in write/search |
| `bench/src/adapters/nark.rs` | Add `model_cache` field + `stage_into` call in `setup()`; switch `as_millis()` → `as_micros()`; constructor takes `Option<PathBuf>` |
| `bench/src/adapters/mod.rs` | Register `"vector"`; factory now takes `Option<&Path>` model cache |
| `bench/src/main.rs` | Add `mod model_cache;`; call `cache_root()` + `ensure_ready()` before adapter loop; pass cache through factory |
| `bench/tests/smoke.rs` | Run `fts5,nark,vector`; assert schema_version "2"; assert zero errors for all three |
| `bench/scripts/regression-check.sh` | Add query-count check + schema-version warning |
| `.github/workflows/bench-pr.yml` | Add `actions/cache@v4` step for the model files |
| `bench/results/main/ir-fts5-default.json` | Regenerated (schema v2; field renames) |
| `bench/results/main/ir-nark-default.json` | Regenerated (schema v2; now with embeddings) |
| `bench/results/main/ir-vector-default.json` | NEW — vector baseline |

---

## Task 1: nark binary+library hybrid

**Files:**
- Modify: `nark/Cargo.toml`
- Create: `nark/src/lib.rs`
- Modify: `bench/Cargo.toml`

The canonical Rust pattern: `lib.rs` independently declares `pub mod X;` for each module the library exposes. `main.rs` is unchanged — its existing `mod X;` declarations stay. Both compilation units pull in the module sources separately. Do NOT switch `main.rs` to `use nark::X` — that creates double-init conflicts with `ort::init_from`.

- [ ] **Step 1.1: Add `[lib]` target to `nark/Cargo.toml`**

Read `nark/Cargo.toml` first to find the right insertion point (immediately after the `[package]` block, before `[dependencies]`).

Insert:

```toml
[lib]
name = "nark"
path = "src/lib.rs"
```

- [ ] **Step 1.2: Create `nark/src/lib.rs`**

```rust
//! nark library — re-exports modules for cross-crate consumers (currently
//! the in-workspace `bench/` crate). `main.rs` is the primary entry point
//! and does not depend on this lib; the lib exists so other crates can
//! `use nark::embed::OnnxProvider` etc. without forking the code.

pub mod cli;
pub mod config;
pub mod db;
pub mod embed;
pub mod registry;
pub mod types;
pub mod vault;
```

- [ ] **Step 1.3: Verify nark still builds as both binary and library**

```bash
cargo build -p nark
cargo build -p nark --lib
```

Expected: both succeed. Pre-existing dead_code warnings are OK. New warnings should NOT appear from this change alone.

- [ ] **Step 1.4: Add nark + dirs to `bench/Cargo.toml`**

Read `bench/Cargo.toml` and add these two entries to the `[dependencies]` table:

```toml
nark = { path = ".." }
dirs = "6"
```

- [ ] **Step 1.5: Verify bench still builds**

```bash
cargo build -p nark-bench
```

Expected: completes with no errors. The newly-imported `nark` and `dirs` crates aren't used yet, so the build may warn about unused deps — that's fine, they're used in Task 4.

- [ ] **Step 1.6: Commit**

```bash
git add nark/Cargo.toml nark/src/lib.rs bench/Cargo.toml Cargo.lock
git commit -m "feat(nark): expose library target for in-workspace consumers

Adds [lib] target and src/lib.rs that re-exports the seven crate modules
via pub mod. main.rs is unchanged — both compilation units pull in the
modules independently per the standard Rust binary+lib hybrid pattern.

bench/ gains nark and dirs as dependencies. Both are unused at this
commit; Task 2 wires them up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Expose `onnx_dylib_name()` and add `install_into()` in nark

**Files:**
- Modify: `nark/src/embed/mod.rs:159` — add `pub` to `onnx_dylib_name`
- Modify: `nark/src/embed/download.rs` — add `pub fn install_into`

These are the two functions the bench's `model_cache` (Task 4) needs to call.

- [ ] **Step 2.1: Make `onnx_dylib_name` public**

Read `nark/src/embed/mod.rs` around line 159 to find the function. Change:

```rust
fn onnx_dylib_name() -> &'static str {
```

to:

```rust
pub fn onnx_dylib_name() -> &'static str {
```

(Body unchanged.)

- [ ] **Step 2.2: Add `pub fn install_into` to `nark/src/embed/download.rs`**

Read `nark/src/embed/download.rs` to confirm `download_ort(vault_dir: &Path) -> Result<()>` (around line 81) and `download_model(vault_dir: &Path) -> Result<()>` (around line 109) both exist. They do — they're the private helpers `run_init` already calls.

Add this function near the existing `pub fn run_init`:

```rust
/// Library-friendly variant of `run_init`: downloads the ORT dylib and
/// nomic-embed model into `target`, producing the standard
/// `<target>/lib/...` and `<target>/models/<MODEL_NAME>/...` layout that
/// `init_embedding` expects.
///
/// Unlike `run_init`, this emits no progress bars or interactive log
/// messages and gives no "Run `nark embed build`" instructions — it is
/// intended for programmatic callers (such as the bench harness) that
/// need the same file layout `nark embed init` produces but want a
/// silent, library-shaped API.
pub fn install_into(target: &Path) -> Result<()> {
    std::fs::create_dir_all(target)?;
    download_ort(target)?;
    download_model(target)?;
    Ok(())
}
```

- [ ] **Step 2.3: Verify nark still builds**

```bash
cargo build -p nark
cargo build -p nark --lib
```

Expected: both succeed.

- [ ] **Step 2.4: Verify the new symbols are accessible from the lib**

Run a quick check that the symbols are exposed via the library:

```bash
cargo doc -p nark --lib --no-deps 2>&1 | tail -5
```

Expected: no errors. Documentation generation succeeding means the public API surface is consistent.

(There is no good way to write a unit test for `install_into` because it hits the network. Task 8's integration smoke test will exercise it via the model cache.)

- [ ] **Step 2.5: Commit**

```bash
git add nark/src/embed/mod.rs nark/src/embed/download.rs
git commit -m "feat(nark): expose embed pub API needed by bench

- src/embed/mod.rs:159: add pub to onnx_dylib_name()
- src/embed/download.rs: add pub fn install_into(target: &Path) that
  calls download_ort + download_model directly, bypassing run_init's
  interactive progress UX (which is misleading in library callers).

These two symbols are consumed by bench's model_cache module (added
in a subsequent commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Schema bump (v1 → v2) + microsecond latency

**Files:**
- Modify: `bench/src/protocol.rs`
- Modify: `bench/src/result.rs`
- Modify: `bench/src/adapters/fts5.rs`
- Modify: `bench/src/adapters/nark.rs`
- Modify: `bench/src/tasks/ir.rs`

Mechanical field renames + one schema version bump. Lands before the new adapters so they're written against the new types from the start.

- [ ] **Step 3.1: Rename latency fields in `bench/src/protocol.rs`**

Read `bench/src/protocol.rs`. Find both `WriteMetrics` and `SearchMetrics` structs. Rename `latency_ms` to `latency_us` in each:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteMetrics {
    pub latency_us: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub latency_us: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}
```

- [ ] **Step 3.2: Rename phase latency fields and bump schema_version in `bench/src/result.rs`**

Read `bench/src/result.rs`. In `PhaseMetrics`, rename `latency_p50_ms` to `latency_p50_us` and `latency_p99_ms` to `latency_p99_us`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseMetrics {
    pub latency_p50_us: u64,
    pub latency_p99_us: u64,
    pub llm_tokens_in_total: u64,
    pub llm_tokens_out_total: u64,
}
```

In `BenchResult::new`, change:

```rust
schema_version: "1".to_string(),
```

to:

```rust
schema_version: "2".to_string(),
```

- [ ] **Step 3.3: Switch FTS5 adapter to `as_micros()`**

Read `bench/src/adapters/fts5.rs`. Find both occurrences of:

```rust
latency_ms: t0.elapsed().as_millis() as u64,
```

(one in `write`, one in `search`). Change both to:

```rust
latency_us: t0.elapsed().as_micros() as u64,
```

- [ ] **Step 3.4: Switch nark adapter to `as_micros()`**

Read `bench/src/adapters/nark.rs`. Find both occurrences of:

```rust
latency_ms: t0.elapsed().as_millis() as u64,
```

(one in `write`, one in `search`). Change both to:

```rust
latency_us: t0.elapsed().as_micros() as u64,
```

- [ ] **Step 3.5: Switch tasks/ir.rs to `as_micros()` and update PhaseMetrics field references**

Read `bench/src/tasks/ir.rs`. Find the lines:

```rust
result.perf.write.latency_p50_ms = percentile(&write_latencies, 0.5);
result.perf.write.latency_p99_ms = percentile(&write_latencies, 0.99);
// ...
result.perf.search.latency_p50_ms = percentile(&search_latencies, 0.5);
result.perf.search.latency_p99_ms = percentile(&search_latencies, 0.99);
```

Rename to `latency_p50_us` / `latency_p99_us`:

```rust
result.perf.write.latency_p50_us = percentile(&write_latencies, 0.5);
result.perf.write.latency_p99_us = percentile(&write_latencies, 0.99);
// ...
result.perf.search.latency_p50_us = percentile(&search_latencies, 0.5);
result.perf.search.latency_p99_us = percentile(&search_latencies, 0.99);
```

The aggregation inputs `m.latency_ms` (read from the Adapter trait's return) also need renaming. Find:

```rust
write_latencies.push(m.latency_ms);
// ...
search_latencies.push(m.latency_ms);
```

Change to:

```rust
write_latencies.push(m.latency_us);
// ...
search_latencies.push(m.latency_us);
```

- [ ] **Step 3.6: Run all unit tests**

```bash
cargo test -p nark-bench
```

Expected: all 19 tests pass (16 unit + 1 ignored adapter smoke + 2 result tests, plus the build). No new tests added in this task; existing tests don't reference the renamed fields directly.

- [ ] **Step 3.7: Run the integration smoke test**

```bash
cargo build -p nark --release
cargo test -p nark-bench --release --test smoke
```

Expected: smoke test passes. (It doesn't assert specific field names, only checks `recall_at_5` is a number.)

- [ ] **Step 3.8: Commit**

```bash
git add bench/src/protocol.rs bench/src/result.rs bench/src/adapters/fts5.rs bench/src/adapters/nark.rs bench/src/tasks/ir.rs
git commit -m "refactor(bench): schema v2 — microsecond latency

Rename latency_ms → latency_us in protocol metric structs.
Rename latency_p{50,99}_ms → latency_p{50,99}_us in PhaseMetrics.
Bump BenchResult.schema_version 1 → 2.
Switch all adapters and the task runner from .as_millis() to .as_micros().

Millisecond resolution showed all zeros for in-process adapters
(FTS5 writes, future vector adapter cosine). Microseconds is honest
without being absurd.

This is a breaking schema change. The committed baselines under
bench/results/main/ remain on schema v1 until Task 10 regenerates them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `bench/src/model_cache.rs`

**Files:**
- Create: `bench/src/model_cache.rs`
- Modify: `bench/src/main.rs` — add `mod model_cache;`

Owns the shared model cache lifecycle. Three public functions: `cache_root()`, `ensure_ready()`, `stage_into()`.

- [ ] **Step 4.1: Create `bench/src/model_cache.rs`**

```rust
//! Shared model cache for adapters that need nark's ONNX embedding model.
//!
//! Default cache location: `~/.cache/nark-bench/models/`. Override with the
//! `NARK_BENCH_MODEL_CACHE` env var. First call to `ensure_ready` downloads
//! the ORT dylib + nomic-embed model (~270MB, 30–60s on a fast connection).
//! Subsequent calls are instant (marker-file existence check).
//!
//! Adapters that need embeddings call `stage_into(cache, workdir)` during
//! `setup()` to hard-link the model files into their per-run workdir, where
//! nark's `init_embedding` expects them.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const CACHE_ENV: &str = "NARK_BENCH_MODEL_CACHE";

/// Resolve the cache root. Honors `NARK_BENCH_MODEL_CACHE` env var; falls
/// back to `~/.cache/nark-bench/models/`.
pub fn cache_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(CACHE_ENV) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().context("no home directory")?;
    Ok(home.join(".cache/nark-bench/models"))
}

/// Idempotent: downloads ORT dylib + nomic-embed model into `cache` if the
/// expected marker files aren't already present.
pub fn ensure_ready(cache: &Path) -> Result<()> {
    let lib_marker = cache.join("lib").join(nark::embed::onnx_dylib_name());
    let model_marker = cache
        .join("models")
        .join(nark::embed::MODEL_NAME)
        .join("model.onnx");
    if lib_marker.exists() && model_marker.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(cache)?;
    nark::embed::download::install_into(cache)
        .context("failed to download embedding model into cache")?;
    Ok(())
}

/// Stage model files from `cache` into `workdir` via hard-links (cheap, no
/// copying). Falls back to copy if hard-link fails (e.g. cross-volume).
///
/// Staged files are treated as read-only. Currently safe only for serial
/// callers — the `!to.exists()` check in `link_tree` is TOCTOU under
/// concurrent invocation. The runner dispatches adapters sequentially so
/// this is fine today.
pub fn stage_into(cache: &Path, workdir: &Path) -> Result<()> {
    for sub in &["lib", "models"] {
        link_tree(&cache.join(sub), &workdir.join(sub))?;
    }
    Ok(())
}

fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            link_tree(&from, &to)?;
        } else if !to.exists() {
            std::fs::hard_link(&from, &to)
                .or_else(|_| std::fs::copy(&from, &to).map(|_| ()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn cache_root_honors_env_var() {
        // SAFETY: tests in the same process share env; isolate via tempdir-derived path
        let tmp = tempdir().unwrap();
        let path_str = tmp.path().to_string_lossy().to_string();
        // SAFETY: single-threaded test, no other test reads CACHE_ENV
        unsafe { std::env::set_var(CACHE_ENV, &path_str); }
        let resolved = cache_root().unwrap();
        unsafe { std::env::remove_var(CACHE_ENV); }
        assert_eq!(resolved, PathBuf::from(&path_str));
    }

    #[test]
    fn stage_into_hardlinks_files_recursively() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();

        // Build a small fixture tree:
        //   src/lib/foo.txt
        //   src/models/nested/bar.txt
        std::fs::create_dir_all(src.path().join("lib")).unwrap();
        std::fs::create_dir_all(src.path().join("models").join("nested")).unwrap();
        std::fs::write(src.path().join("lib").join("foo.txt"), b"hello").unwrap();
        std::fs::write(src.path().join("models").join("nested").join("bar.txt"), b"world").unwrap();

        stage_into(src.path(), dst.path()).unwrap();

        assert!(dst.path().join("lib").join("foo.txt").exists());
        assert!(dst.path().join("models").join("nested").join("bar.txt").exists());
        // Content matches
        assert_eq!(std::fs::read(dst.path().join("lib").join("foo.txt")).unwrap(), b"hello");
    }

    #[test]
    fn stage_into_is_idempotent() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        std::fs::create_dir_all(src.path().join("lib")).unwrap();
        std::fs::create_dir_all(src.path().join("models")).unwrap();
        std::fs::write(src.path().join("lib").join("a"), b"x").unwrap();

        // First call creates
        stage_into(src.path(), dst.path()).unwrap();
        // Second call should not error on existing files
        stage_into(src.path(), dst.path()).unwrap();

        assert!(dst.path().join("lib").join("a").exists());
    }
}
```

- [ ] **Step 4.2: Wire `mod model_cache;` into `bench/src/main.rs`**

Read `bench/src/main.rs`. After the existing `mod tasks;` line, add:

```rust
mod model_cache;
```

The full top of `main.rs` should now look like:

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod protocol;
mod metrics;
mod adapters;
mod result;
mod tasks;
mod model_cache;
```

- [ ] **Step 4.3: Run the new unit tests**

```bash
cargo test -p nark-bench model_cache::tests
```

Expected: 3 tests pass.

- [ ] **Step 4.4: Verify the workspace still builds**

```bash
cargo build -p nark-bench
```

Expected: completes with no errors. There will be a dead_code warning that `ensure_ready` is unused — that's expected; Task 7 uses it.

- [ ] **Step 4.5: Commit**

```bash
git add bench/src/model_cache.rs bench/src/main.rs
git commit -m "feat(bench): model_cache module — own the shared ONNX cache

Three pub functions: cache_root() resolves location (env var or
~/.cache/nark-bench/models/); ensure_ready() downloads on first call
via nark::embed::download::install_into; stage_into() hard-links
model files into a per-adapter workdir.

Hard-link with copy fallback gives correct cross-volume behavior.
Staged files are treated read-only; currently serial-only safe
(documented inline).

Three unit tests cover env-var override, recursive staging, and
idempotency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Vector adapter

**Files:**
- Create: `bench/src/adapters/vector.rs`
- Modify: `bench/src/adapters/mod.rs`

Pure-Rust in-process vector baseline. Uses nark's `OnnxProvider` to embed (same model nark uses), stores vectors in a `HashMap`, brute-force cosine on search via nark's existing `cosine_similarity`.

- [ ] **Step 5.1: Write the failing tests first (TDD)**

Create `bench/src/adapters/vector.rs` with the cosine tests as failing tests against the eventual structure:

```rust
//! Pure-Rust vector baseline. Embeds every doc on write, brute-force cosine
//! on search. Uses the same OnnxProvider as nark to keep model choice
//! constant (apples-to-apples comparison).

use anyhow::{anyhow, Context, Result};
use nark::embed::{cosine_similarity, init_embedding, EmbeddingProvider, OnnxProvider};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::model_cache;
use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

pub struct VectorAdapter {
    model_cache: PathBuf,
    provider: Option<OnnxProvider>,
    embeddings: HashMap<String, Vec<f32>>,
    workdir: Option<PathBuf>,
}

impl VectorAdapter {
    pub fn new(model_cache: PathBuf) -> Self {
        Self {
            model_cache,
            provider: None,
            embeddings: HashMap::new(),
            workdir: None,
        }
    }
}

impl Adapter for VectorAdapter {
    fn name(&self) -> &str { "vector" }

    fn version(&self) -> Result<String> {
        // Reports the intended model identifier. Actual init success is
        // signalled by setup() returning Ok(()). The bench harness only
        // calls version() after setup() succeeds, so this is safe.
        Ok(format!("vector {}", nark::embed::MODEL_NAME))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        todo!()
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        todo!()
    }

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        todo!()
    }

    fn teardown(&mut self) -> Result<()> {
        self.provider = None;
        self.embeddings.clear();
        self.workdir = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nark::embed::l2_normalize;

    #[test]
    fn cosine_identical_unit_vectors_returns_one() {
        let v = l2_normalize(&[1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected ~1.0, got {}", sim);
    }

    #[test]
    fn cosine_perpendicular_unit_vectors_returns_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected ~0.0, got {}", sim);
    }

    #[test]
    fn cosine_opposite_unit_vectors_returns_negative_one() {
        let a = l2_normalize(&[1.0, 2.0, 3.0]);
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "expected ~-1.0, got {}", sim);
    }
}
```

- [ ] **Step 5.2: Add `pub mod vector;` to `bench/src/adapters/mod.rs`**

Read `bench/src/adapters/mod.rs`. After the existing `pub mod nark;` line, add:

```rust
pub mod vector;
```

(The `make_adapter` factory will be updated in Task 7 — leave it pointing at fts5 + nark for now.)

- [ ] **Step 5.3: Verify cosine tests pass**

```bash
cargo test -p nark-bench adapters::vector::tests
```

Expected: 3 tests pass. (They exercise `nark::embed::cosine_similarity` and `nark::embed::l2_normalize`, both of which already exist as `pub` functions — no implementation needed beyond the import. The `todo!()`s in the adapter methods are irrelevant to these tests.)

- [ ] **Step 5.4: Implement `setup` — stage model + initialize provider + defensive normalization probe**

Replace the `todo!()` in `setup` with:

```rust
fn setup(&mut self, workdir: &Path) -> Result<()> {
    model_cache::stage_into(&self.model_cache, workdir)
        .context("failed to stage embedding model into vector adapter workdir")?;
    let mut provider = init_embedding(workdir)
        .ok_or_else(|| anyhow!("ONNX init returned None after staging — model files corrupt?"))?;

    // Defensive normalization probe: confirm provider outputs are L2-normalized.
    // We rely on this assumption when calling nark::embed::cosine_similarity
    // (which is dot-product-only). If a future change breaks the normalization
    // contract, this assertion catches it loudly instead of silent degradation.
    let probe = provider.embed_document("probe")
        .context("normalization probe: failed to embed test string")?;
    let norm: f32 = probe.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 1e-3 {
        return Err(anyhow!(
            "OnnxProvider output is not L2-normalized (norm = {:.6}); \
             cosine_similarity will return incorrect results. \
             Check nark::embed::OnnxProvider::run_inference for an l2_normalize call.",
            norm
        ));
    }

    self.provider = Some(provider);
    self.workdir = Some(workdir.to_path_buf());
    self.embeddings.clear();
    Ok(())
}
```

- [ ] **Step 5.5: Implement `write` — embed doc, store in HashMap**

Replace the `todo!()` in `write` with:

```rust
fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
    let t0 = Instant::now();
    let provider = self.provider.as_mut().ok_or_else(|| anyhow!("setup not called"))?;
    let embedding = provider.embed_document(&doc.body)
        .with_context(|| format!("failed to embed doc {}", doc.id))?;
    self.embeddings.insert(doc.id.clone(), embedding);
    Ok(WriteMetrics {
        latency_us: t0.elapsed().as_micros() as u64,
        llm_tokens_in: 0,
        llm_tokens_out: 0,
    })
}
```

- [ ] **Step 5.6: Implement `search` — embed query, brute-force cosine, top-k**

Replace the `todo!()` in `search` with:

```rust
fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
    let t0 = Instant::now();
    let provider = self.provider.as_mut().ok_or_else(|| anyhow!("setup not called"))?;
    let query_emb = provider.embed_query(query)?;
    let mut scored: Vec<(String, f32)> = self.embeddings.iter()
        .map(|(id, emb)| (id.clone(), cosine_similarity(&query_emb, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let results: Vec<SearchResult> = scored.into_iter().take(k)
        .map(|(doc_id, score)| SearchResult { doc_id, score, snippet: None })
        .collect();
    Ok((results, SearchMetrics {
        latency_us: t0.elapsed().as_micros() as u64,
        llm_tokens_in: 0,
        llm_tokens_out: 0,
    }))
}
```

- [ ] **Step 5.7: Build and run unit tests**

```bash
cargo build -p nark-bench
cargo test -p nark-bench adapters::vector::tests
```

Expected: build succeeds, 3 cosine tests pass. The `setup/write/search` methods aren't exercised by unit tests (would require model files); they're covered by the integration smoke test in Task 8.

- [ ] **Step 5.8: Commit**

```bash
git add bench/src/adapters/vector.rs bench/src/adapters/mod.rs
git commit -m "feat(bench): vector adapter — in-process cosine baseline

Pure-Rust adapter using nark::embed::OnnxProvider to embed (same model
as nark) and brute-force cosine via the existing pub nark::embed::
cosine_similarity (L2-normalized dot product).

Stores embeddings in a HashMap<doc_id, Vec<f32>> — appropriate for the
corpus sizes the bench targets (10s-100s of docs). HNSW/FAISS only
matter at 10k+.

setup() includes a defensive normalization probe that asserts the
provider produces L2-normalized vectors, catching any future change
to OnnxProvider that breaks the cosine_similarity contract.

Three cosine unit tests cover identical/perpendicular/opposite unit
vectors. The setup/write/search round-trip is covered by Task 8's
integration smoke test.

Not yet registered in make_adapter — Task 7 wires the factory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: nark adapter — stage model in `setup()`

**Files:**
- Modify: `bench/src/adapters/nark.rs`

Add a `model_cache: Option<PathBuf>` field. When set, `setup()` stages the model files into the workdir before `nark init`, so subsequent `nark write` calls produce embeddings inline. On staging failure, fall back to BM25-only with a stderr warning.

- [ ] **Step 6.1: Add `model_cache` field to `NarkAdapter`**

Read `bench/src/adapters/nark.rs`. Find the `pub struct NarkAdapter` definition. Add a fifth field:

```rust
pub struct NarkAdapter {
    workdir: Option<PathBuf>,
    staging: Option<PathBuf>,
    nark_bin: Option<PathBuf>,
    uuid_to_bench_id: HashMap<String, String>,
    /// Path to the shared model cache. If `None`, nark runs in BM25-only
    /// mode (no embedding files staged into workdir).
    model_cache: Option<PathBuf>,
}
```

- [ ] **Step 6.2: Update `NarkAdapter::new` signature and initializer**

Find the existing `impl NarkAdapter { pub fn new() -> Self { ... } }`. Change the signature to take `Option<PathBuf>` and initialize the new field:

```rust
impl NarkAdapter {
    pub fn new(model_cache: Option<PathBuf>) -> Self {
        Self {
            workdir: None,
            staging: None,
            nark_bin: None,
            uuid_to_bench_id: HashMap::new(),
            model_cache,
        }
    }
    // ... run_nark stays unchanged
}
```

Also update the `impl Default for NarkAdapter` block:

```rust
impl Default for NarkAdapter {
    fn default() -> Self { Self::new(None) }
}
```

- [ ] **Step 6.3: Add staging call in `setup()`**

Find the existing `fn setup(&mut self, workdir: &Path) -> Result<()>` in the `impl Adapter for NarkAdapter` block. After the `self.uuid_to_bench_id.clear();` line and before the `self.run_nark(&["init"])?;` line, insert:

```rust
// Stage model files into workdir so nark write produces embeddings
// inline. If staging fails (cache missing, download failed, etc),
// fall back to BM25-only — adapter still functions, just without
// the cosine component. We log to stderr so operator sees what happened.
if let Some(cache) = &self.model_cache {
    if let Err(e) = crate::model_cache::stage_into(cache, &workdir) {
        eprintln!("nark adapter: model staging failed, continuing without embeddings: {}", e);
    }
}
```

(Note: the existing `setup` already binds `workdir` as a `PathBuf` and assigns it; the staging call uses that local binding.)

- [ ] **Step 6.4: Update existing ignored smoke test to pass `None` (no cache)**

The existing ignored test `nark_smoke_write_then_search` calls `NarkAdapter::new()`. Change to `NarkAdapter::new(None)`:

```rust
#[test]
#[ignore = "requires built nark binary; run via the integration smoke test"]
fn nark_smoke_write_then_search() {
    let dir = tempdir().unwrap();
    let mut a = NarkAdapter::new(None);  // <- was: NarkAdapter::new()
    a.setup(dir.path()).unwrap();
    // ... rest unchanged
}
```

- [ ] **Step 6.5: Verify build and existing ignored test still passes**

```bash
cargo build -p nark-bench
cargo test -p nark-bench adapters::nark::tests -- --ignored
```

Expected: build succeeds; 1 ignored test passes (`nark_smoke_write_then_search`), confirming the BM25-only fallback path still works.

- [ ] **Step 6.6: Verify the version-sync test still passes**

```bash
cargo test -p nark-bench adapters::nark::tests::nark_version_matches_workspace
```

Expected: 1 test passes.

- [ ] **Step 6.7: Commit**

```bash
git add bench/src/adapters/nark.rs
git commit -m "feat(bench): nark adapter stages model files in setup()

NarkAdapter gains a model_cache: Option<PathBuf> field. When set,
setup() calls model_cache::stage_into(cache, workdir) before nark init,
so subsequent nark write calls find the model files and produce
embeddings inline.

Fall-back semantics on staging failure: log to stderr, continue in
BM25-only mode. Keeps the bench resilient when the cache can't be
populated (offline, intentional clear, etc) and matches the design's
fall-back rationale.

Constructor signature changes from new() to new(model_cache:
Option<PathBuf>); Default impl uses None. The existing ignored unit
test updated to pass None. Task 7 wires the factory to pass an
actual cache.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Factory + runner wiring

**Files:**
- Modify: `bench/src/adapters/mod.rs`
- Modify: `bench/src/main.rs`

Update `make_adapter` to take an optional cache path; pass it through. Update `main.rs` to resolve the cache root, ensure it's ready, and pass it into the factory.

- [ ] **Step 7.1: Update `make_adapter` in `bench/src/adapters/mod.rs`**

Read the current file. Replace its contents with:

```rust
pub mod fts5;
pub mod nark;
pub mod vector;

use crate::protocol::Adapter;
use anyhow::{anyhow, bail, Result};
use std::path::Path;

/// Construct an adapter by short name.
///
/// `model_cache`: path to the populated ONNX model cache. Required for
/// the `vector` adapter (errors if None). The `nark` adapter uses it
/// when provided to enable embeddings; if None, nark runs BM25-only.
/// The `fts5` adapter ignores it.
pub fn make_adapter(name: &str, model_cache: Option<&Path>) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        "nark" => Ok(Box::new(nark::NarkAdapter::new(model_cache.map(Path::to_path_buf)))),
        "vector" => {
            let cache = model_cache.ok_or_else(|| anyhow!(
                "vector adapter requires a model cache; ensure NARK_BENCH_MODEL_CACHE env var \
                 is set or that the runner has called model_cache::ensure_ready"
            ))?;
            Ok(Box::new(vector::VectorAdapter::new(cache.to_path_buf())))
        }
        other => bail!("unknown adapter: {}", other),
    }
}
```

- [ ] **Step 7.2: Update `bench/src/main.rs` to populate + pass the cache**

Read `bench/src/main.rs`. Find the existing `Commands::Run` match arm (in `fn main`). The current loop looks like:

```rust
for system in systems.split(',') {
    let system = system.trim();
    if system.is_empty() { continue; }
    let mut adapter = adapters::make_adapter(system)?;
    let result = tasks::ir::run_ir_task(adapter.as_mut(), &corpus_root, "default")?;
    let path = result.write_to_disk(&output)?;
    eprintln!("wrote {}", path.display());
}
```

Replace it with:

```rust
let cache = model_cache::cache_root()?;
model_cache::ensure_ready(&cache)?;
for system in systems.split(',') {
    let system = system.trim();
    if system.is_empty() { continue; }
    let mut adapter = adapters::make_adapter(system, Some(&cache))?;
    let result = tasks::ir::run_ir_task(adapter.as_mut(), &corpus_root, "default")?;
    let path = result.write_to_disk(&output)?;
    eprintln!("wrote {}", path.display());
}
```

The two new lines (`let cache = ...` and `ensure_ready(...)`) go immediately after the existing `if !corpus_root.exists() { ... }` bail and before the `for system in systems.split(',')` loop. The factory call inside the loop gains `Some(&cache)` as its second arg.

- [ ] **Step 7.3: Build**

```bash
cargo build -p nark-bench
```

Expected: completes with no errors.

- [ ] **Step 7.4: Run all unit tests**

```bash
cargo test -p nark-bench
```

Expected: all bench unit tests pass (22 + 3 ignored, depending on Task 5's exact count). No new tests in this task.

- [ ] **Step 7.5: Commit**

```bash
git add bench/src/adapters/mod.rs bench/src/main.rs
git commit -m "feat(bench): wire model cache through factory and runner

make_adapter signature changes from (name) to (name, model_cache:
Option<&Path>). vector requires a cache (errors if None); nark uses
it when present; fts5 ignores it.

main.rs calls model_cache::cache_root() + ensure_ready() before the
adapter loop, then passes Some(&cache) into each make_adapter call.
ensure_ready is idempotent and only downloads on first invocation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Integration smoke test — all three adapters, zero errors

**Files:**
- Modify: `bench/tests/smoke.rs`

Updated to run `fts5,nark,vector`, assert schema_version "2", and require zero errors for all three. This will trigger a model download on the first run; tolerable for a one-shot integration test.

- [ ] **Step 8.1: Replace the test**

Read `bench/tests/smoke.rs`. Replace the existing test body with the updated version:

```rust
//! Integration smoke test — runs the full harness against synthetic-tiny
//! for all three adapters (fts5, nark, vector). This is what CI Lane 1
//! (the PR check) actually executes.
//!
//! On first run after the model cache is empty, this downloads the
//! ONNX runtime + nomic-embed model (~270MB, 30-60s). Subsequent runs
//! are fast.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn smoke_fts5_nark_vector_all_run_against_synthetic_tiny() {
    let out_dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("nark-bench").unwrap();
    cmd.current_dir(workspace_root())
        .args(["run", "--task", "ir",
               "--systems", "fts5,nark,vector",
               "--corpus", "synthetic-tiny",
               "--output"]).arg(out_dir.path());

    let out = cmd.assert().success().get_output().clone();
    let stderr = String::from_utf8_lossy(&out.stderr);
    for system in ["fts5", "nark", "vector"] {
        assert!(
            stderr.contains(&format!("ir-{}-default.json", system)),
            "stderr did not mention {} result: {}", system, stderr
        );
    }

    for system in ["fts5", "nark", "vector"] {
        let path = out_dir.path().join(format!("ir-{}-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read {}: {}", path.display(), e)
        });
        let v: Value = serde_json::from_str(&content).unwrap();

        // Schema bumped to "2" in Task 3
        assert_eq!(v["schema_version"], "2", "{} schema_version mismatch", system);
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert!(v["ir"]["recall_at_5"].is_number());

        // Metrics in valid range
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5), "{} recall_at_5 out of range: {}", system, r5);

        // Per-class breakdowns
        assert!(v["ir_per_class"]["single_hop"].is_object(),
            "{} missing single_hop breakdown", system);

        // All three adapters should produce zero errors. FTS5's q10
        // sanitizer (Phase 1a fix) handles the column-prefix issue;
        // nark with embeddings and vector both run all queries cleanly.
        let errors = v["errors"].as_array().expect("errors field required");
        assert!(
            errors.is_empty(),
            "{} run had unexpected errors (regression?): {:?}", system, errors
        );
    }
}
```

- [ ] **Step 8.2: Ensure nark is built in release**

```bash
cargo build -p nark --release
```

Expected: completes.

- [ ] **Step 8.3: Run the smoke test (first run downloads the model)**

```bash
cargo test -p nark-bench --release --test smoke
```

Expected: 1 test passes. First run takes 30-60s due to model download; subsequent runs are seconds. Watch stderr for "Downloading..." messages confirming the cache is being populated.

If the test fails because nark's results have errors, inspect the result JSON to diagnose: stage failure? Embed model not properly initialized? The error message in `result.errors` will tell you the phase and the specific error.

- [ ] **Step 8.4: Commit**

```bash
git add bench/tests/smoke.rs
git commit -m "test(bench): integration smoke covers all three adapters + zero errors

Updates smoke test to run fts5,nark,vector against synthetic-tiny.
Asserts schema_version is \"2\" (Task 3 bump), recall_at_5 in [0,1]
for each, per-class breakdowns present, and zero errors for all three
adapters. The zero-errors assertion is the regression gate: any future
change that re-introduces silent query failures (FTS5 hyphen issue,
embedding init failure, etc) breaks the build.

First run downloads the ONNX runtime + nomic-embed model (~270MB,
30-60s). Subsequent runs use the cache.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Regression script enhancements + filename sanitization

**Files:**
- Modify: `bench/scripts/regression-check.sh`
- Modify: `bench/src/result.rs`

Three small additions: query-count check + schema-version warning in the script, filename sanitization in the result writer.

- [ ] **Step 9.1: Add query-count + schema-version checks to regression script**

Read `bench/scripts/regression-check.sh`. Find the `for new_file in "$NEW_DIR"/ir-*.json; do` loop. Inside the loop, right after the `else continue;` for the bootstrap branch, insert this block before the per-metric for loop:

```bash
  # Schema version warning (non-fatal): comparison still works because the
  # regression check only reads .ir.* metrics, but a mismatch usually means
  # the baseline needs regenerating.
  new_schema=$(jq -r '.schema_version' "$new_file")
  base_schema=$(jq -r '.schema_version' "$base_file")
  if [[ "$new_schema" != "$base_schema" ]]; then
    printf 'WARNING: %s schema version mismatch (new=%s, baseline=%s) — re-bootstrap baseline if intentional\n' \
      "$(basename "$new_file")" "$new_schema" "$base_schema"
  fi

  # Query-count check (fatal): catches the case where a future change makes
  # more queries error out, which would otherwise be invisible since per-query
  # averages can look fine even when fewer queries are contributing.
  new_q=$(jq -r '.ir.queries' "$new_file")
  base_q=$(jq -r '.ir.queries' "$base_file")
  if [[ "$new_q" -lt "$base_q" ]]; then
    printf 'REGRESSION: %s query count dropped from %d to %d (likely new adapter errors)\n' \
      "$(basename "$new_file")" "$base_q" "$new_q"
    fail=1
  fi
```

- [ ] **Step 9.2: Add filename sanitization to `write_to_disk` in `bench/src/result.rs`**

Read `bench/src/result.rs`. Find the `write_to_disk` function. Add a private `sanitize` helper above it (or inline as a closure) and use it for the three components:

```rust
fn sanitize_for_filename(s: &str) -> String {
    s.replace('/', "_").replace(std::path::MAIN_SEPARATOR, "_")
}

impl BenchResult {
    // ... new() unchanged ...

    pub fn write_to_disk(&self, output_dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let filename = format!(
            "{}-{}-{}.json",
            sanitize_for_filename(&self.task),
            sanitize_for_filename(&self.system),
            sanitize_for_filename(&self.config),
        );
        let path = output_dir.join(filename);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, format!("{}\n", json))?;
        Ok(path)
    }
}
```

(The `sanitize_for_filename` function lives at module scope above the `impl BenchResult` block, not inside it, because it doesn't need access to `self`.)

- [ ] **Step 9.3: Verify the script with a smoke run**

```bash
mkdir -p /tmp/bench-task9-smoke
find /tmp/bench-task9-smoke -name '*.json' -delete 2>/dev/null
cargo run -p nark-bench --release -- run --task ir --systems fts5,nark,vector --corpus synthetic-tiny --output /tmp/bench-task9-smoke
bench/scripts/regression-check.sh /tmp/bench-task9-smoke
```

Expected: script runs without errors. It may print "no baseline" for the vector file (Task 10 will lock in the baseline). The existing fts5 and nark baselines (still schema v1 at this point in the plan) will print "WARNING: ... schema version mismatch" — that's the new behavior working correctly. Exit code 0.

Clean up the smoke output:

```bash
rm -rf /tmp/bench-task9-smoke
```

- [ ] **Step 9.4: Verify the result unit tests still pass**

```bash
cargo test -p nark-bench result::tests
```

Expected: 2 tests pass (the existing percentile tests are unaffected).

- [ ] **Step 9.5: Commit**

```bash
git add bench/scripts/regression-check.sh bench/src/result.rs
git commit -m "feat(bench): regression script — query-count + schema warning; sanitize filenames

regression-check.sh:
- New fatal check: if new run's .ir.queries is less than baseline's,
  print REGRESSION and fail. Catches silent adapter-error regressions.
- New non-fatal warning: if schema_version differs between new and
  baseline, print WARNING. Comparison still works because the script
  only reads .ir.* metrics, but mismatch usually means baseline needs
  regenerating.

result.rs:
- write_to_disk now passes each filename component through
  sanitize_for_filename (replaces / and MAIN_SEPARATOR with _).
  Defends against future config/system names containing path
  separators.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Regenerate baseline + CI cache step

**Files:**
- Delete: `bench/results/main/ir-fts5-default.json` (old schema v1)
- Delete: `bench/results/main/ir-nark-default.json` (old schema v1)
- Create: `bench/results/main/ir-fts5-default.json` (new schema v2)
- Create: `bench/results/main/ir-nark-default.json` (new schema v2, with embeddings)
- Create: `bench/results/main/ir-vector-default.json` (new)
- Modify: `.github/workflows/bench-pr.yml`

Final step. Produces the new baseline that all future PRs gate against.

- [ ] **Step 10.1: Delete the schema-v1 baselines**

```bash
rm bench/results/main/ir-fts5-default.json
rm bench/results/main/ir-nark-default.json
```

- [ ] **Step 10.2: Produce a fresh baseline run**

```bash
cargo build -p nark --release
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus synthetic-tiny \
  --output bench/results/main
```

Expected: prints three "wrote bench/results/main/ir-{fts5,nark,vector}-default.json" lines on stderr. Model download happens here if not already cached.

- [ ] **Step 10.3: Inspect the new baselines**

```bash
jq '.schema_version, .system_version, .ir, (.errors | length)' bench/results/main/ir-fts5-default.json
jq '.schema_version, .system_version, .ir, (.errors | length)' bench/results/main/ir-nark-default.json
jq '.schema_version, .system_version, .ir, (.errors | length)' bench/results/main/ir-vector-default.json
```

Sanity-check:
- All three show `schema_version: "2"`
- `system_version` reads `"sqlite-fts5 <ver>"`, `"nark 0.13.0"`, `"vector nomic-embed-text-v1.5"` respectively
- All three have `queries: 10` and `errors.length == 0`
- nark's `recall_at_5` should now be **higher than its Phase 1a baseline of 0.1** (embeddings engaged). Typical: 0.3–0.6 range. If still ~0.1, the staging path failed silently — check stderr from step 10.2 for the "model staging failed" warning.
- vector's `recall_at_5` should also be in a reasonable range (0.2–0.7). The synthetic corpus' synonym class should benefit most.
- fts5's numbers should be unchanged from Phase 1a (same algorithm, same data; only the latency field renamed).

If nark's numbers look suspicious (e.g. exactly the same as the Phase 1a BM25-only baseline), the staging silently failed. Diagnose by running:

```bash
RUST_BACKTRACE=1 cargo run -p nark-bench --release -- run --task ir --systems nark --corpus synthetic-tiny --output /tmp/nark-debug 2>&1 | head -30
```

The "model staging failed" warning (if present) will show the underlying error.

- [ ] **Step 10.4: Run the regression check against the new baseline**

```bash
mkdir -p /tmp/bench-task10-verify
find /tmp/bench-task10-verify -name '*.json' -delete 2>/dev/null
cargo run -p nark-bench --release -- run --task ir --systems fts5,nark,vector --corpus synthetic-tiny --output /tmp/bench-task10-verify
bench/scripts/regression-check.sh /tmp/bench-task10-verify
echo "regression-check exit code: $?"
```

Expected: exit code 0 (numbers match baseline; no regressions; no schema warnings since we just regenerated).

Clean up:

```bash
rm -rf /tmp/bench-task10-verify
```

- [ ] **Step 10.5: Add the actions/cache step to CI**

Read `.github/workflows/bench-pr.yml`. Add a new step between "Install jq" and "Build nark" (or anywhere before "Build nark-bench" / "Run benchmark"):

```yaml
      - name: Cache nark-bench embedding model
        uses: actions/cache@v4
        with:
          path: ~/.cache/nark-bench/models
          key: nark-bench-model-${{ hashFiles('src/embed/download.rs', 'src/embed/mod.rs') }}
          restore-keys: |
            nark-bench-model-
```

Place it immediately after the existing "Install jq" step. The hash key invalidates only when the embedding logic / model selection changes.

- [ ] **Step 10.6: Validate the workflow YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/bench-pr.yml'))" && echo OK
```

Expected: `OK`.

- [ ] **Step 10.7: Commit**

```bash
git add bench/results/main/ir-fts5-default.json bench/results/main/ir-nark-default.json bench/results/main/ir-vector-default.json .github/workflows/bench-pr.yml
git commit -m "bench: regenerate baseline for Phase 1b + add CI model cache

New baselines on schema v2:
- ir-fts5-default.json: rankings unchanged from Phase 1a; latency
  fields renamed to *_us.
- ir-nark-default.json: nark now runs with embeddings via the model
  cache; recall improves materially from the BM25-only ~0.1 baseline.
- ir-vector-default.json: new — vector-only baseline using the same
  embedding model nark uses, brute-force cosine, no ranking layers.

CI workflow gains an actions/cache@v4 step keyed on the hash of
src/embed/download.rs and src/embed/mod.rs. First PR after this lands
takes ~30-60s for the cold-cache model download; all subsequent PRs
hit the cache.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Wrap-up

After all 10 tasks land, Phase 1b is complete. The branch now contains:

- A nark binary+library hybrid (`nark/src/lib.rs`); other crates in this workspace can `use nark::embed::OnnxProvider` etc. directly.
- A new pure-Rust `VectorAdapter` that uses nark's `OnnxProvider` in-process.
- The existing `NarkAdapter` now runs with embeddings when a model cache is provided (falls back to BM25-only with a stderr warning if not).
- A `model_cache` module that downloads the ONNX runtime + nomic-embed model once and hard-links it into per-adapter workdirs.
- Schema v2 with microsecond latency. All baselines regenerated.
- Three smoke-test-gated adapters; zero errors required across all three.
- Regression script catches query-count drops and warns on schema-version drift.
- CI workflow caches the model for ~zero-cost reuse on subsequent PRs.

The Phase 1a regression gate continues to work — any change that hurts Recall@5, MRR, or nDCG@10 by >2% (now also: any change that drops the query count) fails the PR check.

## Follow-up plans (not part of Phase 1b)

- **Phase 1c — nark-self real-vault corpus.** Gated on Sean's one-time anonymization scrub of ~150 notes from his actual vault. Drop the files into `bench/datasets/ir/nark-self/` and add a `--corpus nark-self` invocation. No code changes needed.
- **Phase 2 — LongMemEval + LOCOMO + LLM judge.** Adds Task B, the `claude-cli` / `codex-cli` / `api` judge backends, and the judge cache.
- **Phase 3 — Python competitor adapters.** mem0 / Letta / Graphiti shims; Letta HTTP adapter; docker-compose harness.
- **Phase 4 — Report generator.** Auto-generated `BENCHMARKS.md` from `bench/results/latest/` + README badges.
- **Phase 5 — Task C agent replay.** Real anonymized transcripts; labelling helper.
- **Phase 6 — Task D Librarian curation.** Gated on the Librarian agent shipping.
