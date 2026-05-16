# nark Bench — Phase 1b (Vector Baseline + Embeddings-Enabled nark) Design

**Date:** 2026-05-16
**Status:** Design — awaiting user review before plan
**Author:** Sean (with Claude as collaborator)
**Parent design:** `docs/superpowers/specs/2026-05-13-nark-benchmarking-design.md`
**Predecessor:** Phase 1a shipped on branch `feat/bench-phase-1a` (commits `a947e0c`, `4f85018`, `7ae7ea3`).

---

## 1. Goal & framing

Phase 1a shipped the bench harness skeleton: two adapters (FTS5 baseline, nark via CLI subprocess), classical IR metrics, a synthetic-tiny corpus, the regression gate. Both adapters currently run in BM25-only mode — nark's fresh temp vault has no embeddings, so its full hybrid pipeline never engages. The result: nark and FTS5 score similarly on synthetic-tiny (~0.1–0.2 Recall@5), and the comparison story "nark's hybrid beats raw vector" can't actually be told.

Phase 1b enables that story. Three pieces:

1. **Vector adapter** — a pure-Rust baseline that embeds every doc with the same model nark uses, stores embeddings in-memory, and brute-force scans on search. The point: when nark wins, it should win on its hybrid pipeline (BM25 + cosine + graph + engagement blend), not because it has a better embedding model than the comparison.
2. **nark adapter with embeddings enabled** — adapter's `setup()` stages the model into the workdir so subsequent `nark write` calls produce embeddings inline. nark's search then exercises its full pipeline.
3. **Deferred code-review fixes** from Phase 1a's post-merge review: microsecond latency (schema bump to v2), query-count regression tracking, filename sanitization.

Out of scope for Phase 1b (already explicitly deferred):
- `nark-self` real-vault corpus (requires Sean's one-time anonymization scrub of ~150 notes — human work)
- LongMemEval / LOCOMO datasets (Phase 2)
- LLM-judge backends — `claude -p`, `codex exec`, panel mode (Phase 2)
- Competitor adapters — mem0, Letta, Graphiti (Phase 3)
- Report generator → `BENCHMARKS.md` + README badges (Phase 4)

## 2. Why these design choices

**Same embedding model across vector and nark.** Both adapters use `nomic-embed-text-v1.5` via nark's existing `EmbeddingProvider`. Different models would conflate "nark's pipeline helps" with "nomic beats bge-small." Keeping the model constant isolates the variable the bench is trying to measure.

**nark becomes a binary+library hybrid.** Adding `lib.rs` to nark is a small one-time change but it's load-bearing: once done, the bench (and Phase 2's LLM-judge, Phase 3's competitor adapters, anything else in the workspace) can `use nark::embed::OnnxProvider` directly. No subprocess overhead, no code duplication, no risk of drift. nark's binary surface stays unchanged.

**In-process embedding for the vector adapter.** Loading the ONNX model directly into the bench binary makes embedding latency a measurable, comparable axis without subprocess startup noise. Brute-force cosine over 10–200 documents is microseconds; HNSW/FAISS only matter at corpus sizes we don't reach in Phase 1b or Phase 1c.

**Shared model cache.** Both adapters need the same model files (ORT dylib + nomic ONNX + tokenizer). Each adapter staging its own copy from scratch wastes 30–60 seconds per benchmark run and gigabytes of disk. A single cache at `~/.cache/nark-bench/models/` is populated once, then hard-linked into each adapter's per-run workdir.

## 3. Repo changes

### 3.1 `nark/` — binary+library hybrid

`nark/Cargo.toml` gains a `[lib]` target alongside the existing implicit binary:

```toml
[lib]
name = "nark"
path = "src/lib.rs"
```

New file `nark/src/lib.rs`:

```rust
pub mod cli;
pub mod config;
pub mod db;
pub mod embed;
pub mod registry;
pub mod types;
pub mod vault;
```

`nark/src/main.rs` changes one line — `mod cli;` becomes `use nark::cli;` (and similarly for the other modules it references). The binary continues to publish unchanged.

**A small pub-API addition in nark may be needed.** The bench's `model_cache` needs to call something like `nark::embed::download::install_into(target_dir: &Path)` — a function that downloads the ORT dylib + nomic model into the standard `<dir>/lib/...` and `<dir>/models/<MODEL_NAME>/...` layout. If `nark/src/embed/download.rs` doesn't already expose a public function with that exact shape (today its public surface is mostly internal helpers used by `nark embed init`), the implementer adds a thin wrapper — typically <30 lines that just calls the existing private routine with the target directory as `vault_dir`.

### 3.2 `bench/` — new and modified files

```
bench/
├── Cargo.toml                   # add: nark = { path = ".." }, dirs = "6"
└── src/
    ├── model_cache.rs           # NEW — owns the shared model cache lifecycle
    ├── adapters/
    │   ├── mod.rs               # MODIFIED — factory takes Option<&Path> cache
    │   ├── vector.rs            # NEW — in-process cosine baseline
    │   ├── nark.rs              # MODIFIED — setup() stages model files
    │   └── fts5.rs              # unchanged
    ├── result.rs                # MODIFIED — schema_version "2"; *_us fields
    ├── protocol.rs              # MODIFIED — WriteMetrics/SearchMetrics renames
    ├── tasks/ir.rs              # MODIFIED — uses .as_micros() instead of .as_millis()
    └── main.rs                  # MODIFIED — calls model_cache::ensure_ready
```

```
bench/scripts/regression-check.sh   # MODIFIED — adds query-count check
bench/tests/smoke.rs                # MODIFIED — runs fts5,nark,vector; asserts zero errors
bench/results/main/ir-*.json        # REGENERATED — schema v2, vector adapter included
.github/workflows/bench-pr.yml      # MODIFIED — adds actions/cache step for model files
```

## 4. The model cache

### `bench/src/model_cache.rs`

Owns the shared cache. Three functions: locate the cache root, ensure it's populated, stage into a workdir.

```rust
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const CACHE_ENV: &str = "NARK_BENCH_MODEL_CACHE";

pub fn cache_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(CACHE_ENV) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().context("no home dir")?;
    Ok(home.join(".cache/nark-bench/models"))
}

/// Idempotent: downloads ORT dylib + nomic-embed model into the cache root
/// if not already present.
pub fn ensure_ready(cache: &Path) -> Result<()> {
    let lib_marker = cache.join("lib").join(nark::embed::download::onnx_dylib_name());
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
/// copying). Falls back to copy if hard-link fails (cross-volume etc).
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
```

**Fall-back behavior**: if hard-linking fails (cross-volume), we transparently copy. Both produce identical functional behavior; hard-link is just faster.

**Cache key**: changing the embedding model means changing the lib_marker / model_marker filenames. Since those marker filenames are derived from `nark::embed::MODEL_NAME`, a model upgrade in nark automatically invalidates this cache (the markers stop matching and a fresh download triggers).

## 5. The vector adapter

### `bench/src/adapters/vector.rs`

```rust
//! Pure-Rust vector baseline. Embeds every doc on write, brute-force cosine
//! on search. Uses the same OnnxProvider as nark to keep model choice
//! constant (apples-to-apples comparison).

use anyhow::{anyhow, Context, Result};
use nark::embed::{init_embedding, EmbeddingProvider, OnnxProvider};
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
        // Pin to nark's MODEL_NAME constant so the recorded version stays
        // accurate when nark upgrades models.
        Ok(format!("vector {}", nark::embed::MODEL_NAME))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        model_cache::stage_into(&self.model_cache, workdir)
            .context("failed to stage embedding model into vector adapter workdir")?;
        let provider = init_embedding(workdir)
            .ok_or_else(|| anyhow!("ONNX init returned None after staging — model files corrupt?"))?;
        self.provider = Some(provider);
        self.workdir = Some(workdir.to_path_buf());
        self.embeddings.clear();
        Ok(())
    }

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

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        let t0 = Instant::now();
        let provider = self.provider.as_mut().ok_or_else(|| anyhow!("setup not called"))?;
        let query_emb = provider.embed_query(query)?;
        let mut scored: Vec<(String, f32)> = self.embeddings.iter()
            .map(|(id, emb)| (id.clone(), cosine_similarity(&query_emb, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results = scored.into_iter().take(k)
            .map(|(doc_id, score)| SearchResult { doc_id, score, snippet: None })
            .collect();
        Ok((results, SearchMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        }))
    }

    fn teardown(&mut self) -> Result<()> {
        self.provider = None;
        self.embeddings.clear();
        self.workdir = None;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}
```

## 6. nark adapter changes

The single behavioral change: `setup()` stages the embedding model into the workdir before calling `nark init`. The existing `nark write` flow then produces embeddings inline (nark's CLI auto-detects the model files and embeds via its `init_provider`).

### `bench/src/adapters/nark.rs` — `setup()` modification + constructor

```rust
pub struct NarkAdapter {
    workdir: Option<PathBuf>,
    staging: Option<PathBuf>,
    nark_bin: Option<PathBuf>,
    uuid_to_bench_id: HashMap<String, String>,
    /// Path to the shared model cache. If None, nark runs in BM25-only mode
    /// (no embedding files staged into workdir).
    model_cache: Option<PathBuf>,
}

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
}

// In Adapter impl:
fn setup(&mut self, workdir: &Path) -> Result<()> {
    let workdir = workdir.to_path_buf();
    let staging = workdir.join("_staging");
    std::fs::create_dir_all(&staging)?;
    let nark_bin = locate_nark_bin()?;
    self.workdir = Some(workdir.clone());
    self.staging = Some(staging);
    self.nark_bin = Some(nark_bin);
    self.uuid_to_bench_id.clear();

    // Stage model files into workdir so nark write produces embeddings inline.
    // If staging fails (e.g. cache missing or download failed), fall back to
    // BM25-only — adapter still functions, just without the cosine component.
    if let Some(cache) = &self.model_cache {
        if let Err(e) = crate::model_cache::stage_into(cache, &workdir) {
            eprintln!("nark adapter: model staging failed, continuing without embeddings: {}", e);
        }
    }

    self.run_nark(&["init"])?;
    Ok(())
}
```

**Rationale for fall-back vs hard-fail**: keeps the bench resilient in environments where the cache can't be populated (e.g. offline CI run without cache hit, or someone intentionally clearing the cache to test BM25-only behavior). The fall-back is logged to stderr so the operator sees what happened. The smoke test will catch any unintended fall-back by asserting `errors.is_empty()` AND by comparing measured cosine scores; if embeddings silently failed, the cosine component is zero and that would shift the baseline.

## 7. Factory + runner integration

### `bench/src/adapters/mod.rs`

```rust
pub mod fts5;
pub mod nark;
pub mod vector;

use crate::protocol::Adapter;
use anyhow::{anyhow, bail, Result};
use std::path::Path;

pub fn make_adapter(name: &str, model_cache: Option<&Path>) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        "nark" => Ok(Box::new(nark::NarkAdapter::new(model_cache.map(Path::to_path_buf)))),
        "vector" => {
            let cache = model_cache.ok_or_else(|| anyhow!("vector adapter requires model cache; ensure NARK_BENCH_MODEL_CACHE or default cache is set up"))?;
            Ok(Box::new(vector::VectorAdapter::new(cache.to_path_buf())))
        }
        other => bail!("unknown adapter: {}", other),
    }
}
```

### `bench/src/main.rs`

```rust
// inside Commands::Run handler
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

`ensure_ready` is idempotent — first call downloads, subsequent calls return instantly after the marker-file existence check.

## 8. Deferred code-review fixes

### 8.1 Microsecond latency — schema_version bump "1" → "2"

`bench/src/protocol.rs`:

```rust
pub struct WriteMetrics {
    pub latency_us: u64,        // was latency_ms
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}
pub struct SearchMetrics {
    pub latency_us: u64,        // was latency_ms
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}
```

`bench/src/result.rs`:

```rust
pub struct BenchResult {
    pub schema_version: String,  // "2" — was "1"
    // ...rest unchanged
}

pub struct PhaseMetrics {
    pub latency_p50_us: u64,    // was latency_p50_ms
    pub latency_p99_us: u64,    // was latency_p99_ms
    pub llm_tokens_in_total: u64,
    pub llm_tokens_out_total: u64,
}
```

`BenchResult::new` sets `schema_version = "2"`. The existing FTS5 adapter's `latency_ms` writes get renamed to `latency_us` and switch to `t0.elapsed().as_micros() as u64`. Same for the nark adapter's existing call sites. Same for `tasks/ir.rs::run_ir_task` which already sorts and percentile-computes — no algorithmic change, just field renames.

The smoke test's `recall_at_5` field assertion stays valid; it doesn't reference latency fields. The regression script doesn't reference latency fields either (only `.ir.*` metrics) so no change needed there.

**Migration**: there is none. We're pre-v1.0 and the schema is only consumed by code in this repo. The next baseline regeneration will be schema v2.

### 8.2 Query-count check in `bench/scripts/regression-check.sh`

Add before the metric loop:

```bash
new_q=$(jq -r '.ir.queries' "$new_file")
base_q=$(jq -r '.ir.queries' "$base_file")
if [[ "$new_q" -lt "$base_q" ]]; then
  printf 'REGRESSION: %s query count dropped from %d to %d (likely new adapter errors)\n' \
    "$(basename "$new_file")" "$base_q" "$new_q"
  fail=1
fi
```

This catches the case where a future change makes more queries error out, which would otherwise be invisible (the per-query averages might look fine even if fewer queries are contributing).

### 8.3 Filename sanitization in `BenchResult::write_to_disk`

```rust
fn sanitize(s: &str) -> String {
    s.replace('/', "_").replace(std::path::MAIN_SEPARATOR, "_")
}

pub fn write_to_disk(&self, output_dir: &Path) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(output_dir)?;
    let filename = format!(
        "{}-{}-{}.json",
        sanitize(&self.task), sanitize(&self.system), sanitize(&self.config)
    );
    let path = output_dir.join(filename);
    let json = serde_json::to_string_pretty(self)?;
    std::fs::write(&path, format!("{}\n", json))?;
    Ok(path)
}
```

Trivial defense against path-traversal in field names (e.g. a future config name like `nark-self/v2`).

## 9. Tests

### 9.1 New unit tests

In `bench/src/adapters/vector.rs`:

```rust
#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_perpendicular_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_opposite_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
}

#[test]
fn cosine_zero_vector_returns_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
}

#[test]
#[ignore = "requires staged model files; covered by integration smoke"]
fn vector_smoke_write_then_search() {
    // Setup with model_cache pointing at a real cache; write 2 docs, search,
    // assert top-1 is the more semantically similar doc.
}
```

In `bench/src/adapters/nark.rs`:

```rust
#[test]
#[ignore = "requires nark binary + staged model files; covered by integration smoke"]
fn nark_with_embeddings_hits_synonym_match() {
    // Setup with embeddings enabled, write a doc about "automobile",
    // search "car", assert at least one hit. Without embeddings this
    // would return nothing because there's no keyword overlap.
}
```

### 9.2 Smoke integration test updated

`bench/tests/smoke.rs`:

```rust
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
        assert!(stderr.contains(&format!("ir-{}-default.json", system)),
            "stderr did not mention {} result: {}", system, stderr);
    }

    for system in ["fts5", "nark", "vector"] {
        let path = out_dir.path().join(format!("ir-{}-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();

        assert_eq!(v["schema_version"], "2");
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert!(v["ir"]["recall_at_5"].is_number());
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5));
        assert!(v["ir_per_class"]["single_hop"].is_object());

        let errors = v["errors"].as_array().expect("errors field required");
        assert!(errors.is_empty(), "{} run had unexpected errors: {:?}", system, errors);
    }
}
```

### 9.3 Existing tests

All existing unit tests continue to pass after field renames (`latency_ms` → `latency_us`). Total: 19 from Phase 1a + 4 new cosine + 2 new ignored = 25 unit tests, 1 integration smoke.

## 10. Baseline regeneration

After all changes land, regenerate `bench/results/main/`:

| File | What changes |
|---|---|
| `ir-fts5-default.json` | schema_version "2"; `latency_*_us` fields. Rankings unchanged. |
| `ir-nark-default.json` | schema_version "2"; **rankings change materially** — nark now has embeddings, recall should improve from ~0.1 toward something noticeably higher (we expect single-hop and synonym classes to improve most) |
| `ir-vector-default.json` | **new file** — vector adapter's baseline |

The regression script's first run after baseline regeneration is a bootstrap pass (existing behavior — copies new files into `bench/results/main/` if no baseline present). After that, future PRs gate against the new baseline.

## 11. CI workflow update

`.github/workflows/bench-pr.yml` gains an `actions/cache` step before the benchmark run:

```yaml
- name: Cache nark-bench embedding model
  uses: actions/cache@v4
  with:
    path: ~/.cache/nark-bench/models
    key: nark-bench-model-${{ hashFiles('src/embed/download.rs', 'src/embed/mod.rs') }}
    restore-keys: |
      nark-bench-model-
```

First PR after this lands: cache miss → ~30–60s model download. All subsequent PRs: cache hit, no download. Cache invalidates only when `src/embed/download.rs` or `src/embed/mod.rs` change (i.e. when nark's model choice or download logic changes).

The workflow's existing `src/**` path filter already covers nark's embed module, so no path-filter change is needed.

## 12. Phasing within Phase 1b

Suggested order of work for the implementation plan:

1. **nark library extraction** — add `lib.rs`, verify nark binary still builds and runs.
2. **Pub-API exposure in `nark::embed::download`** — add `install_into(target: &Path)` if not present.
3. **Schema bump + microsecond latency** — small, mechanical, isolated; lands first to avoid coupling with new-adapter work.
4. **Model cache module** — `bench/src/model_cache.rs` with unit tests against a fake cache layout.
5. **Vector adapter** — new file, cosine unit tests, ignored smoke test.
6. **Nark adapter setup() change** — stage model files; verify nark with embeddings produces nonzero cosine scores.
7. **Runner integration** — `main.rs` calls `ensure_ready`; `make_adapter` takes the cache.
8. **Smoke integration test update** — run all three systems; zero-errors assertion.
9. **Query-count check + filename sanitization** — small fixes.
10. **Regenerate baseline + CI cache step + commit.**

Each step ends with a clean build + tests passing.

## 13. Open items

- **`nark embed download` public API surface** — needs to expose `install_into(target: &Path)` and `onnx_dylib_name()` as public. Implementer checks what's already `pub` and adds thin wrappers if needed. Estimated: <30 lines in `nark/src/embed/download.rs`.
- **Embedding scores on synthetic-tiny** — the corpus was designed to be tough for BM25; with embeddings, recall should improve markedly (synonym class especially). If it doesn't, that's diagnostic info, not a bug — the synthetic queries may need expansion before nark-self lands.
- **First-run download in CI** — 30–60s on first PR after merge; tolerable, but worth flagging in the rollout PR description so reviewers don't wonder why CI is slow once.
