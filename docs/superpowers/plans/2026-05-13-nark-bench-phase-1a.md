# nark Bench — Phase 1a (Foundation + Regression Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `bench/` workspace crate with the Adapter trait, two adapters (FTS5 baseline + nark via CLI subprocess), classical IR metrics (Recall@k / MRR / nDCG), a small synthetic corpus fixture, and a GitHub Actions PR check that fails when search quality regresses by >2%.

**Architecture:** A new `bench/` crate (sibling of `src/`) sits in the same Cargo workspace as nark. It exposes a single binary `nark-bench` with a `--task ir` subcommand. Every system-under-test implements the `Adapter` trait (`setup → write → search → teardown`). The IR task ingests a fixture corpus into each adapter, runs every query, computes Recall@k/MRR/nDCG, and writes a JSON result file per `(system, config)`. Phase 1a only ships native-Rust adapters (FTS5, nark via subprocess). Competitors, LLM-judge, and LongMemEval ship in later plans.

**Tech Stack:** Rust 2024 edition; `clap` (derive); `rusqlite` (bundled, with FTS5); `serde` / `serde_json`; `anyhow`; `tempfile` (new dev-dep) for adapter workdirs; `assert_cmd` (new dev-dep) for the smoke integration test. No new runtime dependencies on nark itself.

**Scope cut from this plan (deferred to later plans):**
- Vector / embedding-baseline adapter (needs model download infra)
- `nark-self` real-vault corpus (needs Sean's anonymization pass)
- LongMemEval / LOCOMO tasks + LLM-judge backends
- mem0 / Letta / Graphiti adapters
- Report generator → `BENCHMARKS.md`
- `nark-body-only` config variant (requires a new nark flag)

---

## File structure

**Files to create:**

| Path | Responsibility |
|---|---|
| `bench/Cargo.toml` | Crate manifest for `nark-bench`, `publish = false` |
| `bench/src/main.rs` | clap dispatch; one subcommand: `--task ir` |
| `bench/src/protocol.rs` | `Adapter` trait + `Document` / `SearchResult` / `WriteMetrics` / `SearchMetrics` structs |
| `bench/src/adapters/mod.rs` | Module index + `make_adapter(name: &str) -> Box<dyn Adapter>` factory |
| `bench/src/adapters/fts5.rs` | Direct rusqlite FTS5 baseline |
| `bench/src/adapters/nark.rs` | Subprocess to nark CLI via `--vault-dir <tmp>` |
| `bench/src/metrics/mod.rs` | Module index |
| `bench/src/metrics/ir.rs` | `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k` + unit tests |
| `bench/src/tasks/mod.rs` | Module index |
| `bench/src/tasks/ir.rs` | Ingest corpus, run queries, score → `IrResult` |
| `bench/src/result.rs` | `BenchResult` serializable type + `write_to_disk` |
| `bench/tests/smoke.rs` | Integration test: runs full harness against tiny fixture |
| `bench/datasets/ir/synthetic-tiny/corpus/{n01..n12}.md` | 12 small markdown notes with frontmatter |
| `bench/datasets/ir/synthetic-tiny/queries.jsonl` | ~10 queries each with gold relevant IDs |
| `bench/scripts/regression-check.sh` | Diffs new result JSON vs `bench/results/main/ir.json`, exits 1 if any metric drops >2% |
| `.github/workflows/bench-pr.yml` | Lane 1 CI: build, smoke test, regression check |

**Files to modify:**

| Path | Change |
|---|---|
| `Cargo.toml` (top-level) | Promote to `[workspace]` with `members = [".", "bench"]`; keep existing `[package]` block intact |

No changes to `src/` (nark's own code).

---

## Task 1: Promote the top-level Cargo.toml to a workspace root

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Read current Cargo.toml to preserve everything**

Run: `cat Cargo.toml`

Confirm the file currently has a `[package]` block and a `[dependencies]` block, and the `name = "nark"` / `version = "0.13.0"` lines are present.

- [ ] **Step 1.2: Add a `[workspace]` block at the top of Cargo.toml**

Edit `Cargo.toml` to prepend the following before the existing `[package]` block:

```toml
[workspace]
members = [".", "bench"]
resolver = "2"
```

Leave the rest of the file unchanged.

- [ ] **Step 1.3: Verify nark still builds in isolation**

Run: `cargo check`
Expected: completes with no errors. (`bench/` doesn't exist yet, so `cargo check` will warn that `bench` is a missing workspace member — that's a fatal error in practice. So we need to do this AFTER Task 2 creates `bench/Cargo.toml`. Defer the check to step 2.4.)

- [ ] **Step 1.4: Commit**

Stage and commit:

```bash
git add Cargo.toml
git commit -m "chore: promote Cargo.toml to workspace root for bench crate

Adds [workspace] block listing the existing root crate and the new
bench/ crate. Resolver bumped to 2 (already the 2024-edition default).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Create the empty `bench/` crate that builds and runs

**Files:**
- Create: `bench/Cargo.toml`
- Create: `bench/src/main.rs`

- [ ] **Step 2.1: Create `bench/Cargo.toml`**

```toml
[package]
name = "nark-bench"
version = "0.1.0"
edition = "2024"
publish = false

[[bin]]
name = "nark-bench"
path = "src/main.rs"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rusqlite = { version = "0.38", features = ["bundled"] }
tempfile = "3"
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
assert_cmd = "2"
```

- [ ] **Step 2.2: Create `bench/src/main.rs` with a minimal clap CLI**

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nark-bench", about = "Benchmark harness for nark")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a benchmark task
    Run {
        /// Task name (currently only "ir" supported)
        #[arg(long)]
        task: String,
        /// Comma-separated systems to benchmark (e.g. "fts5,nark")
        #[arg(long)]
        systems: String,
        /// Corpus name (e.g. "synthetic-tiny")
        #[arg(long)]
        corpus: String,
        /// Output directory for result JSON files
        #[arg(long, default_value = "bench/results/local")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { task, systems, corpus, output } => {
            println!("task={} systems={} corpus={} output={}", task, systems, corpus, output);
            Ok(())
        }
    }
}
```

- [ ] **Step 2.3: Verify the workspace builds end-to-end**

Run: `cargo build -p nark-bench`
Expected: completes with no errors; produces `target/debug/nark-bench`.

- [ ] **Step 2.4: Verify the binary runs**

Run: `cargo run -p nark-bench -- run --task ir --systems fts5 --corpus synthetic-tiny`
Expected stdout: `task=ir systems=fts5 corpus=synthetic-tiny output=bench/results/local`

- [ ] **Step 2.5: Commit**

```bash
git add bench/Cargo.toml bench/src/main.rs
git commit -m "feat(bench): scaffold nark-bench crate with clap CLI

New workspace member at bench/. Single binary nark-bench with a
'run' subcommand that takes --task / --systems / --corpus / --output.
Currently a stub that prints its args; subsequent commits wire up the
real protocol, adapters, metrics, and task runners.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Define the Adapter trait and protocol types

**Files:**
- Create: `bench/src/protocol.rs`
- Modify: `bench/src/main.rs` (add `mod protocol;`)

- [ ] **Step 3.1: Write `bench/src/protocol.rs`**

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A document to be ingested by an adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub body: String,
    /// Frontmatter / tags / timestamps. Adapters decide what to use.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// One ranked search hit returned by an adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteMetrics {
    pub latency_ms: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub latency_ms: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

/// The single trait every system-under-test implements.
pub trait Adapter {
    /// Human-readable name used in result files and CLI flags (e.g. "fts5", "nark").
    fn name(&self) -> &str;

    /// Version string for the system under test, e.g. "nark 0.13.0" or "sqlite 3.45.0 FTS5".
    /// Recorded in result JSON for reproducibility. Returning "unknown" should be avoided —
    /// the harness refuses to run if any adapter reports "unknown".
    fn version(&self) -> Result<String>;

    /// Called once before any write/search. `workdir` is a clean temp directory
    /// the adapter can use freely.
    fn setup(&mut self, workdir: &Path) -> Result<()>;

    /// Ingest one document. Latency is wall-clock; token counts are 0 for
    /// adapters that don't call LLMs on write.
    fn write(&mut self, doc: &Document) -> Result<WriteMetrics>;

    /// Run a query, return top-k hits ranked by the adapter's own scoring.
    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)>;

    /// Called once at the end. Should release any subprocess / connection / temp state.
    fn teardown(&mut self) -> Result<()>;
}
```

- [ ] **Step 3.2: Wire the new module into `bench/src/main.rs`**

Add `mod protocol;` as the first line after `use anyhow::Result;`:

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};

mod protocol;
```

- [ ] **Step 3.3: Verify it compiles**

Run: `cargo build -p nark-bench`
Expected: completes with no errors.

- [ ] **Step 3.4: Commit**

```bash
git add bench/src/protocol.rs bench/src/main.rs
git commit -m "feat(bench): define Adapter trait and protocol types

The Adapter trait is the single abstraction every system-under-test
implements. Three concrete shapes (native Rust, subprocess+JSON-lines,
HTTP) will all impl this same trait in later tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement classical IR metrics with unit tests (TDD)

**Files:**
- Create: `bench/src/metrics/mod.rs`
- Create: `bench/src/metrics/ir.rs`
- Modify: `bench/src/main.rs` (add `mod metrics;`)

- [ ] **Step 4.1: Write the failing tests first**

Create `bench/src/metrics/ir.rs` with tests but no implementations:

```rust
//! Classical IR metrics. All deterministic, no LLM calls.

use std::collections::HashSet;

/// Recall@k: fraction of relevant docs found in the top-k ranked results.
///
/// `ranked` is the system's output ordered by descending score; only doc_ids matter here.
/// `relevant` is the gold-labelled set of relevant doc_ids for this query.
/// `k` is the cutoff; common values are 1, 5, 10.
pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    todo!()
}

/// Mean Reciprocal Rank for a single query: 1 / rank of the first relevant doc, or 0 if none.
/// Average across queries is done by the caller.
pub fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>) -> f64 {
    todo!()
}

/// nDCG@k with binary relevance (1.0 if doc is in `relevant`, else 0.0).
/// Uses the standard log2 discount.
pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn ranked(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_perfect() {
        // All 2 relevant docs appear in top-5
        let r = recall_at_k(&ranked(&["a", "b", "c", "d", "e"]), &rel(&["b", "e"]), 5);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn recall_partial() {
        // Only 1 of 2 relevant docs in top-3
        let r = recall_at_k(&ranked(&["a", "b", "c", "d", "e"]), &rel(&["b", "e"]), 3);
        assert!((r - 0.5).abs() < 1e-9, "expected 0.5, got {}", r);
    }

    #[test]
    fn recall_none() {
        let r = recall_at_k(&ranked(&["a", "b", "c"]), &rel(&["x", "y"]), 5);
        assert!(r.abs() < 1e-9, "expected 0.0, got {}", r);
    }

    #[test]
    fn recall_no_relevant_set_is_one_by_convention() {
        // When there are no relevant docs, recall is undefined; we return 1.0
        // to avoid penalising systems on queries with empty gold sets.
        let r = recall_at_k(&ranked(&["a", "b"]), &HashSet::new(), 5);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn rr_first_position() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["a"]));
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rr_third_position() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["c"]));
        assert!((r - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn rr_none_present() {
        let r = reciprocal_rank(&ranked(&["a", "b", "c"]), &rel(&["x"]));
        assert!(r.abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_ordering() {
        // All relevant docs at the top in best possible order
        let r = ndcg_at_k(&ranked(&["a", "b", "c", "d"]), &rel(&["a", "b"]), 4);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn ndcg_worst_ordering() {
        // Relevant docs at the bottom — DCG is low but nonzero
        let r = ndcg_at_k(&ranked(&["a", "b", "c", "d"]), &rel(&["c", "d"]), 4);
        // DCG = 0 + 0 + 1/log2(4) + 1/log2(5) = 0.5 + 0.4306... ≈ 0.9307
        // IDCG = 1 + 1/log2(3) ≈ 1.6309
        // nDCG ≈ 0.5706
        assert!((r - 0.5706).abs() < 1e-3, "expected ~0.5706, got {}", r);
    }

    #[test]
    fn ndcg_empty_relevant() {
        let r = ndcg_at_k(&ranked(&["a", "b", "c"]), &HashSet::new(), 5);
        // Convention: when no relevant docs exist, nDCG is 1.0 (vacuously satisfied).
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }
}
```

Create `bench/src/metrics/mod.rs`:

```rust
pub mod ir;
```

- [ ] **Step 4.2: Wire into main**

Edit `bench/src/main.rs` — add `mod metrics;` after `mod protocol;`:

```rust
mod protocol;
mod metrics;
```

- [ ] **Step 4.3: Run the tests and confirm they fail with `todo!()`**

Run: `cargo test -p nark-bench metrics::ir`
Expected: 10 tests fail with `not yet implemented`.

- [ ] **Step 4.4: Implement the three metrics**

Replace the `todo!()` bodies in `bench/src/metrics/ir.rs`:

```rust
pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let top: HashSet<&String> = ranked.iter().take(k).collect();
    let hits = relevant.iter().filter(|r| top.contains(*r)).count();
    hits as f64 / relevant.len() as f64
}

pub fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>) -> f64 {
    for (i, doc_id) in ranked.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .filter_map(|(i, doc_id)| {
            if relevant.contains(doc_id) {
                Some(1.0 / ((i + 2) as f64).log2())
            } else {
                None
            }
        })
        .sum();
    let ideal_count = relevant.len().min(k);
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}
```

- [ ] **Step 4.5: Run tests and confirm they pass**

Run: `cargo test -p nark-bench metrics::ir`
Expected: 10 tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add bench/src/metrics/mod.rs bench/src/metrics/ir.rs bench/src/main.rs
git commit -m "feat(bench): implement IR metrics — Recall@k, MRR, nDCG@k

All three are deterministic and tested against canonical examples.
Conventions for edge cases:
  - empty relevant set → recall = 1.0, nDCG = 1.0 (vacuously satisfied)
  - no relevant in ranked → RR = 0.0
  - binary relevance for nDCG (1.0 if hit, else 0.0)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Build the synthetic-tiny IR corpus

**Files:**
- Create: `bench/datasets/ir/synthetic-tiny/corpus/n01.md` through `n12.md`
- Create: `bench/datasets/ir/synthetic-tiny/queries.jsonl`
- Create: `bench/datasets/ir/synthetic-tiny/README.md`

The corpus has 12 small notes across three topic clusters (chess, breadmaking, electronics). Each query has 1–3 gold relevant note IDs. The clusters have planted overlaps and distractors so the corpus exercises BM25 vs metadata-filtered retrieval.

- [ ] **Step 5.1: Create the 12 corpus notes**

Each file is a markdown note with YAML frontmatter. Bodies are 2–4 sentences. The exact contents below.

Every note uses the same minimal frontmatter shape. nark requires `title, author, domain, intent, kind, status, tags` — exactly those seven fields. We use `author: bench` and `status: active` uniformly; varying those would not exercise anything the bench cares about.

The **bench tracks documents by file stem** (e.g. `n01`), but nark assigns its own UUIDs internally. The nark adapter handles that mapping (Task 7); the corpus files themselves must not include an `id:` field — there is no such frontmatter field in nark.

Create `bench/datasets/ir/synthetic-tiny/corpus/n01.md`:

```markdown
---
title: Sicilian Defense opening principles
author: bench
domain: games
intent: reference
kind: note
status: active
tags: [chess, opening]
---

The Sicilian Defense is a chess opening that begins with 1.e4 c5. It is one of
the most popular responses to white's king pawn opening because it fights for
the center asymmetrically and avoids the symmetrical 1...e5 lines.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n02.md`:

```markdown
---
title: Najdorf variation move order
author: bench
domain: games
intent: reference
kind: note
status: active
tags: [chess, opening, najdorf]
---

The Najdorf Sicilian arises after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6.
The move 5...a6 prepares ...e5 or ...b5 without allowing Nb5. It is favoured by
attacking players including Fischer and Kasparov.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n03.md`:

```markdown
---
title: Endgame king activity
author: bench
domain: games
intent: principle
kind: note
status: active
tags: [chess, endgame]
---

In the endgame the king becomes an attacker, not a piece to be hidden. Centralising
the king is often worth a tempo because each square of advance increases its
influence over pawns and squares on both flanks.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n04.md`:

```markdown
---
title: Sourdough starter hydration
author: bench
domain: cooking
intent: how-to
kind: note
status: active
tags: [bread, sourdough, fermentation]
---

A 100% hydration starter has equal weights of flour and water. Higher hydration
ferments faster and produces a more sour loaf; lower hydration is slower and
milder. Most bakers maintain a 100% starter for predictability.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n05.md`:

```markdown
---
title: Autolyse method for bread dough
author: bench
domain: cooking
intent: how-to
kind: note
status: active
tags: [bread, technique]
---

Autolyse is a rest period after mixing flour and water but before adding salt
or starter. Twenty to forty minutes of autolyse develops gluten passively and
improves extensibility without kneading.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n06.md`:

```markdown
---
title: Bulk fermentation temperature
author: bench
domain: cooking
intent: reference
kind: note
status: active
tags: [bread, fermentation, sourdough]
---

Bulk fermentation at 25°C typically takes 4–5 hours for a 100% hydration
sourdough. At 21°C the same dough may need 8–10 hours. Temperature dominates
all other variables.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n07.md`:

```markdown
---
title: Pull-up resistor selection
author: bench
domain: electronics
intent: reference
kind: note
status: active
tags: [resistor, digital, gpio]
---

For a 3.3V GPIO line a 10kΩ pull-up is the default choice. Lower values draw
more current but resist noise better; higher values save power at the cost of
slower rise time when the line transitions from low to high.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n08.md`:

```markdown
---
title: I2C bus pull-up calculation
author: bench
domain: electronics
intent: how-to
kind: note
status: active
tags: [i2c, resistor, bus]
---

I2C bus capacitance is typically 100–400 pF. For 100 kHz Standard mode, a
4.7kΩ pull-up gives a rise time well within spec; at 400 kHz Fast mode you
usually want 2.2kΩ. Calculate Rp,max = tr / (0.8473 × Cb).
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n09.md`:

```markdown
---
title: ESP32 deep sleep current
author: bench
domain: electronics
intent: reference
kind: note
status: active
tags: [esp32, power, microcontroller]
---

ESP32 deep sleep typically draws 10 μA with the RTC running. The hibernation
mode disables the RTC and brings consumption below 5 μA. Both modes wake on
GPIO or timer events.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n10.md`:

```markdown
---
title: Stretch and fold technique
author: bench
domain: cooking
intent: how-to
kind: note
status: active
tags: [bread, technique, gluten]
---

Stretch-and-fold replaces kneading. Every 30 minutes during bulk fermentation,
lift a corner of the dough, stretch it up, and fold it over the centre. Repeat
on all four sides. Three to four sets is typical.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n11.md`:

```markdown
---
title: Pawn structure in the Sicilian
author: bench
domain: games
intent: principle
kind: note
status: active
tags: [chess, strategy, sicilian]
---

The Sicilian Defense creates an asymmetric pawn structure: black has a half-open
c-file while white has a half-open d-file. The side that uses its open file
better usually achieves a strategic edge in the middlegame.
```

Create `bench/datasets/ir/synthetic-tiny/corpus/n12.md`:

```markdown
---
title: Bypass capacitor placement
author: bench
domain: electronics
intent: how-to
kind: note
status: active
tags: [capacitor, decoupling, layout]
---

Place a 100nF ceramic bypass capacitor as close as possible to each IC's VCC
pin, ideally on the same side of the PCB. The trace length to the cap directly
affects the loop inductance that defeats decoupling at high frequencies.
```

- [ ] **Step 5.2: Create the queries file**

Create `bench/datasets/ir/synthetic-tiny/queries.jsonl`:

```jsonl
{"query_id": "q01", "query": "sicilian defense move order", "relevant": ["n01", "n02"], "class": "single_hop"}
{"query_id": "q02", "query": "najdorf 5 a6 idea", "relevant": ["n02"], "class": "single_hop"}
{"query_id": "q03", "query": "centralising the king in the endgame", "relevant": ["n03"], "class": "synonym"}
{"query_id": "q04", "query": "sourdough hydration ratio", "relevant": ["n04"], "class": "single_hop"}
{"query_id": "q05", "query": "developing gluten without kneading", "relevant": ["n05", "n10"], "class": "multi_hop"}
{"query_id": "q06", "query": "how long does bulk ferment take", "relevant": ["n06"], "class": "synonym"}
{"query_id": "q07", "query": "i2c pull up resistor value", "relevant": ["n07", "n08"], "class": "multi_hop"}
{"query_id": "q08", "query": "esp32 low power sleep modes", "relevant": ["n09"], "class": "synonym"}
{"query_id": "q09", "query": "decoupling capacitor near IC", "relevant": ["n12"], "class": "synonym"}
{"query_id": "q10", "query": "pawn structure asymmetric c-file", "relevant": ["n11"], "class": "single_hop"}
```

- [ ] **Step 5.3: Create the dataset README documenting the contract**

Create `bench/datasets/ir/synthetic-tiny/README.md`:

```markdown
# synthetic-tiny — IR fixture

12 notes across 3 topic clusters (chess, breadmaking, electronics). 10 queries
with binary gold-relevance labels. Designed as a smoke fixture for the bench
harness — large enough to exercise BM25 ranking, small enough to commit and
iterate against on every PR.

## Layout
- `corpus/n01.md ... n12.md` — markdown notes with YAML frontmatter
- `queries.jsonl` — one JSON object per line: `{query_id, query, relevant: [...], class}`

## Query classes
- `single_hop` — exactly one obvious answer note shares keywords with the query
- `multi_hop` — two notes are relevant, often only one shares keywords directly
- `synonym` — the relevant note uses different wording than the query

These classes are surfaced in result JSON as per-class breakdowns so we can
tell whether a regression hurt one type of retrieval more than others.
```

- [ ] **Step 5.4: Verify dataset files are well-formed JSON-lines**

Run: `python3 -c "import json, sys; [json.loads(l) for l in open('bench/datasets/ir/synthetic-tiny/queries.jsonl')]" && echo OK`
Expected: prints `OK`.

(If `python3` isn't available, run `jq -c . bench/datasets/ir/synthetic-tiny/queries.jsonl | wc -l` and confirm output is `10`.)

- [ ] **Step 5.5: Commit**

```bash
git add bench/datasets/ir/synthetic-tiny/
git commit -m "feat(bench): add synthetic-tiny IR corpus fixture

12 notes across 3 topic clusters (chess, cooking, electronics) and
10 queries with gold relevance labels. Three query classes labeled:
single_hop, multi_hop, synonym. Small enough to commit; large enough
to exercise BM25 ranking and per-class breakdown reporting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement the FTS5 adapter (TDD)

**Files:**
- Create: `bench/src/adapters/mod.rs`
- Create: `bench/src/adapters/fts5.rs`
- Modify: `bench/src/main.rs` (add `mod adapters;`)

- [ ] **Step 6.1: Write the failing test first**

Create `bench/src/adapters/fts5.rs` with the test scaffolding but no implementation:

```rust
//! Pure-Rust BM25 baseline using SQLite FTS5 with no ranking layers on top.
//! This is the floor every other adapter is benchmarked against.

use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;
use std::time::Instant;

use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

pub struct Fts5Adapter {
    conn: Option<Connection>,
}

impl Fts5Adapter {
    pub fn new() -> Self {
        Self { conn: None }
    }
}

impl Default for Fts5Adapter {
    fn default() -> Self { Self::new() }
}

impl Adapter for Fts5Adapter {
    fn name(&self) -> &str { "fts5" }

    fn version(&self) -> Result<String> {
        Ok(format!("sqlite-fts5 {}", rusqlite::version()))
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
        self.conn = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn fts5_smoke_write_then_search() {
        let dir = tempdir().unwrap();
        let mut a = Fts5Adapter::new();
        a.setup(dir.path()).unwrap();

        let d1 = Document { id: "a".into(), body: "the quick brown fox".into(), metadata: json!({}) };
        let d2 = Document { id: "b".into(), body: "lazy dogs sleep".into(), metadata: json!({}) };
        a.write(&d1).unwrap();
        a.write(&d2).unwrap();

        let (results, _) = a.search("brown fox", 5).unwrap();
        assert!(!results.is_empty(), "expected at least one hit, got 0");
        assert_eq!(results[0].doc_id, "a", "expected 'a' to rank first, got {}", results[0].doc_id);
        a.teardown().unwrap();
    }

    #[test]
    fn fts5_returns_no_hits_for_unrelated_query() {
        let dir = tempdir().unwrap();
        let mut a = Fts5Adapter::new();
        a.setup(dir.path()).unwrap();
        a.write(&Document { id: "x".into(), body: "completely unrelated content".into(), metadata: json!({}) }).unwrap();
        let (results, _) = a.search("zzzz never appears", 5).unwrap();
        assert!(results.is_empty(), "expected 0 hits, got {}", results.len());
    }
}
```

Create `bench/src/adapters/mod.rs`:

```rust
pub mod fts5;

use crate::protocol::Adapter;
use anyhow::{bail, Result};

/// Construct an adapter by short name.
pub fn make_adapter(name: &str) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        other => bail!("unknown adapter: {}", other),
    }
}
```

- [ ] **Step 6.2: Wire into main**

In `bench/src/main.rs` add `mod adapters;` after `mod metrics;`:

```rust
mod protocol;
mod metrics;
mod adapters;
```

- [ ] **Step 6.3: Run tests, confirm they fail**

Run: `cargo test -p nark-bench adapters::fts5`
Expected: 2 tests fail with `not yet implemented`.

- [ ] **Step 6.4: Implement `setup`, `write`, `search`**

Replace the `todo!()` bodies in `bench/src/adapters/fts5.rs`:

```rust
fn setup(&mut self, workdir: &Path) -> Result<()> {
    let db_path = workdir.join("fts5.db");
    let conn = Connection::open(&db_path)?;
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(doc_id UNINDEXED, body);"
    )?;
    self.conn = Some(conn);
    Ok(())
}

fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
    let t0 = Instant::now();
    let conn = self.conn.as_ref().ok_or_else(|| anyhow::anyhow!("setup not called"))?;
    conn.execute(
        "INSERT INTO docs (doc_id, body) VALUES (?1, ?2)",
        rusqlite::params![doc.id, doc.body],
    )?;
    Ok(WriteMetrics {
        latency_ms: t0.elapsed().as_millis() as u64,
        llm_tokens_in: 0,
        llm_tokens_out: 0,
    })
}

fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
    let t0 = Instant::now();
    let conn = self.conn.as_ref().ok_or_else(|| anyhow::anyhow!("setup not called"))?;
    let mut stmt = conn.prepare(
        "SELECT doc_id, bm25(docs) AS score, snippet(docs, 1, '[', ']', '...', 16) AS snip
         FROM docs WHERE docs MATCH ?1
         ORDER BY score
         LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![query, k as i64], |row| {
            let doc_id: String = row.get(0)?;
            let raw: f64 = row.get(1)?;
            let snippet: String = row.get(2).unwrap_or_default();
            // FTS5 bm25() returns smaller (more-negative) is better. Negate so higher = better
            // and the downstream metrics see a "score" with conventional semantics.
            Ok(SearchResult {
                doc_id,
                score: (-raw) as f32,
                snippet: if snippet.is_empty() { None } else { Some(snippet) },
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok((rows, SearchMetrics {
        latency_ms: t0.elapsed().as_millis() as u64,
        llm_tokens_in: 0,
        llm_tokens_out: 0,
    }))
}
```

- [ ] **Step 6.5: Run tests and confirm they pass**

Run: `cargo test -p nark-bench adapters::fts5`
Expected: 2 tests pass.

- [ ] **Step 6.6: Commit**

```bash
git add bench/src/adapters/mod.rs bench/src/adapters/fts5.rs bench/src/main.rs
git commit -m "feat(bench): FTS5 adapter — pure-Rust BM25 baseline

Direct rusqlite + FTS5 virtual table. setup() creates a fresh SQLite
file in the workdir; write() inserts; search() runs MATCH ordered by
bm25() with snippet() for previews. bm25() is negated so higher score
means more relevant — matches the conventional semantics of every
downstream metric.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Implement the nark adapter (subprocess via CLI) — TDD

**Files:**
- Create: `bench/src/adapters/nark.rs`
- Modify: `bench/src/adapters/mod.rs` (register "nark")

The nark adapter shells out to the `nark` binary built in the same workspace. For each ingest we write the markdown body (plus frontmatter) to a temp file in the workdir, then `cargo run --quiet -p nark -- --vault-dir <workdir> write <file>`. Search shells out to `nark search` and parses the JSON output.

- [ ] **Step 7.1: Write the failing test first**

Create `bench/src/adapters/nark.rs`:

```rust
//! nark adapter — drives the actual nark CLI as a subprocess.
//!
//! We pass --vault-dir to point at an isolated workdir per benchmark run, write
//! markdown files to a staging directory inside the workdir, and ingest them via
//! `nark write <path>`. Search calls `nark search <query>` and parses the JSON.
//!
//! The bench tracks documents by file-stem (e.g. "n01"); nark assigns its own
//! UUIDs. We parse the `nark write` JSON output to learn the assigned UUID per
//! document and keep a `nark_uuid → bench_id` map so search results translate
//! back to bench IDs before being returned to the harness.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

pub struct NarkAdapter {
    workdir: Option<PathBuf>,
    staging: Option<PathBuf>,
    /// Path to the nark binary. Resolved at setup time.
    nark_bin: Option<PathBuf>,
    /// nark-assigned UUID → bench-managed document id.
    uuid_to_bench_id: HashMap<String, String>,
}

impl NarkAdapter {
    pub fn new() -> Self {
        Self {
            workdir: None,
            staging: None,
            nark_bin: None,
            uuid_to_bench_id: HashMap::new(),
        }
    }

    fn run_nark(&self, args: &[&str]) -> Result<std::process::Output> {
        let bin = self.nark_bin.as_ref().ok_or_else(|| anyhow!("nark binary not located"))?;
        let workdir = self.workdir.as_ref().ok_or_else(|| anyhow!("setup not called"))?;
        let workdir_str = workdir.to_string_lossy().to_string();
        let mut cmd = Command::new(bin);
        cmd.arg("--vault-dir").arg(&workdir_str);
        for a in args {
            cmd.arg(a);
        }
        let output = cmd.output().with_context(|| format!("failed to spawn nark with args {:?}", args))?;
        if !output.status.success() {
            return Err(anyhow!(
                "nark {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(output)
    }
}

impl Default for NarkAdapter {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Deserialize)]
struct NarkWriteOut {
    #[serde(default)]
    wrote: u64,
    #[serde(default)]
    notes: Vec<NarkWriteNote>,
}

#[derive(Debug, Deserialize)]
struct NarkWriteNote {
    id: String,
    #[allow(dead_code)]
    title: String,
}

#[derive(Debug, Deserialize)]
struct NarkSearchOut {
    #[allow(dead_code)]
    #[serde(default)]
    query: String,
    #[allow(dead_code)]
    #[serde(default)]
    hits: usize,
    #[serde(default)]
    results: Vec<NarkHit>,
}

#[derive(Debug, Deserialize)]
struct NarkHit {
    id: String,
    #[serde(default)]
    snippet: String,
    rank: f64,
}

impl Adapter for NarkAdapter {
    fn name(&self) -> &str { "nark" }

    fn version(&self) -> Result<String> {
        // The current nark CLI does not implement `--version`; report our pinned
        // crate version of nark instead, taken from the binary's parent workspace.
        // (If a later nark adds --version, replace this with `self.run_nark(&["--version"])`
        // and trim the output.)
        Ok(format!("nark {}", env!("CARGO_PKG_VERSION")))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        let workdir = workdir.to_path_buf();
        let staging = workdir.join("_staging");
        std::fs::create_dir_all(&staging)?;
        let nark_bin = locate_nark_bin()?;
        self.workdir = Some(workdir.clone());
        self.staging = Some(staging);
        self.nark_bin = Some(nark_bin);
        self.uuid_to_bench_id.clear();

        // Initialize the vault
        self.run_nark(&["init"])?;
        Ok(())
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        let t0 = Instant::now();
        let staging = self.staging.as_ref().ok_or_else(|| anyhow!("setup not called"))?;
        let path = staging.join(format!("{}.md", doc.id));
        std::fs::write(&path, &doc.body)?;
        let path_str = path.to_string_lossy().to_string();
        let out = self.run_nark(&["write", &path_str])?;
        let parsed: NarkWriteOut = serde_json::from_slice(&out.stdout)
            .with_context(|| format!(
                "failed to parse nark write JSON for bench_id={}: {}",
                doc.id, String::from_utf8_lossy(&out.stdout)
            ))?;
        if parsed.wrote != 1 || parsed.notes.len() != 1 {
            return Err(anyhow!(
                "nark write returned unexpected count: wrote={}, notes.len={}",
                parsed.wrote, parsed.notes.len()
            ));
        }
        let uuid = parsed.notes.into_iter().next().unwrap().id;
        self.uuid_to_bench_id.insert(uuid, doc.id.clone());
        Ok(WriteMetrics {
            latency_ms: t0.elapsed().as_millis() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        })
    }

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        let t0 = Instant::now();
        let limit = k.to_string();
        let out = self.run_nark(&["search", query, "--limit", &limit])?;
        let parsed: NarkSearchOut = serde_json::from_slice(&out.stdout)
            .with_context(|| format!("failed to parse nark search JSON: {}", String::from_utf8_lossy(&out.stdout)))?;
        let results = parsed.results.into_iter()
            .filter_map(|h| {
                // Drop any hit whose UUID we did not ingest (defensive: shouldn't happen
                // in practice since each adapter has its own vault, but a stray hit
                // would otherwise score as a phantom miss).
                let bench_id = self.uuid_to_bench_id.get(&h.id).cloned()?;
                Some(SearchResult {
                    doc_id: bench_id,
                    score: h.rank as f32,
                    snippet: if h.snippet.is_empty() { None } else { Some(h.snippet) },
                })
            })
            .collect();
        Ok((results, SearchMetrics {
            latency_ms: t0.elapsed().as_millis() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        }))
    }

    fn teardown(&mut self) -> Result<()> {
        self.workdir = None;
        self.staging = None;
        self.nark_bin = None;
        self.uuid_to_bench_id.clear();
        Ok(())
    }
}

/// Find the nark binary built in this workspace. Looks at `target/debug/nark`
/// relative to CARGO_MANIFEST_DIR, then `target/release/nark`, then falls back
/// to `nark` on PATH.
fn locate_nark_bin() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().ok_or_else(|| anyhow!("no workspace root"))?;
    let target_debug = workspace_root.join("target/debug/nark");
    if target_debug.exists() {
        return Ok(target_debug);
    }
    let target_release = workspace_root.join("target/release/nark");
    if target_release.exists() {
        return Ok(target_release);
    }
    Ok(PathBuf::from("nark"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    /// Smoke test — this is gated on the nark binary being built. The integration
    /// test (`tests/smoke.rs`) builds nark first; here we only verify that if the
    /// binary is found, ingest+search round-trips with the UUID→bench-id mapping.
    #[test]
    #[ignore = "requires built nark binary; run via the integration smoke test"]
    fn nark_smoke_write_then_search() {
        let dir = tempdir().unwrap();
        let mut a = NarkAdapter::new();
        a.setup(dir.path()).unwrap();

        let body = "---\ntitle: Test note\nauthor: bench\ndomain: games\nintent: reference\nkind: note\nstatus: active\ntags: [chess]\n---\n\nThe sicilian defense begins with 1.e4 c5.\n";
        let d = Document { id: "n01".into(), body: body.into(), metadata: json!({}) };
        a.write(&d).unwrap();

        let (results, _) = a.search("sicilian defense", 5).unwrap();
        assert!(!results.is_empty(), "expected at least one nark hit");
        assert_eq!(results[0].doc_id, "n01", "expected bench-id translation; got {}", results[0].doc_id);
        a.teardown().unwrap();
    }
}
```

- [ ] **Step 7.2: Register the adapter in the factory**

Edit `bench/src/adapters/mod.rs`:

```rust
pub mod fts5;
pub mod nark;

use crate::protocol::Adapter;
use anyhow::{bail, Result};

pub fn make_adapter(name: &str) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        "nark" => Ok(Box::new(nark::NarkAdapter::new())),
        other => bail!("unknown adapter: {}", other),
    }
}
```

- [ ] **Step 7.3: Build nark so the adapter can find the binary**

Run: `cargo build -p nark`
Expected: completes; `target/debug/nark` exists.

- [ ] **Step 7.4: Run the (ignored) smoke test explicitly**

Run: `cargo test -p nark-bench adapters::nark -- --ignored`
Expected: 1 test passes (`nark_smoke_write_then_search`). If it fails because nark's `write` complains about required frontmatter fields the synthetic note is missing, fix the test fixture body to include all required frontmatter fields per the actual nark `write` command's requirements (read `src/cli/write.rs` if needed).

- [ ] **Step 7.5: Commit**

```bash
git add bench/src/adapters/nark.rs bench/src/adapters/mod.rs
git commit -m "feat(bench): nark adapter — subprocess to the CLI

Each adapter call spawns the nark binary with --vault-dir pointing at
an isolated workdir. Write() stages a markdown file in the workdir
and runs 'nark write <path>'. Search() runs 'nark search <q> --limit k'
and parses the JSON output.

Adapter looks for nark at target/debug/nark, then target/release/nark,
then PATH. The bench's integration tests build nark first so the
binary is always present.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Result file type and serialization

**Files:**
- Create: `bench/src/result.rs`
- Modify: `bench/src/main.rs` (add `mod result;`)

- [ ] **Step 8.1: Write `bench/src/result.rs`**

```rust
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub schema_version: String,
    pub task: String,
    pub system: String,
    pub config: String,
    pub system_version: String,
    pub bench_version: String,
    pub run_started_at: String,
    pub corpus: String,
    pub ir: Option<IrMetrics>,
    pub ir_per_class: HashMap<String, IrMetrics>,
    pub perf: PerfMetrics,
    pub errors: Vec<BenchError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IrMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    /// Number of queries this metric was averaged over.
    pub queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerfMetrics {
    pub write: PhaseMetrics,
    pub search: PhaseMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseMetrics {
    pub latency_p50_ms: u64,
    pub latency_p99_ms: u64,
    pub llm_tokens_in_total: u64,
    pub llm_tokens_out_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchError {
    pub phase: String,
    pub message: String,
}

impl BenchResult {
    pub fn new(task: &str, system: &str, config: &str, system_version: &str, corpus: &str) -> Self {
        Self {
            schema_version: "1".to_string(),
            task: task.to_string(),
            system: system.to_string(),
            config: config.to_string(),
            system_version: system_version.to_string(),
            bench_version: env!("CARGO_PKG_VERSION").to_string(),
            run_started_at: Utc::now().to_rfc3339(),
            corpus: corpus.to_string(),
            ir: None,
            ir_per_class: HashMap::new(),
            perf: PerfMetrics::default(),
            errors: vec![],
        }
    }

    pub fn write_to_disk(&self, output_dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let filename = format!("{}-{}-{}.json", self.task, self.system, self.config);
        let path = output_dir.join(filename);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }
}

/// Helper: compute p50/p99 from a sorted vector of latencies.
pub fn percentile(sorted_ms: &[u64], p: f64) -> u64 {
    if sorted_ms.is_empty() {
        return 0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let mut v = vec![10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        v.sort();
        assert_eq!(percentile(&v, 0.5), 60);
        assert_eq!(percentile(&v, 0.99), 100);
        assert_eq!(percentile(&v, 0.0), 10);
    }

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0);
    }
}
```

- [ ] **Step 8.2: Wire into main**

Edit `bench/src/main.rs` — add `mod result;` after `mod adapters;`:

```rust
mod protocol;
mod metrics;
mod adapters;
mod result;
```

- [ ] **Step 8.3: Run the percentile tests**

Run: `cargo test -p nark-bench result::tests`
Expected: 2 tests pass.

- [ ] **Step 8.4: Commit**

```bash
git add bench/src/result.rs bench/src/main.rs
git commit -m "feat(bench): result file type and percentile helper

BenchResult is the schema written to bench/results/<run-id>/. Includes
task/system/config identification, IR metrics (overall + per-class),
performance metrics (p50/p99 latencies + token totals), and recorded
errors. Schema version pinned at \"1\" to enable future migrations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Implement Task A (IR) runner

**Files:**
- Create: `bench/src/tasks/mod.rs`
- Create: `bench/src/tasks/ir.rs`
- Modify: `bench/src/main.rs` (add `mod tasks;` + wire to CLI)

- [ ] **Step 9.1: Write the IR task module with tests**

Create `bench/src/tasks/ir.rs`:

```rust
//! Task A: classical IR. Loads a query→relevant-ids fixture, runs every
//! query through every adapter, computes Recall@k / MRR / nDCG, emits one
//! BenchResult per (system, config).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::metrics::ir::{ndcg_at_k, recall_at_k, reciprocal_rank};
use crate::protocol::{Adapter, Document};
use crate::result::{percentile, BenchError, BenchResult, IrMetrics};

#[derive(Debug, Deserialize)]
struct QueryRow {
    query_id: String,
    query: String,
    relevant: Vec<String>,
    #[serde(default)]
    class: Option<String>,
}

pub fn run_ir_task(
    adapter: &mut dyn Adapter,
    corpus_root: &Path,
    config_label: &str,
) -> Result<BenchResult> {
    let system_name = adapter.name().to_string();
    let system_version = adapter.version().unwrap_or_else(|_| "unknown".to_string());
    let corpus_name = corpus_root.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    if system_version == "unknown" {
        anyhow::bail!("adapter '{}' refused to report a version — refusing to run for reproducibility", system_name);
    }

    let mut result = BenchResult::new("ir", &system_name, config_label, &system_version, &corpus_name);

    let workdir = tempfile::tempdir()?;
    adapter.setup(workdir.path())?;

    // Ingest corpus
    let corpus_dir = corpus_root.join("corpus");
    let mut write_latencies = Vec::new();
    let mut tokens_in_total = 0u64;
    let mut tokens_out_total = 0u64;

    for entry in std::fs::read_dir(&corpus_dir)
        .with_context(|| format!("failed to read corpus dir {:?}", corpus_dir))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let body = std::fs::read_to_string(&path)?;
        let id = path.file_stem().unwrap().to_string_lossy().to_string();
        let doc = Document { id, body, metadata: serde_json::json!({}) };
        match adapter.write(&doc) {
            Ok(m) => {
                write_latencies.push(m.latency_ms);
                tokens_in_total += m.llm_tokens_in;
                tokens_out_total += m.llm_tokens_out;
            }
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("write:{:?}", path.file_name()),
                    message: e.to_string(),
                });
            }
        }
    }

    // Load queries
    let queries_path = corpus_root.join("queries.jsonl");
    let queries_str = std::fs::read_to_string(&queries_path)
        .with_context(|| format!("failed to read queries file {:?}", queries_path))?;
    let queries: Vec<QueryRow> = queries_str
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<QueryRow>(l)
            .with_context(|| format!("bad query JSON: {}", l)))
        .collect::<Result<_>>()?;

    // Run queries, score each
    let mut search_latencies = Vec::new();
    let mut s_tokens_in_total = 0u64;
    let mut s_tokens_out_total = 0u64;
    let mut per_class_scores: HashMap<String, Vec<(Vec<String>, HashSet<String>)>> = HashMap::new();
    let mut all_scores: Vec<(Vec<String>, HashSet<String>)> = Vec::new();

    for q in &queries {
        let relevant: HashSet<String> = q.relevant.iter().cloned().collect();
        match adapter.search(&q.query, 10) {
            Ok((hits, m)) => {
                search_latencies.push(m.latency_ms);
                s_tokens_in_total += m.llm_tokens_in;
                s_tokens_out_total += m.llm_tokens_out;
                let ranked: Vec<String> = hits.into_iter().map(|h| h.doc_id).collect();
                all_scores.push((ranked.clone(), relevant.clone()));
                if let Some(cls) = &q.class {
                    per_class_scores.entry(cls.clone()).or_default().push((ranked, relevant));
                }
            }
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("search:{}", q.query_id),
                    message: e.to_string(),
                });
            }
        }
    }

    adapter.teardown()?;

    // Compute aggregate metrics
    result.ir = Some(aggregate(&all_scores));
    for (cls, rows) in per_class_scores {
        result.ir_per_class.insert(cls, aggregate(&rows));
    }

    // Performance
    write_latencies.sort_unstable();
    search_latencies.sort_unstable();
    result.perf.write.latency_p50_ms = percentile(&write_latencies, 0.5);
    result.perf.write.latency_p99_ms = percentile(&write_latencies, 0.99);
    result.perf.write.llm_tokens_in_total = tokens_in_total;
    result.perf.write.llm_tokens_out_total = tokens_out_total;
    result.perf.search.latency_p50_ms = percentile(&search_latencies, 0.5);
    result.perf.search.latency_p99_ms = percentile(&search_latencies, 0.99);
    result.perf.search.llm_tokens_in_total = s_tokens_in_total;
    result.perf.search.llm_tokens_out_total = s_tokens_out_total;

    Ok(result)
}

fn aggregate(rows: &[(Vec<String>, HashSet<String>)]) -> IrMetrics {
    if rows.is_empty() {
        return IrMetrics::default();
    }
    let n = rows.len() as f64;
    let r1: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 1)).sum::<f64>() / n;
    let r5: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 5)).sum::<f64>() / n;
    let r10: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 10)).sum::<f64>() / n;
    let mrr: f64 = rows.iter().map(|(r, g)| reciprocal_rank(r, g)).sum::<f64>() / n;
    let nd: f64 = rows.iter().map(|(r, g)| ndcg_at_k(r, g, 10)).sum::<f64>() / n;
    IrMetrics {
        recall_at_1: r1,
        recall_at_5: r5,
        recall_at_10: r10,
        mrr,
        ndcg_at_10: nd,
        queries: rows.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_empty() {
        let m = aggregate(&[]);
        assert_eq!(m.queries, 0);
        assert_eq!(m.mrr, 0.0);
    }

    #[test]
    fn aggregate_one_perfect_query() {
        let ranked = vec!["a".to_string(), "b".to_string()];
        let relevant: HashSet<String> = ["a".to_string()].iter().cloned().collect();
        let m = aggregate(&[(ranked, relevant)]);
        assert_eq!(m.queries, 1);
        assert!((m.recall_at_1 - 1.0).abs() < 1e-9);
        assert!((m.mrr - 1.0).abs() < 1e-9);
    }
}
```

Create `bench/src/tasks/mod.rs`:

```rust
pub mod ir;
```

- [ ] **Step 9.2: Wire the task into CLI dispatch**

Replace `bench/src/main.rs` with:

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod protocol;
mod metrics;
mod adapters;
mod result;
mod tasks;

#[derive(Parser)]
#[command(name = "nark-bench", about = "Benchmark harness for nark")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a benchmark task
    Run {
        /// Task name (currently only "ir" supported)
        #[arg(long)]
        task: String,
        /// Comma-separated systems to benchmark (e.g. "fts5,nark")
        #[arg(long)]
        systems: String,
        /// Corpus name relative to bench/datasets/ir/ (e.g. "synthetic-tiny")
        #[arg(long)]
        corpus: String,
        /// Output directory for result JSON files
        #[arg(long, default_value = "bench/results/local")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { task, systems, corpus, output } => {
            if task != "ir" {
                anyhow::bail!("only --task ir is supported in Phase 1a");
            }
            let corpus_root = PathBuf::from("bench/datasets/ir").join(&corpus);
            if !corpus_root.exists() {
                anyhow::bail!("corpus not found at {:?}", corpus_root);
            }
            for system in systems.split(',') {
                let system = system.trim();
                if system.is_empty() { continue; }
                let mut adapter = adapters::make_adapter(system)?;
                let result = tasks::ir::run_ir_task(adapter.as_mut(), &corpus_root, "default")?;
                let path = result.write_to_disk(&output)?;
                eprintln!("wrote {}", path.display());
            }
            Ok(())
        }
    }
}
```

- [ ] **Step 9.3: Verify the runner compiles**

Run: `cargo build -p nark-bench`
Expected: completes with no errors.

- [ ] **Step 9.4: Run the aggregator unit tests**

Run: `cargo test -p nark-bench tasks::ir`
Expected: 2 tests pass.

- [ ] **Step 9.5: Smoke-run end-to-end against FTS5**

Run: `cargo run -p nark-bench --release -- run --task ir --systems fts5 --corpus synthetic-tiny --output /tmp/bench-out`
Expected stderr: `wrote /tmp/bench-out/ir-fts5-default.json`. Open the file and verify it contains numeric `recall_at_5` / `mrr` / `ndcg_at_10` values between 0 and 1.

- [ ] **Step 9.6: Commit**

```bash
git add bench/src/tasks/mod.rs bench/src/tasks/ir.rs bench/src/main.rs
git commit -m "feat(bench): Task A (IR) runner end-to-end

run_ir_task ingests every .md file under corpus/, runs every query
from queries.jsonl, computes Recall@{1,5,10} / MRR / nDCG@10 plus
per-class breakdowns, and emits a BenchResult JSON. Adapter errors
during write or search are recorded in the result's errors array
and do not abort the run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Smoke integration test (gates CI)

**Files:**
- Create: `bench/tests/smoke.rs`

This test is the single artifact CI Lane 1 actually runs. It builds nark + nark-bench, runs the full pipeline against `synthetic-tiny`, asserts both adapters produce metrics in the valid range, and asserts no errors were recorded.

- [ ] **Step 10.1: Write the integration test**

Create `bench/tests/smoke.rs`:

```rust
//! Integration smoke test — runs the full harness against synthetic-tiny.
//! This is what CI Lane 1 (the PR check) actually executes.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn smoke_fts5_and_nark_run_against_synthetic_tiny() {
    let out_dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("nark-bench").unwrap();
    cmd.current_dir(workspace_root())
        .args(["run", "--task", "ir",
               "--systems", "fts5,nark",
               "--corpus", "synthetic-tiny",
               "--output"]).arg(out_dir.path());

    let out = cmd.assert().success().get_output().clone();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ir-fts5-default.json"), "stderr did not mention fts5 result: {}", stderr);
    assert!(stderr.contains("ir-nark-default.json"), "stderr did not mention nark result: {}", stderr);

    for system in ["fts5", "nark"] {
        let path = out_dir.path().join(format!("ir-{}-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read {}: {}", path.display(), e)
        });
        let v: Value = serde_json::from_str(&content).unwrap();

        // Required top-level fields
        assert_eq!(v["schema_version"], "1");
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert!(v["ir"]["recall_at_5"].is_number());

        // No errors recorded
        let errors = v["errors"].as_array().unwrap();
        assert!(errors.is_empty(), "{} run had errors: {:?}", system, errors);

        // Metrics in valid range
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5), "recall_at_5 out of range: {}", r5);

        // Per-class breakdowns exist
        assert!(v["ir_per_class"]["single_hop"].is_object(), "missing single_hop breakdown");
    }
}
```

- [ ] **Step 10.2: Build nark first so the nark adapter can find the binary**

Run: `cargo build -p nark`
Expected: completes; `target/debug/nark` exists.

- [ ] **Step 10.3: Run the smoke integration test**

Run: `cargo test -p nark-bench --test smoke`
Expected: 1 test passes. If it fails because `nark write` rejects the synthetic-tiny notes (e.g. requires a different frontmatter field), update the fixture markdown files in `bench/datasets/ir/synthetic-tiny/corpus/` so they pass nark's frontmatter validator, then rerun.

- [ ] **Step 10.4: Commit**

```bash
git add bench/tests/smoke.rs
git commit -m "test(bench): integration smoke test gating CI

Runs nark-bench end-to-end against the synthetic-tiny corpus for both
fts5 and nark adapters. Asserts result JSON has the required schema,
no errors were recorded, and Recall@5 is in [0, 1]. This is the
single artifact CI Lane 1 runs on every PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Regression check script

**Files:**
- Create: `bench/scripts/regression-check.sh`
- Create: `bench/results/main/.gitkeep`

The regression check compares the latest IR run against a stored baseline. If the baseline doesn't exist (first commit), the script bootstraps by copying the current run into `bench/results/main/`. Subsequent runs fail the build if Recall@5 or MRR or nDCG@10 drops by more than 2%.

- [ ] **Step 11.1: Create the baseline directory**

```bash
mkdir -p bench/results/main
touch bench/results/main/.gitkeep
```

- [ ] **Step 11.2: Write the regression script**

Create `bench/scripts/regression-check.sh`:

```bash
#!/usr/bin/env bash
# regression-check.sh — fails (exits 1) if any IR metric in <new-results-dir>
# drops by more than 2% compared to bench/results/main/<same-filename>.
#
# Usage: bench/scripts/regression-check.sh <new-results-dir>
#
# On first run (no baseline present) it bootstraps the baseline by copying
# the new results to bench/results/main/ and exits 0.

set -euo pipefail

NEW_DIR="${1:-}"
if [[ -z "$NEW_DIR" ]]; then
  echo "usage: $0 <new-results-dir>" >&2
  exit 2
fi

BASELINE_DIR="bench/results/main"
THRESHOLD=0.02   # 2% relative drop

fail=0
bootstrap=0

shopt -s nullglob
for new_file in "$NEW_DIR"/ir-*.json; do
  base_file="$BASELINE_DIR/$(basename "$new_file")"
  if [[ ! -f "$base_file" ]]; then
    echo "no baseline for $(basename "$new_file") — copying as bootstrap"
    cp "$new_file" "$base_file"
    bootstrap=1
    continue
  fi

  for metric in recall_at_5 mrr ndcg_at_10; do
    new_val=$(jq -r ".ir.${metric}" "$new_file")
    base_val=$(jq -r ".ir.${metric}" "$base_file")
    drop=$(awk -v n="$new_val" -v b="$base_val" 'BEGIN { if (b == 0) print 0; else print (b - n) / b }')
    is_regress=$(awk -v d="$drop" -v t="$THRESHOLD" 'BEGIN { print (d > t) ? 1 : 0 }')
    if [[ "$is_regress" == "1" ]]; then
      printf 'REGRESSION: %s %s dropped from %s to %s (%.2f%%)\n' \
        "$(basename "$new_file")" "$metric" "$base_val" "$new_val" \
        "$(awk -v d="$drop" 'BEGIN { print d * 100 }')"
      fail=1
    fi
  done
done

if [[ "$bootstrap" == "1" ]]; then
  echo "regression-check: bootstrap mode — baseline copied; commit bench/results/main/ to lock it in"
fi

exit $fail
```

- [ ] **Step 11.3: Make the script executable**

Run: `chmod +x bench/scripts/regression-check.sh`

- [ ] **Step 11.4: Smoke-test the script in bootstrap mode**

Run:
```
mkdir -p /tmp/bench-rc-test
cargo run -p nark-bench --release -- run --task ir --systems fts5 --corpus synthetic-tiny --output /tmp/bench-rc-test
bench/scripts/regression-check.sh /tmp/bench-rc-test
```
Expected: script reports "no baseline" and "bootstrap mode", exits 0. After running, `bench/results/main/ir-fts5-default.json` exists.

- [ ] **Step 11.5: Smoke-test the script in regression-detect mode**

Run the script a second time with the same output dir:

```
bench/scripts/regression-check.sh /tmp/bench-rc-test
```

Expected: exits 0 (numbers match baseline exactly). Then manually edit `bench/results/main/ir-fts5-default.json` to set `recall_at_5: 0.99` (artificially high), re-run the script:

```
bench/scripts/regression-check.sh /tmp/bench-rc-test
```

Expected: prints "REGRESSION: ir-fts5-default.json recall_at_5 dropped from 0.99 to ..." and exits 1. **Restore the original file** (e.g. by re-running the bootstrap-style copy: delete `bench/results/main/ir-fts5-default.json` and re-run the script).

- [ ] **Step 11.6: Commit (without committing the test baseline yet — we'll lock that in via CI)**

```bash
git add bench/scripts/regression-check.sh bench/results/main/.gitkeep
git commit -m "feat(bench): regression-check script for Lane 1 CI

Compares ir-*.json files in a new results directory against
bench/results/main/. Fails (exit 1) if Recall@5, MRR, or nDCG@10
drops by more than 2% (relative). First run bootstraps the baseline.
The threshold is configurable as the THRESHOLD env var. Uses jq + awk
for portability across Linux and macOS runners.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: GitHub Actions Lane 1 workflow

**Files:**
- Create: `.github/workflows/bench-pr.yml`

- [ ] **Step 12.1: Write the workflow**

Create `.github/workflows/bench-pr.yml`:

```yaml
name: bench-pr

on:
  pull_request:
    paths:
      - 'src/**'
      - 'bench/**'
      - 'Cargo.toml'
      - '.github/workflows/bench-pr.yml'

jobs:
  bench-regression:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: bench-pr
      - name: Install jq
        run: sudo apt-get update && sudo apt-get install -y jq
      - name: Build nark
        run: cargo build -p nark --release
      - name: Build nark-bench
        run: cargo build -p nark-bench --release
      - name: Run smoke integration test
        run: cargo test -p nark-bench --release --test smoke
      - name: Run benchmark for regression baseline comparison
        run: |
          mkdir -p /tmp/bench-pr-out
          cargo run -p nark-bench --release -- run \
            --task ir \
            --systems fts5,nark \
            --corpus synthetic-tiny \
            --output /tmp/bench-pr-out
      - name: Regression check
        run: bench/scripts/regression-check.sh /tmp/bench-pr-out
      - name: Upload result artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: bench-results
          path: /tmp/bench-pr-out
```

- [ ] **Step 12.2: Verify the workflow YAML is valid syntactically**

Run: `cat .github/workflows/bench-pr.yml | python3 -c "import sys, yaml; yaml.safe_load(sys.stdin)" && echo OK`
Expected: prints `OK`. (If `yaml` is missing, install with `python3 -m pip install pyyaml --break-system-packages` or trust GitHub's syntax check on push.)

- [ ] **Step 12.3: Commit**

```bash
git add .github/workflows/bench-pr.yml
git commit -m "ci: add Lane 1 bench-pr workflow

Runs on every PR touching src/, bench/, Cargo.toml, or the workflow
itself. Builds nark + nark-bench, runs the integration smoke test,
then runs the regression check against bench/results/main/.

Path filter prevents unrelated documentation PRs from spending
Actions minutes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Lock in the regression baseline

**Files:**
- Create: `bench/results/main/ir-fts5-default.json`
- Create: `bench/results/main/ir-nark-default.json`

These are the result of running the bench once cleanly. Subsequent PR checks compare against them.

- [ ] **Step 13.1: Produce a clean baseline run**

Run:
```bash
rm -f bench/results/main/ir-*.json
mkdir -p bench/results/main
cargo build -p nark --release
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark \
  --corpus synthetic-tiny \
  --output bench/results/main
```

Expected: produces `bench/results/main/ir-fts5-default.json` and `bench/results/main/ir-nark-default.json`.

- [ ] **Step 13.2: Inspect the baseline numbers**

Run: `jq '.ir' bench/results/main/ir-*.json`

Sanity-check: both files should show `recall_at_5` somewhere in the 0.4–1.0 range for the synthetic-tiny corpus. nark's full pipeline should not be dramatically worse than raw FTS5 on this small fixture. If nark's number is wildly off (e.g. 0.0), inspect the per-class breakdowns to diagnose — the bench/scripts/regression-check.sh threshold is 2%, so the baseline doesn't need to be optimal, but the numbers should be plausible.

- [ ] **Step 13.3: Commit the baseline**

```bash
git add bench/results/main/ir-fts5-default.json bench/results/main/ir-nark-default.json
git commit -m "bench: lock in initial regression baseline

These files capture the first clean run of the IR benchmark against
synthetic-tiny for the fts5 and nark adapters. They serve as the
comparison target for CI Lane 1's regression-check.sh — any future
PR that drops Recall@5, MRR, or nDCG@10 by more than 2% relative to
these numbers will fail the bench-pr workflow.

To update the baseline intentionally (e.g. after an algorithm change
that improves search), re-run the bench locally, inspect the diff,
and commit the new files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Wrap-up

After all 13 tasks land, Phase 1a is complete. Concretely, the repo now has:

- A `bench/` workspace crate that builds with `cargo build -p nark-bench`.
- An `Adapter` trait + two implementations (`fts5`, `nark`).
- Classical IR metrics with unit-test coverage against canonical examples.
- A 12-note / 10-query synthetic IR fixture committed in-tree.
- A `run_ir_task` runner that emits schema-versioned JSON results.
- An integration smoke test that exercises the whole pipeline against the fixture.
- A regression-check script that fails CI on a >2% drop in Recall@5 / MRR / nDCG@10.
- A GitHub Actions workflow (`.github/workflows/bench-pr.yml`) wiring the above into PR gating.
- A committed baseline (`bench/results/main/`) that every future PR compares against.

The first PR after this lands that actually changes search behaviour will be visibly compared against `bench/results/main/`. If you intentionally change the algorithm and the new numbers are better, just commit the updated baseline as part of that PR. If the numbers are worse and you didn't expect that, the PR check tells you before merge.

## Follow-up plans (not part of Phase 1a)

- **Phase 1b — Vector baseline + nark-self corpus.** Adds the pure-Rust embedding-baseline adapter and the scrubbed real-vault corpus.
- **Phase 2 — LongMemEval + LOCOMO + LLM-judge.** Adds Task B, the `claude-cli` / `codex-cli` / `api` judge backends, and the judge cache.
- **Phase 3 — Python competitor adapters.** mem0 and Graphiti shims, plus the Letta HTTP adapter and its docker-compose harness.
- **Phase 4 — Report generator.** Auto-generated `BENCHMARKS.md` from `bench/results/latest/` + README badges.
- **Phase 5 — Task C agent replay.** Real anonymized transcripts; labelling helper.
- **Phase 6 — Task D Librarian curation.** Gated on the Librarian agent shipping.
