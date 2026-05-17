# nark Bench — Phase 1c (nark-self Scaffold + Baseline Naming Fix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed the corpus name in result-file filenames so multiple corpora can coexist, document the corpus contract, and scaffold an empty `bench/datasets/ir/nark-self/` directory so future scrubbed-vault notes can drop in without revisiting plumbing.

**Architecture:** Single one-line change in `bench/src/result.rs::write_to_disk` adds `<corpus>` as a fourth filename component (already-existing field, just newly included). The integration smoke test is updated to reference the new pattern plus a defensive assertion that the *old* pattern is absent. Existing baselines are regenerated under the new naming. Two new READMEs document the general corpus contract and the nark-self-specific scrub methodology.

**Tech Stack:** Rust 2024; existing bench/ infrastructure unchanged (no new deps). Pure markdown for the docs.

**Spec reference:** `docs/superpowers/specs/2026-05-17-phase-1c-design.md`

**Branch:** `feat/bench-phase-1a` (continuation; Phase 1a, 1b already shipped on this branch).

---

## File structure

**Files to create:**

| Path | Responsibility |
|---|---|
| `bench/datasets/ir/README.md` | General corpus contract — applies to every corpus |
| `bench/datasets/ir/nark-self/README.md` | nark-self methodology (scrub + query construction) |
| `bench/datasets/ir/nark-self/corpus/.gitkeep` | Marker so git tracks the empty corpus directory |
| `bench/results/main/ir-fts5-synthetic-tiny-default.json` | Regenerated baseline (was `ir-fts5-default.json`) |
| `bench/results/main/ir-nark-synthetic-tiny-default.json` | Regenerated baseline (was `ir-nark-default.json`) |
| `bench/results/main/ir-vector-synthetic-tiny-default.json` | Regenerated baseline (was `ir-vector-default.json`) |

**Files to modify:**

| Path | Change |
|---|---|
| `bench/src/result.rs` | `write_to_disk` filename gains `<corpus>` as a fourth component |
| `bench/tests/smoke.rs` | Filename references updated; defensive assertion that old pattern is absent |

**Files to delete:**

| Path | Reason |
|---|---|
| `bench/results/main/ir-fts5-default.json` | Replaced by new-naming version |
| `bench/results/main/ir-nark-default.json` | Replaced by new-naming version |
| `bench/results/main/ir-vector-default.json` | Replaced by new-naming version |

No changes to: `nark/` (the main crate), `bench/Cargo.toml`, `.github/workflows/`, `bench/scripts/regression-check.sh` (its glob `ir-*.json` matches both old and new patterns), any other adapter or task module.

---

## Task 1: Embed corpus name in result filenames

**Files:**
- Modify: `bench/src/result.rs` — the `write_to_disk` method on `BenchResult`

`BenchResult.corpus` is already populated correctly today (set during `BenchResult::new`). This task just adds it as a fourth filename component. `sanitize_for_filename` from Phase 1b applies uniformly.

- [ ] **Step 1.1: Read the current `write_to_disk` method**

Read `bench/src/result.rs` and locate the `write_to_disk` method on `impl BenchResult`. Confirm it currently constructs the filename as:

```rust
let filename = format!(
    "{}-{}-{}.json",
    sanitize_for_filename(&self.task),
    sanitize_for_filename(&self.system),
    sanitize_for_filename(&self.config),
);
```

If the current method's body differs from this shape (e.g. has been refactored since Phase 1b), STOP and report BLOCKED rather than guessing — the change is too small to be worth doing if the surrounding code has shifted.

- [ ] **Step 1.2: Update the filename construction**

Edit the same block to:

```rust
let filename = format!(
    "{}-{}-{}-{}.json",
    sanitize_for_filename(&self.task),
    sanitize_for_filename(&self.system),
    sanitize_for_filename(&self.corpus),
    sanitize_for_filename(&self.config),
);
```

The rest of `write_to_disk` (path join, JSON serialization, trailing newline write) stays unchanged.

- [ ] **Step 1.3: Verify the workspace builds**

Run: `cargo build -p nark-bench`
Expected: completes with no errors. No warnings introduced by this change.

- [ ] **Step 1.4: Run unit tests**

Run: `cargo test -p nark-bench --lib`
Expected: all 25 unit tests pass. The percentile tests in `result::tests` don't exercise `write_to_disk` so they continue to pass unchanged. No new unit test added (the integration smoke test in Task 2 covers the filename behavior end-to-end).

- [ ] **Step 1.5: Commit**

```bash
git add bench/src/result.rs
git commit -m "$(cat <<'EOF'
feat(bench): embed corpus name in result filenames

BenchResult.write_to_disk now produces filenames matching
ir-<system>-<corpus>-<config>.json (was ir-<system>-<config>.json).
Without this, two corpora would overwrite each other's baselines —
e.g. synthetic-tiny and future nark-self would both write to
ir-fts5-default.json in the same output directory.

BenchResult.corpus is already populated correctly (Phase 1a set it
from corpus_root.file_name()). This change just makes the filename
use it. sanitize_for_filename from Phase 1b applies uniformly so
corpus names with path separators get neutralized.

The regression script's ir-*.json glob still matches both old and
new filename patterns; no change needed there. Cascading impacts
(smoke test references, baseline regen) handled in subsequent
commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update integration smoke test for new filenames + defensive assertion

**Files:**
- Modify: `bench/tests/smoke.rs`

Six filename references update to the new pattern. A new defensive assertion confirms the *old* pattern is NOT produced — guards against any future regression that reverts the naming.

- [ ] **Step 2.1: Read the current smoke test**

Read `bench/tests/smoke.rs`. Locate the two for-loops over `["fts5", "nark", "vector"]`:
- The first loop asserts stderr contains each filename.
- The second loop reads each result file and asserts on its JSON contents.

Confirm both loops currently use `format!("ir-{}-default.json", system)`.

- [ ] **Step 2.2: Replace the test contents**

Replace the entire body of `bench/tests/smoke.rs` with:

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

    // New-naming files should be mentioned in stderr.
    for system in ["fts5", "nark", "vector"] {
        let new_name = format!("ir-{}-synthetic-tiny-default.json", system);
        assert!(
            stderr.contains(&new_name),
            "stderr did not mention {} result file '{}': {}",
            system, new_name, stderr
        );
    }

    // Defensive: old-naming files must NOT exist. Catches a regression that
    // accidentally reverts write_to_disk to the pre-Phase-1c filename shape.
    for system in ["fts5", "nark", "vector"] {
        let old = out_dir.path().join(format!("ir-{}-default.json", system));
        assert!(
            !old.exists(),
            "old (pre-Phase-1c) filename pattern present (regression?): {}",
            old.display()
        );
    }

    for system in ["fts5", "nark", "vector"] {
        let path = out_dir.path().join(format!("ir-{}-synthetic-tiny-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read {}: {}", path.display(), e)
        });
        let v: Value = serde_json::from_str(&content).unwrap();

        // Schema bumped to "2" in Phase 1b.
        assert_eq!(v["schema_version"], "2", "{} schema_version mismatch", system);
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert_eq!(v["corpus"], "synthetic-tiny");
        assert!(v["ir"]["recall_at_5"].is_number());

        // Metrics in valid range.
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5), "{} recall_at_5 out of range: {}", system, r5);

        // Per-class breakdowns present.
        assert!(v["ir_per_class"]["single_hop"].is_object(),
            "{} missing single_hop breakdown", system);

        // Zero errors required — Phase 1b's gate stays in force.
        let errors = v["errors"].as_array().expect("errors field required");
        assert!(
            errors.is_empty(),
            "{} run had unexpected errors (regression?): {:?}", system, errors
        );
    }
}
```

Notable additions vs the Phase 1b version:
- All filename references updated from `ir-{}-default.json` to `ir-{}-synthetic-tiny-default.json`.
- New defensive loop asserting the old filenames do NOT exist.
- New `assert_eq!(v["corpus"], "synthetic-tiny")` confirms the JSON content's `corpus` field matches.

- [ ] **Step 2.3: Ensure nark is built in release**

Run: `cargo build -p nark --release`
Expected: completes (likely already cached from prior work).

- [ ] **Step 2.4: Run the smoke test**

Run: `cargo test -p nark-bench --release --test smoke`
Expected: 1 test passes. If the test fails with "could not read .../ir-fts5-synthetic-tiny-default.json" — that means Task 1's filename change didn't land or the build didn't pick it up. Run `cargo build -p nark-bench --release` and retry.

- [ ] **Step 2.5: Commit**

```bash
git add bench/tests/smoke.rs
git commit -m "$(cat <<'EOF'
test(bench): smoke test uses new <system>-<corpus>-<config> filenames

Updates the integration smoke test to:
- Reference the new filename pattern ir-<system>-synthetic-tiny-default.json
  (six refs: three in stderr-contains, three in file reads).
- Assert v["corpus"] == "synthetic-tiny" in the JSON content.
- Defensively assert the OLD filename pattern is absent. If a future
  refactor accidentally reverts write_to_disk to the pre-Phase-1c shape,
  this catches it before the regression script's bootstrap mode silently
  creates a divergent baseline.

Test still passes against the Phase 1b nark and vector baseline numbers
(recall_at_5 = 1.0). Phase 1c does not change the schema or the
per-adapter behavior; only the filename shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Regenerate synthetic-tiny baselines under new naming

**Files:**
- Delete: `bench/results/main/ir-fts5-default.json`
- Delete: `bench/results/main/ir-nark-default.json`
- Delete: `bench/results/main/ir-vector-default.json`
- Create: `bench/results/main/ir-fts5-synthetic-tiny-default.json`
- Create: `bench/results/main/ir-nark-synthetic-tiny-default.json`
- Create: `bench/results/main/ir-vector-synthetic-tiny-default.json`

The IR metric numbers should be identical to Phase 1b's baselines (same code, same data, only filenames change). FTS5 should land at recall_at_5 ≈ 0.2; nark and vector should land at 1.0.

- [ ] **Step 3.1: Delete the old-naming baselines**

```bash
rm bench/results/main/ir-fts5-default.json
rm bench/results/main/ir-nark-default.json
rm bench/results/main/ir-vector-default.json
```

Confirm via `ls bench/results/main/`: should now show only `.gitkeep`.

- [ ] **Step 3.2: Produce fresh baselines**

```bash
cargo build -p nark --release
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus synthetic-tiny \
  --output bench/results/main
```

Expected stderr: three "wrote bench/results/main/ir-{fts5,nark,vector}-synthetic-tiny-default.json" lines.

- [ ] **Step 3.3: Inspect the baselines**

```bash
jq '.schema_version, .corpus, .ir, (.errors | length)' bench/results/main/ir-fts5-synthetic-tiny-default.json
echo "---"
jq '.schema_version, .corpus, .ir, (.errors | length)' bench/results/main/ir-nark-synthetic-tiny-default.json
echo "---"
jq '.schema_version, .corpus, .ir, (.errors | length)' bench/results/main/ir-vector-synthetic-tiny-default.json
```

Sanity-check each file:
- `schema_version` = `"2"`
- `corpus` = `"synthetic-tiny"`
- `errors.length` = `0`
- FTS5: `recall_at_5 ≈ 0.2`, `queries = 10`
- nark: `recall_at_5 = 1.0`, `queries = 10`
- vector: `recall_at_5 = 1.0`, `queries = 10`

If any number is materially off from these expectations, STOP and investigate before proceeding. Particularly: if nark's `recall_at_5` is ~0.1 (BM25-only fallback), the embedding stage failed silently — check stderr for "model staging failed" warnings.

- [ ] **Step 3.4: Verify the regression script round-trips against the new baselines**

```bash
mkdir -p /tmp/p1c-task3-verify
find /tmp/p1c-task3-verify -name '*.json' -delete 2>/dev/null
cargo run -p nark-bench --release -- run --task ir --systems fts5,nark,vector --corpus synthetic-tiny --output /tmp/p1c-task3-verify
bench/scripts/regression-check.sh /tmp/p1c-task3-verify
echo "regression-check exit: $?"
rm -rf /tmp/p1c-task3-verify
```

Expected: regression-check exit 0. No warnings (schema and query count match the baselines we just produced).

- [ ] **Step 3.5: Commit**

```bash
git add bench/results/main/
git commit -m "$(cat <<'EOF'
bench: regenerate synthetic-tiny baselines under new filename pattern

The Phase 1c filename change produces ir-<system>-synthetic-tiny-default.json
instead of ir-<system>-default.json. Same code path, same data, same
metric numbers — only the filename shape differs.

Baseline numbers preserved from Phase 1b:
- fts5: recall_at_5 = 0.2 (BM25-only baseline, sanitizer engaged)
- nark: recall_at_5 = 1.0 (full hybrid with embeddings)
- vector: recall_at_5 = 1.0 (pure embedding baseline)
- all three: errors = [], queries = 10

Regression-check.sh round-trip confirmed (exit 0). Old-naming files
deleted in the same commit; only new-naming baselines remain alongside
the existing .gitkeep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: General corpus contract README

**Files:**
- Create: `bench/datasets/ir/README.md`

The doc anyone reads when adding a new corpus to the bench (general; not nark-self-specific). Lives at the datasets-level so it applies to synthetic-tiny, nark-self, and any future corpus.

- [ ] **Step 4.1: Create `bench/datasets/ir/README.md`**

```markdown
# nark-bench IR Corpora

This directory holds named corpora that the bench harness runs Task A
(classical IR) against. Each subdirectory is one corpus, with its own
notes, queries, and methodology README.

## Existing corpora

- **`synthetic-tiny/`** — 12 hand-written notes across 3 topic clusters
  (chess, cooking, electronics) and 10 queries with binary gold labels.
  Designed as a smoke fixture: large enough to exercise BM25 ranking,
  small enough to commit and iterate against on every PR.
- **`nark-self/`** — scaffold for ~150 anonymized real notes from Sean's
  vault and ~50 hand-labeled queries. Currently empty (the scrub work
  is manual and pending). See `nark-self/README.md` for methodology.

## Directory layout (contract for any new corpus)

```
<corpus-name>/
├── README.md             # methodology, anonymization (if applicable), provenance
├── corpus/
│   ├── n01.md            # one markdown note per file
│   ├── n02.md            # ...
│   └── ...
└── queries.jsonl         # one JSON object per line
```

The bench tracks each document by file stem (e.g. `n01` for `corpus/n01.md`).
The nark adapter assigns its own UUIDs internally and maintains a mapping
back to bench IDs — transparent to corpus authors. Other adapters use the
file stem directly.

## Frontmatter requirements

Every `.md` file in `corpus/` must have YAML frontmatter with these
**seven required fields** (nark's `Frontmatter` struct rejects anything
missing):

```yaml
---
title: Sicilian Defense opening principles
author: bench
domain: games
intent: reference
kind: note
status: active
tags: [chess, opening]
---
```

Valid `status` values: `active`, `draft`, `deprecated`, `retracted`.
Other fields are free-form strings (no enum validation in nark for
domain/kind/intent — they're treated as taxonomy labels).

Body comes after the frontmatter block. There is no upper bound; nark
truncates to ~8192 tokens during embedding.

## queries.jsonl format

One JSON object per line, no leading or trailing whitespace:

```jsonl
{"query_id": "q01", "query": "sicilian defense move order", "relevant": ["n01", "n02"], "class": "single_hop"}
{"query_id": "q02", "query": "najdorf variation idea", "relevant": ["n02"], "class": "single_hop"}
```

Fields:
- `query_id` (string, required) — used in result JSON's `errors` array if a
  query fails (e.g. `"search:q07"`).
- `query` (string, required) — the natural-language query sent to each
  adapter's `search()` method.
- `relevant` (array of strings, required) — file stems (without `.md`) of
  notes that are considered correct answers. Empty arrays are tolerated
  (by convention recall = 1.0 for queries with no relevant docs).
- `class` (string, optional) — query class label. Surfaces as a per-class
  breakdown in result JSON's `ir_per_class` map. Each corpus defines its
  own class scheme.

## How to invoke

```bash
# From the workspace root
cargo build -p nark --release
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus <corpus-name> \
  --output bench/results/main
```

Produces one result JSON per `(system, corpus, config)` combination at
`bench/results/main/ir-<system>-<corpus>-<config>.json`. Today `config`
is always `default`; future ablations (e.g. nark-body-only) will vary it.

## Sizing guidance

| Corpus | Notes | Queries | Use case |
|---|---|---|---|
| synthetic-tiny | 12 | 10 | Smoke fixture; CI regression gate; iterate fast |
| nark-self (target) | ~150 | ~50 | Realistic regression detection |
| LongMemEval (future Phase 2) | 500 questions × variable | n/a | Public scoreboard comparability |

Larger corpora improve statistical confidence in absolute scores. For
regression detection (i.e. "did my last change make search worse?"),
even a 12-note smoke fixture catches most pipeline-level regressions.

## Adding a new corpus

1. Create `bench/datasets/ir/<name>/`.
2. Create `corpus/` subdirectory; drop in `.md` files with valid frontmatter.
3. Create `queries.jsonl` with at least one query.
4. Create a per-corpus `README.md` documenting methodology and provenance.
5. Run the bench with `--corpus <name> --output bench/results/main` to
   produce the initial baseline.
6. Run `bench/scripts/regression-check.sh bench/results/main` to confirm
   the new files are valid (bootstrap mode kicks in for the new files).
7. Commit the baseline JSONs alongside the corpus content.

For Sean specifically: nark-self is the target real-vault corpus.
Methodology and scrub procedure in `nark-self/README.md`.
```

- [ ] **Step 4.2: Verify the markdown renders correctly**

The README is pure markdown — no programmatic validation needed. Visually scan the file once after writing to confirm:
- All triple-backtick code blocks have matching close fences
- Table renders as a table (pipes line up)
- Inline code spans are intact

- [ ] **Step 4.3: Commit**

```bash
git add bench/datasets/ir/README.md
git commit -m "$(cat <<'EOF'
docs(bench): add general corpus contract README

Documents the contract for adding a new IR corpus to the bench:
directory layout, the seven required frontmatter fields, queries.jsonl
format with field semantics, invocation pattern, sizing guidance, and
the step-by-step procedure for landing a new corpus + baseline.

Lives at bench/datasets/ir/README.md so it applies to existing
synthetic-tiny and future nark-self / LongMemEval / etc. Per-corpus
specifics belong in each corpus's own README (e.g. nark-self/README.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: nark-self directory scaffold

**Files:**
- Create: `bench/datasets/ir/nark-self/README.md`
- Create: `bench/datasets/ir/nark-self/corpus/.gitkeep`

Empty corpus directory + methodology README. No queries.jsonl deliberately — its absence makes `--corpus nark-self` fail fast until Sean does the scrub work.

- [ ] **Step 5.1: Create the empty corpus directory and `.gitkeep`**

```bash
mkdir -p bench/datasets/ir/nark-self/corpus
touch bench/datasets/ir/nark-self/corpus/.gitkeep
```

Confirm via `ls bench/datasets/ir/nark-self/corpus/`: should show only `.gitkeep`.

- [ ] **Step 5.2: Create `bench/datasets/ir/nark-self/README.md`**

```markdown
# nark-self IR Corpus — Methodology

**Status:** Empty as of Phase 1c. Awaiting Sean's one-time anonymization
scrub of ~150 real notes plus ~50 hand-labeled queries.

`cargo run -p nark-bench -- --corpus nark-self` will fail until
`queries.jsonl` exists and `corpus/` contains at least one `.md` file.
The failure mode is "failed to read queries file" — that's the
intended signal that this corpus is not yet populated.

## Purpose

Measure nark's hybrid search pipeline against a corpus that reflects
how Sean *actually uses nark*, not the synthetic smoke fixture.

synthetic-tiny is small, synonym-heavy, and demonstrably favors pure
embedding (both nark and vector hit recall@5 = 1.0 on it). nark-self
should expose cases where nark's pipeline (BM25 + cosine + graph +
engagement blend) pulls ahead of raw vector — particularly on
synthesis queries and queries that reference versioned content.

## Source

Notes drawn from `~/.ark/objects/md/` at a documented point in time.
Filter criteria when doing the scrub:

- Research / reference / how-to notes from approximately the last 6-12 months
- Exclude any note tagged with personally-sensitive domains
  (e.g. `tags: [personal-finance, family, medical]`)
- Aim for diversity across domains so per-class breakdowns are
  meaningful

Record the source `~/.ark` commit SHA (if version-tracked) or the
date range in this README once the scrub completes.

## Anonymization checklist

The bar: **would you be comfortable if this corpus were public on GitHub?**

For each note:

- [ ] Real names of people → pseudonyms (Alice, Bob, Carol, ...)
- [ ] Specific company names → "Corp A", "Corp B", ...
- [ ] URLs → redact to bare domain or remove
- [ ] Specific dollar amounts → replace with `$X` or `$XXX` (order of magnitude OK)
- [ ] Specific dates → year-month only (`2026-05` not `2026-05-17`)
- [ ] Email addresses → remove or replace with `name@example.com`
- [ ] API keys, tokens, credentials → remove entirely (even if test-only)
- [ ] Internal codenames → replace with descriptive but generic equivalents
- [ ] Verbatim quotes from non-public sources → paraphrase
- [ ] Frontmatter `author` field → set to `bench` (matches synthetic-tiny convention)

After anonymizing, do a pass with fresh eyes to catch anything missed.

## Sizing target

- ~150 notes (range 100–200 is fine for v1)
- ~50 queries (range 30–80 is fine for v1)

Smaller is fine to start. Add more queries over time as you notice
real-world questions where you'd want to test how nark handles them.

## Query construction

Write queries reflecting how you *actually search* nark. The goal is
ecological validity, not coverage of every nark feature.

Sources of good real queries:
- Replay your own session history if any is preserved
- Note questions you've asked Claude/Codex while researching in nark
- The kinds of queries you'd issue if dropped into a vault cold

Avoid:
- Crafting queries specifically to test a feature (that's synthetic-tiny's
  job — it has classes for `single_hop` / `multi_hop` / `synonym`
  to make pipeline-component-level diagnosis possible)
- Verbatim copies of note titles (too easy; doesn't measure retrieval)

## Proposed query classes for nark-self

Each query gets an optional `class` field. Suggested taxonomy:

- **`factual`** — direct fact lookup; one note clearly answers
- **`synthesis`** — combining 2+ notes; exercises nark's hybrid
- **`temporal`** — asks about something that changed; exercises nark's
  versioning (currently surfaces latest version, but queries about
  history would benefit from richer query parsing in future phases)
- **`recall`** — find a half-remembered note by partial description;
  exercises the cosine pathway most heavily

Refine the taxonomy as you write queries. The class names are not
enforced; the bench just produces per-class breakdowns based on
whatever class strings appear in `queries.jsonl`.

## Validation checklist

Before committing the corpus, confirm:

- [ ] Every `corpus/*.md` has valid frontmatter (parses without errors
      via `nark write <file>` against a scratch vault)
- [ ] `queries.jsonl` parses as JSONL (one valid JSON object per line,
      no trailing whitespace)
- [ ] Every entry in any query's `relevant` array corresponds to an
      actual file in `corpus/`
- [ ] No personally-sensitive content remains (final pass)
- [ ] File count and query count match the target ranges above

## Running

Once the corpus is populated:

```bash
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus nark-self \
  --output bench/results/main
```

Produces three new baseline files at
`bench/results/main/ir-{fts5,nark,vector}-nark-self-default.json`.
Run `bench/scripts/regression-check.sh bench/results/main` to confirm
the regression script accepts them (bootstrap mode on first run).

Commit the new baselines alongside the corpus content.

## CI status

**Not in CI Lane 1.** The PR check workflow runs `--corpus synthetic-tiny`
only. nark-self runs are intentionally manual until the corpus has stable
baselines worth gating on.

Future work (deferred): the CLI doesn't yet accept comma-separated
corpus values, so adding nark-self to CI requires either (a) running
the bench twice with different `--corpus` flags, or (b) adding
multi-corpus support to the runner. Cross that bridge when nark-self
has settled.
```

- [ ] **Step 5.3: Verify the directory tree is correct**

```bash
ls -la bench/datasets/ir/nark-self/
ls -la bench/datasets/ir/nark-self/corpus/
```

Expected:
- `nark-self/` contains `README.md` and `corpus/`
- `nark-self/corpus/` contains only `.gitkeep`
- No `queries.jsonl` anywhere

- [ ] **Step 5.4: Confirm the "fail fast" behavior works as documented**

```bash
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5 \
  --corpus nark-self \
  --output /tmp/p1c-task5-confirm 2>&1 | tail -10
```

Expected: the run fails with an error message about reading the queries file (e.g. "failed to read queries file ...nark-self/queries.jsonl"). This is the intended signal — the README documents it.

Clean up: `rm -rf /tmp/p1c-task5-confirm` (the empty output dir may or may not exist depending on where the failure occurred).

- [ ] **Step 5.5: Commit**

```bash
git add bench/datasets/ir/nark-self/
git commit -m "$(cat <<'EOF'
docs(bench): scaffold nark-self corpus directory + methodology README

Adds bench/datasets/ir/nark-self/ with an empty corpus/ directory
(marker file .gitkeep) and a README.md documenting:

- Purpose: real-vault corpus to exercise nark's hybrid pipeline on
  realistic data, complementing the synthetic-tiny smoke fixture
- Source filter criteria (date range, sensitivity exclusion)
- Anonymization checklist with concrete substitution rules
- Sizing targets (~150 notes, ~50 queries)
- Query construction guidance + proposed class taxonomy
- Validation checklist
- Running instructions for once the corpus is populated
- CI status: not in Lane 1 until nark-self stabilizes

No queries.jsonl is created; --corpus nark-self failing with
"queries file missing" is the intended signal that the scrub work
hasn't happened yet. The README documents this explicitly.

Future Sean drops scrubbed notes into corpus/ and writes queries.jsonl
by hand; no further code changes needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Wrap-up

After all 5 tasks land, Phase 1c is complete. Concretely:

- `BenchResult::write_to_disk` produces `ir-<system>-<corpus>-<config>.json`. Multiple corpora can now coexist in the same output directory without overwriting each other.
- The integration smoke test references the new naming and defensively asserts the old naming is absent.
- `bench/results/main/` holds the three regenerated baselines under new naming; old-naming files are gone. Regression script round-trip exits 0.
- `bench/datasets/ir/README.md` exists with the general corpus contract.
- `bench/datasets/ir/nark-self/` exists with its methodology README and an empty corpus directory.
- `cargo run -p nark-bench -- --corpus nark-self` fails fast with a clear "queries file missing" error — exactly the right behavior since the corpus is intentionally empty.

The CI workflow continues to gate PRs on the synthetic-tiny baseline. Future Sean can drop ~150 scrubbed notes + ~50 queries into `bench/datasets/ir/nark-self/` and produce nark-self baselines without revisiting any plumbing.

## Follow-up plans (not part of Phase 1c)

- **The nark-self corpus content itself** — manual scrub work; not a code change.
- **CI inclusion of nark-self** — deferred; needs comma-separated `--corpus` support in the CLI or two separate CI workflow runs.
- **Corpus version tracking** — currently no story for "this baseline was generated against nark-self vN." Could embed a content hash in the result JSON later.
- **Phase 2 — LongMemEval + LOCOMO + LLM judge** — separate plan; substantial work.
- **Phase 3 — Python competitor adapters** (mem0, Letta, Graphiti).
- **Phase 4 — Report generator → `BENCHMARKS.md`** + README badges.
