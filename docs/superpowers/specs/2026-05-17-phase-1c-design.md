# nark Bench — Phase 1c (nark-self Scaffold + Baseline Naming Fix) Design

**Date:** 2026-05-17
**Status:** Design — awaiting user review before plan
**Author:** Sean (with Claude as collaborator)
**Parent design:** `docs/superpowers/specs/2026-05-13-nark-benchmarking-design.md`
**Predecessors:**
- Phase 1a: bench harness foundation (commits `a947e0c`, `4f85018`, `7ae7ea3`)
- Phase 1b: vector adapter + embeddings-enabled nark + code-review fixes (commits `28b128d`, `70dec03`)

---

## 1. Goal & framing

Phase 1c is the smallest possible shippable slice that unblocks the future `nark-self` corpus. The corpus itself — ~150 anonymized notes from Sean's real vault plus ~50 hand-labeled queries — is human work he does at his own pace, not part of this phase. Phase 1c ships the *infrastructure and documentation* needed so that "drop the files in and it Just Works" is literally true once the scrub is done.

Three concrete goals:

1. **Fix the baseline-naming collision.** Today `BenchResult.write_to_disk` produces filenames like `ir-fts5-default.json` — the corpus name is not part of the filename, so any second corpus would overwrite synthetic-tiny's baselines. Adding `corpus` as a filename component is a one-file change with three cascading impacts (smoke test, baseline regen).
2. **Document the corpus contract.** A general "what is a bench IR corpus" README at `bench/datasets/ir/README.md` covers directory layout, frontmatter requirements, queries.jsonl format. This is the doc anyone (Sean or future contributor) reads when adding a corpus.
3. **Scaffold the nark-self directory** with a methodology-specific README and an empty `corpus/` directory. When the scrub work happens later, files drop in without revisiting plumbing.

**Out of scope (explicit):**

- The corpus content itself (notes + queries + labels) — pure manual work.
- A scrub helper script or LLM-assist labeler — Sean chose the minimal-scope option during brainstorming.
- A `queries.jsonl` template file — deliberately omitted so that `--corpus nark-self` fails fast with "queries file missing" until real queries exist.
- CI workflow changes — Lane 1 keeps running synthetic-tiny only. nark-self runs are manual `cargo run` invocations.
- Phase 2 work (LongMemEval, LLM judge, etc.) — separate plan.

## 2. Why these design choices

**Embed `corpus` in the filename, not in `config`.** Phase 1b's deferred scope includes a `nark-body-only` ablation that would vary the `config` dimension (e.g. `config="full"` vs `config="body-only"`). Conflating corpus with config now would collapse that distinction. Keeping each field semantically distinct is future-proof.

**Filename pattern `ir-<system>-<corpus>-<config>.json`, not subdirectories.** The regression script's `ir-*.json` glob already matches the new names without modification. Subdirectories per corpus would require updating the script's iteration logic. Flat is simpler and the contract surface is smaller.

**No `queries.jsonl` template file.** An empty queries file would cause `--corpus nark-self` to silently succeed with 0 queries and produce a degenerate baseline (all metrics 0.0). Better to fail loudly with "failed to read queries file" — that's a clear "you haven't done the scrub yet" signal, not a subtle data quality issue.

**The general corpus README lives at `bench/datasets/ir/README.md`, not the nark-self README.** General contract (frontmatter requirements, queries.jsonl format, how to invoke) applies to every corpus including the existing synthetic-tiny. The nark-self README is purely methodology-specific.

## 3. Repo changes

```
bench/
├── results/main/
│   ├── ir-fts5-default.json                              # DELETE
│   ├── ir-nark-default.json                              # DELETE
│   ├── ir-vector-default.json                            # DELETE
│   ├── ir-fts5-synthetic-tiny-default.json               # CREATE (regenerated)
│   ├── ir-nark-synthetic-tiny-default.json               # CREATE (regenerated)
│   └── ir-vector-synthetic-tiny-default.json             # CREATE (regenerated)
├── src/
│   └── result.rs                                          # MODIFY write_to_disk
├── tests/
│   └── smoke.rs                                           # MODIFY filename refs + defensive assertion
└── datasets/ir/
    ├── README.md                                          # NEW — general corpus contract
    └── nark-self/                                         # NEW directory tree
        ├── README.md                                      # NEW — nark-self methodology
        └── corpus/
            └── .gitkeep                                   # NEW
```

No changes to `nark/` (the main crate). No CI workflow changes. No schema_version bump (the JSON contents are identical to Phase 1b's; only filenames change).

## 4. The naming change in detail

### Current behavior (Phase 1b)

`bench/src/result.rs::write_to_disk` constructs the filename as:

```rust
let filename = format!(
    "{}-{}-{}.json",
    sanitize_for_filename(&self.task),
    sanitize_for_filename(&self.system),
    sanitize_for_filename(&self.config),
);
```

For a run with `--corpus synthetic-tiny --systems fts5`, `self.corpus = "synthetic-tiny"` and `self.config = "default"`. Filename: `ir-fts5-default.json`. The corpus name appears nowhere in the filename, so a `--corpus nark-self` run would produce `ir-fts5-default.json` too, overwriting the synthetic-tiny baseline.

### New behavior (Phase 1c)

Add `corpus` as a fourth component:

```rust
let filename = format!(
    "{}-{}-{}-{}.json",
    sanitize_for_filename(&self.task),
    sanitize_for_filename(&self.system),
    sanitize_for_filename(&self.corpus),
    sanitize_for_filename(&self.config),
);
```

For the same `--corpus synthetic-tiny --systems fts5` run, filename becomes `ir-fts5-synthetic-tiny-default.json`. A `--corpus nark-self` run produces `ir-fts5-nark-self-default.json` — coexists peacefully.

`sanitize_for_filename` already exists from Phase 1b — replaces `/` and `MAIN_SEPARATOR` with `_`. Applies uniformly to the corpus component to defend against future corpus names with path separators.

### Cascading impacts

- **Existing baselines**: the three files at `bench/results/main/ir-{fts5,nark,vector}-default.json` need to be deleted and regenerated under the new naming. Same code, same data, same metrics — only filenames change.
- **Integration smoke test**: `bench/tests/smoke.rs` references the old filename pattern in stderr-contains assertions and result-file reads (six references total: three in the stderr loop, three in the per-system validation loop). All six update to the new pattern.
- **Defensive smoke assertion**: a new assertion verifies the *old* filename pattern is NOT produced. If `write_to_disk` ever silently reverted to the old format, this catches it:

  ```rust
  for system in ["fts5", "nark", "vector"] {
      let old = out_dir.path().join(format!("ir-{}-default.json", system));
      assert!(!old.exists(), "old filename pattern present (regression?): {}", old.display());
  }
  ```

- **Regression script**: no change. `bench/scripts/regression-check.sh` iterates `"$NEW_DIR"/ir-*.json` — the new filename pattern matches the same glob.
- **CI workflow**: no change. The benchmark step writes to `/tmp/bench-pr-out`, the script reads from there, the baseline lives in `bench/results/main/`. All paths unchanged; just different filenames inside those directories.
- **`BenchResult.corpus` field**: already populated correctly today (Phase 1a set it from `corpus_root.file_name()`). The schema doesn't change; only the filename uses it.

## 5. Documentation content

### `bench/datasets/ir/README.md` (general corpus contract)

Sections:

- **Purpose** — what an IR corpus is in this bench, why it matters for regression detection vs absolute quality measurement.
- **Directory layout** — `<corpus>/corpus/*.md` for notes, `<corpus>/queries.jsonl` for queries, `<corpus>/README.md` for methodology.
- **Frontmatter requirements** — nark requires `title, author, domain, intent, kind, status, tags` (the 7 fields; `status` must be one of `active|draft|deprecated|retracted`). Document this clearly with a copyable example block.
- **queries.jsonl format** — one JSON object per line: `{"query_id": "q01", "query": "...", "relevant": ["n01", "n02"], "class": "single_hop"}`. The `class` field is optional but produces per-class breakdowns in the result JSON.
- **How to invoke** — full `cargo run` example with all flags.
- **bench_id ↔ note tracking** — bench tracks docs by file stem; nark assigns UUIDs internally; the nark adapter maintains the mapping (transparent to corpus authors).
- **Sizing** — synthetic-tiny is 12 notes / 10 queries; nark-self target is ~150 notes / ~50 queries; both work for regression detection. Larger corpora improve statistical confidence in absolute scores.
- **Adding a new corpus** — concrete steps: create directory, drop notes in `corpus/`, write `queries.jsonl`, write a per-corpus README, run with `--corpus <name>`, commit the resulting baseline.

### `bench/datasets/ir/nark-self/README.md` (nark-self methodology)

Sections:

- **Purpose** — measure nark's hybrid pipeline against a corpus that reflects how Sean actually uses nark. Diagnostic complement to synthetic-tiny (which is intentionally synonym-heavy and favors pure embedding).
- **Source** — drawn from `~/.ark` at a documented point in time. Filter criteria: research/reference notes from the last N months, excluding obviously-sensitive domains like personal finance.
- **Anonymization checklist** — names → pseudonyms (Alice/Bob); company names → "Corp A"; URLs → redacted to domain; specific dollar amounts → "$X"; specific dates → year-month only. The bar: "would I be comfortable if this corpus were public?"
- **Sizing target** — ~150 notes, ~50 queries. Smaller is fine for v1.
- **Query construction guidance** — write queries reflecting how you actually search nark. Not crafted to test specific capabilities (that's synthetic-tiny's job). Capture real queries from session logs if possible.
- **Proposed query classes for nark-self** — `factual` (direct fact lookup), `synthesis` (combining 2+ notes), `temporal` (exercises nark's versioning), `recall` (find a half-remembered note by partial description).
- **Validation checklist** — frontmatter parses via `nark write`, queries.jsonl is valid JSONL, every `relevant` entry corresponds to a real file stem in `corpus/`.
- **CI status** — NOT in CI Lane 1. Manual `cargo run -p nark-bench -- --corpus nark-self` invocation only. CI inclusion can come later once the corpus exists and stability is established.
- **Status** — empty as of Phase 1c. Until Sean does the scrub, `cargo run -p nark-bench -- --corpus nark-self` fails with "queries file missing" — expected behavior.

### `bench/datasets/ir/nark-self/corpus/.gitkeep`

Empty file. Exists only to make git track the empty `corpus/` directory.

## 6. Testing

### Existing tests

All 25 unit tests + 1 integration smoke continue to pass after the filename change (with the smoke test updated to the new pattern, plus the new defensive assertion).

### No new unit tests

`sanitize_for_filename` is already tested implicitly via its existing uses in Phase 1b. The corpus name component is just another string passed through the same function. No new test surface.

### Baseline verification

After regenerating:

```bash
# Inspect the new baselines exist and the old ones don't
ls bench/results/main/ir-*.json
# Expect: ir-{fts5,nark,vector}-synthetic-tiny-default.json
# Do NOT expect: ir-{fts5,nark,vector}-default.json

# Confirm content matches Phase 1b numbers
jq '.ir' bench/results/main/ir-nark-synthetic-tiny-default.json
# Expect: recall_at_5 = 1.0, queries = 10, errors = []
```

Then a regression-check round-trip:

```bash
mkdir -p /tmp/p1c-verify
cargo run -p nark-bench --release -- run --task ir \
  --systems fts5,nark,vector --corpus synthetic-tiny \
  --output /tmp/p1c-verify
bench/scripts/regression-check.sh /tmp/p1c-verify
# Exit 0 expected
rm -rf /tmp/p1c-verify
```

## 7. Phasing

Five tight tasks for the implementation plan:

1. **Update `BenchResult::write_to_disk`** to embed corpus in filename. Single edit to `bench/src/result.rs`. Build clean.
2. **Update integration smoke test.** Six filename refs switched to new pattern; new defensive assertion that old-naming files are absent. Test passes.
3. **Regenerate synthetic-tiny baselines.** Delete old, run benchmark with new code, verify content matches Phase 1b numbers, regression-check exits 0.
4. **Create `bench/datasets/ir/README.md`** — the general corpus contract.
5. **Create nark-self scaffold** — `bench/datasets/ir/nark-self/README.md` (methodology) + `bench/datasets/ir/nark-self/corpus/.gitkeep`.

Each task ends with a clean build + tests passing. Tasks 1-3 affect code; tasks 4-5 are pure documentation.

## 8. What success looks like

After Phase 1c lands:

- `cargo test -p nark-bench --release --test smoke` passes with the new filenames; defensive assertion catches any future regression to old naming.
- `bench/results/main/` has the three new-naming baselines; old-naming files are gone.
- `bench/scripts/regression-check.sh` succeeds against the new baselines without modification.
- `bench/datasets/ir/README.md` exists with the general corpus contract.
- `bench/datasets/ir/nark-self/README.md` exists with the methodology Sean follows when doing the scrub.
- `bench/datasets/ir/nark-self/corpus/` exists (empty, with `.gitkeep`).
- Future Sean can drop ~150 scrubbed `.md` files into `corpus/`, hand-write `queries.jsonl`, then `cargo run -p nark-bench -- --task ir --systems fts5,nark,vector --corpus nark-self --output bench/results/main` to produce nark-self baselines — no plumbing changes required.
- `cargo run -p nark-bench -- --corpus nark-self` (run today, before scrub work happens) fails fast with "failed to read queries file" — the right signal.

## 9. Open items

- **When the corpus eventually ships** (post-Phase-1c), running it will write three new baseline files to `bench/results/main/`. The regression-check script will bootstrap them on first run. Worth re-reading the queries vs notes mapping at that time to make sure the methodology delivers the diagnostic value we want (i.e. nark's hybrid actually pulling ahead of vector on `synthesis` / `temporal` classes).
- **CI inclusion of nark-self** — deferred. Adding it to the CI workflow is a one-line change to `.github/workflows/bench-pr.yml` (`--systems fts5,nark,vector --corpus synthetic-tiny,nark-self`) but the CLI doesn't currently support comma-separated corpus values. Cross that bridge when nark-self has stable baselines worth gating on.
- **Corpus version tracking** — nark-self will evolve (notes added/removed over time). No story yet for "this baseline was generated against nark-self vN." A future improvement could embed a corpus content hash in the result JSON. Not needed for Phase 1c.
